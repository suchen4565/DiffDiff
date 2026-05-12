import math

import lightning as L
import torch
import torch.nn.functional as F

import src.backbone


class DiffDiff(L.LightningModule):
    """Differencing Diffusion for Time Series Forecasting.

    Non-isotropic forward process:
        x_t = sqrt(alpha_bar_t) * A_t @ x_0 + sqrt(1 - alpha_bar_t) * eps
    where A_t = (1 - lambda_t) * I + lambda_t * D_2, D_2 is second-order differencing.

    When lambda_max=0, degenerates to standard DDPM.
    """

    def __init__(
        self,
        backbone_config: dict,
        ns_path: str,
        norm=True,
        lr=2e-4,
        alpha=1e-5,
        lambda_max=0.5,
        lambda_schedule="linear",
        d2_normalize=False,
        diff_order=2,
        label_len=0,
        loss_weight_future=1.0,
        norm_source="target",
        label_replace="none",
        lr_sched="constant",          # "constant" | "cosine" | "warmup_cosine"
        warmup_ratio=0.0,             # fraction of total steps used for linear warmup
        min_lr_ratio=0.01,            # cosine floor as fraction of base lr
        max_steps=None,               # set by Lightning at fit time; needed for cosine
        **kwargs,
    ) -> None:
        super().__init__()
        bb_class = getattr(src.backbone, backbone_config["name"])
        self.backbone = bb_class(**backbone_config)
        self.seq_length = self.backbone.seq_length
        self.norm = norm
        self.lr = lr
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.lambda_schedule = lambda_schedule
        self.d2_normalize = d2_normalize
        self.diff_order = diff_order
        self.label_len = label_len
        self.loss_weight_future = loss_weight_future
        self.norm_source = norm_source
        self.label_replace = label_replace
        self.lr_sched = lr_sched
        self.warmup_ratio = warmup_ratio
        self.min_lr_ratio = min_lr_ratio
        self.loss_fn = F.mse_loss

        noise_schedule = torch.load(ns_path)
        self.register_buffer("alpha_bars", noise_schedule["alpha_bars"])
        self.register_buffer("betas", noise_schedule["betas"])
        self.register_buffer("sigmas", torch.zeros(len(self.alpha_bars)))
        self.T = len(self.alpha_bars)

        # Build transition matrices A_t = (1-λ)I + λD
        if lambda_max > 0:
            D = self._build_diff_matrix(self.seq_length, order=diff_order, normalize=d2_normalize)
            lambdas = self._build_lambda_schedule(lambda_max, lambda_schedule, self.T)
            I = torch.eye(self.seq_length)
            self.register_buffer("A", (1 - lambdas) * I + lambdas * D)

        # Pre-compute loss weights for label window
        if label_len > 0 and loss_weight_future > 1.0:
            w = torch.ones(1, self.seq_length, 1)
            w[:, label_len:, :] = loss_weight_future
            self.register_buffer("loss_weights", w)

        self.save_hyperparameters()

    @staticmethod
    def _build_diff_matrix(seq_len, order=2, normalize=False):
        """Build differencing matrix D of a given order (S x S).

        Structure for any order:
          Row 0: all zeros (erases level info).
          Row k (1 <= k < order): k-th order diff coefficients.
          Row k (k >= order): full `order`-th order diff coefficients.

        order=1: D_1 — Row 1+: [-1, 1]              (velocity)
        order=2: D_2 — Row 1: [-1,1], Row 2+: [1,-2,1]  (default, acceleration)
        order=3: D_3 — Row 1: [-1,1], Row 2: [1,-2,1], Row 3+: [-1,3,-3,1] (jerk)
        """
        D = torch.zeros(seq_len, seq_len)
        for row in range(1, seq_len):
            k = min(row, order)
            for j in range(k + 1):
                col = row - k + j
                if 0 <= col < seq_len:
                    D[row, col] = ((-1) ** (k - j)) * math.comb(k, j)
        if normalize:
            _, S, _ = torch.linalg.svd(D)
            D = D / S[0]
        return D

    @staticmethod
    def _build_D2(seq_len, normalize=False):
        """Build second-order differencing matrix. Kept for backward compatibility."""
        return DiffDiff._build_diff_matrix(seq_len, order=2, normalize=normalize)

    @staticmethod
    def _build_lambda_schedule(lambda_max, schedule, T):
        """Build lambda schedule for differencing interpolation.

        Returns shape (T, 1, 1) for broadcasting with A_t matrices.
        """
        t_frac = torch.linspace(0, 1, T)
        if schedule == "linear":
            lambdas = lambda_max * t_frac
        elif schedule == "cosine":
            lambdas = lambda_max * (1 - torch.cos(math.pi * t_frac)) / 2
        elif schedule == "diff_first":
            lambdas = lambda_max * torch.clamp(2 * t_frac, max=1.0)
        elif schedule == "sqrt":
            lambdas = lambda_max * torch.sqrt(t_frac)
        else:
            raise ValueError(f"Unknown lambda schedule: {schedule}")
        return lambdas.view(-1, 1, 1)

    def on_load_checkpoint(self, checkpoint):
        """Handle backward-compatible key remapping (gate → gate_net)."""
        sd = checkpoint.get("state_dict", {})
        remapped = {}
        for k, v in sd.items():
            if ".cond_embed.gate." in k and ".cond_embed.gate_net." not in k:
                remapped[k.replace(".cond_embed.gate.", ".cond_embed.gate_net.")] = v
            else:
                remapped[k] = v
        checkpoint["state_dict"] = remapped

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.alpha)
        if self.lr_sched == "constant":
            return opt

        # Steps-per-epoch × max_epochs (works under Lightning 2.x).
        # estimated_stepping_batches accounts for early stopping budget.
        try:
            total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            total_steps = 10000  # safe default if trainer not yet attached
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))

        if self.lr_sched == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=total_steps, eta_min=self.lr * self.min_lr_ratio,
            )
        elif self.lr_sched == "warmup_cosine":
            warm = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps,
            )
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, total_steps - warmup_steps),
                eta_min=self.lr * self.min_lr_ratio,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warm, cos], milestones=[warmup_steps],
            )
        else:
            raise ValueError(f"Unknown lr_sched: {self.lr_sched}")

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step", "frequency": 1},
        }

    @torch.no_grad()
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        """Forward process: x_t = sqrt(ab) * A_t @ x + sqrt(1-ab) * eps"""
        ab = self.alpha_bars[t].view(-1, 1, 1)
        if self.lambda_max > 0:
            x_signal = self.A[t] @ x
        else:
            x_signal = x
        return torch.sqrt(ab) * x_signal + torch.sqrt(1 - ab) * torch.randn_like(x)

    def _norm_step(self, x, cond=None):
        """Normalize x. If norm_source='condition', use condition stats (no future leakage)."""
        if not self.norm:
            return x, (None, None)
        if self.norm_source == "condition" and cond is not None:
            c_mean = cond.mean(dim=1, keepdim=True)
            c_std = torch.sqrt(cond.var(dim=1, keepdim=True, unbiased=False) + 1e-6)
            return (x - c_mean) / c_std, (c_mean, c_std)
        return self._normalize(x)

    def training_step(self, batch, batch_idx):
        x = batch.pop("x")
        cond = batch.get("c", None)
        x, (x_mean, x_std) = self._norm_step(x, cond)
        loss, mean_loss, std_loss = self._get_loss(x, batch, x_mean, x_std)
        total_loss = loss + mean_loss + std_loss
        # Gate L1 regularization (AdaptiveConditionEmbed / MultiOrderConditionEmbed)
        gate_loss = 0
        if hasattr(self.backbone, "cond_embed") and hasattr(self.backbone.cond_embed, "gate_l1_weight"):
            w = self.backbone.cond_embed.gate_l1_weight
            if w > 0:
                for attr in ("_last_gate", "_last_gate1", "_last_gate2"):
                    g = getattr(self.backbone.cond_embed, attr, None)
                    if g is not None:
                        gate_loss = gate_loss + w * g.mean()
                total_loss = total_loss + gate_loss
        log_dict = {
            "recon_loss": loss,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "gate_loss": gate_loss,
            "loss": total_loss,
        }
        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return log_dict

    def validation_step(self, batch, batch_idx):
        x = batch.pop("x")
        cond = batch.get("c", None)
        x, (x_mean, x_std) = self._norm_step(x, cond)
        loss, mean_loss, std_loss = self._get_loss(x, batch, x_mean, x_std)
        log_dict = {
            "recon_loss": loss,
            "mean_loss": mean_loss,
            "std_loss": std_loss,
            "val_loss": loss + mean_loss + std_loss,
        }
        self.log_dict(
            log_dict, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0]
        )
        return log_dict

    def predict_step(self, batch, batch_idx):
        assert self._sample_ready
        x_real = batch.pop("x")
        cond = batch.get("c", None)

        # Prepare label replacement if enabled
        label_norm = None
        cond_mu, cond_std = None, None
        if self.label_len > 0 and self.label_replace != "none" and \
           self.condition == "fcst" and cond is not None:
            label_raw = x_real[:, : self.label_len, :]
            if self.label_replace == "x0_cond" or self.norm_source == "condition":
                # Use condition stats (consistent at train & inference time)
                cond_mu = cond.mean(dim=1, keepdim=True)
                cond_std = torch.sqrt(
                    cond.var(dim=1, keepdim=True, unbiased=False) + 1e-6
                )
                label_norm = (label_raw - cond_mu) / cond_std

        all_sample_x = []
        for _ in range(self.n_sample):
            x = self._init_noise(x_real, cond)
            x, mu, std = self._sample_loop(x, cond, label_norm=label_norm)

            if (cond is None) or (self.condition == "sr"):
                mu, std = self.init_mean, self.init_std
            elif self.norm_source == "condition" and cond_mu is not None:
                mu, std = cond_mu, cond_std
            else:
                if not self.norm:
                    mu, std = 0, 1

            out_x = x * std + mu
            assert out_x.shape == x_real.shape
            all_sample_x.append(out_x)

        all_sample_x = torch.stack(all_sample_x)
        return all_sample_x

    def _sample_loop(self, x: torch.Tensor, c: torch.Tensor, label_norm=None):
        for i in range(len(self.sample_Ts)):
            t = self.sample_Ts[i]
            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            t_tensor = torch.tensor(t, device=x.device).expand(x.shape[0])

            if (c is None) or (self.condition == "sr"):
                x_hat, x_mean, x_std = self.backbone(x, t_tensor, None, train=False)
            else:
                # Classifier-free guidance
                x_concat = torch.concat([x, x], dim=0)
                c_null = torch.zeros_like(c)
                c_concat = torch.concat([c, c_null], dim=0)
                t_concat = torch.concat([t_tensor, t_tensor], dim=0)

                x_hat, x_mean, x_std = self.backbone(
                    x_concat, t_concat, c_concat, train=False
                )

                cond_x_hat, uncond_x_hat = torch.split(x_hat, len(x_hat) // 2, dim=0)
                cond_x_mean, uncond_x_mean = torch.split(x_mean, len(x_hat) // 2, dim=0)
                cond_x_std, uncond_x_std = torch.split(x_std, len(x_hat) // 2, dim=0)

                x_hat = self.w_cond * cond_x_hat + (1 - self.w_cond) * uncond_x_hat
                x_mean = self.w_cond * cond_x_mean + (1 - self.w_cond) * uncond_x_mean
                x_std = self.w_cond * cond_x_std + (1 - self.w_cond) * uncond_x_std

            # E1a: Replace label in x̂_0 space using condition stats
            if label_norm is not None and self.label_replace in ("x0_cond",):
                x_hat = x_hat.clone()
                x_hat[:, : self.label_len, :] = label_norm

            x = self.reverse(x=x, x_hat=x_hat, t=t, prev_t=prev_t)

            # E1b (repaint): Replace label in x_t space after reverse step
            if label_norm is not None and self.label_replace == "repaint" and \
               prev_t is not None and prev_t > 0:
                ab_prev = self.alpha_bars[prev_t]
                # Degrade known label to timestep prev_t via forward process
                label_full = x.clone()
                label_full[:, : self.label_len, :] = label_norm[:, :, :]
                if self.lambda_max > 0:
                    Aprev_label = (self.A[prev_t] @ label_full)[:, : self.label_len, :]
                else:
                    Aprev_label = label_norm
                x_label_t = (
                    torch.sqrt(ab_prev) * Aprev_label
                    + torch.sqrt(1 - ab_prev) * torch.randn_like(Aprev_label)
                )
                x[:, : self.label_len, :] = x_label_t

        return x, x_mean, x_std

    def _get_loss(self, x, condition: dict = None, x_mean=None, x_std=None):
        batch_size = x.shape[0]
        cond = condition.get("c", None)

        t = torch.randint(0, self.T, (batch_size,)).to(x.device)
        x_noisy = self.degrade(x, t)

        x_hat, x_mean_hat, x_std_hat = self.backbone(x_noisy, t, cond, train=True)

        if hasattr(self, "loss_weights"):
            loss = (self.loss_weights * (x_hat - x) ** 2).mean()
        else:
            loss = self.loss_fn(x_hat, x)

        if (x_mean is not None) and (cond is not None):
            mean_loss = F.mse_loss(x_mean_hat, x_mean)
        else:
            mean_loss = 0

        if (x_std is not None) and (cond is not None):
            std_loss = F.mse_loss(x_std_hat, x_std)
        else:
            std_loss = 0

        return loss, mean_loss, std_loss

    def _init_noise(self, x: torch.Tensor, condition: torch.Tensor = None):
        first_step = self.sample_Ts[0]

        if self.condition is None:
            if self.norm:
                x_norm, (self.init_mean, self.init_std) = self._normalize(x)
            else:
                x_norm, self.init_mean, self.init_std = x, 0, 1
            t = torch.ones((x.shape[0],), device=x.device, dtype=torch.int) * first_step
            x_T = self.degrade(x_norm, t)

        elif self.condition == "fcst":
            assert condition is not None
            # Terminal distribution is N(0, I) regardless of A_t
            # (alpha_bar_T -> 0 erases the signal)
            x_T = torch.randn_like(x)

        elif self.condition == "sr":
            assert condition is not None
            if self.norm:
                x_norm, (self.init_mean, self.init_std) = self._normalize(condition)
            else:
                x_norm, self.init_mean, self.init_std = condition, 0, 1
            ab = self.alpha_bars[first_step]
            x_T = x_norm + torch.sqrt(1 - ab) * torch.randn_like(condition)

        return x_T

    def config_sampling(
        self,
        n_sample: int = 1,
        w_cond: float = 1,
        sigmas: torch.Tensor = None,
        sample_steps=None,
        condition=None,
        **kwargs,
    ):
        assert condition in ["fcst", "sr", None]
        self.condition = condition
        self.n_sample = n_sample
        self.w_cond = w_cond
        self.sample_Ts = list(range(self.T)) if sample_steps is None else sample_steps
        self.sample_Ts.sort(reverse=True)
        if sigmas is not None:
            self.sigmas.copy_(sigmas)
        else:
            self.sigmas.zero_()

        # Validate step ordering
        for i in range(len(self.sample_Ts)):
            t = self.sample_Ts[i]
            prev_t = None if t == 0 else self.sample_Ts[i + 1]
            if t != 0:
                assert t > prev_t

        self._sample_ready = True
        print("Config success")

    def _normalize(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-6)
        x_norm = (x - mean) / stdev
        return x_norm, (mean, stdev)

    def reverse(self, x, x_hat, t, prev_t):
        """DDIM reverse step with A_t transition matrices.

        x_{t-1} = sqrt(ab_prev) * A_{prev} @ x_hat
                 + sqrt((1-ab_prev-sigma^2)/(1-ab_t)) * (x - sqrt(ab_t) * A_t @ x_hat)
                 + sigma * eps
        """
        if t == 0:
            return x_hat

        ab_t = self.alpha_bars[t]
        ab_prev = self.alpha_bars[prev_t]
        sigma = self.sigmas[t]

        # Compute A_t @ x_hat and A_{prev} @ x_hat
        if self.lambda_max > 0:
            At_x_hat = self.A[t] @ x_hat
            Aprev_x_hat = self.A[prev_t] @ x_hat
        else:
            At_x_hat = x_hat
            Aprev_x_hat = x_hat

        # Predicted noise
        eps_hat = (x - torch.sqrt(ab_t) * At_x_hat) / torch.sqrt(1 - ab_t)

        # DDIM step (sigma=0 for deterministic, >0 for stochastic)
        dir_coeff = torch.sqrt(torch.clamp(1 - ab_prev - sigma**2, min=1e-8))
        x_prev = (
            torch.sqrt(ab_prev) * Aprev_x_hat
            + dir_coeff * eps_hat
            + sigma * torch.randn_like(x)
        )
        return x_prev

import sys

import torch
from torch import nn
from torchvision.ops import MLP
from ..layers.revin import RevIN
from ..layers.Embed import SinusoidalPosEmb

thismodule = sys.modules[__name__]


# TODO: build backbone function
def build_backbone(bb_config):
    bb_config_c = bb_config.copy()
    bb_net = getattr(thismodule, bb_config_c.pop("name"))
    return bb_net(**bb_config_c)


class ConditionEmbed(nn.Module):
    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.norm = norm
        # self.embedder = nn.Linear(seq_length, hidden_size)
        if norm:
            self.rev = RevIN(seq_channels)
        self.input_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.pred_dec = torch.nn.Linear(hidden_size, latent_dim)
        self.dropout_prob = cond_dropout_prob

        # self.mean_dec = torch.nn.Linear(target_seq_length, 1)
        # self.std_dec = torch.nn.Linear(target_seq_length, 1)

        self.seq_channels = seq_channels

    # def forward(self, observed_data, **kwargs):

    #     if self.norm:
    #         x_norm = self.rev(observed_data, "norm")
    #     else:
    #         x_norm = observed_data

    #     latents = self.input_enc(x_norm.permute(0, 2, 1))
    #     latents = self.pred_dec(latents).permute(0, 2, 1)

    #     # if self.norm:
    #     # latents = self.rev(latents, "denorm")

    #     return latents
    #     # mean_pred = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
    #     # std_pred = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)

    #     # return {"latents": latents, "mean_pred": mean_pred, "std_pred": std_pred}

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, force_drop_ids=None, **kwargs):
        use_dropout = self.dropout_prob > 0

        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        latents = self.input_enc(x_norm.permute(0, 2, 1))
        latents = self.pred_dec(latents).permute(0, 2, 1)

        # if self.norm:
        # latents = self.rev(latents, "denorm")

        return latents
        # mean_pred = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
        # std_pred = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)

        # return {"latents": latents, "mean_pred": mean_pred, "std_pred": std_pred}


class DualBranchConditionEmbed(nn.Module):
    """Dual-branch history encoder: static (values) + dynamic (differences).

    Mirrors the forward process's information separation by encoding level/trend
    and temporal changes through separate pathways, merged with a t-dependent gate.
    """

    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = norm
        self.dropout_prob = cond_dropout_prob
        self.seq_channels = seq_channels

        if norm:
            self.rev = RevIN(seq_channels)

        # Static branch: encodes raw history (level/trend)
        self.static_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.static_proj = nn.Linear(hidden_size, latent_dim)

        # Dynamic branch: encodes first-order differences (changes/dynamics)
        self.dynamic_enc = MLP(
            in_channels=seq_length - 1,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.dynamic_proj = nn.Linear(hidden_size, latent_dim)

        # t-dependent merge gate: sigmoid output controls dynamic vs static weight
        self.merge_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, t_emb=None, force_drop_ids=None, **kwargs):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        x_perm = x_norm.permute(0, 2, 1)  # (B, C, S)

        # Static branch: raw values
        static_h = self.static_proj(self.static_enc(x_perm))  # (B, C, d_model)

        # Dynamic branch: first-order differences
        x_diff = x_perm[:, :, 1:] - x_perm[:, :, :-1]  # (B, C, S-1)
        dynamic_h = self.dynamic_proj(self.dynamic_enc(x_diff))  # (B, C, d_model)

        # Merge with t-dependent gate
        if t_emb is not None:
            # t_emb: (B, 1, d_model) — squeeze to (B, d_model) for gate, then expand
            gate = self.merge_gate(t_emb.squeeze(1)).unsqueeze(1)  # (B, 1, d_model)
            latent = gate * dynamic_h + (1 - gate) * static_h
        else:
            latent = (static_h + dynamic_h) / 2

        # Return (B, d_model, C) to match ConditionEmbed's output convention
        return latent.permute(0, 2, 1)


class ConcatConditionEmbed(nn.Module):
    """Concat-fusion ablation variant of AdaptiveConditionEmbed.

    Identical encoders (main + diff) but the gating mechanism is removed:
    fusion is performed by concatenating the two branches and projecting
    through a small MLP. Tests whether the stage-adaptive sigmoid gate is
    necessary or whether simply increasing model capacity (the doubled feature
    dimension after concat) explains DiffDiff's gain.
    """

    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
        **_kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = norm
        self.dropout_prob = cond_dropout_prob
        self.seq_channels = seq_channels

        if norm:
            self.rev = RevIN(seq_channels)

        # Main path: same as ConditionEmbed
        self.input_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.pred_dec = nn.Linear(hidden_size, latent_dim)

        # Diff residual path (lightweight: half hidden size, same as Adaptive)
        diff_hidden = hidden_size // 2
        self.diff_enc = MLP(
            in_channels=seq_length - 1,
            hidden_channels=[diff_hidden, diff_hidden],
        )
        self.diff_proj = nn.Linear(diff_hidden, latent_dim)

        # Concat-fusion MLP: 2d -> d -> d (replaces sigmoid gate)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, t_emb=None, force_drop_ids=None, **kwargs):
        # t_emb is accepted for API compatibility but unused (no gating)
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        x_perm = x_norm.permute(0, 2, 1)  # (B, C, S)

        # Main path
        h_main = self.pred_dec(self.input_enc(x_perm))  # (B, C, d_model)

        # Diff residual path
        x_diff = x_perm[:, :, 1:] - x_perm[:, :, :-1]  # (B, C, S-1)
        h_diff = self.diff_proj(self.diff_enc(x_diff))  # (B, C, d_model)

        # Concat-fusion (no gate)
        fused = torch.cat([h_main, h_diff], dim=-1)  # (B, C, 2*d_model)
        latent = self.fuse_mlp(fused)                # (B, C, d_model)

        return latent.permute(0, 2, 1)  # (B, d_model, C) match ConditionEmbed convention


class AdaptiveConditionEmbed(nn.Module):
    """Adaptive residual conditioning: main encoder + data-aware diff residual.

    The main path is identical to ConditionEmbed (backward compatible).
    A lightweight differencing residual is gated by BOTH timestep and data content,
    allowing the model to learn per-sample when dynamics separation is beneficial.

    - Non-stationary data + low noise: gate → high (use diff branch, like E7)
    - Stationary data + high noise: gate → low (standard encoder)
    """

    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
        gate_init_bias=-2.0,
        gate_l1_weight=0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = norm
        self.dropout_prob = cond_dropout_prob
        self.seq_channels = seq_channels
        self.gate_l1_weight = gate_l1_weight

        if norm:
            self.rev = RevIN(seq_channels)

        # Main path: same as ConditionEmbed
        self.input_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.pred_dec = nn.Linear(hidden_size, latent_dim)

        # Diff residual path (lightweight: half hidden size)
        diff_hidden = hidden_size // 2
        self.diff_enc = MLP(
            in_channels=seq_length - 1,
            hidden_channels=[diff_hidden, diff_hidden],
        )
        self.diff_proj = nn.Linear(diff_hidden, latent_dim)

        # Data-aware gate: conditioned on BOTH t_emb and data summary
        self.gate_net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.gate_net[-2].bias, gate_init_bias)

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, t_emb=None, force_drop_ids=None, **kwargs):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        x_perm = x_norm.permute(0, 2, 1)  # (B, C, S)

        # Main path
        h_main = self.pred_dec(self.input_enc(x_perm))  # (B, C, d_model)

        # Diff residual path
        x_diff = x_perm[:, :, 1:] - x_perm[:, :, :-1]  # (B, C, S-1)
        h_diff = self.diff_proj(self.diff_enc(x_diff))  # (B, C, d_model)

        # Data-aware gate: [t_emb, data_summary] → gate
        if t_emb is not None:
            data_summary = h_main.mean(dim=1)  # (B, d_model)
            t_pool = t_emb.squeeze(1)  # (B, d_model)
            gate_input = torch.cat([t_pool, data_summary], dim=-1)  # (B, 2*d_model)
            gate = self.gate_net(gate_input).unsqueeze(1)  # (B, 1, d_model)
            self._last_gate = gate  # store for regularization / analysis
            latent = h_main + gate * h_diff
        else:
            self._last_gate = None
            latent = h_main

        # Return (B, d_model, C) to match ConditionEmbed's output convention
        return latent.permute(0, 2, 1)


class MultiOrderConditionEmbed(nn.Module):
    """Multi-order conditioning: main encoder + 1st-order diff + 2nd-order diff,
    each independently gated.

    Aligns conditioning structure with the DiffDiff forward process D_2 matrix:
    - Main path: encodes raw values (level/trend)
    - 1st-order path: encodes velocity (analogous to D_2 row 1)
    - 2nd-order path: encodes acceleration (analogous to D_2 rows 2+)
    """

    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
        gate_init_bias=-2.0,
        gate_l1_weight=0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = norm
        self.dropout_prob = cond_dropout_prob
        self.seq_channels = seq_channels
        self.gate_l1_weight = gate_l1_weight

        if norm:
            self.rev = RevIN(seq_channels)

        # Main path: same as ConditionEmbed
        self.input_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.pred_dec = nn.Linear(hidden_size, latent_dim)

        # 1st-order diff path (lightweight)
        diff_hidden = hidden_size // 2
        self.diff1_enc = MLP(
            in_channels=seq_length - 1,
            hidden_channels=[diff_hidden, diff_hidden],
        )
        self.diff1_proj = nn.Linear(diff_hidden, latent_dim)

        # 2nd-order diff path (lightweight)
        self.diff2_enc = MLP(
            in_channels=seq_length - 2,
            hidden_channels=[diff_hidden, diff_hidden],
        )
        self.diff2_proj = nn.Linear(diff_hidden, latent_dim)

        # Independent gates for each order
        def _make_gate(bias):
            net = nn.Sequential(
                nn.Linear(latent_dim * 2, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.Sigmoid(),
            )
            nn.init.constant_(net[-2].bias, bias)
            return net

        self.gate1_net = _make_gate(gate_init_bias)
        self.gate2_net = _make_gate(gate_init_bias)

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, t_emb=None, force_drop_ids=None, **kwargs):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        x_perm = x_norm.permute(0, 2, 1)  # (B, C, S)

        # Main path
        h_main = self.pred_dec(self.input_enc(x_perm))  # (B, C, d_model)

        # 1st-order diff: x[t] - x[t-1]
        x_diff1 = x_perm[:, :, 1:] - x_perm[:, :, :-1]  # (B, C, S-1)
        h_diff1 = self.diff1_proj(self.diff1_enc(x_diff1))  # (B, C, d_model)

        # 2nd-order diff: diff of diff
        x_diff2 = x_diff1[:, :, 1:] - x_diff1[:, :, :-1]  # (B, C, S-2)
        h_diff2 = self.diff2_proj(self.diff2_enc(x_diff2))  # (B, C, d_model)

        if t_emb is not None:
            data_summary = h_main.mean(dim=1)  # (B, d_model)
            t_pool = t_emb.squeeze(1)  # (B, d_model)
            gate_input = torch.cat([t_pool, data_summary], dim=-1)  # (B, 2*d_model)

            gate1 = self.gate1_net(gate_input).unsqueeze(1)  # (B, 1, d_model)
            gate2 = self.gate2_net(gate_input).unsqueeze(1)  # (B, 1, d_model)

            self._last_gate1 = gate1
            self._last_gate2 = gate2
            latent = h_main + gate1 * h_diff1 + gate2 * h_diff2
        else:
            self._last_gate1 = None
            self._last_gate2 = None
            latent = h_main

        return latent.permute(0, 2, 1)  # (B, d_model, C)


class CrossAttnConditionEmbed(nn.Module):
    """Cross-attention conditioning: timestep-conditioned query attends to history views.

    Unlike AdaptiveConditionEmbed (sigmoid gate, additive residual), this uses
    softmax attention over [raw, diff] view tokens. The convex combination
    constraint (weights sum to 1) means the model CANNOT add noise — it must
    CHOOSE between views, making it inherently safer on smooth data.
    """

    def __init__(
        self,
        seq_channels,
        seq_length,
        hidden_size,
        latent_dim,
        cond_dropout_prob=0.5,
        norm=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = norm
        self.dropout_prob = cond_dropout_prob
        self.seq_channels = seq_channels

        if norm:
            self.rev = RevIN(seq_channels)

        # View 1: raw history encoding
        self.raw_enc = MLP(
            in_channels=seq_length,
            hidden_channels=[hidden_size, hidden_size],
        )
        self.raw_proj = nn.Linear(hidden_size, latent_dim)

        # View 2: diff history encoding (lightweight)
        diff_hidden = hidden_size // 2
        self.diff_enc = MLP(
            in_channels=seq_length - 1,
            hidden_channels=[diff_hidden, diff_hidden],
        )
        self.diff_proj = nn.Linear(diff_hidden, latent_dim)

        # Data-aware query projection: combine t_emb with data summary
        self.query_proj = nn.Linear(latent_dim * 2, latent_dim)

        # Cross-attention (single head for 2 tokens; multi-head is overkill)
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads=1, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(latent_dim)

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        self.drop_ids = drop_ids.unsqueeze(1).unsqueeze(1)
        labels = torch.where(self.drop_ids, torch.zeros_like(labels), labels)
        return labels

    def forward(self, observed_data, train, t_emb=None, force_drop_ids=None, **kwargs):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            observed_data = self.token_drop(observed_data, force_drop_ids)

        if self.norm:
            x_norm = self.rev(observed_data, "norm")
        else:
            x_norm = observed_data

        x_perm = x_norm.permute(0, 2, 1)  # (B, C, S)

        # Encode two views
        h_raw = self.raw_proj(self.raw_enc(x_perm))  # (B, C, d_model)
        x_diff = x_perm[:, :, 1:] - x_perm[:, :, :-1]  # (B, C, S-1)
        h_diff = self.diff_proj(self.diff_enc(x_diff))  # (B, C, d_model)

        if t_emb is not None:
            B, C, D = h_raw.shape
            # Pool views over channels → 2 view tokens
            views = torch.stack(
                [h_raw.mean(dim=1), h_diff.mean(dim=1)], dim=1
            )  # (B, 2, d_model)

            # Data-aware query: [t_emb, raw_summary]
            data_summary = h_raw.mean(dim=1)  # (B, d_model)
            t_pool = t_emb.squeeze(1)  # (B, d_model)
            query = self.query_proj(
                torch.cat([t_pool, data_summary], dim=-1)
            ).unsqueeze(1)  # (B, 1, d_model)

            # Cross-attention: query selects from views
            attn_out, attn_weights = self.cross_attn(query, views, views)
            # attn_out: (B, 1, d_model), attn_weights: (B, 1, 2)
            self._attn_weights = attn_weights.detach()

            # Combine: raw encoding + attention-weighted context
            context = self.attn_norm(attn_out)  # (B, 1, d_model)
            latent = h_raw + context.expand(-1, C, -1)
        else:
            self._attn_weights = None
            latent = h_raw

        # Return (B, d_model, C) to match ConditionEmbed's output convention
        return latent.permute(0, 2, 1)


class MLPBackbone(nn.Module):
    """
    ## MLP backbone for denoising a time-domian data
    """

    def __init__(
        self,
        seq_channels: int,
        seq_length: int,
        d_model: int,
        d_mlp: int,
        # hidden_size: int,
        n_layers: int = 3,
        dropout: float = 0.1,
        cond_dropout_prob=0.5,
        cond_seq_len=None,
        cond_seq_chnl=None,
        norm=True,
        freq_denoise=False,
        t_aware_cond=False,
        dual_branch_cond=False,
        adaptive_cond=False,
        crossattn_cond=False,
        multi_order_cond=False,
        concat_cond=False,
        gate_init_bias=-2.0,
        gate_l1_weight=0.0,
        **kwargs,
    ) -> None:
        """
        * `seq_channels` is the number of channels in the time series. $1$ for uni-variable.
        * `seq_length` is the number of timesteps in the time series.
        * `hidden_size` is the hidden size
        * `n_layers` is the number of MLP
        """
        super().__init__()
        self.dropout_prob = cond_dropout_prob
        self.freq_denoise = freq_denoise
        self.norm = norm
        self.t_aware_cond = t_aware_cond
        self.dual_branch_cond = dual_branch_cond
        self.adaptive_cond = adaptive_cond
        self.crossattn_cond = crossattn_cond
        self.multi_order_cond = multi_order_cond
        self.concat_cond = concat_cond

        # condition
        # TODO: uncond
        if (cond_seq_chnl is not None) and (cond_seq_len is not None):
            if multi_order_cond:
                self.cond_embed = MultiOrderConditionEmbed(
                    seq_channels=cond_seq_chnl,
                    seq_length=cond_seq_len,
                    hidden_size=d_mlp // 2,
                    latent_dim=d_model,
                    norm=norm,
                    cond_dropout_prob=cond_dropout_prob,
                    gate_init_bias=gate_init_bias,
                    gate_l1_weight=gate_l1_weight,
                )
            elif crossattn_cond:
                self.cond_embed = CrossAttnConditionEmbed(
                    seq_channels=cond_seq_chnl,
                    seq_length=cond_seq_len,
                    hidden_size=d_mlp // 2,
                    latent_dim=d_model,
                    norm=norm,
                    cond_dropout_prob=cond_dropout_prob,
                )
            elif adaptive_cond:
                self.cond_embed = AdaptiveConditionEmbed(
                    seq_channels=cond_seq_chnl,
                    seq_length=cond_seq_len,
                    hidden_size=d_mlp // 2,
                    latent_dim=d_model,
                    norm=norm,
                    cond_dropout_prob=cond_dropout_prob,
                    gate_init_bias=gate_init_bias,
                    gate_l1_weight=gate_l1_weight,
                )
            elif concat_cond:
                self.cond_embed = ConcatConditionEmbed(
                    seq_channels=cond_seq_chnl,
                    seq_length=cond_seq_len,
                    hidden_size=d_mlp // 2,
                    latent_dim=d_model,
                    norm=norm,
                    cond_dropout_prob=cond_dropout_prob,
                )
            elif dual_branch_cond:
                self.cond_embed = DualBranchConditionEmbed(
                    seq_channels=cond_seq_chnl,
                    seq_length=cond_seq_len,
                    hidden_size=d_mlp // 2,
                    latent_dim=d_model,
                    norm=norm,
                    cond_dropout_prob=cond_dropout_prob,
                )
            else:
                self.cond_embed = ConditionEmbed(
                    seq_channels=cond_seq_chnl,
                    seq_length=cond_seq_len,
                    hidden_size=d_mlp // 2,
                    latent_dim=d_model,
                    norm=norm,
                    cond_dropout_prob=cond_dropout_prob,
                )
            self.pred_dec = torch.nn.Linear(d_model, seq_length)
            self.mean_dec = torch.nn.Linear(seq_length, 1)
            self.std_dec = torch.nn.Linear(seq_length, 1)

        # Timestep-aware conditioning: AdaLN-style modulation of condition embedding
        if t_aware_cond:
            self.cond_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 2 * d_model),
            )

        # self.freq_denoise = freq_denoise
        self.embedder = nn.Linear(seq_length, d_model)
        self.unembedder = nn.Linear(d_model, seq_length)
        self.pe = SinusoidalPosEmb(d_model)
        self.net = nn.ModuleList(
            [
                MLP(
                    in_channels=d_model,
                    hidden_channels=[d_mlp, d_model],
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.seq_channels = seq_channels
        self.seq_length = seq_length

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor = None,
        train=True,
        force_drop_ids=None,
    ):
        # use_dropout = self.dropout_prob > 0
        # if (train and use_dropout) or (force_drop_ids is not None):
        #     condition = self.token_drop(condition, force_drop_ids)

        input_freq = False
        if torch.is_complex(x):
            input_freq = True
            fft_len = x.shape[1]
            x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        elif self.freq_denoise:
            x = torch.fft.rfft(x, dim=1, norm="ortho")
            fft_len = x.shape[1]
            x = torch.concat([x.real, x.imag[:, 1:-1, :]], dim=1)
        x = self.embedder(x.permute(0, 2, 1))
        t = self.pe(t).unsqueeze(1)
        if condition is not None:
            if self.multi_order_cond or self.crossattn_cond or self.adaptive_cond or self.dual_branch_cond or self.concat_cond:
                c = self.cond_embed(
                    condition, train, t_emb=t, force_drop_ids=force_drop_ids
                ).permute(0, 2, 1)
            else:
                c = self.cond_embed(condition, train, force_drop_ids).permute(0, 2, 1)
            y_pred = self.pred_dec(c).permute(0, 2, 1)
            if self.norm:
                y_pred_denorm = self.cond_embed.rev(y_pred, "denorm")
            else:
                y_pred_denorm = y_pred
            x_mean = self.mean_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
            x_std = self.std_dec(y_pred_denorm.permute(0, 2, 1)).permute(0, 2, 1)
            # Timestep-aware conditioning: modulate c with t via AdaLN
            if self.t_aware_cond:
                mod = self.cond_modulation(t)  # (B, 1, 2*d_model)
                scale, shift = mod.chunk(2, dim=-1)
                c = c * (1 + scale) + shift
            x = x + (t + c)
        else:
            x_mean, x_std = (
                torch.zeros_like(x)[:, [0], :],
                torch.ones_like(x)[:, [0], :],
            )
            y_pred = 0
            x = x + t
        for layer in self.net:
            x = x + layer(x)

        x = self.unembedder(x).permute(0, 2, 1)

        if self.freq_denoise or input_freq:
            # ! for test
            x_re = x[:, :fft_len, :]
            x_im = torch.concat(
                [
                    torch.zeros_like(x[:, [0], :]),
                    x[:, fft_len:, :],
                    torch.zeros_like(x[:, [0], :]),
                ],
                dim=1,
            )
            x = torch.stack([x_re, x_im], dim=-1)
            x = torch.view_as_complex(x)
        if self.freq_denoise:
            x = torch.fft.irfft(x, dim=1, norm="ortho")
        x = x + y_pred

        return x, x_mean, x_std

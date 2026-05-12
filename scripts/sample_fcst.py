import argparse
import json
import os

import numpy as np
import torch
import yaml

from lightning import Trainer
from lightning.fabric import seed_everything

from src import models
from src.datamodule.data_factory import data_provider
from src.utils.filters import MovingAvgTime, get_factors
from src.utils.metrics import calculate_metrics
from src.utils.parser import exp_parser, build_exp_suffix


def main(args):
    seed_everything(9, workers=True)

    data_config = yaml.safe_load(open(f"configs/dataset/{args.data_config}.yaml", "r"))
    data_config = exp_parser(data_config, vars(args))
    print(data_config)

    exp_suffix = build_exp_suffix(data_config, args.exp_tag)
    model_name_full = args.model_name + exp_suffix

    root_path = os.path.join(
        args.save_dir, f"{args.data_config}_{data_config['pred_len']}_S"
    )
    exp_path = os.path.join(root_path, model_name_full)
    print(f"[SAMPLE] Loading from: {model_name_full}")
    exp_config = json.load(open(os.path.join(exp_path, "config.json"), "rb"))
    model_name = exp_config["diff_config"]["name"]
    model_class = getattr(models, model_name)
    seq_length = data_config["pred_len"]
    diffusion_steps = exp_config["diff_config"]["T"]
    out_name = [
        f"cond_{args.condition}",
        f"fast_{args.fast_sample}",
        f"dtm_{args.deterministic}",
        f"nsample_{args.n_sample}",
        f"wc_{args.w_cond}" if args.w_cond != 1.0 else "",
    ]
    out_name = [s for s in out_name if s]
    out_name = "_".join(out_name)
    if args.out_suffix:
        out_name += f"_{args.out_suffix}"
    if (not args.smoke_test) and os.path.exists(
        os.path.join(exp_path, f"{out_name}.npy"),
    ):
        print("already have metrics — skipping.")
        return 0
    _, test_dl = data_provider(data_config, "test")

    noise_schedule_name = exp_config["diff_config"].get("noise_schedule", "std")

    if args.fast_sample:
        if noise_schedule_name == "std":
            kernel_size = get_factors(seq_length)
            kernel_size = np.array(kernel_size)
            sample_steps = []
            noise_schedule = torch.load(
                os.path.join(root_path, f"std_sched_{diffusion_steps}.pt")
            )
            all_K = noise_schedule["alphas"]
            for i in kernel_size:
                ideal_K = MovingAvgTime(i, seq_length).K.unsqueeze(0)
                close_id = torch.argmin(
                    ((all_K.flatten(1) - ideal_K.flatten(1)) ** 2).mean(dim=1)
                ).int()
                sample_steps.append(close_id.item())
            if 0 not in sample_steps:
                sample_steps.append(0)
            sample_steps = sorted(set(sample_steps))
        else:
            n_fast = args.ddim_steps if args.ddim_steps is not None else 10
            sample_steps = np.round(np.linspace(0, diffusion_steps - 1, n_fast)).astype(int).tolist()
            if 0 not in sample_steps:
                sample_steps.append(0)
            sample_steps = sorted(set(sample_steps))
    else:
        sample_steps = None

    print("sample_steps:\t", sample_steps)

    trainer = Trainer(
        devices=[args.gpu], enable_progress_bar=False, fast_dev_run=args.smoke_test
    )
    avg_m = []

    seed_start = args.seed_start
    for i in range(seed_start, seed_start + args.num_train):
        read_d = os.path.join(exp_path, f"best_model_path_{i}.pt")
        best_model_path = torch.load(read_d)
        best_model_path = os.path.join(exp_path, os.path.basename(best_model_path))
        diff = model_class.load_from_checkpoint(
            best_model_path, map_location=f"cuda:{args.gpu}"
        )
        if hasattr(args, "label_replace") and args.label_replace is not None:
            diff.label_replace = args.label_replace

        if args.deterministic:
            sigmas = torch.zeros_like(diff.betas)
        elif args.sigma_schedule == "low":
            sigmas = torch.linspace(0.1, 0.05, len(diff.betas))
        elif args.sigma_schedule == "high":
            sigmas = torch.linspace(0.4, 0.2, len(diff.betas))
        else:
            sigmas = torch.linspace(0.2, 0.1, len(diff.betas))

        diff.config_sampling(
            args.n_sample,
            w_cond=args.w_cond,
            sigmas=sigmas,
            sample_steps=sample_steps,
            condition=args.condition,
        )

        y_real = []
        y_pred = trainer.predict(diff, test_dl)
        y_pred = torch.concat(y_pred, dim=1)
        for b in test_dl:
            y_real.append(b["x"])
            if args.smoke_test:
                break
        y_real = torch.concat(y_real)

        label_len = data_config.get("label_len", 0)
        if label_len > 0:
            y_pred = y_pred[:, :, label_len:, :]
            y_real = y_real[:, label_len:, :]

        m, y_pred_q, y_pred_point = calculate_metrics(y_pred, y_real)
        print(m)
        avg_m.append(m)

        if i == seed_start:
            np.save(os.path.join(exp_path, f"{out_name}_pred.npy"), y_pred)
            np.save(os.path.join(exp_path, "truth.npy"), y_real)
        if args.smoke_test:
            break
    avg_m = np.array(avg_m)
    print(avg_m.mean(axis=0))
    if not args.smoke_test:
        np.save(os.path.join(exp_path, f"{out_name}.npy"), avg_m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDiff sampling entry point.")
    parser.add_argument("--save_dir", type=str, default="./savings/fcst")
    parser.add_argument("-dc", "--data_config", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--fast_sample", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--w_cond", type=float, default=0.5)

    parser.add_argument("--root_path", type=str, default=None,
                        help="Override data root_path from dataset YAML.")
    parser.add_argument("--condition", type=str, choices=["sr", "fcst"])
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--label_len", type=int)
    parser.add_argument("--exp_tag", type=str, default=None)
    parser.add_argument("--num_train", type=int, default=5)
    parser.add_argument("--seed_start", type=int, default=0)
    parser.add_argument("--label_replace", type=str, default=None,
                        choices=["none", "x0_cond", "repaint"])
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--out_suffix", type=str, default=None)
    parser.add_argument("--sigma_schedule", type=str, default="default",
                        choices=["low", "default", "high"])
    args = parser.parse_args()
    main(args)

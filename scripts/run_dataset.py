"""Train and evaluate DiffDiff on a single dataset across all horizons and seeds.

For one dataset, runs all 4 horizons (optionally concurrently on a single GPU
as subprocesses); each horizon trains all seeds serially, then performs one
sampling pass that aggregates the per-seed metrics into a single N-row file.
Compares against the expected numbers in the optimal yaml.

Example:
    python scripts/run_dataset.py --dataset electricity --gpu 0 --parallel_horizons
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


PYTHON_BIN = sys.executable


def load_optimal(dataset):
    with open(f"configs/optimal/{dataset}.yaml", "r") as f:
        return yaml.safe_load(f)


def args_to_cli(d):
    """Convert {key: value} to ["--key", "value", ...]. True booleans become
    bare flags; False booleans are dropped."""
    out = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                out += [f"--{k}"]
        else:
            out += [f"--{k}", str(v)]
    return out


def train_cmd(dataset, horizon, seed, gpu, save_dir, opt, exp_tag):
    cfg = opt["shared"]
    cmd = [
        PYTHON_BIN, "scripts/train_fcst.py",
        "-mc", cfg["model_config"],
        "-dc", dataset,
        "--save_dir", save_dir,
        "--gpu", str(gpu),
        "--seed", str(seed),
        "--pred_len", str(horizon),
        "--label_len", str(cfg["label_len"]),
        "--batch_size", str(cfg["batch_size"]),
        "--condition", cfg["condition"],
        "--exp_tag", exp_tag,
    ]
    cmd += args_to_cli(opt["train_args"])
    return cmd


def sample_cmd(dataset, horizon, gpu, save_dir, opt, exp_tag, seed_start, num_seeds):
    cfg = opt["shared"]
    model_name = f"{cfg['model_config']}_bs{cfg['batch_size']}_cond{cfg['condition']}"
    cmd = [
        PYTHON_BIN, "scripts/sample_fcst.py",
        "-dc", dataset,
        "--model_name", model_name,
        "--save_dir", save_dir,
        "--gpu", str(gpu),
        "--condition", cfg["condition"],
        "--pred_len", str(horizon),
        "--label_len", str(cfg["label_len"]),
        "--batch_size", str(cfg["batch_size"]),
        "--n_sample", str(cfg["n_sample"]),
        "--ddim_steps", str(cfg["ddim_steps"]),
        "--fast_sample",
        "--num_train", str(num_seeds),
        "--seed_start", str(seed_start),
        "--exp_tag", exp_tag,
    ]
    cmd += args_to_cli(opt["sample_args"])
    return cmd


def run_subprocess(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as logf:
        logf.write(f"\n>>> {' '.join(cmd)}\n")
        logf.flush()
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            raise RuntimeError(f"FAILED cmd[0]={cmd[1]} rc={r.returncode}")


def run_horizon(dataset, horizon, gpu, save_dir, opt, exp_tag, log_dir, seeds):
    """Train all seeds for `horizon` serially on `gpu`, then a single sample pass."""
    log_path = Path(log_dir) / f"{dataset}_H{horizon}.log"
    for s in seeds:
        run_subprocess(
            train_cmd(dataset, horizon, s, gpu, save_dir, opt, exp_tag),
            log_path,
        )
    # Contiguous seed block required by sample_fcst.py's seed_start+num_train loop.
    assert seeds == list(range(min(seeds), max(seeds) + 1)), \
        f"seeds must be contiguous: got {seeds}"
    run_subprocess(
        sample_cmd(dataset, horizon, gpu, save_dir, opt, exp_tag,
                   seed_start=min(seeds), num_seeds=len(seeds)),
        log_path,
    )


def aggregate_horizon(dataset, horizon, save_dir, opt, exp_tag):
    """Read the single per-horizon metric .npy (rows = seeds; cols = MAE,MSE,MQL)."""
    cfg = opt["shared"]
    sa = opt["sample_args"]
    model_name = f"{cfg['model_config']}_bs{cfg['batch_size']}_cond{cfg['condition']}"
    suffix = f"_lw{cfg['label_len']}_{exp_tag}"
    exp_path = Path(save_dir) / f"{dataset}_{horizon}_S" / (model_name + suffix)

    deterministic = sa.get("deterministic", False)
    w_cond = sa.get("w_cond", 1.0)
    out_name_parts = [
        f"cond_{cfg['condition']}",
        f"fast_True",
        f"dtm_{deterministic}",
        f"nsample_{cfg['n_sample']}",
    ]
    if w_cond != 1.0:
        out_name_parts.append(f"wc_{w_cond}")
    out_name = "_".join(out_name_parts)

    p = exp_path / f"{out_name}.npy"
    if not p.exists():
        raise FileNotFoundError(f"missing metrics: {p}")
    arr = np.load(p)  # (n_seeds, 3) — (MAE, MSE, MQL)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0, ddof=0)
    return {
        "MAE": (float(mean[0]), float(std[0])),
        "MSE": (float(mean[1]), float(std[1])),
        "CRPS": (float(mean[2]), float(std[2])),
    }, arr.shape[0]


def compare_to_expected(observed, expected, tol_pct):
    rows = []
    for metric in ["MSE", "MAE", "CRPS"]:
        obs_mean, _ = observed[metric]
        exp_mean = expected[metric]
        pct = 100.0 * abs(obs_mean - exp_mean) / max(abs(exp_mean), 1e-12)
        rows.append((metric, obs_mean, exp_mean, pct, pct <= tol_pct))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                   choices=["electricity", "ettm2", "exchange_rate",
                            "traffic", "weather", "solar", "wind"])
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--save_dir", default="./savings/fcst")
    p.add_argument("--log_dir", default="./logs")
    p.add_argument("--seeds", default="0,1,2,3,4",
                   help="Contiguous seed range (e.g. 0,1,2,3,4).")
    p.add_argument("--horizons", default=None,
                   help="Override pred_lens. Comma-separated, e.g. 96,192,336,720.")
    p.add_argument("--exp_tag", default="release",
                   help="Tag appended to save folder.")
    p.add_argument("--parallel_horizons", action="store_true",
                   help="Run all horizons concurrently on the same GPU as subprocesses.")
    p.add_argument("--tol_pct", type=float, default=1.0,
                   help="Pass threshold: |obs-expected|/expected <= tol_pct (default 1%).")
    p.add_argument("--skip_aggregate", action="store_true",
                   help="Internal: used by parallel_horizons children to avoid racing on summary.json.")
    args = p.parse_args()

    opt = load_optimal(args.dataset)
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.horizons is not None:
        horizons = [int(h) for h in args.horizons.split(",")]
    else:
        horizons = opt["shared"]["pred_lens"]

    t0 = time.time()
    print(f"[run_dataset] dataset={args.dataset} gpu={args.gpu} seeds={seeds} "
          f"horizons={horizons} parallel_horizons={args.parallel_horizons}")

    if args.parallel_horizons:
        procs = []
        for h in horizons:
            p_cmd = [PYTHON_BIN, __file__,
                     "--dataset", args.dataset,
                     "--gpu", str(args.gpu),
                     "--save_dir", args.save_dir,
                     "--log_dir", args.log_dir,
                     "--seeds", args.seeds,
                     "--horizons", str(h),
                     "--exp_tag", args.exp_tag,
                     "--tol_pct", str(args.tol_pct),
                     "--skip_aggregate"]
            procs.append(subprocess.Popen(p_cmd))
        rc = [p.wait() for p in procs]
        if any(r != 0 for r in rc):
            raise RuntimeError(f"parallel horizons failed: rc={rc}")
    else:
        for h in horizons:
            run_horizon(args.dataset, h, args.gpu, args.save_dir, opt,
                        args.exp_tag, args.log_dir, seeds)

    if args.skip_aggregate:
        return

    elapsed = time.time() - t0
    print(f"[run_dataset] training/sampling done in {elapsed/60:.1f} min")

    print("\n" + "=" * 80)
    print(f"Validation: {args.dataset}  (tol={args.tol_pct}%)")
    print("=" * 80)
    summary = {}
    for h in opt["shared"]["pred_lens"]:
        if h not in horizons:
            continue
        observed, n_seeds = aggregate_horizon(args.dataset, h, args.save_dir, opt,
                                              args.exp_tag)
        expected = opt["expected"][f"H{h}"]
        rows = compare_to_expected(observed, expected, tol_pct=args.tol_pct)
        summary[h] = {"observed": observed, "n_seeds": n_seeds, "rows": rows}

        print(f"\nH={h} ({n_seeds} seeds):")
        print(f"  {'metric':<6}{'observed':>14}{'expected':>14}{'diff%':>10}  status")
        for metric, obs, exp_, pct, ok in rows:
            marker = "PASS" if ok else "FAIL"
            print(f"  {metric:<6}{obs:>14.4f}{exp_:>14.4f}{pct:>9.2f}%  {marker}")

    out_json = Path(args.log_dir) / f"{args.dataset}_summary.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(
            {h: {"n_seeds": s["n_seeds"],
                 "metrics": {m: {"mean": v[0], "std": v[1]}
                             for m, v in s["observed"].items()}}
             for h, s in summary.items()},
            f, indent=2,
        )
    print(f"\n[run_dataset] summary -> {out_json}")


if __name__ == "__main__":
    main()

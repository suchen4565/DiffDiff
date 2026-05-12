"""Aggregate logs/*_summary.json across the 7 datasets and compare to expected."""

import json
from pathlib import Path

import yaml


DATASETS = ["electricity", "ettm2", "exchange_rate",
            "traffic", "weather", "solar", "wind"]


def main():
    log_dir = Path("./logs")
    print(f"{'dataset':<15}{'H':>5}{'metric':>7}"
          f"{'observed':>14}{'expected':>14}{'diff%':>10}  status")
    print("-" * 75)

    n_fail = 0
    n_total = 0
    for ds in DATASETS:
        sum_path = log_dir / f"{ds}_summary.json"
        opt_path = Path(f"configs/optimal/{ds}.yaml")
        if not sum_path.exists():
            print(f"{ds:<15}  -- summary missing: {sum_path}")
            continue
        with open(sum_path) as f:
            obs_all = json.load(f)
        with open(opt_path) as f:
            opt = yaml.safe_load(f)

        for h_str, obs_entry in obs_all.items():
            expected = opt["expected"][f"H{h_str}"]
            metrics = obs_entry["metrics"]
            for metric in ["MSE", "MAE", "CRPS"]:
                obs_mean = metrics[metric]["mean"]
                exp_mean = expected[metric]
                pct = 100.0 * abs(obs_mean - exp_mean) / max(abs(exp_mean), 1e-12)
                ok = pct <= 1.0
                marker = "PASS" if ok else "FAIL"
                n_total += 1
                if not ok:
                    n_fail += 1
                print(f"{ds:<15}{h_str:>5}{metric:>7}"
                      f"{obs_mean:>14.4f}{exp_mean:>14.4f}{pct:>9.2f}%  {marker}")

    print("-" * 75)
    print(f"Total cells: {n_total}, failed (>1%): {n_fail}")


if __name__ == "__main__":
    main()

# DiffDiff

Reference implementation of **DiffDiff**, a differencing-based diffusion model
for time series forecasting.

This repository contains the code needed to train DiffDiff from scratch and
reproduce the main-table numbers across 7 datasets. The MA-TSD baseline source
is included for architectural reference.

## Repository layout

```
DiffDiff_release/
├── configs/
│   ├── model/
│   │   ├── DiffDiff.yaml
│   │   └── MATSD_fcst.yaml
│   ├── dataset/         # 7 dataset YAMLs (S-feature, OT target)
│   └── optimal/         # per-dataset configs + expected metrics
├── src/
│   ├── models/          # DiffDiff + MATSD (Lightning modules)
│   ├── backbone/        # MLPBackbone
│   ├── layers/          # RevIN, sinusoidal time embedding
│   ├── datamodule/      # CSV-based dataloaders
│   └── utils/           # parser, schedule, metrics, filters
├── scripts/
│   ├── train_fcst.py    # single-config training entry
│   ├── sample_fcst.py   # DDIM sampling + metrics
│   ├── run_dataset.py   # one dataset across all horizons and seeds
│   ├── compare_all.py   # aggregate + compare to expected
│   └── reproduce.sh     # all 7 datasets in parallel
├── requirements.txt
└── LICENSE
```

## Setup

```bash
conda create -n diffdiff python=3.10 -y
conda activate diffdiff
pip install -r requirements.txt
```

Tested with PyTorch 2.1+ and Lightning 2.1+.

## Data

The 7 datasets are standard public benchmarks:
`electricity`, `ETTm2`, `exchange_rate`, `traffic`, `weather`, `solar`, `wind`.

Place the CSV files under `./data/` (default path in `configs/dataset/*.yaml`):

```
./data/
  electricity.csv
  ETTm2.csv
  exchange_rate.csv
  traffic.csv
  weather.csv
  solar.csv
  wind_farms.csv
```

Each dataset YAML uses `features: S` (univariate, OT column). To use a
different path, pass `--root_path /your/data/dir/` to the training and
sampling scripts.

## Quick start: single configuration

Train one (dataset, horizon, seed) combination:

```bash
python scripts/train_fcst.py \
    -mc DiffDiff \
    -dc electricity \
    --condition fcst \
    --pred_len 96 --label_len 48 --batch_size 64 \
    --seed 0 --exp_tag debug \
    --d_model 64 --d_mlp 256 --n_layers 3 --dropout 0.1 \
    --lr 2e-4 --lr_sched cosine --min_lr_ratio 0.01 \
    --grad_clip 0.5 --early_stop 8 \
    --gate_init_bias 0 --gate_l1_weight 1e-4
```

Sample from the trained checkpoint:

```bash
python scripts/sample_fcst.py \
    -dc electricity --condition fcst \
    --model_name DiffDiff_bs64_condfcst \
    --pred_len 96 --label_len 48 --batch_size 64 \
    --fast_sample --ddim_steps 10 --n_sample 100 \
    --deterministic --w_cond 1.0 \
    --num_train 1 --seed_start 0 --exp_tag debug
```

## Full reproduction (7 datasets × 4 horizons × 5 seeds)

The orchestrator reads `configs/optimal/<dataset>.yaml`, runs all horizons
(concurrently on the same GPU) across 5 seeds, then compares observed metrics
to the expected values stored in the same YAML (1% tolerance).

Single dataset:

```bash
python scripts/run_dataset.py --dataset electricity --gpu 0 --parallel_horizons
```

All 7 datasets (one per GPU, GPUs 0-6):

```bash
bash scripts/reproduce.sh
```

After completion, summarize across all datasets:

```bash
python scripts/compare_all.py
```

Each cell prints `PASS` if `|observed − expected| / expected ≤ 1%`.

## Per-dataset configurations

`configs/optimal/<dataset>.yaml` contains, for each dataset:

- `shared`: model, batch size, sampling steps, horizons.
- `train_args`: architecture and optimizer settings (override
  `configs/model/DiffDiff.yaml`).
- `sample_args`: inference-time settings (deterministic vs stochastic σ
  schedule, classifier-free guidance weight).
- `expected`: 5-seed MSE/MAE/CRPS mean and std at each horizon.

Shared invariants across datasets:

| Knob | Value |
|---|---|
| Diffusion steps `T` | 50 |
| Forward gain bound `λ_max` | 0.5 |
| Forward gain schedule | linear |
| Future loss weight `w_fut` | 5 |
| Noise schedule | cosine |
| Label length `L_l` | 48 |
| Look-back length `L` | 96 |
| Batch size | 64 |
| Max epochs | 100 |
| DDIM sampling steps | 10 |
| Samples for CRPS | 100 |
| Adaptive condition gating | on |

## MA-TSD baseline

`src/models/matsd.py` and `configs/model/MATSD_fcst.yaml` are included for
architectural reference. MA-TSD uses the same `MLPBackbone` as DiffDiff with a
moving-average forward process instead of differencing. To train MA-TSD,
replace `-mc DiffDiff` with `-mc MATSD_fcst` in the train command above.

## Outputs

Each `(dataset, horizon, seed)` run writes:

- `savings/fcst/<dataset>_<H>_S/<model_name>_<suffix>/best_model_path_<seed>.pt`
  — Lightning checkpoint pointer.
- `cond_fcst_fast_True_dtm_<bool>_nsample_100[_wc_<w>][_<out_suffix>].npy` —
  per-seed `(MAE, MSE, MQL)` triple.
- `*_pred.npy`, `truth.npy` — first-seed prediction and ground-truth tensors.

`logs/<dataset>_summary.json` is written once aggregation completes.

## License

See `LICENSE`.

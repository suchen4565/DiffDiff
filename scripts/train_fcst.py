import argparse
import json
import os

import torch
import yaml
from lightning import Trainer
from lightning.fabric import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.datamodule.data_factory import data_provider
from src import models
from src.utils.parser import exp_parser, build_exp_suffix
from src.utils.schedule import get_schedule


def prepare_train(model_config, data_config, args, n):
    root_pth = args["save_dir"]

    _, train_dl = data_provider(data_config, "train")
    _, val_dl = data_provider(data_config, "val")

    data_folder = os.path.join(
        root_pth,
        f"{args['data_config']}_{data_config['pred_len']}_{data_config['features']}",
    )
    # Apply CLI overrides to configs before model construction
    if data_config.get("label_len", 0) > 0:
        model_config["diff_config"]["label_len"] = data_config["label_len"]
    for key in ["lambda_max", "loss_weight_future", "d2_normalize", "lambda_schedule",
                 "norm_source", "label_replace", "diff_order", "T", "noise_schedule"]:
        if args.get(key) is not None:
            model_config["diff_config"][key] = args[key]
    for key in ["cond_dropout_prob", "adaptive_cond", "gate_init_bias", "gate_l1_weight",
                "d_model", "d_mlp", "n_layers", "dropout"]:
        if args.get(key) is not None:
            model_config["bb_config"][key] = args[key]
    for key in ["lr", "alpha", "epochs", "early_stop",
                "lr_sched", "warmup_ratio", "min_lr_ratio"]:
        if args.get(key) is not None:
            model_config["train_config"][key] = args[key]

    exp_suffix = build_exp_suffix(data_config, args.get("exp_tag"))
    save_folder = os.path.join(
        data_folder,
        args["model_config"]
        + f"_bs{data_config['batch_size']}_cond{data_config['condition']}{exp_suffix}",
    )
    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "config.json"), "w") as w:
        json.dump(model_config, w, indent=2)

    df_ = model_config["diff_config"].pop("name")
    df = getattr(models, df_)

    batch = next(iter(train_dl))
    target_seq_length, target_seq_channels = (batch["x"].shape[1], batch["x"].shape[2])
    model_config["bb_config"]["seq_channels"] = target_seq_channels
    model_config["bb_config"]["seq_length"] = target_seq_length

    if data_config["condition"] is not None:
        seq_length, seq_channels = (batch["c"].shape[1], batch["c"].shape[2])
        model_config["bb_config"]["cond_seq_chnl"] = seq_channels
        model_config["bb_config"]["cond_seq_len"] = seq_length

    ns_name = model_config["diff_config"].pop("noise_schedule")
    n_steps = model_config["diff_config"]["T"]
    ns_path = get_schedule(
        ns_name,
        n_steps,
        check_pth=data_folder,
        train_dl=train_dl,
    )
    return model_config, ns_path, df, save_folder, train_dl, val_dl


def main(args, n):
    seed_everything(n, workers=True)

    data_config = yaml.safe_load(
        open(f'configs/dataset/{args["data_config"]}.yaml', "r")
    )
    data_config = exp_parser(data_config, args)

    model_config = yaml.safe_load(
        open(f'configs/model/{args["model_config"]}.yaml', "r")
    )
    model_config = exp_parser(model_config, args)

    model_config, ns_path, df, save_folder, train_dl, val_dl = prepare_train(
        model_config, data_config, args, n
    )
    sched_kwargs = {}
    for k in ("lr_sched", "warmup_ratio", "min_lr_ratio"):
        if k in model_config["train_config"]:
            sched_kwargs[k] = model_config["train_config"][k]
    diff = df(
        backbone_config=model_config["bb_config"],
        ns_path=ns_path,
        lr=model_config["train_config"]["lr"],
        alpha=model_config["train_config"]["alpha"],
        **sched_kwargs,
        **model_config["diff_config"],
    )

    bb = model_config["bb_config"]
    tc = model_config["train_config"]
    print(f"[CONFIG] model={type(diff).__name__} seq_len={diff.seq_length} T={diff.T} "
          f"lambda_max={getattr(diff, 'lambda_max', 'N/A')} label_len={getattr(diff, 'label_len', 0)} "
          f"loss_wt={getattr(diff, 'loss_weight_future', 'N/A')} "
          f"d_model={bb['d_model']} d_mlp={bb['d_mlp']} n_layers={bb['n_layers']} "
          f"dropout={bb['dropout']} lr={tc['lr']} alpha={tc['alpha']} "
          f"save={os.path.basename(save_folder)}")

    es = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=model_config["train_config"]["early_stop"],
    )
    mc = ModelCheckpoint(monitor="val_loss", dirpath=save_folder, save_top_k=1)
    trainer_kwargs = dict(
        max_epochs=model_config["train_config"]["epochs"],
        deterministic=True,
        devices=[args["gpu"]],
        callbacks=[es, mc],
        default_root_dir=save_folder,
        fast_dev_run=args["smoke_test"],
        enable_progress_bar=args["smoke_test"],
        check_val_every_n_epoch=model_config["train_config"]["val_step"],
    )
    if args.get("grad_clip") is not None:
        trainer_kwargs["gradient_clip_val"] = args["grad_clip"]
    trainer = Trainer(**trainer_kwargs)

    trainer.fit(diff, train_dl, val_dl)

    torch.save(mc.best_model_path, os.path.join(save_folder, f"best_model_path_{n}.pt"))
    print(mc.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiffDiff training entry point.")
    parser.add_argument("-mc", "--model_config", required=True,
                        help="Model YAML name under configs/model/.")
    parser.add_argument("-dc", "--data_config", type=str, required=True,
                        help="Dataset YAML name under configs/dataset/.")
    parser.add_argument("--save_dir", default="./savings/fcst", type=str,
                        help="Root directory for checkpoints and metrics.")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_train", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed only. Overrides num_train loop.")

    # Data overrides
    parser.add_argument("--root_path", type=str, default=None,
                        help="Override data root_path from dataset YAML.")
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--condition", type=str)
    parser.add_argument("--kernel_size", type=int)
    parser.add_argument("--label_len", type=int)
    parser.add_argument("--exp_tag", type=str, default=None)

    # diff_config overrides
    parser.add_argument("--lambda_max", type=float, default=None)
    parser.add_argument("--loss_weight_future", type=float, default=None)
    parser.add_argument("--d2_normalize", action="store_true", default=None)
    parser.add_argument("--lambda_schedule", type=str, default=None,
                        choices=["linear", "cosine", "diff_first", "sqrt"])
    parser.add_argument("--diff_order", type=int, default=None, choices=[1, 2, 3, 4])
    parser.add_argument("--norm_source", type=str, default=None,
                        choices=["target", "condition"])
    parser.add_argument("--label_replace", type=str, default=None,
                        choices=["none", "x0_cond", "repaint"])
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--noise_schedule", type=str, default=None,
                        choices=["linear", "cosine"])

    # bb_config overrides
    parser.add_argument("--cond_dropout_prob", type=float, default=None)
    parser.add_argument("--adaptive_cond", action="store_true", default=None)
    parser.add_argument("--gate_init_bias", type=float, default=None)
    parser.add_argument("--gate_l1_weight", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_mlp", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # train_config overrides
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early_stop", type=int, default=None)
    parser.add_argument("--lr_sched", type=str, default=None,
                        choices=["constant", "cosine", "warmup_cosine"])
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--min_lr_ratio", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None,
                        help="Lightning gradient_clip_val (max-norm).")

    args = parser.parse_args()
    if args.seed is not None:
        main(vars(args), args.seed)
    else:
        for i in range(args.num_train):
            main(vars(args), i)
            if args.smoke_test:
                break

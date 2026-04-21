import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from oasis2_coronal_experiment import (
    DATA_ROOT,
    DEFAULT_SEEDS,
    RESULTS_DIR,
    ExperimentConfig,
    run_experiment,
)


DEFAULT_LRS = [0.0005, 0.001, 0.002]


def lr_stem(lr):
    return f"hybrid_lr_{lr:g}".replace(".", "_")


def summarize_lr_result(output_dir, lr):
    stem = lr_stem(lr)
    result_csv = output_dir / f"{stem}.csv"
    diagnostics_csv = output_dir / f"{stem}_diagnostics.csv"

    results = pd.read_csv(result_csv)
    diagnostics = pd.read_csv(diagnostics_csv)

    original_final = results[
        (results["row_type"] == "final") & (results["eval_split"] == "original")
    ].copy()
    for column in ["macro_f1", "balanced_acc", "auc"]:
        original_final[column] = pd.to_numeric(original_final[column], errors="coerce")

    for column in ["epoch", "qnn_grad_norm", "qnn_weight_update_norm"]:
        diagnostics[column] = pd.to_numeric(diagnostics[column], errors="coerce")

    stuck_trials = int(original_final["convergence_status"].eq("stuck").sum())
    converged_trials = int(original_final["convergence_status"].eq("converged").sum())
    final_epoch = diagnostics["epoch"].max()
    final_epoch_diagnostics = diagnostics[diagnostics["epoch"] == final_epoch]

    return {
        "lr": lr,
        "mean_macro_f1": np.nanmean(original_final["macro_f1"]),
        "mean_balanced_acc": np.nanmean(original_final["balanced_acc"]),
        "mean_auc": np.nanmean(original_final["auc"]),
        "stuck_trials": stuck_trials,
        "converged_trials": converged_trials,
        "failure_rate": stuck_trials / len(original_final),
        "mean_qnn_grad_norm": np.nanmean(diagnostics["qnn_grad_norm"]),
        "mean_qnn_update_norm": np.nanmean(diagnostics["qnn_weight_update_norm"]),
        "final_epoch_qnn_grad_norm": np.nanmean(
            final_epoch_diagnostics["qnn_grad_norm"]
        ),
        "final_epoch_qnn_update_norm": np.nanmean(
            final_epoch_diagnostics["qnn_weight_update_norm"]
        ),
    }


def write_summary(output_dir, rows):
    summary_csv = output_dir / "hybrid_lr_sweep_summary.csv"
    fieldnames = [
        "lr",
        "mean_macro_f1",
        "mean_balanced_acc",
        "mean_auc",
        "stuck_trials",
        "converged_trials",
        "failure_rate",
        "mean_qnn_grad_norm",
        "mean_qnn_update_norm",
        "final_epoch_qnn_grad_norm",
        "final_epoch_qnn_update_norm",
    ]
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a hybrid-only learning-rate sweep on OASIS-2 coronal."
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "lr_sweep",
    )
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--test-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.seeds) < args.trials:
        raise ValueError(
            f"Need at least {args.trials} seeds, but only received {len(args.seeds)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for lr in args.lrs:
        stem = lr_stem(lr)
        print(f"\n=== hybrid lr sweep | lr={lr:g} | stem={stem} ===")
        config = ExperimentConfig(
            model_name="hybrid",
            data_root=args.data_root,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            lr=lr,
            epochs=args.epochs,
            trials=args.trials,
            seeds=args.seeds,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            device=torch.device(args.device),
            result_stem=stem,
        )
        run_experiment(config)
        summary_rows.append(summarize_lr_result(args.output_dir, lr))

    summary_csv = write_summary(args.output_dir, summary_rows)
    print(f"\nWrote learning-rate sweep summary: {summary_csv}")


if __name__ == "__main__":
    main()

import os, sys, pathlib
sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

"""
Plot LANTERN-style ranking confusion matrices for the trained models.

Each model's test-set predictions are binned into `num_classes` percentile bins
(true bins on the y-axis, predicted bins on the x-axis), the matrix is
row-normalised by the true bin, and the title reports the *ranking accuracy* =
mean of the row-normalised diagonal (the average rate at which a compound's
predicted percentile bin matches its true bin). This matches Fig. 6 of the
AGILE benchmark.

Outputs (saved under results/confusion_matrices/<split>/):
  * confusion_<tag>_<split>.<fmt>   -- one per model
  * confusion_grid_<split>.<fmt>    -- all models in a single figure

Run (in the `mofpgnn` env, from the repo root):
    python scripts/plot_confusion_grid.py --split random
    python scripts/plot_confusion_grid.py --split Murcko_scaffold
"""

import argparse
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from utils.confusion_matrix_utils import (
    convert_to_class,
    ranking_accuracy,
    create_and_plot_confusion_matrix,
)

# (dir_prefix, short_tag, pretty_label) for the five reported models.
# dir_prefix is matched against results/<prefix>-<split>-<timestamp>/ ;
# the most recent matching run is used. The KPGT models come from
# finetune_kpgt.py (kpgt_finetune[-cmf|-bmf]); the baselines from
# run_kpgt_pipeline.py (morgan_only[-binary]).
MODELS = [
    ("kpgt_finetune",        "A",  "KPGT (fine-tune)"),
    ("morgan_only",          "B1", "CMF-only"),
    ("morgan_only-binary",   "B2", "BMF-only"),
    ("kpgt_finetune-cmf",    "C1", "KPGT + CMF"),
    ("kpgt_finetune-bmf",    "C2", "KPGT + BMF"),
]


def find_run_dir(results_root, prefix, split):
    """Return the latest <results_root>/<prefix>-<split>-* directory, or None."""
    pattern = os.path.join(results_root, f"{prefix}-{split}-*")
    matches = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    return matches[-1] if matches else None


def load_test_preds(run_dir):
    df = pd.read_csv(os.path.join(run_dir, "test_results.csv"))
    return df["true"].to_numpy(), df["pred"].to_numpy()


def plot_grid(results_root, split, num_classes, save_path, save_format,
              vmax, font_scale, ncols=3):
    sn.set_theme(style="whitegrid", context="paper", font_scale=font_scale)

    panels = []
    for prefix, tag, label in MODELS:
        run_dir = find_run_dir(results_root, prefix, split)
        if run_dir is None:
            print(f"  [skip] no run found for {tag} ({prefix}-{split}-*)")
            continue
        labels, preds = load_test_preds(run_dir)
        label_cls, pred_cls = convert_to_class(labels, preds, num_classes)
        acc = ranking_accuracy(label_cls, pred_cls, num_classes)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(label_cls, pred_cls,
                              labels=list(range(num_classes)), normalize="true")
        panels.append((tag, label, cm, acc))
        print(f"  {tag}: {label:<24} Accuracy = {acc * 100:.1f}%")

    if not panels:
        print("No runs found - nothing to plot.")
        return

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    tmp = np.linspace(100, 0, num_classes + 1)
    tick_labels = [f"{int(x)}%" for x in tmp[:-1]] + ["Top"]

    for ax, (tag, label, cm, acc) in zip(axes, panels):
        sn.heatmap(cm, ax=ax, annot=True, fmt=".2f", cmap="Blues",
                   vmin=0, vmax=vmax, cbar=False, square=True)
        ax.set_title(f"{tag}: {label}\nAccuracy: {acc * 100:.1f}%")
        ax.set_xlabel("Predicted percentile bin")
        ax.set_ylabel("Actual percentile bin")
        ax.set_xticks(np.arange(num_classes) + 0.5)
        ax.set_yticks(np.arange(num_classes) + 0.5)
        ax.set_xticklabels(tick_labels[:-1], rotation=0)
        ax.set_yticklabels(tick_labels[:-1], rotation=0)

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(f"{save_path}.{save_format}", format=save_format,
                dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid saved -> {save_path}.{save_format}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot LANTERN-style ranking confusion matrices.")
    parser.add_argument("--split", default="random",
                        help="random / Murcko_scaffold")
    parser.add_argument("--results_root", default="results",
                        help="Directory holding the per-model run folders")
    parser.add_argument("--num_classes", type=int, default=6,
                        help="Number of percentile bins (AGILE uses 6)")
    parser.add_argument("--save_format", default="png", help="png / jpg / svg")
    parser.add_argument("--vmax", type=float, default=0.75,
                        help="Upper colour-scale limit for the heatmaps")
    parser.add_argument("--font_scale", type=float, default=1.4)
    parser.add_argument("--no_individual", action="store_true",
                        help="Skip the per-model figures, only build the grid")
    args = parser.parse_args()

    out_dir = os.path.join(args.results_root, "confusion_matrices", args.split)
    os.makedirs(out_dir, exist_ok=True)

    # Per-model figures (uses the shared util so titles/style match the pipeline)
    if not args.no_individual:
        for prefix, tag, label in MODELS:
            run_dir = find_run_dir(args.results_root, prefix, args.split)
            if run_dir is None:
                print(f"[skip] no run found for {tag} ({prefix}-{args.split}-*)")
                continue
            labels, preds = load_test_preds(run_dir)
            save_path = os.path.join(out_dir, f"confusion_{tag}_{args.split}")
            create_and_plot_confusion_matrix(
                labels, preds,
                num_classes=args.num_classes,
                save_path=save_path,
                save_format=args.save_format,
                vmax=args.vmax,
                font_scale=3.2,
            )

    # Combined grid
    grid_path = os.path.join(out_dir, f"confusion_grid_{args.split}")
    plot_grid(args.results_root, args.split, args.num_classes,
              grid_path, args.save_format, args.vmax, args.font_scale)


if __name__ == "__main__":
    main()

from pathlib import Path
import pandas as pd
import re

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TEX_PATH = RESULTS_DIR / "chemprop_results.tex"

# Map CSV model names → LaTeX row labels (must match table order)
MODEL_MAP = {
    "random": [
        ("MLP(Morgan)", "MLP (Morgan)"),
        ("MLP(Morgan+Expert)", "MLP (Morgan + Expert RDKit)"),
        ("Chemprop(pure)", "Chemprop (pure D-MPNN)"),
        ("Chemprop+Morgan(e2e)", "Chemprop + Morgan (end-to-end)"),
        ("Chemprop+Morgan(frozen)", "Chemprop + Morgan (frozen enc)"),
        ("Chemprop+Morgan(best_HP)", "Chemprop + Morgan (best HP)"),
        ("Ablation:FP-only_MLP", "Ablation: FP-only MLP"),
        ("Ablation:Expert-only_MLP", "Ablation: Expert-only MLP"),
    ],
    "scaffold": [
        ("MLP(Morgan)", "MLP (Morgan)"),
        ("MLP(Morgan+Expert)", "MLP (Morgan + Expert RDKit)"),
        ("Chemprop(pure)", "Chemprop (pure D-MPNN)"),
        ("Chemprop+Morgan(e2e)", "Chemprop + Morgan (end-to-end)"),
        ("Chemprop+Morgan(frozen)", "Chemprop + Morgan (frozen enc)"),
        ("Chemprop+Morgan(best_HP)", "Chemprop + Morgan (best HP)"),
        ("Ablation:FP-only_MLP", "Ablation: FP-only MLP"),
        ("Ablation:Expert-only_MLP", "Ablation: Expert-only MLP"),
    ],
}

COLS = [
    "R2",
    "RMSE",
    "MAE",
    "r",
    "top5%_recovery",
    "top10%_recovery",
    "top20%_recovery",
    "percentile_bin_acc",
]


def bold_best(col_vals, higher_better):
    """Return list of strings, bolding the best numeric value."""
    # Only consider actual floats/ints — skip None and pre-formatted strings like \text{err}
    valid = [
        (i, v)
        for i, v in enumerate(col_vals)
        if v is not None and isinstance(v, (int, float))
    ]
    best_i = (
        max(valid, key=lambda x: x[1] if higher_better else -x[1])[0] if valid else -1
    )
    result = []
    for i, v in enumerate(col_vals):
        if v is None:
            result.append("\\tbd")
        elif isinstance(v, str):
            result.append(v)  # already a LaTeX snippet, e.g. \text{err}
        else:
            s = f"{v:.4f}"
            result.append(f"\\textbf{{{s}}}" if i == best_i else s)
    return result


def build_rows(df, model_pairs):
    # higher_better flags per column
    hb = [True, False, False, True, True, True, True, True]
    rows_data = []
    for csv_name, _ in model_pairs:
        row = df[df["model"] == csv_name]
        if row.empty:
            rows_data.append([None] * len(COLS))
        else:
            vals = [row.iloc[0].get(c, None) for c in COLS]
            # Guard: if R² < -1 the model diverged — treat whole row as invalid
            r2 = vals[0]
            if r2 is not None and r2 < -1.0:
                print(
                    f"  WARNING: {csv_name} has R²={r2:.2f} (diverged) — shown as error"
                )
                vals = ["\\text{err}"] + [None] * (len(COLS) - 1)
            rows_data.append(vals)

    col_formatted = []
    for ci, (col, hb_flag) in enumerate(zip(COLS, hb)):
        col_vals = [r[ci] for r in rows_data]
        col_formatted.append(bold_best(col_vals, hb_flag))

    lines = []
    for ri, (_, tex_label) in enumerate(model_pairs):
        vals = [col_formatted[ci][ri] for ci in range(len(COLS))]
        lines.append(f"{tex_label:<40} & " + " & ".join(vals) + r" \\")
    return "\n".join(lines)


def fill_table(tex: str, split: str, df: pd.DataFrame) -> str:
    pairs = MODEL_MAP[split]
    new_body = build_rows(df, pairs)
    section_tag = "random_test" if split == "random" else "scaffold_test"
    pattern = rf"(\\label{{tab:{section_tag}}}.*?\\midrule\n)(.*?)(\\bottomrule)"
    # Use a lambda so re.sub never interprets backslashes in new_body as escape seqs
    return re.sub(
        pattern,
        lambda m: m.group(1) + new_body + "\n" + m.group(3),
        tex,
        flags=re.DOTALL,
    )


def main():
    tex = TEX_PATH.read_text()
    changed = False
    for split in ("random", "scaffold"):
        csv_path = RESULTS_DIR / f"results_{split}.csv"
        if not csv_path.exists():
            print(f"  {csv_path.name} not found — skipping {split} table.")
            continue
        df = pd.read_csv(csv_path)
        tex = fill_table(tex, split, df)
        changed = True
        print(f"  Filled {split} table from {csv_path.name}")

    if changed:
        TEX_PATH.write_text(tex)
        print(f"\nUpdated: {TEX_PATH}")
    else:
        print("No CSV results found. Run baseline_pipeline.py first.")


if __name__ == "__main__":
    main()

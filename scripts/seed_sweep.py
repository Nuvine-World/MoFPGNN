import os, sys, re, subprocess, argparse
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Run scripts/finetune_kpgt.py over several seeds and report mean +/- std of the
test metrics -- the right way to compare models on a 110-molecule test set,
where any single run carries ~0.01-0.02 R2 of noise.

Each seed is run as a fresh subprocess (clean memory/RNG), and the per-run test
metrics are read from the results/<run>/metrics.txt the fine-tune writes. The
summary is printed and saved to results/seed_sweep-<tag>-<split>-<ts>.txt.

Example (KPGT + CMF, 5 seeds, 50 epochs, random split):
    python scripts/seed_sweep.py \
        --kpgt_dir ./KPGT --checkpoint ./KPGT/models/pretrained/base/base.pth \
        --split random --morgan count --n_epochs 50 --seeds 0 1 2 3 4
"""

# Point-estimate metrics to aggregate (skip the _95CI string fields).
METRIC_KEYS = ["R2", "RMSE", "MAE", "Pearson_r",
               "EF@5%", "NDCG@5%", "HitRate@5%",
               "EF@10%", "NDCG@10%", "HitRate@10%", "mean_rel_error"]


def parse_test_metrics(metrics_path):
    """Read the 'Test:' block of a metrics.txt into {key: float}."""
    out, section = {}, None
    with open(metrics_path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = re.match(r"^(Train|Val|Test):\s*$", line.strip())
            if m:
                section = m.group(1)
                continue
            if section == "Test":
                mm = re.match(r"^\s+(\S+):\s+(.+)$", line)
                if mm:
                    key, val = mm.group(1), mm.group(2).strip()
                    try:
                        out[key] = float(val)
                    except ValueError:
                        pass  # CI strings like "[0.0, 16.0]"
    return out


def run_one(cmd):
    """Run a finetune command, stream its output, return its results dir."""
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    results_dir = None
    for line in proc.stdout:
        print(line, end="")
        m = re.search(r"All results saved to:\s*(\S+)", line)
        if m:
            results_dir = m.group(1).strip()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"finetune run failed (exit code {proc.returncode})")
    if results_dir is None:
        raise RuntimeError("could not find the run's results directory in its output")
    return results_dir


def main():
    p = argparse.ArgumentParser(description="Seed sweep for finetune_kpgt.py (mean +/- std)")
    p.add_argument("--kpgt_dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--split", default="random", choices=["random", "Murcko_scaffold"])
    p.add_argument("--morgan", default="none", choices=["none", "count", "binary"])
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = p.parse_args()

    ckpt = args.checkpoint or os.path.join(
        args.kpgt_dir, "models", "pretrained", "base", "base.pth")
    finetune = os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_kpgt.py")

    runs = {}  # seed -> {metric: value}
    for seed in args.seeds:
        cmd = [sys.executable, finetune,
               "--kpgt_dir", args.kpgt_dir, "--checkpoint", ckpt,
               "--split", args.split, "--morgan", args.morgan,
               "--n_epochs", str(args.n_epochs), "--lr", str(args.lr),
               "--seed", str(seed)]
        rdir = run_one(cmd)
        runs[seed] = parse_test_metrics(os.path.join(rdir, "metrics.txt"))

    # Aggregate
    tag = {"none": "KPGT", "count": "KPGT+CMF", "binary": "KPGT+BMF"}[args.morgan]
    header = (f"Seed sweep: {tag} | split={args.split} | "
              f"epochs={args.n_epochs} | lr={args.lr} | seeds={args.seeds}")

    lines = [header, "=" * len(header), ""]
    # Per-seed test R2 / RMSE quick view
    lines.append(f"{'seed':>6} | {'R2':>8} | {'RMSE':>8}")
    lines.append("-" * 30)
    for s in args.seeds:
        r2 = runs[s].get("R2", float("nan"))
        rmse = runs[s].get("RMSE", float("nan"))
        lines.append(f"{s:>6} | {r2:>8.4f} | {rmse:>8.4f}")
    lines.append("")
    # mean +/- std per metric
    lines.append(f"{'metric':>14} | {'mean':>9} | {'std':>8} | {'min':>8} | {'max':>8}")
    lines.append("-" * 58)
    for k in METRIC_KEYS:
        vals = np.array([runs[s][k] for s in args.seeds if k in runs[s]], dtype=float)
        if len(vals) == 0:
            continue
        lines.append(f"{k:>14} | {vals.mean():>9.4f} | {vals.std():>8.4f} | "
                     f"{vals.min():>8.4f} | {vals.max():>8.4f}")

    summary = "\n".join(lines)
    print("\n" + summary + "\n")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fp_suffix = {"none": "", "count": "-cmf", "binary": "-bmf"}[args.morgan]
    out_path = os.path.join(
        "results", f"seed_sweep-kpgt{fp_suffix}-{args.split}-{ts}.txt")
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(summary + "\n")
    print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()

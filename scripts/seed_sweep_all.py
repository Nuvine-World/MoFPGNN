import os, sys, re, csv, shutil, argparse, subprocess
from datetime import datetime
import numpy as np

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
    """Run one pipeline command, stream its output, return its results dir."""
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
        raise RuntimeError(f"run failed (exit code {proc.returncode})")
    if results_dir is None:
        raise RuntimeError("could not find the run's results directory in its output")
    return results_dir


# stem -> (human label, filename tag). config file = config/<stem>_config.yaml.
CONFIG_META = {
    "morgan_only":                   ("CMF only",   "morgan-cmf"),
    "morgan_binary_only":            ("BMF only",   "morgan-bmf"),
    "kpgt_pretrained_regressor":     ("KPGT only",  "kpgt"),
    "kpgt_morgan_pretrained_hybrid": ("KPGT + CMF", "kpgt-cmf"),
    "kpgt_morgan_binary_hybrid":     ("KPGT + BMF", "kpgt-bmf"),
}
DEFAULT_CONFIGS = list(CONFIG_META)
SPLITS = ["random"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 23, 42]


def _cfg_epochs_lr(cfg_path):
    """Best-effort read of `epochs` / `lr` from a YAML config (for the header)."""
    try:
        import yaml
        with open(cfg_path) as f:
            c = yaml.safe_load(f) or {}
        return c.get("epochs"), c.get("lr")
    except Exception:
        return None, None


def _cell_summary(label, split, seeds, runs, epochs=None, lr=None):
    """Human-readable per-seed + mean/std/min/max block for one (config, split)."""
    header = f"Seed sweep: {label} | split={split}"
    if epochs is not None:
        header += f" | epochs={epochs}"
    if lr is not None:
        header += f" | lr={lr}"
    header += f" | seeds={list(seeds)}"

    lines = [header, "=" * len(header), ""]
    lines.append(f"{'seed':>6} | {'R2':>8} | {'RMSE':>8}")
    lines.append("-" * 30)
    for s in seeds:
        r2 = runs[s].get("R2", float("nan"))
        rmse = runs[s].get("RMSE", float("nan"))
        lines.append(f"{s:>6} | {r2:>8.4f} | {rmse:>8.4f}")
    lines.append("")
    lines.append(f"{'metric':>14} | {'mean':>9} | {'std':>8} | {'min':>8} | {'max':>8}")
    lines.append("-" * 58)
    for k in METRIC_KEYS:
        vals = np.array([runs[s][k] for s in seeds
                         if k in runs[s] and not np.isnan(runs[s][k])], dtype=float)
        if len(vals) == 0:
            continue
        lines.append(f"{k:>14} | {vals.mean():>9.4f} | {vals.std():>8.4f} | "
                     f"{vals.min():>8.4f} | {vals.max():>8.4f}")
    return "\n".join(lines)


def _write_per_run_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "config", "split", "seed"] + METRIC_KEYS)
        for r in rows:
            w.writerow([r["model"], r["config"], r["split"], r["seed"]]
                       + [r["metrics"].get(k, "") for k in METRIC_KEYS])


def _write_summary(csv_path, txt_path, agg, splits, configs, seeds):
    # summary.csv : one row per (model, split), mean & std of each metric
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["model", "split", "n_seeds"]
        for k in METRIC_KEYS:
            head += [f"{k}_mean", f"{k}_std"]
        w.writerow(head)
        for (_, label, _) in configs:
            for split in splits:
                a = agg.get((label, split))
                if a is None:
                    continue  # cell not run yet (summary is refreshed mid-sweep)
                row = [label, split, a["n"]]
                for k in METRIC_KEYS:
                    row += [f"{a[k][0]:.4f}", f"{a[k][1]:.4f}"] if k in a else ["", ""]
                w.writerow(row)

    # summary.txt : compact R2/RMSE/MAE/Pearson/MRE table, grouped by split
    key5 = ["R2", "RMSE", "MAE", "Pearson_r", "mean_rel_error"]
    lines = [f"Full seed sweep -- mean +/- std over seeds={list(seeds)}",
             "=" * 60, ""]
    for split in splits:
        lines.append(f"### {split} split")
        lines.append(f"{'model':<12} | " + " | ".join(f"{k:>17}" for k in key5))
        lines.append("-" * (12 + 3 + len(key5) * 20))
        for (_, label, _) in configs:
            a = agg.get((label, split))
            cells = []
            for k in key5:
                if a is not None and k in a:
                    cells.append(f"{a[k][0]:>7.4f}+/-{a[k][1]:<7.4f}")
                else:
                    cells.append(f"{'n/a':>17}")
            lines.append(f"{label:<12} | " + " | ".join(cells))
        lines.append("")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(
        description="Seed sweep over the five paper models on the random split")
    p.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    p.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="config stems to run (default: the 5 canonical paper configs)")
    p.add_argument("--config_dir", default="config")
    p.add_argument("--results_dir", default="results",
                   help="where the per-config seed_sweep-*.txt summaries are written")
    p.add_argument("--out_dir", default=None,
                   help="dir for preserved per-run results + CSVs "
                        "(default: <results_dir>/seed_sweep_all-<timestamp>)")
    args = p.parse_args()

    configs = [(stem, *CONFIG_META.get(stem, (stem, stem))) for stem in args.configs]

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(args.results_dir, exist_ok=True)
    out_dir = args.out_dir or os.path.join(args.results_dir, f"seed_sweep_all-{ts}")
    os.makedirs(out_dir, exist_ok=True)

    pipeline = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_kpgt_pipeline.py")
    per_run_rows = []
    agg = {}   # (label, split) -> {metric: (mean, std), "n": n_seeds_ok}

    total = len(configs) * len(args.splits) * len(args.seeds)
    done = 0
    for stem, label, tag in configs:
        cfg = os.path.join(args.config_dir, f"{stem}_config.yaml")
        if not os.path.exists(cfg):
            print(f"WARNING: config not found, skipping: {cfg}")
            continue
        epochs, lr = _cfg_epochs_lr(cfg)
        for split in args.splits:
            runs = {}  # seed -> {metric: value}
            for seed in args.seeds:
                done += 1
                print(f"\n########## [{done}/{total}] {label} | {split} | seed {seed} ##########")
                cmd = [sys.executable, "-u", pipeline,
                       "--config", cfg, "--split", split, "--seed", str(seed)]
                try:
                    rdir = run_one(cmd)
                    metrics = parse_test_metrics(os.path.join(rdir, "metrics.txt"))
                    # Seed-tag the run dir: pipeline names collide within a minute.
                    dest = os.path.join(out_dir, f"{stem}-{split}-seed{seed}")
                    if os.path.abspath(rdir) != os.path.abspath(dest):
                        if os.path.exists(dest):
                            shutil.rmtree(dest)
                        shutil.move(rdir, dest)
                except Exception as e:
                    print(f"!! run failed ({label} | {split} | seed {seed}): {e}")
                    metrics = {k: float("nan") for k in METRIC_KEYS}
                runs[seed] = metrics
                per_run_rows.append({"model": label, "config": stem,
                                     "split": split, "seed": seed, "metrics": metrics})
                # Persist raw rows after every run so a crash never loses progress.
                _write_per_run_csv(os.path.join(out_dir, "per_run.csv"), per_run_rows)

            # Aggregate this cell
            a = {}
            for k in METRIC_KEYS:
                vals = np.array([runs[s][k] for s in args.seeds
                                 if k in runs[s] and not np.isnan(runs[s][k])], dtype=float)
                if len(vals):
                    a[k] = (float(vals.mean()), float(vals.std()))
            a["n"] = max((sum(1 for s in args.seeds
                              if not np.isnan(runs[s].get(k, float("nan"))))
                          for k in METRIC_KEYS), default=0)
            agg[(label, split)] = a

            # Per-config summary in the established naming convention, under results/.
            cell = _cell_summary(label, split, args.seeds, runs, epochs=epochs, lr=lr)
            print("\n" + cell + "\n")
            cell_ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
            cell_path = os.path.join(args.results_dir,
                                     f"seed_sweep-{tag}-{split}-{cell_ts}.txt")
            with open(cell_path, "w") as f:
                f.write(cell + "\n")
            print(f"  Wrote {cell_path}")

            # Refresh the combined summaries after each completed cell.
            _write_summary(os.path.join(out_dir, "summary.csv"),
                           os.path.join(out_dir, "summary.txt"),
                           agg, args.splits, configs, args.seeds)

    print(f"\nPer-config summaries : {args.results_dir}/seed_sweep-<tag>-<split>-<ts>.txt")
    print(f"Per-run results+CSVs : {out_dir}/")
    print(f"  <config>-<split>-seed<seed>/  full run dirs (metrics.txt, csvs, plots)")
    print(f"  per_run.csv / summary.csv / summary.txt")


if __name__ == "__main__":
    main()

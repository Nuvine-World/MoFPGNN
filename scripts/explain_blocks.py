import os, sys, csv, argparse, pathlib
from copy import deepcopy

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import torch

from utils.io_tools import load_yaml
from utils.utils import seed_everything
from pipeline.kpgt_pipeline import _load_pretrained_hybrid_datasets, _normalise_labels
from pipeline.kpgt_trainer import KPGTTrainer
from models.hybrid_model import PretrainedKPGTMorganHybrid


KPGT_DIM_DEFAULT = 2304


def _r2(pred, true):
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - np.mean(true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _tensors(ds, device):
    """Stack a hybrid dataset into (kpgt, morgan, labels) tensors."""
    X = torch.cat([d.x.view(1, -1) for d in ds._data_list]).to(device)
    M = torch.cat([d.morgan_fp.view(1, -1) for d in ds._data_list]).to(device)
    y = torch.cat([d.y.view(1, -1) for d in ds._data_list]).view(-1).cpu().numpy()
    return X, M, y


def _predict(model, X, M, mean, std, batch=256):
    """Forward pass -> predictions in the ORIGINAL label scale."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            out.append(model(X[i:i + batch], M[i:i + batch]).cpu())
    return torch.cat(out).view(-1).numpy() * std + mean


def permutation_importance(model, X, M, y_true, mean, std, rng, n_repeats=20):
    """R2 drop when each block is shuffled across molecules."""
    base = _r2(_predict(model, X, M, mean, std), y_true)
    drops = {}
    for name in ("kpgt", "morgan"):
        vals = []
        for _ in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(X.shape[0])).to(X.device)
            if name == "kpgt":
                r = _r2(_predict(model, X[perm], M, mean, std), y_true)
            else:
                r = _r2(_predict(model, X, M[perm], mean, std), y_true)
            vals.append(base - r)
        drops[name] = float(np.mean(vals))
    return base, drops


def occlusion_importance(model, X, M, y_true, mean, std):
    """R2 drop when each block is replaced by the training mean (zeros)."""
    base = _r2(_predict(model, X, M, mean, std), y_true)
    zX, zM = torch.zeros_like(X), torch.zeros_like(M)
    return base, {
        "kpgt": base - _r2(_predict(model, zX, M, mean, std), y_true),
        "morgan": base - _r2(_predict(model, X, zM, mean, std), y_true),
    }


def integrated_gradients(model, X, M, steps=64, batch=128):
    """
    Integrated gradients with the mean molecule (zeros, post z-scoring) as
    baseline. Returns the summed |attribution| carried by each block.
    """
    model.eval()
    totX = torch.zeros(X.shape[1], device=X.device)
    totM = torch.zeros(M.shape[1], device=M.device)
    for i in range(0, X.shape[0], batch):
        xb, mb = X[i:i + batch], M[i:i + batch]
        gX = torch.zeros_like(xb)
        gM = torch.zeros_like(mb)
        for s in range(1, steps + 1):
            a = float(s) / steps                      # baseline is 0 -> path is a*x
            xs = (a * xb).clone().requires_grad_(True)
            ms = (a * mb).clone().requires_grad_(True)
            out = model(xs, ms).sum()
            gx, gm = torch.autograd.grad(out, [xs, ms])
            gX += gx
            gM += gm
        # (x - baseline) * average gradient along the path
        totX += (xb * gX / steps).abs().sum(dim=0)
        totM += (mb * gM / steps).abs().sum(dim=0)
    return float(totX.sum()), float(totM.sum())


def shap_importance(model, X, M, n_background=100, n_explain=110):
    """Optional SHAP GradientExplainer; returns per-block summed |SHAP|."""
    try:
        import shap
    except ImportError:
        return None
    kd = X.shape[1]

    class Joint(torch.nn.Module):
        def __init__(self, m, kd):
            super().__init__()
            self.m, self.kd = m, kd

        def forward(self, z):
            return self.m(z[:, :self.kd], z[:, self.kd:])

    Z = torch.cat([X, M], dim=1)
    joint = Joint(model, kd).eval()
    bg = Z[:min(n_background, Z.shape[0])]
    try:
        expl = shap.GradientExplainer(joint, bg)
        sv = expl.shap_values(Z[:min(n_explain, Z.shape[0])])
        sv = sv[0] if isinstance(sv, list) else sv
        sv = np.abs(np.asarray(sv))
        return float(sv[:, :kd].sum()), float(sv[:, kd:].sum())
    except Exception as e:
        print(f"    (shap failed: {e})")
        return None


def run_seed(cfg_path, split, seed, n_repeats):
    config = load_yaml(cfg_path)
    config["split"] = split
    config["split_path"] = f"data/splits/{config.get('dataset','AGILE')}/{split}.npy"
    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    config["seed"] = seed          # reaches the trainer's DataLoader generator
    seed_everything(seed)

    train_ds, val_ds, test_ds = _load_pretrained_hybrid_datasets(config)
    mean, std = _normalise_labels(train_ds, val_ds, test_ds)

    model = PretrainedKPGTMorganHybrid(
        kpgt_dim=config.get("kpgt_dim", KPGT_DIM_DEFAULT),
        morgan_dim=config.get("morgan_dim", 2048),
        mlp_hidden=config.get("mlp_head", {}).get("hidden_layers", [512, 256]),
        dropout=config.get("mlp_head", {}).get("dropout", 0.3),
    )
    trainer = KPGTTrainer(model, config, label_mean=mean, label_std=std)
    trainer.train(train_ds, val_ds)

    device = torch.device(config["device"])
    model = trainer.model.to(device).eval()
    X, M, y_norm = _tensors(test_ds, device)
    y_true = y_norm * std + mean          # labels were z-scored in place

    rng = np.random.default_rng(seed)
    rows = []

    base, drop = permutation_importance(model, X, M, y_true, mean, std, rng, n_repeats)
    tot = drop["kpgt"] + drop["morgan"]
    rows.append(dict(method="permutation", base_r2=base,
                     kpgt=drop["kpgt"], morgan=drop["morgan"],
                     kpgt_share=drop["kpgt"] / tot if tot > 0 else float("nan")))

    base, drop = occlusion_importance(model, X, M, y_true, mean, std)
    tot = drop["kpgt"] + drop["morgan"]
    rows.append(dict(method="occlusion", base_r2=base,
                     kpgt=drop["kpgt"], morgan=drop["morgan"],
                     kpgt_share=drop["kpgt"] / tot if tot > 0 else float("nan")))

    aX, aM = integrated_gradients(model, X, M)
    rows.append(dict(method="intgrad", base_r2=base, kpgt=aX, morgan=aM,
                     kpgt_share=aX / (aX + aM) if (aX + aM) > 0 else float("nan")))

    sh = shap_importance(model, X, M)
    if sh is not None:
        sX, sM = sh
        rows.append(dict(method="shap", base_r2=base, kpgt=sX, morgan=sM,
                         kpgt_share=sX / (sX + sM) if (sX + sM) > 0 else float("nan")))

    for r in rows:
        r["seed"] = seed
    return rows


def main():
    p = argparse.ArgumentParser(
        description="Block-level XAI: KPGT vs Morgan contribution in the hybrids")
    p.add_argument("--config", default="config/kpgt_morgan_pretrained_hybrid_config.yaml")
    p.add_argument("--split", default="random",
                   choices=["random"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n_repeats", type=int, default=20,
                   help="permutation shuffles per block per seed")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    tag = "cmf" if "binary" not in os.path.basename(args.config) else "bmf"
    out_dir = args.out_dir or os.path.join("results", f"xai-{tag}-{args.split}")
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for seed in args.seeds:
        print(f"\n########## XAI | {tag} | {args.split} | seed {seed} ##########")
        all_rows += run_seed(args.config, args.split, seed, args.n_repeats)
        with open(os.path.join(out_dir, "block_attribution.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "method", "base_r2",
                                              "kpgt", "morgan", "kpgt_share"])
            w.writeheader()
            w.writerows(all_rows)

    # Aggregate
    methods = []
    for r in all_rows:
        if r["method"] not in methods:
            methods.append(r["method"])
    lines = [f"Block attribution: KPGT vs Morgan | model=KPGT+{tag.upper()} | "
             f"split={args.split} | seeds={args.seeds}", "=" * 78, ""]
    lines.append(f"{'method':>12} | {'KPGT share':>18} | {'KPGT effect':>16} | {'Morgan effect':>16}")
    lines.append("-" * 78)
    for m in methods:
        sub = [r for r in all_rows if r["method"] == m]
        sh = np.array([r["kpgt_share"] for r in sub], dtype=float)
        kv = np.array([r["kpgt"] for r in sub], dtype=float)
        mv = np.array([r["morgan"] for r in sub], dtype=float)
        lines.append(f"{m:>12} | {100*np.nanmean(sh):8.1f}% +/-{100*np.nanstd(sh, ddof=1):5.1f}% | "
                     f"{kv.mean():8.4f}+/-{kv.std(ddof=1):6.4f} | "
                     f"{mv.mean():8.4f}+/-{mv.std(ddof=1):6.4f}")
    lines += ["",
              "permutation/occlusion effect = drop in test R2 when that block is destroyed.",
              "intgrad/shap effect = summed |attribution| carried by that block.",
              "KPGT share = KPGT effect / (KPGT effect + Morgan effect); 50% = equal reliance."]
    summary = "\n".join(lines)
    print("\n" + summary + "\n")
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print(f"Saved to {out_dir}/")


if __name__ == "__main__":
    main()

import os
import sys
import pathlib
import shutil
import warnings
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime

sys.path.insert(0, os.path.dirname(pathlib.Path(__file__).parent.absolute()))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from utils.io_tools import load_pickle  # noqa: F401  (kept for parity with other scripts)
from utils.metrics import compute_all_metrics, save_metrics_to_file

warnings.filterwarnings("ignore")

"""
End-to-end fine-tuning of the pretrained KPGT (LiGhT) graph transformer on AGILE.

This is the "KPGT-only" baseline as a FULL fine-tune (the whole transformer is
trained, not a frozen-embedding probe). The recipe mirrors the reference KPGT
fine-tuning protocol: the pretrained `base.pth` backbone with a fresh 3-layer
GELU regression head, Adam (lr 4e-5, weight_decay 0), a polynomial-decay LR
schedule with ~10% warmup, gradient clipping at 5.0, MSE on z-scored labels
(train-set mean/std), batch size 32, and best-validation-RMSE selection with an
early-stopping patience of 20 epochs.

The heavy model/data code (the LiGhT model, the molecule dataset, the collator,
the tokenizer/featurizer, and preprocessing) is imported from a separately
cloned KPGT repository -- exactly like `scripts/extract_kpgt_embeddings.py`
does -- via `--kpgt_dir`. Nothing in the KPGT clone is modified; all new logic
lives here in MoFPGNN.

Requirements (run in the `KPGT` conda env, which has dgl + the KPGT deps):
  * a cloned KPGT repo:        git clone https://github.com/lihan97/kpgt.git
  * the pretrained checkpoint: base.pth  (figshare link printed on error)

Outputs (under results/kpgt_finetune-<split>-<timestamp>/):
  * metrics.txt                     (R2/RMSE/MAE/Pearson + screening + rel-error)
  * {train,val,test}_results.csv    (true,pred in original mTP scale)
  * loss_curve / regression / confusion plots (via utils.visual_utils)

Example:
  python scripts/finetune_kpgt.py \
      --kpgt_dir /path/to/kpgt \
      --checkpoint /path/to/base.pth \
      --split random --seed 23
"""


# --------------------------------------------------------------------------- #
# LR schedule (re-implemented here so the script does not depend on a specific
# module path inside the KPGT clone). Linear warmup then polynomial decay.
# --------------------------------------------------------------------------- #
class PolynomialDecayLR(_LRScheduler):
    def __init__(self, optimizer, warmup_updates, tot_updates, lr, end_lr,
                 power=1.0, last_epoch=-1):
        self.warmup_updates = max(1, int(warmup_updates))
        self.tot_updates = max(self.warmup_updates + 1, int(tot_updates))
        self.lr = lr
        self.end_lr = end_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            factor = self._step_count / float(self.warmup_updates)
            lr = factor * self.lr
        elif self._step_count >= self.tot_updates:
            lr = self.end_lr
        else:
            pct_remaining = 1 - (self._step_count - self.warmup_updates) / (
                self.tot_updates - self.warmup_updates)
            lr = (self.lr - self.end_lr) * pct_remaining ** self.power + self.end_lr
        return [lr for _ in self.optimizer.param_groups]


def _init_params(module):
    """Small-normal init for the fresh regression head, matching KPGT."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def _build_predictor(d_input, n_tasks, n_layers, dropout, device, d_hidden):
    """3-layer (by default) GELU MLP head placed on the [g||fp||md] readout."""
    n_layers = int(n_layers)
    if n_layers == 1:
        predictor = nn.Linear(int(d_input), n_tasks)
    else:
        layers = [nn.Linear(int(d_input), int(d_hidden)), nn.Dropout(dropout), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(int(d_hidden), int(d_hidden)), nn.Dropout(dropout), nn.GELU()]
        layers += [nn.Linear(int(d_hidden), n_tasks)]
        predictor = nn.Sequential(*layers)
    predictor.apply(_init_params)
    return predictor.to(device)


class GatedFusion(nn.Module):
    """
    Gated late fusion of the KPGT readout with a Morgan fingerprint. The Morgan
    fingerprint is encoded to `emb_dim`, then a sigmoid gate (conditioned on the
    graph readout) scales it before concatenation:

        m   = enc(morgan)                  # 2048 -> emb_dim (Linear-GELU-Dropout)
        g   = sigmoid(W[readout || m])     # per-dim gate in (0, 1)
        out = [ readout || g * m ]         # head input, dim = kpgt_dim + emb_dim

    The gate lets the model down-weight the fingerprint when the graph already
    explains the molecule, which makes the hybrid more robust than a raw concat
    (this was the best-performing fusion in our experiments).
    """

    def __init__(self, kpgt_dim, morgan_dim, emb_dim=256, dropout=0.2):
        super().__init__()
        self.morgan_enc = nn.Sequential(
            nn.Linear(morgan_dim, emb_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(
            nn.Linear(kpgt_dim + emb_dim, emb_dim), nn.Sigmoid())
        self.out_dim = kpgt_dim + emb_dim

    def forward(self, readout, morgan):
        m = self.morgan_enc(morgan)
        g = self.gate(torch.cat([readout, m], dim=1))
        return torch.cat([readout, g * m], dim=1)


class KPGTFineTuneModel(nn.Module):
    """
    Wraps the pretrained LiGhT backbone with a fresh regression head, optionally
    fusing an external Morgan fingerprint via gated late fusion.

    The 2304-dim graph readout is taken from `LiGhT.generate_fps` (the full
    transformer forward, gradients enabled, so the whole backbone is fine-tuned).
    For the hybrids the (z-scored) Morgan fingerprint is combined with the readout
    through a `GatedFusion` block before the head. With no Morgan, this is the
    plain KPGT fine-tune (head input = 2304).
    """

    def __init__(self, kpgt_model, head, morgan_lookup=None, fusion=None):
        super().__init__()
        self.kpgt = kpgt_model        # LiGhT with pretraining heads deleted
        self.head = head              # fresh GELU MLP (in MoFPGNN)
        self.morgan_lookup = morgan_lookup   # dict smiles -> scaled np.float32 fp
        self.fusion = fusion          # GatedFusion (hybrids) or None (KPGT-only)

    def forward(self, smiles_list, g, ecfp, md):
        g_feats = self.kpgt.generate_fps(g, ecfp, md)   # (B, 2304), grad-enabled
        if self.morgan_lookup is not None:
            m = np.stack([self.morgan_lookup[s] for s in smiles_list]).astype(np.float32)
            m = torch.from_numpy(m).to(g_feats.device)
            g_feats = self.fusion(g_feats, m)
        return self.head(g_feats)


def _build_morgan_lookup(morgan_path, smiles_order, train_idx):
    """
    Load a Morgan fingerprint pickle (smiles -> vector), z-score it on the TRAIN
    molecules (clipped to [-10, 10], matching the project's other feature
    scaling), and return (lookup_dict, morgan_dim).
    """
    from sklearn.preprocessing import StandardScaler
    raw = load_pickle(morgan_path)
    fps = []
    for smi in smiles_order:
        v = raw.get(smi)
        if v is None:
            raise KeyError(f"Morgan fingerprint missing for SMILES: {smi}")
        fps.append(np.asarray(v, dtype=np.float32))
    fps = np.nan_to_num(np.array(fps, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler().fit(fps[train_idx])
    fps = np.clip(scaler.transform(fps).astype(np.float32), -10.0, 10.0)
    lookup = {smi: fps[i] for i, smi in enumerate(smiles_order)}
    return lookup, fps.shape[1]


# --------------------------------------------------------------------------- #
# KPGT clone wiring
# --------------------------------------------------------------------------- #
def _import_kpgt(kpgt_dir):
    """Import the LiGhT model + data utilities from the cloned KPGT repo."""
    sys.path.insert(0, kpgt_dir)
    try:
        import dgl  # noqa: F401  (required by the KPGT model/dataset)
        from src.data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES
        from src.data.finetune_dataset import MoleculeDataset
        from src.data.collator import Collator_tune
        from src.model.light import LiGhTPredictor as LiGhT
        from src.model_config import config_dict
    except ImportError as e:
        print(f"ERROR: cannot import KPGT modules from '{kpgt_dir}'. ({e})")
        print("\n  Make sure you have, in the `KPGT` conda env:")
        print("    1. Cloned the repo:  git clone https://github.com/lihan97/kpgt.git")
        print("    2. Installed deps:   pip install dgl dgllife rdkit")
        sys.exit(1)
    return dict(Vocab=Vocab, N_ATOM_TYPES=N_ATOM_TYPES, N_BOND_TYPES=N_BOND_TYPES,
                MoleculeDataset=MoleculeDataset, Collator_tune=Collator_tune,
                LiGhT=LiGhT, config_dict=config_dict)


def _seed_everything(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import dgl
        dgl.seed(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Data preparation: build the KPGT-format directory from MoFPGNN's data + splits
# --------------------------------------------------------------------------- #
def _prepare_kpgt_data_dir(kpgt_data_root, data_name, csv_path, splits_dir):
    """
    Create <kpgt_data_root>/<data_name>/ with:
        <name>.csv          (columns: smiles, Target  -- nothing else)
        splits/<split>.npy  (copied from MoFPGNN's splits)
    The KPGT preprocessing step then fills in the graph/fp/md caches.
    """
    out_dir = os.path.join(kpgt_data_root, data_name)
    os.makedirs(os.path.join(out_dir, "splits"), exist_ok=True)

    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns or "Target" not in df.columns:
        raise ValueError(f"{csv_path} must contain 'SMILES' and 'Target' columns")
    # KPGT treats every non-'smiles' column as a task -> keep ONLY smiles + Target
    out_df = pd.DataFrame({"smiles": df["SMILES"].values,
                           "Target": df["Target"].values.astype(np.float32)})
    out_csv = os.path.join(out_dir, f"{data_name}.csv")
    out_df.to_csv(out_csv, index=False)

    for fname in os.listdir(splits_dir):
        if fname.endswith(".npy"):
            shutil.copyfile(os.path.join(splits_dir, fname),
                            os.path.join(out_dir, "splits", fname))
    print(f"  Prepared KPGT-format data dir: {out_dir}")
    return out_dir


def _maybe_preprocess(kpgt_dir, kpgt_data_root, data_name, path_length=5,
                      n_jobs=1, force=False):
    """Run KPGT's downstream preprocessing if the caches are missing."""
    cache = os.path.join(kpgt_data_root, data_name, f"{data_name}_{path_length}.pkl")
    fp = os.path.join(kpgt_data_root, data_name, "rdkfp1-7_512.npz")
    md = os.path.join(kpgt_data_root, data_name, "molecular_descriptors.npz")
    if not force and all(os.path.exists(p) for p in (cache, fp, md)):
        print("  KPGT caches already present, skipping preprocessing.")
        return

    print("  Running KPGT downstream preprocessing (graphs + fps + descriptors)...")
    import importlib.util
    from argparse import Namespace
    pp_path = os.path.join(kpgt_dir, "scripts", "preprocess_downstream_dataset.py")
    if not os.path.exists(pp_path):
        raise FileNotFoundError(
            f"KPGT preprocessing script not found at {pp_path}. "
            "Check --kpgt_dir points at the cloned KPGT repository.")
    spec = importlib.util.spec_from_file_location("kpgt_preprocess", pp_path)
    pdd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pdd)
    # Build the args object directly: the official KPGT repo's parse_args() reads
    # sys.argv (takes no positional list), unlike the TBD-main copy. preprocess_dataset
    # only reads .data_path / .dataset / .path_length / .n_jobs, so construct those.
    pp_args = Namespace(data_path=kpgt_data_root, dataset=data_name,
                        path_length=path_length, n_jobs=n_jobs)
    pdd.preprocess_dataset(pp_args)
    print("  Preprocessing complete.")


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def _forward_batch(model, batch, device):
    """Unpack a collated KPGT batch and run the fine-tune model (readout + head)."""
    smiles, g, ecfp, md, labels = batch
    g = g.to(device)
    ecfp = ecfp.to(device)
    md = md.to(device)
    labels = labels.to(device)
    preds = model(smiles, g, ecfp, md)
    return preds, labels


def _rmse_denorm(model, loader, device, mean, std):
    """Validation RMSE on denormalised predictions (model output * std + mean)."""
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            preds, labels = _forward_batch(model, batch, device)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    preds = preds * std + mean            # denormalise to original scale
    return float(np.sqrt(np.mean((labels - preds) ** 2)))


def _predict_denorm(model, loader, device, mean, std):
    """Return (preds, labels) in original mTP scale as flat numpy arrays."""
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            preds, labels = _forward_batch(model, batch, device)
            preds_all.append(preds.cpu())
            labels_all.append(labels.cpu())
    preds = torch.cat(preds_all).numpy() * std + mean
    labels = torch.cat(labels_all).numpy()
    return preds.reshape(-1), labels.reshape(-1)


def finetune(args):
    kp = _import_kpgt(args.kpgt_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1) Build the KPGT-format data dir from MoFPGNN data + splits, then preprocess
    splits_dir = os.path.join("data", "splits", args.data_name)
    csv_path = os.path.join("data", f"{args.data_name}.csv")
    _prepare_kpgt_data_dir(args.kpgt_data_root, args.data_name, csv_path, splits_dir)
    _maybe_preprocess(args.kpgt_dir, args.kpgt_data_root, args.data_name,
                      path_length=5, n_jobs=max(1, args.n_threads),
                      force=args.force_preprocess)

    # 2) Datasets / loaders
    MoleculeDataset = kp["MoleculeDataset"]
    Collator_tune = kp["Collator_tune"]
    config = kp["config_dict"]["base"]
    vocab = kp["Vocab"](kp["N_ATOM_TYPES"], kp["N_BOND_TYPES"])
    collator = Collator_tune(config["path_length"])

    ds_kwargs = dict(root_path=args.kpgt_data_root, dataset=args.data_name,
                     dataset_type="regression", split_name=args.split)
    train_ds = MoleculeDataset(split="train", **ds_kwargs)
    val_ds = MoleculeDataset(split="val", **ds_kwargs)
    test_ds = MoleculeDataset(split="test", **ds_kwargs)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=True, collate_fn=collator,
                              num_workers=args.n_threads, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collator, num_workers=args.n_threads)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collator, num_workers=args.n_threads)

    # Train-set label mean/std for z-scoring (denormalised again at eval time)
    mean_t = train_ds.mean.to(device)
    std_t = train_ds.std.to(device)
    mean_np = train_ds.mean.numpy()
    std_np = train_ds.std.numpy()
    print(f"  Label normalisation: mean={float(mean_np[0]):.4f}, std={float(std_np[0]):.4f}")

    # 2b) Optional Morgan fingerprint to fuse with the KPGT readout
    morgan_lookup, morgan_dim = None, 0
    if args.morgan != "none":
        morgan_path = args.morgan_path or os.path.join(
            "data", "fingerprints", args.data_name,
            "morgan_count.pkl" if args.morgan == "count" else "morgan_binary.pkl")
        if not os.path.exists(morgan_path):
            print(f"ERROR: Morgan fingerprint pickle not found: {morgan_path}")
            print("Generate it first, e.g.: python scripts/extract_morgan_fingerprint.py "
                  f"--fp_type {args.morgan}")
            sys.exit(1)
        df_full = pd.read_csv(csv_path)
        smiles_order = df_full["SMILES"].tolist()
        train_idx = np.load(
            os.path.join(splits_dir, f"{args.split}.npy"), allow_pickle=True)[0]
        morgan_lookup, morgan_dim = _build_morgan_lookup(
            morgan_path, smiles_order, train_idx)
        print(f"  Fusing Morgan FP ({args.morgan}, dim={morgan_dim}) from {morgan_path}")

    # 3) Model: pretrained LiGhT backbone + fresh GELU head (+ optional Morgan fusion)
    LiGhT = kp["LiGhT"]
    kpgt_model = LiGhT(
        d_node_feats=config["d_node_feats"], d_edge_feats=config["d_edge_feats"],
        d_g_feats=config["d_g_feats"], d_fp_feats=train_ds.d_fps,
        d_md_feats=train_ds.d_mds, d_hpath_ratio=config["d_hpath_ratio"],
        n_mol_layers=config["n_mol_layers"], path_length=config["path_length"],
        n_heads=config["n_heads"], n_ffn_dense_layers=config["n_ffn_dense_layers"],
        input_drop=0, attn_drop=args.dropout, feat_drop=args.dropout,
        n_node_types=vocab.vocab_size,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found at: {args.checkpoint}")
        print("Download base.pth from: https://figshare.com/s/d488f30c23946cf6898f")
        sys.exit(1)
    state = torch.load(args.checkpoint, map_location="cpu")
    kpgt_model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})

    # Drop ALL pretraining heads -- we read the readout via generate_fps and apply
    # our own head, so none of the original predictors are used.
    for head in ("predictor", "md_predictor", "fp_predictor", "node_predictor"):
        if hasattr(kpgt_model, head):
            delattr(kpgt_model, head)

    # For the hybrids, fuse the Morgan fingerprint with the readout via gated
    # fusion; the head consumes the fused vector. KPGT-only uses no fusion.
    kpgt_dim = config["d_g_feats"] * 3
    if morgan_lookup is not None:
        fusion = GatedFusion(kpgt_dim, morgan_dim, emb_dim=args.morgan_emb_dim,
                             dropout=args.dropout).to(device)
        head_in = fusion.out_dim
    else:
        fusion = None
        head_in = kpgt_dim

    head = _build_predictor(
        d_input=head_in, n_tasks=train_ds.n_tasks,
        n_layers=args.n_layers, dropout=args.dropout, device=device,
        d_hidden=args.d_hidden_feats)
    model = KPGTFineTuneModel(kpgt_model, head, morgan_lookup=morgan_lookup,
                              fusion=fusion).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    tag = {"none": "KPGT-only", "count": "KPGT + CMF (gated)",
           "binary": "KPGT + BMF (gated)"}[args.morgan]
    print(f"  Model: LiGhT fine-tune [{tag}] | {n_params:.2f}M params")

    # 4) Optimiser / schedule / loss
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_ds) // args.batch_size)
    tot_updates = args.n_epochs * steps_per_epoch
    scheduler = PolynomialDecayLR(optimizer, warmup_updates=tot_updates // 10,
                                  tot_updates=tot_updates, lr=args.lr,
                                  end_lr=1e-9, power=1)
    criterion = nn.MSELoss(reduction="none")

    # 5) Training loop with best-val selection + early stopping
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.n_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            preds, labels = _forward_batch(model, batch, device)
            is_labeled = (~torch.isnan(labels)).float()
            labels = torch.nan_to_num(labels)
            labels_norm = (labels - mean_t) / std_t
            loss = (criterion(preds, labels_norm) * is_labeled).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            scheduler.step()
            running += loss.item() * preds.shape[0]
            n += preds.shape[0]
        avg_train = running / max(n, 1)

        val_rmse = _rmse_denorm(model, val_loader, device, mean_np, std_np)
        train_losses.append(avg_train)
        val_losses.append(val_rmse)
        print(f"Epoch {epoch:<3} | train(norm-MSE): {avg_train:.4f} | "
              f"val RMSE: {val_rmse:.4f} | lr: {optimizer.param_groups[0]['lr']:.2e}")

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
        if epoch - best_epoch >= args.early_stop:
            print(f"\nEarly stopping at epoch {epoch} (no val improvement "
                  f"for {args.early_stop} epochs)")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest model -- val RMSE: {best_val:.4f} (epoch {best_epoch})\n")

    # Save the fine-tuned checkpoint FIRST, so an expensive run is never lost to
    # a downstream (metrics/plotting) error.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    fp_suffix = {"none": "", "count": "-cmf", "binary": "-bmf"}[args.morgan]
    pipeline_name = f"kpgt_finetune{fp_suffix}-{args.split}-{timestamp}"
    results_dir = os.path.join("results", pipeline_name)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("checkpoints", f"{pipeline_name}.pth"))

    # 6) Predictions (original scale) + metrics + saving
    train_preds, train_labels = _predict_denorm(model, train_loader, device, mean_np, std_np)
    val_preds, val_labels = _predict_denorm(model, val_loader, device, mean_np, std_np)
    test_preds, test_labels = _predict_denorm(model, test_loader, device, mean_np, std_np)

    # Always write the raw predictions before computing metrics, so the run's
    # outputs survive even if a metric helper raises.
    for name, preds, labels in [("train", train_preds, train_labels),
                                ("val", val_preds, val_labels),
                                ("test", test_preds, test_labels)]:
        pd.DataFrame({"true": labels, "pred": preds}).to_csv(
            os.path.join(results_dir, f"{name}_results.csv"), index=False)

    train_metrics = compute_all_metrics(train_preds, train_labels, "Train")
    val_metrics = compute_all_metrics(val_preds, val_labels, "Val")
    test_metrics = compute_all_metrics(test_preds, test_labels, "Test")

    print(f"\n{'=' * 50}\nTEST RESULTS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print(f"{'=' * 50}\n")

    save_metrics_to_file(
        {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics},
        os.path.join(results_dir, "metrics.txt"), title=pipeline_name)

    # Train/validation loss curve -- matplotlib only (no seaborn dependency), so
    # it is always produced even if the seaborn-based plots below fail.
    # Both curves are shown in the same units: normalised MSE (z-scored label
    # space). The training loss is already normalised MSE; the validation RMSE is
    # converted via norm_MSE = (RMSE / label_std)^2 so the two are comparable.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        std0 = float(std_np[0])
        val_mse_curve = [(r / std0) ** 2 for r in val_losses]
        epochs_axis = range(1, len(train_losses) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_axis, train_losses, marker="o", ms=3, label="Train loss (norm-MSE)")
        ax.plot(epochs_axis, val_mse_curve, marker="s", ms=3, label="Validation loss (norm-MSE)")
        ax.axvline(best_epoch, color="grey", ls="--", lw=1,
                   label=f"best epoch ({best_epoch})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalised MSE loss")
        ax.set_title(f"{pipeline_name}\nbest val RMSE = {best_val:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "loss_curve.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved loss curve -> {results_dir}/loss_curve.png")
    except Exception as e:
        print(f"Warning: could not plot loss curve: {e}")

    # Regression / confusion plots (shared visual utils; needs seaborn).
    try:
        from utils.visual_utils import generate_all_regression_plots
        cfg = {"pipeline_save_name": pipeline_name}
        generate_all_regression_plots(train_preds, val_preds, test_preds,
                                      train_labels, val_labels, test_labels, cfg)
    except Exception as e:
        print(f"Warning: could not generate regression/confusion plots: {e}")

    print(f"All results saved to: {results_dir}")
    return test_metrics


def get_args():
    p = ArgumentParser(description="End-to-end fine-tune of pretrained KPGT on AGILE")
    p.add_argument("--kpgt_dir", type=str, required=True,
                   help="Path to the cloned KPGT repository (lihan97/kpgt)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to base.pth (default: <kpgt_dir>/models/pretrained/base/base.pth)")
    p.add_argument("--data_name", type=str, default="AGILE")
    p.add_argument("--kpgt_data_root", type=str, default="data/kpgt_finetune",
                   help="Working dir for the KPGT-format data + caches")
    p.add_argument("--split", type=str, default="random",
                   choices=["random", "Murcko_scaffold"])
    p.add_argument("--morgan", type=str, default="none",
                   choices=["none", "count", "binary"],
                   help="Fuse a Morgan fingerprint with the KPGT readout via gated "
                        "fusion: none=KPGT-only, count=KPGT+CMF, binary=KPGT+BMF")
    p.add_argument("--morgan_path", type=str, default=None,
                   help="Override the Morgan pickle path (default: "
                        "data/fingerprints/<data>/morgan_{count,binary}.pkl)")
    p.add_argument("--morgan_emb_dim", type=int, default=256,
                   help="Encoded Morgan dimension in the gated fusion (hybrids only)")
    p.add_argument("--force_preprocess", action="store_true",
                   help="Rebuild the KPGT graph/fp/descriptor caches even if present")

    # LANTERN/KPGT fine-tune recipe defaults
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--n_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--n_layers", type=int, default=3)
    p.add_argument("--d_hidden_feats", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--n_threads", type=int, default=0)
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.kpgt_dir, "models", "pretrained", "base", "base.pth")
    return args


if __name__ == "__main__":
    args = get_args()
    _seed_everything(args.seed)
    finetune(args)

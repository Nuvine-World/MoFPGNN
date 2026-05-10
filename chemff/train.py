from __future__ import annotations

import os
import argparse
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Subset
from torch.amp import autocast, GradScaler
import pandas as pd

from chemff.model import ConcatFFN
from chemff.dataset import MoleculeDataset, collate_fn, load_dataset_from_config
from chemff.chemprop.featurizer import MolecularGraphFeaturizer
from chemff.morgan.featurizer import MorganFeaturizer

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def prepare_dataloaders(cfg: dict, smiles: List[str], labels: List[float]) -> Dict:
    """Build train / val / test DataLoaders with GPU-optimised settings."""
    split_path = f"data/splits/{cfg['dataset']}/{cfg['split_type']}.npy"
    if not os.path.exists(split_path):
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Run: python scripts/split_dataset.py first."
        )
    train_idx, val_idx, test_idx = np.load(split_path, allow_pickle=True)

    # ---- Morgan fingerprints ----
    morgan_cfg = cfg["morgan"]
    morgan_feat = MorganFeaturizer(
        radius=morgan_cfg["radius"],
        n_bits=morgan_cfg["n_bits"],
        use_counts=morgan_cfg["use_counts"],
        use_chirality=morgan_cfg["use_chirality"],
    )

    fp_path = morgan_cfg["fp_path"]
    if os.path.exists(fp_path):
        print(f"Loading pre-computed Morgan fingerprints from {fp_path}")
        morgan_fp_dict = MorganFeaturizer.load(fp_path)
    else:
        print("Computing Morgan fingerprints on the fly...")
        morgan_fp_dict = None

    # ---- Graph featurizer ----
    graph_feat = MolecularGraphFeaturizer()

    # ---- Dataset ----
    full_dataset = MoleculeDataset(
        smiles_list=smiles,
        labels=labels,
        morgan_featurizer=morgan_feat,
        graph_featurizer=graph_feat,
        morgan_fp_dict=morgan_fp_dict,
    )

    batch_size = cfg.get("batch_size", 100)
    device = cfg.get("device", "cpu")

    # GPU-optimised DataLoader settings
    use_pin = device.startswith("cuda")
    num_workers = cfg.get("num_workers", 4 if use_pin else 0)

    loaders = {}
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        subset = Subset(full_dataset, list(idx))
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            collate_fn=collate_fn,
            pin_memory=use_pin,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    return loaders, (train_idx, val_idx, test_idx)


# ---------------------------------------------------------------------------
# Training loop — AMP-enabled
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: ConcatFFN,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    scaler: GradScaler = None,
    use_amp: bool = False,
) -> float:
    """Run one training epoch with optional AMP. Returns average loss."""
    model.train()
    total_loss, n = 0.0, 0
    amp_device = "cuda" if device.startswith("cuda") else device

    for graphs, morgan_fps, labels in loader:
        morgan_fps = morgan_fps.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()

        with autocast(device_type=amp_device, enabled=use_amp):
            preds = model(graphs, morgan_fps)
            loss = criterion(preds, labels)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * morgan_fps.size(0)
        n += morgan_fps.size(0)

    return total_loss / n


def evaluate(
    model: ConcatFFN,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool = False,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run evaluation with optional AMP. Returns (avg_loss, predictions, labels)."""
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [], []
    amp_device = "cuda" if device.startswith("cuda") else device

    with torch.no_grad():
        for graphs, morgan_fps, labels in loader:
            morgan_fps = morgan_fps.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=amp_device, enabled=use_amp):
                preds = model(graphs, morgan_fps)
                loss = criterion(preds, labels)

            total_loss += loss.item() * morgan_fps.size(0)
            n += morgan_fps.size(0)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    all_preds = np.concatenate(all_preds).squeeze()
    all_labels = np.concatenate(all_labels).squeeze()
    return total_loss / n, all_preds, all_labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute R2, RMSE, MAE, Pearson r."""
    r2 = r2_score(labels, preds)
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    pearson_r, _ = pearsonr(preds, labels)
    return {"R2": r2, "RMSE": rmse, "MAE": mae, "Pearson_r": pearson_r}


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------


def train(cfg: dict):
    device = cfg.get("device", "cpu")

    # ---- Load data ----
    _, smiles, labels = load_dataset_from_config(cfg)
    print(f"Dataset: {len(smiles)} molecules")

    # ---- DataLoaders ----
    loaders, split_indices = prepare_dataloaders(cfg, smiles, labels)

    # ---- Model ----
    chemprop_cfg = cfg["chemprop"]
    morgan_cfg = cfg["morgan"]
    ffn_cfg = cfg["ffn"]

    model = ConcatFFN(
        chemprop_hidden_size=chemprop_cfg["hidden_size"],
        chemprop_depth=chemprop_cfg["depth"],
        chemprop_dropout=chemprop_cfg["dropout"],
        morgan_n_bits=morgan_cfg["n_bits"],
        ffn_hidden_layers=ffn_cfg["hidden_layers"],
        ffn_dropout=ffn_cfg["dropout"],
        output_dim=ffn_cfg["output_dim"],
    ).to(device)

    # Optional: torch.compile for PyTorch 2.0+ kernel fusion
    if cfg.get("compile", False) and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Optimizer & loss ----
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    # ---- AMP setup ----
    use_amp = device.startswith("cuda") and cfg.get("use_amp", True)
    scaler = GradScaler("cuda") if use_amp else None
    if use_amp:
        print("AMP (mixed precision) enabled for faster training")

    # ---- Training loop ----
    best_val_loss = float("inf")
    best_model = None
    train_losses, val_losses = [], []

    for epoch in range(cfg["epochs"]):
        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler, use_amp
        )
        val_loss, _, _ = evaluate(model, loaders["val"], criterion, device, use_amp)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = deepcopy(model)

        print(
            f"Epoch {epoch:<3} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    print(f"\nBest Val Loss: {best_val_loss:.4f}")

    # ---- Test evaluation ----
    model = best_model
    _, test_preds, test_labels = evaluate(
        model, loaders["test"], criterion, device, use_amp
    )
    metrics = compute_metrics(test_preds, test_labels)
    print("\n=== Test Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ---- Save checkpoint ----
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints/chemff")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "chemff_model.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved -> {ckpt_path}")

    return metrics, train_losses, val_losses


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ChemFF combined model")
    parser.add_argument("--config", type=str, default="chemff/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)

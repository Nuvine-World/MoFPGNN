from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset, TensorDataset

# Add project root to path so chemff / models / utils are importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

from chemff.chemprop.featurizer import ATOM_FDIM, BOND_FDIM, MolecularGraphFeaturizer
from chemff.dataset import MoleculeDataset, collate_fn
from chemff.model import ConcatFFN, _batch_graphs
from chemff.morgan.featurizer import MorganFeaturizer

# ---------------------------------------------------------------------------
# Official chemprop v2 (preferred; falls back to custom implementation)
# ---------------------------------------------------------------------------
try:
    from chemprop.models import MPNN as ChempropMPNN
    from chemprop.nn import BondMessagePassing, MeanAggregation
    from chemprop.nn import RegressionFFN as ChempropRegressionFFN
    from chemprop.data import (
        MoleculeDatapoint as ChempropDatapoint,
        MoleculeDataset as ChempropMolDataset,
        build_dataloader as chemprop_build_dataloader,
    )

    _CHEMPROP_V2 = True
except ImportError:
    _CHEMPROP_V2 = False
    print("WARNING: chemprop v2 not found; using custom D-MPNN fallback.")

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
DATA_PATH = PROJECT_ROOT / "data" / "AGILE.csv"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits" / "AGILE"
FP_CACHE_PATH = PROJECT_ROOT / "data" / "fingerprints" / "AGILE" / "morgan.pkl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# A100 GPU setup
# ---------------------------------------------------------------------------


def setup_gpu(device: str) -> None:
    """
    Apply every A100-specific kernel and precision flag before any model
    is built or any data is loaded.

    TF32 matmul  — A100 tensor cores compute FP32 matmul at BF16 speed.
    cuDNN bench  — auto-selects the fastest conv algorithm for each input shape.
    float32 prec — 'high' routes matmul through TF32 tensor cores (same as above,
                   but also covers ops that bypass allow_tf32).
    BF16 autocast dtype is set here so _AMP_DTYPE is visible to all helpers.
    """
    if not device.startswith("cuda"):
        return
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 matmul on A100
    torch.backends.cudnn.allow_tf32 = True  # TF32 in cuDNN conv
    torch.backends.cudnn.benchmark = True  # auto-tune cuDNN kernels
    torch.set_float32_matmul_precision("high")  # also enables TF32 path


# AMP dtype: BF16 on A100 (native tensor-core support, no GradScaler needed);
# fall back to FP16 on older GPUs; no-op on CPU.
def _amp_dtype(device: str) -> Optional[torch.dtype]:
    if not device.startswith("cuda"):
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16  # A100 preferred: same exponent range as FP32
    return torch.float16  # older GPUs


# ===========================================================================
# STAGE 1 — Preprocessing
# ===========================================================================


def canonicalise_smiles(smiles: str) -> Optional[str]:
    """Return RDKit canonical SMILES, or None if the molecule is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def compute_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
) -> np.ndarray:
    """
    Count-based Morgan (circular) fingerprints, radius=2, 2048-dim.
    Matches LANTERN Section 4.2: 'count-based circular fingerprints'
    generated via DeepChem CircularFingerprint(size=2048, radius=2,
    is_counts_based=True, chiral=True).
    """
    fps = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetHashedMorganFingerprint(
            mol, radius=radius, nBits=n_bits, useChirality=use_chirality
        )
        nz = fp.GetNonzeroElements()
        for bit, count in nz.items():
            fps[i, bit] = float(count)
    return fps


# Expert descriptor names: stable 210-dim RDKit descriptor set
# (LANTERN Section 4.2: "RDKit's 210-dimensional built-in descriptor module")
_EXPERT_NAMES: List[str] = [name for name, _ in Descriptors.descList]


def compute_expert_descriptors(smiles_list: List[str]) -> np.ndarray:
    """
    Compute the full RDKit descriptor set (~200+ physicochemical descriptors).
    NaN / Inf values are replaced with 0 after standardisation.
    """
    records = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            records.append([np.nan] * len(_EXPERT_NAMES))
            continue
        vals = []
        for _, fn in Descriptors.descList:
            try:
                v = fn(mol)
            except Exception:
                v = np.nan
            vals.append(float(v))
        records.append(vals)

    arr = np.array(records, dtype=np.float32)
    # Replace Inf with NaN, then fill NaN with column median
    arr[~np.isfinite(arr)] = np.nan
    col_median = np.nanmedian(arr, axis=0)
    for j in range(arr.shape[1]):
        mask = np.isnan(arr[:, j])
        arr[mask, j] = col_median[j] if np.isfinite(col_median[j]) else 0.0
    return arr


def load_splits(split_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed 80/10/10 split indices."""
    fname = "random.npy" if split_type == "random" else "Murcko_scaffold.npy"
    splits = np.load(SPLITS_DIR / fname, allow_pickle=True)
    train_idx, val_idx, test_idx = splits[0], splits[1], splits[2]
    return (
        np.array(train_idx, dtype=int),
        np.array(val_idx, dtype=int),
        np.array(test_idx, dtype=int),
    )


def stage1_preprocessing(split_type: str = "random") -> Dict:
    """
    Stage 1: Load dataset, canonicalise SMILES, compute all representations.

    Returns a context dict consumed by all downstream stages.
    """
    print("\n" + "=" * 70)
    print("STAGE 1 — Preprocessing")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} molecules from {DATA_PATH.name}")

    # Canonicalise SMILES
    canonical = [canonicalise_smiles(s) for s in df["SMILES"]]
    valid_mask = [s is not None for s in canonical]
    df = df[valid_mask].reset_index(drop=True)
    smiles_list = [canonical[i] for i, ok in enumerate(valid_mask) if ok]
    labels = df["Target"].tolist()
    print(f"After canonicalisation: {len(smiles_list)} valid molecules")

    # Morgan fingerprints
    print("Computing count-based Morgan fingerprints (r=2, 2048-dim) …")
    # Always recompute keyed by canonical SMILES so fp_dict is consistent.
    # If the canonical-key cache already exists we load it; otherwise compute.
    canon_cache = FP_CACHE_PATH.parent / "morgan_canonical.pkl"
    if canon_cache.exists():
        print(f"  Loading cached fingerprints from {canon_cache}")
        with open(canon_cache, "rb") as f:
            fp_dict: Dict[str, np.ndarray] = pickle.load(f)
        # If any canonical SMILES are missing, recompute them
        missing = [s for s in smiles_list if s not in fp_dict]
        if missing:
            fps_missing = compute_morgan_fingerprints(missing)
            for i, s in enumerate(missing):
                fp_dict[s] = fps_missing[i]
    else:
        morgan_fps = compute_morgan_fingerprints(smiles_list)
        fp_dict = {s: morgan_fps[i] for i, s in enumerate(smiles_list)}
        canon_cache.parent.mkdir(parents=True, exist_ok=True)
        with open(canon_cache, "wb") as f:
            pickle.dump(fp_dict, f)
        print(f"  Saved canonical Morgan fingerprints to {canon_cache}")

    morgan_fps = np.vstack([fp_dict[s] for s in smiles_list]).astype(np.float32)
    print(f"  Morgan FP shape: {morgan_fps.shape}")

    # Expert RDKit descriptors
    expert_cache = RESULTS_DIR / "expert_descriptors.npy"
    if expert_cache.exists():
        print(f"  Loading cached Expert descriptors from {expert_cache}")
        expert_fps = np.load(expert_cache).astype(np.float32)
    else:
        print(f"Computing Expert RDKit descriptors ({len(_EXPERT_NAMES)}-dim) …")
        expert_fps = compute_expert_descriptors(smiles_list)
        np.save(expert_cache, expert_fps)
    print(f"  Expert descriptor shape: {expert_fps.shape}")

    # Load splits
    train_idx, val_idx, test_idx = load_splits(split_type)
    print(
        f"Split ({split_type}): train={len(train_idx)}  "
        f"val={len(val_idx)}  test={len(test_idx)}"
    )

    ctx = {
        "smiles_list": smiles_list,
        "labels": np.array(labels, dtype=np.float32),
        "morgan_fps": morgan_fps,
        "expert_fps": expert_fps,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "split_type": split_type,
        "fp_dict": fp_dict,
    }
    return ctx


# ===========================================================================
# Model definitions
# ===========================================================================


class MLPRegressor(nn.Module):
    """
    7-layer MLP matching LANTERN Section 4.1:
      hidden sizes = [200, 300, 500, 500, 300, 200], ReLU activations.
    Input → 200 → 300 → 500 → 500 → 300 → 200 → 1
    """

    def __init__(self, input_dim: int, dropout: float = 0.0):
        super().__init__()
        sizes = [input_dim, 200, 300, 500, 500, 300, 200, 1]
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ChempropBaseline(nn.Module):
    """
    Pure D-MPNN baseline (no explicit fingerprints).
    Graph encoder → FFN regressor (hidden=500).
    Matches LANTERN Section 4.1: 'hidden dimension of 500 in the regressor'.
    """

    def __init__(
        self,
        hidden_size: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        regressor_hidden: int = 500,
    ):
        super().__init__()
        from chemff.chemprop.model import ChempropEncoder

        self.encoder = ChempropEncoder(
            hidden_size=hidden_size,
            depth=depth,
            dropout=dropout,
            output_dim=hidden_size,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, regressor_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regressor_hidden, 1),
        )

    def forward(
        self,
        graphs,
        morgan_fps: Optional[torch.Tensor] = None,  # ignored for pure baseline
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        f_atoms, f_bonds, b2a, b2revb, a_scope, mol_atom_idx = _batch_graphs(graphs)
        f_atoms = f_atoms.to(device)
        f_bonds = f_bonds.to(device)
        b2a = b2a.to(device)
        b2revb = b2revb.to(device)
        a_scope = a_scope.to(device)
        mol_atom_idx = mol_atom_idx.to(device)
        emb = self.encoder(f_atoms, f_bonds, b2a, b2revb, a_scope, mol_atom_idx)
        return self.regressor(emb)


# ===========================================================================
# Training helpers
# ===========================================================================


class GPUTensorStore:
    """
    Entire dataset lives on GPU as a single resident tensor pair.
    Eliminates all CPU→GPU transfers during training — every epoch is
    pure GPU indexing + matrix multiply with zero PCIe traffic.

    Memory: 1100 × 2048 × 4 bytes ≈ 9 MB (Morgan only).
    With Morgan+Expert (2265-dim): ~10 MB.  Negligible on 80 GB A100.
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        test_idx: np.ndarray,
        scaler: Optional[StandardScaler],
        device: str,
        clip: float = 10.0,
    ):
        if scaler is None:
            scaler = StandardScaler()
            clean_train = np.nan_to_num(
                features[train_idx], nan=0.0, posinf=0.0, neginf=0.0
            )
            scaler.fit(clean_train)
            bad = ~np.isfinite(scaler.scale_) | (scaler.scale_ < 1e-8)
            if bad.any():
                scaler.scale_[bad] = 1.0
                scaler.mean_[bad] = 0.0
        # Clean NaN/Inf BEFORE transform, clip to ±clip, then kill any residual NaN
        X_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        X = np.nan_to_num(
            np.clip(scaler.transform(X_clean).astype(np.float32), -clip, clip),
            nan=0.0,
            posinf=clip,
            neginf=-clip,
        )
        n_bad_np = (~np.isfinite(X)).sum()
        if n_bad_np > 0:
            print(
                f"  [GPUTensorStore] WARNING: {n_bad_np} non-finite values "
                f"in features after numpy cleanup — force-zeroing"
            )
            X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
        # Move once — never touched again until the run ends
        self.X = torch.tensor(X, device=device)
        # Final GPU-side safety net: replace any surviving NaN/Inf on the tensor
        if not torch.isfinite(self.X).all():
            bad_count = (~torch.isfinite(self.X)).sum().item()
            print(
                f"  [GPUTensorStore] WARNING: {bad_count} non-finite values "
                f"on GPU tensor — applying torch.nan_to_num_"
            )
            torch.nan_to_num_(self.X, nan=0.0, posinf=clip, neginf=-clip)
        self.y = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)
        self.train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
        self.val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
        self.test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
        self.scaler = scaler
        self.device = device

    def split(self, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[
            split
        ]
        return self.X[idx], self.y[idx]


def _make_tensor_loaders(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 64,
    scaler: Optional[StandardScaler] = None,
    device: str = "cpu",
    num_workers: int = 4,
) -> Tuple["GPUTensorStore", None, None, StandardScaler]:
    """
    On GPU: returns a GPUTensorStore — the entire dataset lives in GPU VRAM.
    On CPU: falls back to DataLoader with pin_memory for correctness.
    Callers distinguish by checking isinstance(result[0], GPUTensorStore).
    """
    if scaler is None:
        scaler = StandardScaler()
        # Replace any NaN/Inf in features before fitting
        clean_train = np.nan_to_num(
            features[train_idx], nan=0.0, posinf=0.0, neginf=0.0
        )
        scaler.fit(clean_train)
        # Fix zero / NaN / Inf in scale_ (constant or degenerate columns)
        bad = ~np.isfinite(scaler.scale_) | (scaler.scale_ < 1e-8)
        if bad.any():
            scaler.scale_[bad] = 1.0
            scaler.mean_[bad] = 0.0

    # Scale → clip to ±10 → kill residual NaN/Inf.
    # np.clip turns ±Inf into ±10 but leaves NaN intact; nan_to_num cleans that.
    def _scale_clip(X: np.ndarray) -> np.ndarray:
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        Xs = scaler.transform(X_clean).astype(np.float32)
        return np.nan_to_num(
            np.clip(Xs, -10.0, 10.0), nan=0.0, posinf=10.0, neginf=-10.0
        )

    if device.startswith("cuda"):
        store = GPUTensorStore(
            features, labels, train_idx, val_idx, test_idx, scaler, device, clip=10.0
        )
        return store, None, None, scaler

    # CPU fallback — original DataLoader path
    X = _scale_clip(features)
    ds = TensorDataset(torch.tensor(X), torch.tensor(labels).unsqueeze(1))
    kw: dict = {}
    train_loader = DataLoader(
        Subset(ds, train_idx), batch_size=batch_size, shuffle=True, **kw
    )
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=batch_size, **kw)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, **kw)
    return train_loader, val_loader, test_loader, scaler


def _train_mlp(
    model: nn.Module,
    train_data,  # GPUTensorStore (fast path) or DataLoader (CPU fallback)
    val_data,
    lr: float = 2e-4,
    epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    batch_size: int = 512,
) -> Tuple[nn.Module, List[float]]:
    """
    Train an MLP with Adam + MSE + early stopping (LANTERN §4.1).

    GPU fast path (GPUTensorStore):
      - All data already resident in GPU VRAM — zero PCIe traffic per epoch.
      - batch_size=512 fills A100 SMs without spilling L2; full 880-sample
        train set is only ~7 MB so one pass = one kernel launch per layer.
      - BF16 autocast over forward+loss; no GradScaler needed (BF16 range=FP32).
      - set_to_none=True skips the memset on old gradients.

    CPU fallback: standard DataLoader loop with non_blocking transfers.
    """
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    amp_ctx = device.split(":")[0] if use_amp else "cpu"
    grad_scaler = (
        GradScaler(amp_ctx) if (use_amp and amp_dtype == torch.float16) else None
    )

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    train_losses: List[float] = []

    use_gpu_store = isinstance(train_data, GPUTensorStore)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        if use_gpu_store:
            # ---- GPU fast path: shuffle indices on GPU, mini-batch from VRAM ----
            X_tr, y_tr = train_data.split("train")
            n_tr = X_tr.size(0)
            perm = torch.randperm(n_tr, device=device)
            for i in range(0, n_tr, batch_size):
                idx = perm[i : i + batch_size]
                X_b, y_b = X_tr[idx], y_tr[idx]
                optimiser.zero_grad(set_to_none=True)
                with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                    loss = criterion(model(X_b), y_b)
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    grad_scaler.step(optimiser)
                    grad_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimiser.step()
                epoch_loss += loss.item() * len(y_b)
            epoch_loss /= n_tr
        else:
            # ---- CPU fallback: DataLoader with non_blocking transfers ----
            for X_b, y_b in train_data:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                    loss = criterion(model(X_b), y_b)
                if grad_scaler is not None:
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    grad_scaler.step(optimiser)
                    grad_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimiser.step()
                epoch_loss += loss.item() * len(y_b)
            epoch_loss /= len(train_data.dataset)
        train_losses.append(epoch_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            if use_gpu_store:
                X_v, y_v = train_data.split("val")
                with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                    val_loss = criterion(model(X_v), y_v).item() * len(y_v)
                nv = len(y_v)
            else:
                nv = 0
                for X_b, y_b in val_data:
                    X_b = X_b.to(device, non_blocking=True)
                    y_b = y_b.to(device, non_blocking=True)
                    with autocast(
                        device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp
                    ):
                        val_loss += criterion(model(X_b), y_b).item() * len(y_b)
                    nv += len(y_b)
        val_loss /= nv

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, train_losses


def _eval_mlp(
    model: nn.Module,
    test_data,  # GPUTensorStore or DataLoader
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred). GPU path: single forward pass, zero transfers."""
    model.eval()
    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    amp_ctx = device.split(":")[0] if use_amp else "cpu"

    with torch.inference_mode():
        if isinstance(test_data, GPUTensorStore):
            X_te, y_te = test_data.split("test")
            with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                pred = model(X_te).float()
            return y_te.cpu().numpy().ravel(), pred.cpu().numpy().ravel()

        preds, trues = [], []
        for X_b, y_b in test_data:
            X_b = X_b.to(device, non_blocking=True)
            with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                out = model(X_b).float()
            preds.append(out.cpu().numpy())
            trues.append(y_b.numpy())
    return np.vstack(trues).ravel(), np.vstack(preds).ravel()


class GPUGraphCache:
    """
    Pre-compute ALL molecular graph tensors once and pin them in GPU VRAM.

    For 1100 molecules (ionisable lipids, ~40–80 atoms each):
      atom features  ~1100 × 60 atoms × 133 floats ≈ 35 MB
      bond features  ~1100 × 120 bonds × 147 floats ≈ 95 MB
      Morgan FPs     ~1100 × 2048 floats             ≈  9 MB
    Total ≈ 140 MB — less than 0.2 % of 80 GB A100 VRAM.

    Training loop: GPU randperm → GPU gather → ChempropEncoder.
    Zero PCIe traffic after the one-time load.
    """

    def __init__(
        self,
        smiles_list: List[str],
        fp_dict: Dict[str, np.ndarray],
        labels: np.ndarray,
        device: str,
    ):
        from tqdm import tqdm as _tqdm

        print(f"  Pre-computing graph tensors → GPU ({device}) …")
        graph_feat = MolecularGraphFeaturizer()
        graphs = [graph_feat(s) for s in _tqdm(smiles_list, leave=False)]

        # Store per-molecule GPU tensors (fast mini-batch gather below)
        self.device = device
        self.n = len(smiles_list)
        self._f_atoms = [
            torch.tensor(g.f_atoms, dtype=torch.float32, device=device) for g in graphs
        ]
        self._f_bonds = [
            torch.tensor(g.f_bonds, dtype=torch.float32, device=device) for g in graphs
        ]
        self._b2a = [
            torch.tensor(g.b2a, dtype=torch.long, device=device) for g in graphs
        ]
        self._b2revb = [
            torch.tensor(g.b2revb, dtype=torch.long, device=device) for g in graphs
        ]
        self._n_atoms = [g.n_atoms for g in graphs]
        self._n_bonds = [g.n_bonds for g in graphs]

        fps = np.vstack([fp_dict[s] for s in smiles_list]).astype(np.float32)
        self.morgan_fps = torch.tensor(fps, device=device)
        self.labels = torch.tensor(
            labels, dtype=torch.float32, device=device
        ).unsqueeze(1)
        print(f"  Graph cache ready — {self.n} molecules on {device}.")

    def batch(self, indices: List[int]):
        """
        Gather pre-computed GPU tensors for `indices` and concatenate entirely
        on GPU — no Python object construction, no numpy, no PCIe transfer.
        Returns the six tensors expected by ChempropEncoder.forward().
        """
        atom_off = bond_off = 0
        fa, fb, b2a_list, b2rb_list = [], [], [], []
        a_scope_list, mol_atom_list = [], []

        for mol_i, idx in enumerate(indices):
            na = self._n_atoms[idx]
            fa.append(self._f_atoms[idx])
            fb.append(self._f_bonds[idx])
            b2a_list.append(self._b2a[idx] + atom_off)
            b2rb_list.append(self._b2revb[idx] + bond_off)
            a_scope_list.append([atom_off, na])
            mol_atom_list.append(
                torch.full((na,), mol_i, dtype=torch.long, device=self.device)
            )
            atom_off += na
            bond_off += self._n_bonds[idx]

        return (
            torch.cat(fa),
            torch.cat(fb),
            torch.cat(b2a_list),
            torch.cat(b2rb_list),
            torch.tensor(a_scope_list, dtype=torch.long, device=self.device),
            torch.cat(mol_atom_list),
        )


def _make_graph_loaders(
    ctx: Dict,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 128,
    device: str = "cpu",
    num_workers: int = 4,
):
    """
    GPU path  → returns (GPUGraphCache, train_idx, val_idx, test_idx).
               The cache holds all graph tensors in VRAM; callers loop over
               shuffled index batches and call cache.batch(indices).
    CPU path  → returns (DataLoader, DataLoader, DataLoader) as before.
    """
    if device.startswith("cuda"):
        # One-time precompute; cache is shared across all stages
        if "_graph_cache" not in ctx:
            ctx["_graph_cache"] = GPUGraphCache(
                ctx["smiles_list"], ctx["fp_dict"], ctx["labels"], device
            )
        cache = ctx["_graph_cache"]
        idxs = {"train": train_idx, "val": val_idx, "test": test_idx}
        # Return (cache, cache, cache, idx_dict) — callers use idxs["train"] etc.
        return cache, cache, cache, idxs

    # CPU fallback — original DataLoader path with pin_memory
    morgan_feat = MorganFeaturizer(
        radius=2, n_bits=2048, use_counts=True, use_chirality=True
    )
    graph_feat = MolecularGraphFeaturizer()
    full_ds = MoleculeDataset(
        smiles_list=ctx["smiles_list"],
        labels=list(ctx["labels"]),
        morgan_featurizer=morgan_feat,
        graph_featurizer=graph_feat,
        morgan_fp_dict=ctx["fp_dict"],
    )
    nw = num_workers
    kw = dict(
        collate_fn=collate_fn,
        pin_memory=True,
        num_workers=nw,
        persistent_workers=(nw > 0),
        prefetch_factor=(2 if nw > 0 else None),
    )
    return (
        DataLoader(
            Subset(full_ds, list(train_idx)), batch_size=batch_size, shuffle=True, **kw
        ),
        DataLoader(Subset(full_ds, list(val_idx)), batch_size=batch_size, **kw),
        DataLoader(Subset(full_ds, list(test_idx)), batch_size=batch_size, **kw),
        None,  # no split index dict needed for DataLoader path
    )


def _iter_batches(
    cache_or_loader, split_idx, batch_size: int, shuffle: bool, device: str
):
    """
    Unified batch iterator for both GPUGraphCache and DataLoader.
    Yields (f_atoms, f_bonds, b2a, b2revb, a_scope, mol_atom_idx, morgan_fps, labels)
    for the cache path, or (graphs_list, morgan_fps, labels) for the loader path.
    The returned 'is_cache' flag tells the caller which format to expect.
    """
    if isinstance(cache_or_loader, GPUGraphCache):
        cache = cache_or_loader
        n = len(split_idx)
        order = (
            torch.randperm(n, device=device)
            if shuffle
            else torch.arange(n, device=device)
        )
        idx_tensor = torch.tensor(split_idx, dtype=torch.long, device=device)
        for i in range(0, n, batch_size):
            mol_idx = idx_tensor[order[i : i + batch_size]].tolist()
            graph_tensors = cache.batch(mol_idx)
            fps = cache.morgan_fps[mol_idx]
            labels = cache.labels[mol_idx]
            yield True, graph_tensors, fps, labels
    else:
        for graphs, fps, labels in cache_or_loader:
            yield False, graphs, fps.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )


def _train_graph_model(
    model: nn.Module,
    train_data,  # GPUGraphCache (fast) or DataLoader (CPU)
    val_data,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
    frozen_encoder: bool = False,
    batch_size: int = 128,
    train_idx=None,
    val_idx=None,
) -> Tuple[nn.Module, List[float]]:
    """
    Train Chemprop baseline or ConcatFFN hybrid.

    GPU fast path (GPUGraphCache):
      - batch() gathers pre-computed GPU tensors — pure torch.cat on-device.
      - No Python-level featurisation, no DataLoader overhead, no PCIe per batch.
      - batch_size=128 is safe for 80 GB with hidden=300; increase to 256 if headroom allows.
      - BF16 autocast over D-MPNN scatter_add + FFN.
      - Grad clip unscaled before GradScaler.step (FP16 path).
    """
    model = model.to(device)
    if frozen_encoder:
        for name, param in model.named_parameters():
            if "encoder" in name or "chemprop" in name.lower():
                param.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = optim.Adam(trainable, lr=lr)
    criterion = nn.MSELoss()

    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    amp_ctx = device.split(":")[0] if use_amp else "cpu"
    grad_scaler = (
        GradScaler(amp_ctx) if (use_amp and amp_dtype == torch.float16) else None
    )

    use_cache = isinstance(train_data, GPUGraphCache)

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    train_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n = 0.0, 0

        for is_cache, graph_data, fps, y_b in _iter_batches(
            train_data,
            train_idx if use_cache else None,
            batch_size,
            shuffle=True,
            device=device,
        ):
            optimiser.zero_grad(set_to_none=True)
            with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                if is_cache:
                    # Direct call to encoder with pre-batched GPU tensors
                    fa, fb, b2a, b2revb, a_scope, mai = graph_data
                    if hasattr(model, "chemprop_encoder"):  # ConcatFFN
                        emb = model.chemprop_encoder(fa, fb, b2a, b2revb, a_scope, mai)
                        pred = model.ffn_head(torch.cat([emb, fps], dim=1))
                    else:  # ChempropBaseline
                        emb = model.encoder(fa, fb, b2a, b2revb, a_scope, mai)
                        pred = model.regressor(emb)
                else:
                    pred = model(graph_data, fps)
                loss = criterion(pred, y_b)

            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                grad_scaler.step(optimiser)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
                optimiser.step()

            epoch_loss += loss.item() * len(y_b)
            n += len(y_b)
        train_losses.append(epoch_loss / n)

        # ---- Validation ----
        model.eval()
        val_loss, nv = 0.0, 0
        with torch.inference_mode():
            for is_cache, graph_data, fps, y_b in _iter_batches(
                val_data,
                val_idx if use_cache else None,
                batch_size,
                shuffle=False,
                device=device,
            ):
                with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                    if is_cache:
                        fa, fb, b2a, b2revb, a_scope, mai = graph_data
                        if hasattr(model, "chemprop_encoder"):
                            emb = model.chemprop_encoder(
                                fa, fb, b2a, b2revb, a_scope, mai
                            )
                            pred = model.ffn_head(torch.cat([emb, fps], dim=1))
                        else:
                            emb = model.encoder(fa, fb, b2a, b2revb, a_scope, mai)
                            pred = model.regressor(emb)
                    else:
                        pred = model(graph_data, fps)
                    val_loss += criterion(pred, y_b).item() * len(y_b)
                nv += len(y_b)
        val_loss /= nv

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, train_losses


def _eval_graph_model(
    model: nn.Module,
    test_data,
    device: str = "cpu",
    test_idx=None,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single forward pass per batch, all on GPU. Zero PCIe after cache load."""
    model.eval()
    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    amp_ctx = device.split(":")[0] if use_amp else "cpu"

    preds, trues = [], []
    with torch.inference_mode():
        for is_cache, graph_data, fps, y_b in _iter_batches(
            test_data, test_idx, batch_size, shuffle=False, device=device
        ):
            with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                if is_cache:
                    fa, fb, b2a, b2revb, a_scope, mai = graph_data
                    if hasattr(model, "chemprop_encoder"):
                        emb = model.chemprop_encoder(fa, fb, b2a, b2revb, a_scope, mai)
                        out = model.ffn_head(torch.cat([emb, fps], dim=1)).float()
                    else:
                        emb = model.encoder(fa, fb, b2a, b2revb, a_scope, mai)
                        out = model.regressor(emb).float()
                else:
                    out = model(graph_data, fps).float()
            preds.append(out.cpu().numpy())
            trues.append(y_b.cpu().numpy())
    return np.vstack(trues).ravel(), np.vstack(preds).ravel()


# ===========================================================================
# Official chemprop v2 helpers
# ===========================================================================


def _make_chemprop_datapoints(
    smiles_list: List[str],
    labels: np.ndarray,
    morgan_fps: Optional[np.ndarray] = None,
    scaler: Optional[StandardScaler] = None,
) -> List["ChempropDatapoint"]:
    """
    Return a plain Python list of MoleculeDatapoint objects.

    Keeping raw MoleculeDatapoint objects (not a MoleculeDataset) avoids the
    `Datum` issue: indexing into a MoleculeDataset returns internal Datum
    namedtuples that lack the V_f / V_d attributes MoleculeDataset._V_fs
    needs when building a new dataset from slices.
    """
    if morgan_fps is not None and scaler is not None:
        morgan_fps = scaler.transform(morgan_fps).astype(np.float32)

    datapoints = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        y = np.array([float(labels[i])], dtype=np.float32)
        x_d = morgan_fps[i] if morgan_fps is not None else None
        datapoints.append(ChempropDatapoint(mol=mol, y=y, x_d=x_d))
    return datapoints


def _chemprop_loaders(
    datapoints: List["ChempropDatapoint"],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple:
    """Slice the raw datapoints list (never a MoleculeDataset) and build loaders."""
    train_ds = ChempropMolDataset([datapoints[i] for i in train_idx])
    val_ds = ChempropMolDataset([datapoints[i] for i in val_idx])
    test_ds = ChempropMolDataset([datapoints[i] for i in test_idx])
    kw = dict(num_workers=num_workers)
    train_ld = chemprop_build_dataloader(
        train_ds, batch_size=batch_size, shuffle=True, **kw
    )
    val_ld = chemprop_build_dataloader(
        val_ds, batch_size=batch_size, shuffle=False, **kw
    )
    test_ld = chemprop_build_dataloader(
        test_ds, batch_size=batch_size, shuffle=False, **kw
    )
    return train_ld, val_ld, test_ld


def build_chemprop_mpnn(
    d_h: int = 300,
    depth: int = 3,
    dropout: float = 0.0,
    d_morgan: int = 0,
    ffn_hidden: int = 500,
    n_ffn_layers: int = 2,
) -> "ChempropMPNN":
    """
    Build an official chemprop v2 MPNN.
      d_morgan = 0   → pure D-MPNN baseline (no fingerprints).
      d_morgan > 0   → hybrid: Morgan FPs concatenated via X_d before the FFN head.
    FFN input dim = d_h + d_morgan  (chemprop v2 handles X_d concatenation internally).
    """
    mp = BondMessagePassing(d_h=d_h, depth=depth, dropout=dropout)
    agg = MeanAggregation()
    ffn_input = d_h + d_morgan
    try:
        ffn = ChempropRegressionFFN(
            input_dim=ffn_input,
            hidden_dim=ffn_hidden,
            n_layers=n_ffn_layers,
            dropout=dropout,
        )
    except TypeError:
        # Older chemprop v2 sub-releases use n_tasks as first positional arg
        ffn = ChempropRegressionFFN(
            n_tasks=1,
            input_dim=ffn_input,
            hidden_dim=ffn_hidden,
            n_layers=n_ffn_layers,
            dropout=dropout,
        )
    return ChempropMPNN(message_passing=mp, agg=agg, predictor=ffn)


def _train_chemprop_model(
    model: nn.Module,
    train_loader,
    val_loader,
    lr: float = 1e-3,
    epochs: int = 100,
    patience: int = 15,
    device: str = "cpu",
) -> Tuple[nn.Module, List[float]]:
    """
    Train official chemprop v2 MPNN with AMP, grad-clip, and early stopping.
    Batch format: TrainingBatch(bmg, V_d, X_d, targets, weights, ...)
    """
    model = model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    amp_ctx = device.split(":")[0] if use_amp else "cpu"
    grad_scaler = (
        GradScaler(amp_ctx) if (use_amp and amp_dtype == torch.float16) else None
    )

    best_val_loss = float("inf")
    best_state = None
    patience_ctr = 0
    train_losses: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n = 0.0, 0
        for batch in train_loader:
            # Positional unpack: (bmg, V_d, X_d, Y, ...) regardless of field name
            bmg, V_d, X_d, targets = batch[0], batch[1], batch[2], batch[3]
            bmg.to(device)
            if X_d is not None:
                X_d = X_d.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                preds = model(bmg, V_d, X_d)
                loss = criterion(preds, targets)
            if grad_scaler is not None:
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_scaler.step(optimiser)
                grad_scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimiser.step()
            epoch_loss += loss.item() * len(targets)
            n += len(targets)
        train_losses.append(epoch_loss / n)

        model.eval()
        val_loss, nv = 0.0, 0
        with torch.inference_mode():
            for batch in val_loader:
                bmg, V_d, X_d, targets = batch[0], batch[1], batch[2], batch[3]
                bmg.to(device)
                if X_d is not None:
                    X_d = X_d.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                    preds = model(bmg, V_d, X_d)
                    val_loss += criterion(preds, targets).item() * len(targets)
                nv += len(targets)
        val_loss /= nv

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return model, train_losses


def _eval_chemprop_model(
    model: nn.Module,
    test_loader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate official chemprop v2 MPNN; return (y_true, y_pred)."""
    model.eval()
    amp_dtype = _amp_dtype(device)
    use_amp = amp_dtype is not None
    amp_ctx = device.split(":")[0] if use_amp else "cpu"

    preds, trues = [], []
    with torch.inference_mode():
        for batch in test_loader:
            bmg, V_d, X_d, targets = batch[0], batch[1], batch[2], batch[3]
            bmg.to(device)
            if X_d is not None:
                X_d = X_d.to(device, non_blocking=True)
            with autocast(device_type=amp_ctx, dtype=amp_dtype, enabled=use_amp):
                out = model(bmg, V_d, X_d).float()
            preds.append(out.cpu().numpy())
            trues.append(targets.cpu().numpy())
    return np.vstack(trues).ravel(), np.vstack(preds).ravel()


# ===========================================================================
# Evaluation metrics (Section 3.4 / LANTERN Section 4.3)
# ===========================================================================


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """R², RMSE, MAE, Pearson r — four core metrics from Section 3.4."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    return {
        "R2": round(r2, 4),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "r": round(r, 4),
    }


def relative_error_cdf(
    y_true: np.ndarray, y_pred: np.ndarray, n_points: int = 100
) -> Dict[str, np.ndarray]:
    """
    Cumulative distribution of absolute relative error |ŷ-y|/|y|.
    Section 3.4(ii): 'CDF of absolute relative error for tail-end assessment'.
    Returns dict with 'thresholds' and 'cdf_values' arrays.
    """
    # Avoid division by zero
    nonzero = np.abs(y_true) > 1e-8
    rel_err = np.abs(y_pred[nonzero] - y_true[nonzero]) / np.abs(y_true[nonzero])
    thresholds = np.linspace(0, np.percentile(rel_err, 99), n_points)
    cdf = np.array([np.mean(rel_err <= t) for t in thresholds])
    return {"thresholds": thresholds, "cdf_values": cdf}


def top_k_recovery(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[float] = (5, 10, 20),
) -> Dict[str, float]:
    """
    Fraction of true top-k% compounds recovered in predicted top-k%.
    Section 3.4(iii): 'percentile-based ranking accuracy for virtual screening'.
    """
    n = len(y_true)
    results = {}
    for k in k_values:
        top_n = max(1, int(np.ceil(n * k / 100.0)))
        true_top_k = set(np.argsort(y_true)[-top_n:])
        pred_top_k = set(np.argsort(y_pred)[-top_n:])
        recall = len(true_top_k & pred_top_k) / len(true_top_k)
        results[f"top{k}%_recovery"] = round(recall, 4)
    return results


def percentile_bin_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 6
) -> float:
    """
    AGILE-style classification accuracy over equal-width percentile bins.
    LANTERN Section 4.3: 'percentile bins, accuracy = correct bin assignments / N'.
    """
    bin_edges = np.percentile(y_true, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 1e-8  # ensure last value falls in final bin
    true_bins = np.digitize(y_true, bin_edges[1:-1])
    pred_bins = np.digitize(y_pred, bin_edges[1:-1])
    return round(float(np.mean(true_bins == pred_bins)), 4)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_type: str,
) -> Dict:
    """Compute and print the full metric suite for one model."""
    metrics = regression_metrics(y_true, y_pred)
    topk = top_k_recovery(y_true, y_pred)
    bin_acc = percentile_bin_accuracy(y_true, y_pred)
    cdf = relative_error_cdf(y_true, y_pred)

    record = {
        "model": model_name,
        "split": split_type,
        **metrics,
        **topk,
        "percentile_bin_acc": bin_acc,
    }
    print(
        f"  {model_name:<35s} "
        f"R²={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.4f}  "
        f"MAE={metrics['MAE']:.4f}  r={metrics['r']:.4f}  "
        f"top10%={topk['top10%_recovery']:.3f}  "
        f"bin_acc={bin_acc:.3f}"
    )
    return record, cdf


# ===========================================================================
# STAGE 2 — Baseline training
# ===========================================================================


def stage2_baseline_training(ctx: Dict, device: str = "cpu") -> Dict:
    """
    Stage 2: Train pure-feature baselines.
      • MLP(Morgan)         — LANTERN best single-representation model
      • MLP(Morgan+Expert)  — LANTERN top-1 model (R²=0.8161)
      • Chemprop (pure)     — pure D-MPNN, no explicit fingerprints
    """
    print("\n" + "=" * 70)
    print("STAGE 2 — Baseline Training")
    print("=" * 70)

    trained = {}
    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]

    nw = ctx.get("num_workers", 4)
    use_compile = ctx.get("use_compile", False)

    # ---- MLP (Morgan) ----------------------------------------
    print(
        "\n  [2a] MLP(Morgan)  lr=2e-4  epochs=300  patience=30  Adam  MSE  early-stop"
    )
    train_ld, val_ld, test_ld, scaler_morgan = _make_tensor_loaders(
        ctx["morgan_fps"],
        ctx["labels"],
        train_idx,
        val_idx,
        ctx["test_idx"],
        batch_size=64,
        device=device,
        num_workers=nw,
    )
    mlp_morgan = _maybe_compile(MLPRegressor(input_dim=2048), use_compile)
    mlp_morgan, _ = _train_mlp(
        mlp_morgan, train_ld, val_ld, lr=2e-4, epochs=300, patience=30, device=device
    )
    trained["MLP(Morgan)"] = {
        # GPU: train_ld is a GPUTensorStore; _eval_mlp uses .split("test") on it.
        # CPU: test_ld is the test DataLoader; _eval_mlp iterates over it.
        "model": mlp_morgan,
        "loader": train_ld if isinstance(train_ld, GPUTensorStore) else test_ld,
        "type": "mlp",
        "scaler": scaler_morgan,
    }

    # ---- MLP (Morgan + Expert) --------------------------------
    morgan_expert = np.hstack([ctx["morgan_fps"], ctx["expert_fps"]])
    print(
        f"\n  [2b] MLP(Morgan+Expert)  input_dim={morgan_expert.shape[1]}  epochs=300  patience=30"
    )
    train_ld_me, val_ld_me, test_ld_me, scaler_me = _make_tensor_loaders(
        morgan_expert,
        ctx["labels"],
        train_idx,
        val_idx,
        ctx["test_idx"],
        batch_size=64,
        device=device,
        num_workers=nw,
    )
    mlp_me = _maybe_compile(MLPRegressor(input_dim=morgan_expert.shape[1]), use_compile)
    mlp_me, _ = _train_mlp(
        mlp_me, train_ld_me, val_ld_me, lr=2e-4, epochs=300, patience=30, device=device
    )
    trained["MLP(Morgan+Expert)"] = {
        "model": mlp_me,
        "loader": (
            train_ld_me if isinstance(train_ld_me, GPUTensorStore) else test_ld_me
        ),
        "type": "mlp",
        "scaler": scaler_me,
    }

    # ---- Chemprop (pure graph) --------------------------------
    print("\n  [2c] Chemprop(pure)  hidden=300  depth=3  ffn_hidden=500  epochs=100")
    if _CHEMPROP_V2:
        cp_ds = _make_chemprop_datapoints(ctx["smiles_list"], ctx["labels"])
        cp_tr, cp_va, cp_te = _chemprop_loaders(
            cp_ds,
            train_idx,
            val_idx,
            ctx["test_idx"],
            batch_size=128,
            num_workers=nw,
        )
        chemprop_model = _maybe_compile(
            build_chemprop_mpnn(
                d_h=300, depth=3, dropout=0.0, d_morgan=0, ffn_hidden=500
            ),
            use_compile,
        )
        chemprop_model, _ = _train_chemprop_model(
            chemprop_model,
            cp_tr,
            cp_va,
            lr=1e-3,
            epochs=100,
            patience=15,
            device=device,
        )
        trained["Chemprop(pure)"] = {
            "model": chemprop_model,
            "loader": cp_te,
            "type": "chemprop",
        }
    else:
        train_gld, val_gld, test_gld, g_idxs = _make_graph_loaders(
            ctx,
            train_idx,
            val_idx,
            ctx["test_idx"],
            batch_size=128,
            device=device,
            num_workers=nw,
        )
        chemprop_model = _maybe_compile(
            ChempropBaseline(
                hidden_size=300, depth=3, dropout=0.0, regressor_hidden=500
            ),
            use_compile,
        )
        chemprop_model, _ = _train_graph_model(
            chemprop_model,
            train_gld,
            val_gld,
            lr=1e-3,
            epochs=100,
            patience=15,
            device=device,
            batch_size=128,
            train_idx=g_idxs["train"] if g_idxs else None,
            val_idx=g_idxs["val"] if g_idxs else None,
        )
        trained["Chemprop(pure)"] = {
            "model": chemprop_model,
            "loader": test_gld,
            "loader_idx": g_idxs["test"] if g_idxs else None,
            "type": "graph",
        }

    ctx["stage2_models"] = trained
    return ctx


# ===========================================================================
# STAGE 3 — Hybrid model implementation
# ===========================================================================


def stage3_hybrid_model(ctx: Dict, device: str = "cpu") -> Dict:
    """
    Stage 3: Chemprop + Morgan fingerprint late-fusion (Equation 1, LANTERN_EXT).
      (a) End-to-end: train graph encoder and FFN head jointly.
      (b) Two-phase: freeze encoder after Stage-2 training, fine-tune FFN head only.
    """
    print("\n" + "=" * 70)
    print("STAGE 3 — Hybrid Model Implementation")
    print("=" * 70)

    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]
    nw = ctx.get("num_workers", 4)
    use_compile = ctx.get("use_compile", False)
    hybrid_models = {}

    if _CHEMPROP_V2:
        # Fit Morgan scaler on training set (reuse if available from Stage 2)
        morgan_scaler = StandardScaler().fit(ctx["morgan_fps"][train_idx])
        hybrid_ds = _make_chemprop_datapoints(
            ctx["smiles_list"],
            ctx["labels"],
            morgan_fps=ctx["morgan_fps"],
            scaler=morgan_scaler,
        )
        hyb_tr, hyb_va, hyb_te = _chemprop_loaders(
            hybrid_ds,
            train_idx,
            val_idx,
            ctx["test_idx"],
            batch_size=128,
            num_workers=nw,
        )
        ctx["stage3_morgan_scaler"] = morgan_scaler

        # ---- (a) End-to-end joint optimisation ------------------
        print("\n  [3a] Chemprop+Morgan  end-to-end  d_h=300  ffn_hidden=500  X_d=2048")
        hybrid_e2e = _maybe_compile(
            build_chemprop_mpnn(
                d_h=300, depth=3, dropout=0.0, d_morgan=2048, ffn_hidden=500
            ),
            use_compile,
        )
        hybrid_e2e, _ = _train_chemprop_model(
            hybrid_e2e,
            hyb_tr,
            hyb_va,
            lr=1e-3,
            epochs=100,
            patience=15,
            device=device,
        )
        hybrid_models["Chemprop+Morgan(e2e)"] = {
            "model": hybrid_e2e,
            "loader": hyb_te,
            "type": "chemprop",
        }

        # ---- (b) Frozen encoder — two-phase protocol ------------
        print("\n  [3b] Chemprop+Morgan  frozen encoder  (two-phase)")
        hybrid_frozen = build_chemprop_mpnn(
            d_h=300,
            depth=3,
            dropout=0.0,
            d_morgan=2048,
            ffn_hidden=500,
        )
        # Transfer Stage-2 message_passing weights and freeze
        if "stage2_models" in ctx and "Chemprop(pure)" in ctx["stage2_models"]:
            stage2_mp = ctx["stage2_models"]["Chemprop(pure)"]["model"]
            if hasattr(stage2_mp, "message_passing"):
                hybrid_frozen.message_passing.load_state_dict(
                    stage2_mp.message_passing.state_dict(), strict=True
                )
                for p in hybrid_frozen.message_passing.parameters():
                    p.requires_grad_(False)
                print(
                    "    Loaded Stage-2 message_passing weights → frozen for Phase 2."
                )
        hybrid_frozen = _maybe_compile(hybrid_frozen, use_compile)
        hybrid_frozen, _ = _train_chemprop_model(
            hybrid_frozen,
            hyb_tr,
            hyb_va,
            lr=5e-4,
            epochs=100,
            patience=15,
            device=device,
        )
        hybrid_models["Chemprop+Morgan(frozen)"] = {
            "model": hybrid_frozen,
            "loader": hyb_te,
            "type": "chemprop",
        }

    else:
        # Custom fallback path
        train_gld, val_gld, test_gld, g_idxs = _make_graph_loaders(
            ctx,
            train_idx,
            val_idx,
            ctx["test_idx"],
            batch_size=128,
            device=device,
            num_workers=nw,
        )
        print("\n  [3a] Chemprop+Morgan  end-to-end  hidden=300  ffn=[1024,512,256]")
        hybrid_e2e = ConcatFFN(
            chemprop_hidden_size=300,
            chemprop_depth=3,
            chemprop_dropout=0.0,
            morgan_n_bits=2048,
            ffn_hidden_layers=(1024, 512, 256),
            ffn_dropout=0.1,
        )
        hybrid_e2e = _maybe_compile(hybrid_e2e, use_compile)
        hybrid_e2e, _ = _train_graph_model(
            hybrid_e2e,
            train_gld,
            val_gld,
            lr=1e-3,
            epochs=100,
            patience=15,
            device=device,
            frozen_encoder=False,
            batch_size=128,
            train_idx=g_idxs["train"] if g_idxs else None,
            val_idx=g_idxs["val"] if g_idxs else None,
        )
        hybrid_models["Chemprop+Morgan(e2e)"] = {
            "model": hybrid_e2e,
            "loader": test_gld,
            "loader_idx": g_idxs["test"] if g_idxs else None,
            "type": "graph",
        }

        print("\n  [3b] Chemprop+Morgan  frozen encoder  (two-phase)")
        hybrid_frozen = ConcatFFN(
            chemprop_hidden_size=300,
            chemprop_depth=3,
            chemprop_dropout=0.0,
            morgan_n_bits=2048,
            ffn_hidden_layers=(1024, 512, 256),
            ffn_dropout=0.1,
        )
        if "stage2_models" in ctx and "Chemprop(pure)" in ctx["stage2_models"]:
            stage2_enc_state = {
                k.replace("encoder.", ""): v
                for k, v in ctx["stage2_models"]["Chemprop(pure)"]["model"]
                .encoder.state_dict()
                .items()
            }
            hybrid_frozen.chemprop_encoder.load_state_dict(
                stage2_enc_state, strict=False
            )
            print("    Loaded Stage-2 Chemprop encoder weights → frozen for Phase 2.")
        hybrid_frozen = _maybe_compile(hybrid_frozen, use_compile)
        hybrid_frozen, _ = _train_graph_model(
            hybrid_frozen,
            train_gld,
            val_gld,
            lr=5e-4,
            epochs=100,
            patience=15,
            device=device,
            frozen_encoder=True,
            batch_size=128,
            train_idx=g_idxs["train"] if g_idxs else None,
            val_idx=g_idxs["val"] if g_idxs else None,
        )
        hybrid_models["Chemprop+Morgan(frozen)"] = {
            "model": hybrid_frozen,
            "loader": test_gld,
            "loader_idx": g_idxs["test"] if g_idxs else None,
            "type": "graph",
        }

    ctx["stage3_models"] = hybrid_models
    return ctx


# ===========================================================================
# STAGE 4 — Hyperparameter optimisation
# ===========================================================================


def stage4_hp_optimisation(ctx: Dict, device: str = "cpu") -> Dict:
    """
    Stage 4: Grid search over learning rate, batch size, dropout, hidden layers.
    Evaluated on the validation set; best config applied to the hybrid model.
    """
    print("\n" + "=" * 70)
    print("STAGE 4 — Hyperparameter Optimisation  (Chemprop+Morgan hybrid)")
    print("=" * 70)

    hp_grid = {
        "lr": [1e-4, 5e-4, 1e-3],
        "batch_size": [32, 64],
        "dropout": [0.0, 0.1, 0.2],
        "ffn_hidden": [256, 512],  # uniform hidden dim; n_layers=2
    }

    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]
    nw = ctx.get("num_workers", 4)

    best_val_r2 = -np.inf
    best_cfg = None
    results = []

    if _CHEMPROP_V2:
        # Build chemprop dataset once; reuse across all HP configs
        morgan_scaler = ctx.get(
            "stage3_morgan_scaler",
            StandardScaler().fit(ctx["morgan_fps"][train_idx]),
        )
        hp_ds = _make_chemprop_datapoints(
            ctx["smiles_list"],
            ctx["labels"],
            morgan_fps=ctx["morgan_fps"],
            scaler=morgan_scaler,
        )

        for lr, bs, do, fh in product(
            hp_grid["lr"],
            hp_grid["batch_size"],
            hp_grid["dropout"],
            hp_grid["ffn_hidden"],
        ):
            hp_tr, hp_va, hp_te = _chemprop_loaders(
                hp_ds,
                train_idx,
                val_idx,
                ctx["test_idx"],
                batch_size=bs,
                num_workers=nw,
            )
            model = build_chemprop_mpnn(
                d_h=300, depth=3, dropout=do, d_morgan=2048, ffn_hidden=fh
            )
            model, _ = _train_chemprop_model(
                model,
                hp_tr,
                hp_va,
                lr=lr,
                epochs=50,
                patience=10,
                device=device,
            )
            y_true_val, y_pred_val = _eval_chemprop_model(model, hp_va, device=device)
            val_r2 = r2_score(y_true_val, y_pred_val)
            cfg = {"lr": lr, "batch_size": bs, "dropout": do, "ffn_hidden": fh}
            results.append({**cfg, "val_R2": round(val_r2, 4)})
            print(f"  lr={lr}  bs={bs}  do={do}  ffn_h={fh}  val_R²={val_r2:.4f}")
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_cfg = cfg

        print(f"\n  Best HP config: {best_cfg}  (val R²={best_val_r2:.4f})")
        hp_df = pd.DataFrame(results).sort_values("val_R2", ascending=False)
        hp_df.to_csv(RESULTS_DIR / f"hp_search_{ctx['split_type']}.csv", index=False)

        # Retrain best model on train+val for final evaluation
        best_tr, best_va, best_te = _chemprop_loaders(
            hp_ds,
            train_idx,
            val_idx,
            ctx["test_idx"],
            batch_size=best_cfg["batch_size"],
            num_workers=nw,
        )
        best_model = build_chemprop_mpnn(
            d_h=300,
            depth=3,
            dropout=best_cfg["dropout"],
            d_morgan=2048,
            ffn_hidden=best_cfg["ffn_hidden"],
        )
        best_model, _ = _train_chemprop_model(
            best_model,
            best_tr,
            best_va,
            lr=best_cfg["lr"],
            epochs=100,
            patience=15,
            device=device,
        )
        ctx["best_hp_model"] = {
            "model": best_model,
            "loader": best_te,
            "type": "chemprop",
        }

    else:
        # Custom fallback
        train_gld, val_gld, test_gld, g_idxs = _make_graph_loaders(
            ctx,
            train_idx,
            val_idx,
            ctx["test_idx"],
            batch_size=128,
            device=device,
            num_workers=nw,
        )
        tr_idx_hp = g_idxs["train"] if g_idxs else None
        va_idx_hp = g_idxs["val"] if g_idxs else None
        te_idx_hp = g_idxs["test"] if g_idxs else None

        for lr, bs, do, ffn_h in product(
            hp_grid["lr"],
            hp_grid["batch_size"],
            hp_grid["dropout"],
            [(ffn_h, ffn_h // 2) for ffn_h in hp_grid["ffn_hidden"]],
        ):
            model = ConcatFFN(
                chemprop_hidden_size=300,
                chemprop_depth=3,
                chemprop_dropout=do,
                morgan_n_bits=2048,
                ffn_hidden_layers=ffn_h,
                ffn_dropout=do,
            )
            model, _ = _train_graph_model(
                model,
                train_gld,
                val_gld,
                lr=lr,
                epochs=50,
                patience=10,
                device=device,
                batch_size=bs,
                train_idx=tr_idx_hp,
                val_idx=va_idx_hp,
            )
            y_true_val, y_pred_val = _eval_graph_model(
                model, val_gld, device=device, test_idx=va_idx_hp, batch_size=bs
            )
            val_r2 = r2_score(y_true_val, y_pred_val)
            cfg = {"lr": lr, "batch_size": bs, "dropout": do, "ffn_hidden": ffn_h}
            results.append({**cfg, "val_R2": round(val_r2, 4)})
            print(f"  lr={lr}  bs={bs}  do={do}  ffn={ffn_h}  val_R²={val_r2:.4f}")
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_cfg = cfg

        print(f"\n  Best HP config: {best_cfg}  (val R²={best_val_r2:.4f})")
        pd.DataFrame(results).sort_values("val_R2", ascending=False).to_csv(
            RESULTS_DIR / f"hp_search_{ctx['split_type']}.csv", index=False
        )
        best_model = ConcatFFN(
            chemprop_hidden_size=300,
            chemprop_depth=3,
            chemprop_dropout=best_cfg["dropout"],
            morgan_n_bits=2048,
            ffn_hidden_layers=best_cfg["ffn_hidden"],
            ffn_dropout=best_cfg["dropout"],
        )
        best_model, _ = _train_graph_model(
            best_model,
            train_gld,
            val_gld,
            lr=best_cfg["lr"],
            epochs=100,
            patience=15,
            device=device,
            batch_size=best_cfg["batch_size"],
            train_idx=tr_idx_hp,
            val_idx=va_idx_hp,
        )
        ctx["best_hp_model"] = {
            "model": best_model,
            "loader": test_gld,
            "loader_idx": te_idx_hp,
            "type": "graph",
        }

    ctx["best_hp_cfg"] = best_cfg
    return ctx


# ===========================================================================
# STAGE 5 — Evaluation and comparison
# ===========================================================================


def stage5_evaluation(ctx: Dict, device: str = "cpu") -> Dict:
    """
    Stage 5: Evaluate all models under the current split with the full metric suite.
    Head-to-head comparison including the LANTERN Morgan+Expert MLP baseline.
    """
    print("\n" + "=" * 70)
    print(f"STAGE 5 — Evaluation  (split: {ctx['split_type']})")
    print("=" * 70)

    split = ctx["split_type"]
    all_records = []
    all_cdfs = {}

    all_models: Dict[str, Dict] = {}
    all_models.update(ctx.get("stage2_models", {}))
    all_models.update(ctx.get("stage3_models", {}))
    if "best_hp_model" in ctx:
        all_models["Chemprop+Morgan(best_HP)"] = ctx["best_hp_model"]
    if "stage6_models" in ctx:
        all_models.update(ctx["stage6_models"])

    for name, entry in all_models.items():
        model = entry["model"].to(device)
        loader = entry["loader"]
        if entry["type"] == "mlp":
            y_true, y_pred = _eval_mlp(model, loader, device=device)
        elif entry["type"] == "chemprop":
            y_true, y_pred = _eval_chemprop_model(model, loader, device=device)
        else:
            y_true, y_pred = _eval_graph_model(
                model,
                loader,
                device=device,
                test_idx=entry.get("loader_idx"),
                batch_size=128,
            )

        record, cdf = evaluate_model(y_true, y_pred, name, split)
        all_records.append(record)
        all_cdfs[name] = cdf

    results_df = pd.DataFrame(all_records)
    out_csv = RESULTS_DIR / f"results_{split}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"\n  Saved results → {out_csv}")

    ctx["stage5_results"] = results_df
    ctx["stage5_cdfs"] = all_cdfs
    return ctx


# ===========================================================================
# STAGE 6 — Ablation studies
# ===========================================================================


def stage6_ablation(ctx: Dict, device: str = "cpu") -> Dict:
    """
    Stage 6: Isolate contribution of Morgan fingerprints.
      • Fingerprint-only MLP  — captures pure FP signal, no graph encoder
      • Chemprop-only (pure)  — already trained in Stage 2; re-evaluated here
      • Hybrid vs pure graph  — quantify uplift from adding fingerprints
    """
    print("\n" + "=" * 70)
    print("STAGE 6 — Ablation Studies")
    print("=" * 70)

    train_idx = ctx["train_idx"]
    val_idx = ctx["val_idx"]
    ablation_models = {}

    nw = ctx.get("num_workers", 4)

    # ---- Fingerprint-only MLP --------------------------------
    print("\n  [6a] Fingerprint-only MLP(Morgan)  — isolates FP signal")
    train_ld, val_ld, test_ld, _ = _make_tensor_loaders(
        ctx["morgan_fps"],
        ctx["labels"],
        train_idx,
        val_idx,
        ctx["test_idx"],
        batch_size=64,
        device=device,
        num_workers=nw,
    )
    fp_only_mlp = MLPRegressor(input_dim=2048)
    fp_only_mlp, _ = _train_mlp(
        fp_only_mlp, train_ld, val_ld, lr=2e-4, epochs=100, patience=15, device=device
    )
    ablation_models["Ablation:FP-only_MLP"] = {
        "model": fp_only_mlp,
        "loader": train_ld if isinstance(train_ld, GPUTensorStore) else test_ld,
        "type": "mlp",
    }

    # ---- Expert-only MLP -------------------------------------
    print("\n  [6b] Expert-only MLP  — isolates physicochemical descriptors")
    _ef = ctx["expert_fps"]
    _test_ef = _ef[ctx["test_idx"]]
    _train_ef = _ef[train_idx]
    print(f"    Expert descriptor diagnostics:")
    print(
        f"      train  — max={np.nanmax(np.abs(_train_ef)):.3e}  "
        f"NaN={np.isnan(_train_ef).sum()}  Inf={np.isinf(_train_ef).sum()}"
    )
    print(
        f"      test   — max={np.nanmax(np.abs(_test_ef)):.3e}  "
        f"NaN={np.isnan(_test_ef).sum()}  Inf={np.isinf(_test_ef).sum()}"
    )
    train_ld_ex, val_ld_ex, test_ld_ex, _ = _make_tensor_loaders(
        ctx["expert_fps"],
        ctx["labels"],
        train_idx,
        val_idx,
        ctx["test_idx"],
        batch_size=64,
        device=device,
        num_workers=nw,
    )
    expert_only_mlp = MLPRegressor(input_dim=ctx["expert_fps"].shape[1])
    expert_only_mlp, _ = _train_mlp(
        expert_only_mlp,
        train_ld_ex,
        val_ld_ex,
        lr=2e-4,
        epochs=100,
        patience=15,
        device=device,
    )
    ablation_models["Ablation:Expert-only_MLP"] = {
        "model": expert_only_mlp,
        "loader": (
            train_ld_ex if isinstance(train_ld_ex, GPUTensorStore) else test_ld_ex
        ),
        "type": "mlp",
    }

    # Chemprop-only is already in Stage 2; note it here for comparison
    print("\n  [6c] Chemprop(pure) already in Stage 2 — included in Stage 5 table.")

    ctx["stage6_models"] = ablation_models
    return ctx


# ===========================================================================
# STAGE 5 summary comparison helper
# ===========================================================================


def print_comparison_table(ctx: Dict) -> None:
    """Print final head-to-head comparison table against LANTERN reference values."""
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE  (test set)")
    print("=" * 70)

    ref = {
        "MLP(Morgan)": {"R2": 0.7974, "RMSE": 1.5018, "MAE": 1.1507, "r": 0.8959},
        "MLP(Morgan+Expert)": {
            "R2": 0.8161,
            "RMSE": 1.4308,
            "MAE": 1.1003,
            "r": 0.9053,
        },
    }

    df = ctx.get("stage5_results")
    if df is None:
        print("  No Stage-5 results found — run with --stage all first.")
        return

    print(f"  {'Model':<35} {'R²':>7} {'RMSE':>7} {'MAE':>7} {'r':>7}  {'REF R²':>7}")
    print("  " + "-" * 75)
    for _, row in df.iterrows():
        model_name = row["model"]
        r2_ref = ref.get(model_name, {}).get("R2", "—")
        ref_str = f"{r2_ref:.4f}" if isinstance(r2_ref, float) else r2_ref
        print(
            f"  {model_name:<35} {row['R2']:>7.4f} {row['RMSE']:>7.4f} "
            f"{row['MAE']:>7.4f} {row['r']:>7.4f}  {ref_str:>7}"
        )


# ===========================================================================
# CLI entry point
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoFPGNN Baseline Pipeline")
    p.add_argument(
        "--stage",
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "all", "ablation"],
        help="Pipeline stage to run (default: all). Use 'ablation' to run Stage 1 + Stage 6 + Stage 5 only.",
    )
    p.add_argument(
        "--split",
        default="all",
        choices=["random", "scaffold", "all"],
        help="Data split strategy (default: all)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device (default: cuda if available, else cpu)",
    )
    p.add_argument(
        "--skip_hp",
        action="store_true",
        help="Skip Stage 4 HP search (saves time during development)",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile models for kernel fusion (PyTorch ≥ 2.0, A100 recommended)",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes for parallel graph featurisation (default: 4)",
    )
    return p.parse_args()


def _maybe_compile(model: nn.Module, use_compile: bool) -> nn.Module:
    """
    Wrap model with torch.compile if available and requested.
    torch.compile (Dynamo + Inductor) fuses scatter_add, linear, and ReLU
    kernels into single CUDA launches — reduces kernel launch overhead on A100.
    First forward pass triggers compilation (~30-60 s); subsequent runs are fast.
    """
    if use_compile and hasattr(torch, "compile"):
        print("    torch.compile() — tracing model for kernel fusion …")
        return torch.compile(model)
    return model


def run_pipeline(split_type: str, args: argparse.Namespace) -> None:
    device = args.device
    stage = args.stage

    # ---- A100 global setup (must precede any tensor / model creation) ----
    setup_gpu(device)
    amp_info = ""
    if device.startswith("cuda"):
        dt = _amp_dtype(device)
        amp_info = f"  AMP={dt}  TF32=on  cudnn.bench=on"
        if args.compile:
            amp_info += "  torch.compile=on"

    print(f"\n{'#'*70}")
    print(f"# MoFPGNN Baseline Pipeline  |  split={split_type}  |  device={device}")
    if amp_info:
        print(f"#{amp_info}")
    print(f"{'#'*70}")

    ctx = stage1_preprocessing(split_type)
    ctx["num_workers"] = args.num_workers
    ctx["use_compile"] = args.compile

    if stage in ("2", "all"):
        ctx = stage2_baseline_training(ctx, device=device)
    if stage in ("3", "all"):
        ctx = stage3_hybrid_model(ctx, device=device)
    if stage in ("4", "all") and not args.skip_hp:
        ctx = stage4_hp_optimisation(ctx, device=device)
    if stage in ("6", "all", "ablation"):
        ctx = stage6_ablation(ctx, device=device)
    if stage in ("5", "all", "ablation"):
        ctx = stage5_evaluation(ctx, device=device)

    print_comparison_table(ctx)

    summary_path = RESULTS_DIR / f"summary_{split_type}.json"
    if ctx.get("stage5_results") is not None:
        ctx["stage5_results"].to_json(summary_path, orient="records", indent=2)
        print(f"\nSaved summary → {summary_path}")


def main() -> None:
    args = parse_args()
    splits = ["random", "scaffold"] if args.split == "all" else [args.split]
    for split in splits:
        run_pipeline(split_type=split, args=args)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from utils.io_tools import load_yaml, load_pickle
from utils.metrics import compute_all_metrics, save_metrics_to_file
from utils.visual_utils import plot_loss_curves, generate_all_regression_plots

from pipeline.kpgt_trainer import KPGTTrainer

from models.kpgt_encoder import PretrainedEmbeddingRegressor
from models.hybrid_model import PretrainedKPGTMorganHybrid, MorganOnlyMLP


def _setup_device(config):
    """Resolve device string to actual device."""
    requested = config.get("device", "cpu")
    if requested == "cuda":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested


def _build_model(config, device):
    """Instantiate the model based on config."""
    model_type = config.get("model_type", "kpgt_pretrained_regressor")
    mlp_config = config.get("mlp_head", {})

    mlp_hidden = mlp_config.get("hidden_layers", [512, 256])
    dropout_mlp = mlp_config.get("dropout", 0.3)
    morgan_dim = config.get("morgan_dim", 2048)

    if model_type == "kpgt_pretrained_regressor":
        reg_config = config.get("regressor", {})
        embedding_dim = config.get("embedding_dim", 2304)
        model = PretrainedEmbeddingRegressor(
            input_dim=embedding_dim,
            hidden_dim=reg_config.get("hidden_dim", 512),
            dropout=reg_config.get("dropout", 0.3),
        )

    elif model_type == "kpgt_morgan_pretrained_hybrid":
        kpgt_dim = config.get("kpgt_dim", 2304)
        model = PretrainedKPGTMorganHybrid(
            kpgt_dim=kpgt_dim,
            morgan_dim=morgan_dim,
            mlp_hidden=mlp_hidden,
            dropout=dropout_mlp,
        )

    elif model_type == "morgan_only":
        model = MorganOnlyMLP(
            morgan_dim=morgan_dim,
            mlp_hidden=mlp_hidden, dropout=dropout_mlp,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_type} | Total params: {total_params:,} "
          f"| Trainable: {trainable_params:,}")

    return model


def _normalise_labels(train_ds, val_ds, test_ds):
    """
    Z-score normalise labels based on training set statistics.
    Modifies datasets in-place and returns (mean, std).
    """
    train_labels = np.array([d.y.item() for d in train_ds])
    mean = float(np.mean(train_labels))
    std = float(np.std(train_labels))
    if std < 1e-8:
        std = 1.0

    for ds in [train_ds, val_ds, test_ds]:
        for data in ds._data_list:
            data.y = (data.y - mean) / std

    print(f"Label normalisation: mean={mean:.4f}, std={std:.4f}")
    return mean, std


# Pretrained-embedding dataset
class _EmbeddingDataset(torch.utils.data.Dataset):
    """Minimal dataset wrapping pre-extracted embeddings + labels."""

    def __init__(self, embeddings, labels):
        from torch_geometric.data import Data
        self._data_list = []
        for emb, y in zip(embeddings, labels):
            d = Data()
            d.x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            d.y = torch.tensor([y], dtype=torch.float32)
            self._data_list.append(d)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

    # PyG Dataset compat
    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]


def _load_pretrained_embedding_datasets(config):
    """Build train/val/test datasets from pre-extracted embeddings.

    Embeddings are z-score normalised (fit on training set only) to prevent
    large-magnitude descriptor values from causing numerical overflow.
    """
    from sklearn.preprocessing import StandardScaler

    csv_path = config.get("csv_path", "data/AGILE.csv")
    split_path = config["split_path"]
    emb_path = config["embeddings_path"]

    df = pd.read_csv(csv_path)
    smiles_list = df["SMILES"].tolist()
    targets = df["Target"].values.astype(np.float32)

    emb_dict = load_pickle(emb_path)
    emb_dim = None

    # Build ordered arrays
    embeddings = []
    for smi in smiles_list:
        emb = emb_dict.get(smi)
        if emb is None:
            emb = np.zeros_like(next(iter(emb_dict.values())))
        embeddings.append(emb)
        if emb_dim is None:
            emb_dim = len(emb)
    embeddings = np.array(embeddings, dtype=np.float32)

    # Replace any NaN / Inf values that can come from RDKit descriptors
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    # Update embedding_dim in config to match actual data
    config["embedding_dim"] = emb_dim

    # Load splits
    splits = np.load(split_path, allow_pickle=True)
    train_idx, val_idx, test_idx = splits[0], splits[1], splits[2]

    # Feature normalisation (fit on train only)
    scaler = StandardScaler()
    scaler.fit(embeddings[train_idx])
    embeddings = scaler.transform(embeddings).astype(np.float32)
    # Clip extreme values after scaling to prevent outliers
    embeddings = np.clip(embeddings, -10.0, 10.0)

    train_ds = _EmbeddingDataset(embeddings[train_idx], targets[train_idx])
    val_ds = _EmbeddingDataset(embeddings[val_idx], targets[val_idx])
    test_ds = _EmbeddingDataset(embeddings[test_idx], targets[test_idx])

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Embedding dim: {emb_dim}")
    print(f"  Features scaled: mean~0, std~1 (fit on train, clipped to [-10, 10])")

    return train_ds, val_ds, test_ds


# Pretrained-hybrid dataset (KPGT embeddings + Morgan FPs)
class _HybridEmbeddingDataset(torch.utils.data.Dataset):
    """Dataset wrapping KPGT embeddings + Morgan FPs + labels."""

    def __init__(self, kpgt_embeddings, morgan_fps, labels):
        from torch_geometric.data import Data
        self._data_list = []
        for kpgt_emb, morgan_fp, y in zip(kpgt_embeddings, morgan_fps, labels):
            d = Data()
            d.x = torch.tensor(kpgt_emb, dtype=torch.float32).unsqueeze(0)
            d.morgan_fp = torch.tensor(morgan_fp, dtype=torch.float32).unsqueeze(0)
            d.y = torch.tensor([y], dtype=torch.float32)
            self._data_list.append(d)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]


class _MorganOnlyDataset(torch.utils.data.Dataset):
    """Dataset wrapping Morgan fingerprints + labels (no KPGT embeddings)."""

    def __init__(self, morgan_fps, labels):
        from torch_geometric.data import Data
        self._data_list = []
        for morgan_fp, y in zip(morgan_fps, labels):
            d = Data()
            d.morgan_fp = torch.tensor(morgan_fp, dtype=torch.float32).unsqueeze(0)
            d.y = torch.tensor([y], dtype=torch.float32)
            self._data_list.append(d)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        return self._data_list[idx]

    def len(self):
        return len(self._data_list)

    def get(self, idx):
        return self._data_list[idx]


def _load_morgan_only_datasets(config):
    """Build train/val/test datasets from Morgan fingerprints only."""
    from sklearn.preprocessing import StandardScaler

    csv_path = config.get("csv_path", "data/AGILE.csv")
    split_path = config["split_path"]
    morgan_path = config["morgan_path"]

    df = pd.read_csv(csv_path)
    smiles_list = df["SMILES"].tolist()
    targets = df["Target"].values.astype(np.float32)

    morgan_dict = load_pickle(morgan_path)
    morgan_fps = []
    for smi in smiles_list:
        fp = morgan_dict.get(smi)
        if fp is None:
            fp = np.zeros_like(next(iter(morgan_dict.values())))
        morgan_fps.append(fp)
    morgan_fps = np.array(morgan_fps, dtype=np.float32)
    morgan_fps = np.nan_to_num(morgan_fps, nan=0.0, posinf=0.0, neginf=0.0)
    config["morgan_dim"] = morgan_fps.shape[1]

    splits = np.load(split_path, allow_pickle=True)
    train_idx, val_idx, test_idx = splits[0], splits[1], splits[2]

    scaler = StandardScaler()
    scaler.fit(morgan_fps[train_idx])
    morgan_fps = np.clip(
        scaler.transform(morgan_fps).astype(np.float32), -10.0, 10.0
    )

    train_ds = _MorganOnlyDataset(morgan_fps[train_idx], targets[train_idx])
    val_ds = _MorganOnlyDataset(morgan_fps[val_idx], targets[val_idx])
    test_ds = _MorganOnlyDataset(morgan_fps[test_idx], targets[test_idx])

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  Morgan dim: {config['morgan_dim']}")
    print(f"  Features scaled: mean~0, std~1 (fit on train, clipped to [-10, 10])")

    return train_ds, val_ds, test_ds


def _load_pretrained_hybrid_datasets(config):
    """Build train/val/test datasets with both KPGT embeddings AND Morgan FPs.

    Both feature sets are z-score normalised (fit on training set only).
    """
    from sklearn.preprocessing import StandardScaler

    csv_path = config.get("csv_path", "data/AGILE.csv")
    split_path = config["split_path"]
    kpgt_path = config["embeddings_path"]
    morgan_path = config["morgan_path"]

    df = pd.read_csv(csv_path)
    smiles_list = df["SMILES"].tolist()
    targets = df["Target"].values.astype(np.float32)

    # Load KPGT embeddings
    kpgt_dict = load_pickle(kpgt_path)
    kpgt_embeddings = []
    for smi in smiles_list:
        emb = kpgt_dict.get(smi)
        if emb is None:
            emb = np.zeros_like(next(iter(kpgt_dict.values())))
        kpgt_embeddings.append(emb)
    kpgt_embeddings = np.array(kpgt_embeddings, dtype=np.float32)
    kpgt_embeddings = np.nan_to_num(kpgt_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
    config["kpgt_dim"] = kpgt_embeddings.shape[1]

    # Load Morgan fingerprints
    morgan_dict = load_pickle(morgan_path)
    morgan_fps = []
    for smi in smiles_list:
        fp = morgan_dict.get(smi)
        if fp is None:
            fp = np.zeros_like(next(iter(morgan_dict.values())))
        morgan_fps.append(fp)
    morgan_fps = np.array(morgan_fps, dtype=np.float32)
    morgan_fps = np.nan_to_num(morgan_fps, nan=0.0, posinf=0.0, neginf=0.0)
    config["morgan_dim"] = morgan_fps.shape[1]

    # Load splits
    splits = np.load(split_path, allow_pickle=True)
    train_idx, val_idx, test_idx = splits[0], splits[1], splits[2]

    # Normalise KPGT embeddings (fit on train)
    kpgt_scaler = StandardScaler()
    kpgt_scaler.fit(kpgt_embeddings[train_idx])
    kpgt_embeddings = np.clip(
        kpgt_scaler.transform(kpgt_embeddings).astype(np.float32), -10.0, 10.0
    )

    # Normalise Morgan FPs (fit on train)
    morgan_scaler = StandardScaler()
    morgan_scaler.fit(morgan_fps[train_idx])
    morgan_fps = np.clip(
        morgan_scaler.transform(morgan_fps).astype(np.float32), -10.0, 10.0
    )

    train_ds = _HybridEmbeddingDataset(
        kpgt_embeddings[train_idx], morgan_fps[train_idx], targets[train_idx])
    val_ds = _HybridEmbeddingDataset(
        kpgt_embeddings[val_idx], morgan_fps[val_idx], targets[val_idx])
    test_ds = _HybridEmbeddingDataset(
        kpgt_embeddings[test_idx], morgan_fps[test_idx], targets[test_idx])

    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  KPGT dim: {config['kpgt_dim']} | Morgan dim: {config['morgan_dim']}")
    print(f"  Total fused dim: {config['kpgt_dim'] + config['morgan_dim']}")
    print(f"  Both feature sets scaled: mean~0, std~1 (fit on train)")

    return train_ds, val_ds, test_ds


def run_kpgt_pipeline(config_path):
    """
    Main entry point: run the full KPGT pipeline from a YAML config file.
    """
    config = load_yaml(config_path)
    config["device"] = _setup_device(config)

    model_type = config.get("model_type", "kpgt_pretrained_regressor")
    split_name = config.get("split", "random")
    fp_variant = config.get("fp_variant", "")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    name_parts = [model_type]
    if fp_variant:
        name_parts.append(fp_variant)
    name_parts += [split_name, timestamp]
    pipeline_name = "-".join(name_parts)
    config["pipeline_save_name"] = pipeline_name

    print(f"Pipeline: {pipeline_name}")
    print(f"Device: {config['device']}")

    # Load data
    if model_type == "kpgt_morgan_pretrained_hybrid":
        print("Loading pre-extracted KPGT embeddings + Morgan fingerprints...")
        train_ds, val_ds, test_ds = _load_pretrained_hybrid_datasets(config)
    elif model_type == "kpgt_pretrained_regressor":
        print("Loading pre-extracted embeddings...")
        train_ds, val_ds, test_ds = _load_pretrained_embedding_datasets(config)
    elif model_type == "morgan_only":
        print("Loading Morgan fingerprints (ablation)...")
        train_ds, val_ds, test_ds = _load_morgan_only_datasets(config)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Normalise labels
    label_mean, label_std = _normalise_labels(train_ds, val_ds, test_ds)

    # Model
    print("Building model...")
    model = _build_model(config, config["device"])

    # Train
    trainer = KPGTTrainer(model, config, label_mean=label_mean, label_std=label_std)

    print("Training...")
    loss_dict = trainer.train(train_ds, val_ds)

    # Save checkpoint
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{pipeline_name}.pth")
    trainer.save(model_path)

    # Evaluate
    print("Evaluating...")
    train_preds = trainer.predict(train_ds)
    val_preds = trainer.predict(val_ds)
    test_preds = trainer.predict(test_ds)

    train_labels = trainer.get_true_labels(train_ds)
    val_labels = trainer.get_true_labels(val_ds)
    test_labels = trainer.get_true_labels(test_ds)

    train_metrics = compute_all_metrics(train_preds, train_labels, "Train")
    val_metrics = compute_all_metrics(val_preds, val_labels, "Val")
    test_metrics = compute_all_metrics(test_preds, test_labels, "Test")

    print(f"\n{'='*50}")
    print("TEST RESULTS:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")

    # Save
    results_dir = os.path.join("results", pipeline_name)
    os.makedirs(results_dir, exist_ok=True)

    save_metrics_to_file(
        {"Train": train_metrics, "Val": val_metrics, "Test": test_metrics},
        os.path.join(results_dir, "metrics.txt"),
        title=pipeline_name,
    )

    for name, preds, labels in [
        ("train", train_preds, train_labels),
        ("val", val_preds, val_labels),
        ("test", test_preds, test_labels),
    ]:
        df = pd.DataFrame({"true": labels.flatten(), "pred": preds.flatten()})
        df.to_csv(os.path.join(results_dir, f"{name}_results.csv"), index=False)

    print("Generating plots...")
    _save_plots(loss_dict, train_preds, val_preds, test_preds,
                train_labels, val_labels, test_labels, config, results_dir)

    print(f"All results saved to: {results_dir}")
    return test_metrics


def _save_plots(loss_dict, train_preds, val_preds, test_preds,
                train_labels, val_labels, test_labels, config, results_dir):
    """Generate and save loss curves and regression plots."""
    try:
        plot_loss_curves(loss_dict, config, save=True)
        generate_all_regression_plots(
            train_preds, val_preds, test_preds,
            train_labels, val_labels, test_labels,
            config,
        )
    except Exception as e:
        print(f"Warning: could not generate plots via LANTERN utils: {e}")
        _fallback_plots(loss_dict, test_preds, test_labels, results_dir)


def _fallback_plots(loss_dict, test_preds, test_labels, results_dir):
    """Simple fallback plots if LANTERN visual utils fail."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if loss_dict:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(loss_dict.get("Train", []), label="Train")
            ax.plot(loss_dict.get("Validation", []), label="Validation")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Training Loss Curves")
            ax.legend()
            fig.savefig(os.path.join(results_dir, "loss_curve.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(test_labels.flatten(), test_preds.flatten(), alpha=0.5, s=15)
        lims = [
            min(test_labels.min(), test_preds.min()),
            max(test_labels.max(), test_preds.max()),
        ]
        ax.plot(lims, lims, "r--", alpha=0.7)
        ax.set_xlabel("True mTP")
        ax.set_ylabel("Predicted mTP")
        ax.set_title("Test Set: Predicted vs True")
        fig.savefig(os.path.join(results_dir, "prediction_vs_true_test.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"Warning: fallback plot generation failed: {e}")

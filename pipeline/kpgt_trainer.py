import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch_geometric.loader import DataLoader

from utils.utils import make_generator


class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    """Linear warmup for `warmup_epochs`, then cosine decay to `min_lr`."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6,
                 last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * cosine
                    for base_lr in self.base_lrs]


class KPGTTrainer:
    """
    Trainer for KPGT-based models (standalone and hybrid).

    Args:
        model: nn.Module (KPGTRegressor, KPGTMorganHybrid, or MorganOnlyMLP)
        config: dict with training hyperparameters
        label_mean: mean of training labels (for denormalisation)
        label_std: std of training labels (for denormalisation)
    """

    def __init__(self, model, config, label_mean=0.0, label_std=1.0):
        self.model = model
        self.config = config
        self.device = config.get("device", "cpu")
        self.label_mean = label_mean
        self.label_std = label_std

        self.model.to(self.device)

    def train(self, train_dataset, val_dataset):
        """
        Train the model.

        Args:
            train_dataset: MolecularGraphDataset for training
            val_dataset: MolecularGraphDataset for validation

        Returns:
            dict with "Train" and "Validation" loss lists
        """
        batch_size = self.config.get("batch_size", 32)
        epochs = self.config.get("epochs", 300)
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-4)
        patience = self.config.get("patience", 50)
        warmup_epochs = self.config.get("warmup_epochs", 10)
        grad_clip = self.config.get("grad_clip", 1.0)
        loss_fn = self.config.get("loss_fn", "huber")
        # "none"/"constant" holds lr fixed; the Morgan baseline trains that way.
        scheduler_type = self.config.get("scheduler", "cosine")
        # The Morgan-only baseline's DataLoader does NOT shuffle; default stays True.
        shuffle = self.config.get("shuffle", True)
        generator = make_generator(self.config.get("seed", 42)) if shuffle else None
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=shuffle, drop_last=False,
                                  generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False)

        # Collect trainable parameters
        if hasattr(self.model, "get_trainable_params"):
            params = list(self.model.get_trainable_params())
        else:
            params = list(self.model.parameters())

        # AdamW with weight_decay=0 is identical to plain Adam (the baseline's choice).
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if scheduler_type in ("none", "constant", None):
            scheduler = None
        else:
            scheduler = WarmupCosineScheduler(optimizer, warmup_epochs, epochs)

        # Huber loss is more robust to outlier targets than MSE
        if loss_fn == "huber":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_train_loss, n_train = 0.0, 0
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                preds = self._forward(batch)
                loss = criterion(preds.squeeze(-1), batch.y.squeeze(-1))
                loss.backward()
                # grad_clip <= 0 (or None) disables clipping; the baseline does not clip.
                if grad_clip and grad_clip > 0:
                    nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
                optimizer.step()
                epoch_train_loss += loss.item() * batch.num_graphs
                n_train += batch.num_graphs

            avg_train = epoch_train_loss / max(n_train, 1)

            # Validate
            avg_val = self._evaluate_loss(val_loader, criterion)

            if scheduler is not None:
                scheduler.step()
            train_losses.append(avg_train)
            val_losses.append(avg_val)

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_model_state = deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if self.config.get("loss_per_epoch", True):
                print(f"Epoch {epoch:<3} | Train: {avg_train:.4f} | "
                      f"Val: {avg_val:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            else:
                print(f"Epoch {epoch:<3}")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={patience})")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        print(f"\nBest model -- Val Loss: {best_val_loss:.4f}\n")

        return {"Train": train_losses, "Validation": val_losses}

    def predict(self, dataset):
        """
        Generate predictions for a dataset.

        Returns:
            numpy array of predictions, shape (N, 1), in original label scale
        """
        loader = DataLoader(dataset,
                            batch_size=self.config.get("batch_size", 32),
                            shuffle=False)
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self._forward(batch)
                all_preds.append(preds.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        # Denormalise
        preds = preds * self.label_std + self.label_mean
        return preds

    def get_true_labels(self, dataset):
        """Extract ground-truth labels from a dataset, in original scale."""
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        batch = next(iter(loader))
        labels = batch.y.numpy()
        return labels * self.label_std + self.label_mean

    def _forward(self, batch):
        """Forward pass handling pretrained-embedding, pretrained-hybrid, and Morgan-only models."""
        from models.hybrid_model import MorganOnlyMLP, PretrainedKPGTMorganHybrid
        from models.kpgt_encoder import PretrainedEmbeddingRegressor
        if isinstance(self.model, PretrainedKPGTMorganHybrid):
            # KPGT embeddings in data.x, Morgan FPs in data.morgan_fp
            kpgt_emb = batch.x.view(batch.num_graphs, -1).to(self.device)
            morgan_fp = batch.morgan_fp.view(batch.num_graphs, -1).to(self.device)
            return self.model(kpgt_emb, morgan_fp)
        elif isinstance(self.model, MorganOnlyMLP):
            morgan_fp = batch.morgan_fp.view(batch.num_graphs, -1).to(self.device)
            return self.model(morgan_fp)
        elif isinstance(self.model, PretrainedEmbeddingRegressor):
            emb = batch.x.view(batch.num_graphs, -1).to(self.device)
            return self.model(emb)
        return self.model(batch)

    def _evaluate_loss(self, loader, criterion):
        """Compute average loss over a DataLoader."""
        self.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                preds = self._forward(batch)
                loss = criterion(preds.squeeze(-1), batch.y.squeeze(-1))
                total_loss += loss.item() * batch.num_graphs
                n += batch.num_graphs
        return total_loss / max(n, 1)

    def save(self, path):
        """Save model state dict."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to: {path}")

    def load(self, path):
        """Load model state dict."""
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        print(f"Model loaded from: {path}")

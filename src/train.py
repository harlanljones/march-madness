"""Training loop with early stopping and LR scheduling."""

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .model import BracketNet
from .dataset import MarchMadnessDataset


def train_model(
    train_ds: MarchMadnessDataset,
    val_ds: MarchMadnessDataset,
    input_dim: int | None = None,
    epochs: int = 300,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    hidden_dims: tuple[int, ...] = (128, 64, 32),
    dropout: float = 0.3,
    scheduler_factor: float = 0.5,
    scheduler_patience: int = 7,
    epoch_callback=None,
    verbose: bool = True,
    device: str = "cpu",
) -> tuple[BracketNet, dict]:
    """Train BracketNet with early stopping.

    Args:
        epoch_callback: Optional callable(epoch, val_loss) called each epoch.
            May raise optuna.TrialPruned to abort training early.
        verbose: If False, suppress per-epoch logging.

    Returns:
        (best_model, history) where history has train_loss, val_loss lists.
    """
    if input_dim is None:
        input_dim = train_ds.features.shape[1]

    model = BracketNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds))

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_features, val_labels = next(iter(val_loader))
            val_features, val_labels = val_features.to(device), val_labels.to(device)
            val_preds = model(val_features)
            val_loss = criterion(val_preds, val_labels).item()

        avg_train_loss = np.mean(train_losses)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch % 25 == 0 or epoch == 1):
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr_now:.1e}")

        if epoch_callback is not None:
            epoch_callback(epoch, val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"Best validation loss: {best_val_loss:.4f}")
    return model, history


def train_ensemble(
    train_ds: MarchMadnessDataset,
    val_ds: MarchMadnessDataset,
    n_models: int = 5,
    **kwargs,
) -> tuple[list[BracketNet], dict]:
    """Train an ensemble of models with different random seeds."""
    models = []
    best_history = None

    for i in range(n_models):
        print(f"\n--- Ensemble model {i+1}/{n_models} ---")
        torch.manual_seed(42 + i)
        np.random.seed(42 + i)
        model, history = train_model(train_ds, val_ds, **kwargs)
        models.append(model)
        if best_history is None:
            best_history = history

    return models, best_history


def save_model(model: BracketNet, path: str, scaler=None, metadata: dict | None = None):
    """Save model checkpoint with optional scaler and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    if scaler is not None:
        checkpoint["scaler_mean"] = scaler.mean_.tolist()
        checkpoint["scaler_scale"] = scaler.scale_.tolist()

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(path: str, device: str = "cpu") -> tuple[BracketNet, dict]:
    """Load model from checkpoint.

    Infers architecture (hidden_dims) from weight shapes so old checkpoints
    without metadata are still compatible.

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    # Find all Linear layers (2-D weight tensors) by sequential index
    linear_indices = sorted(
        int(k.split(".")[1])
        for k in state_dict
        if k.startswith("net.") and k.endswith(".weight") and state_dict[k].dim() == 2
    )
    input_dim = state_dict["net.0.weight"].shape[1]
    # All hidden layers except the final output layer
    hidden_dims = tuple(state_dict[f"net.{i}.weight"].shape[0] for i in linear_indices[:-1])
    dropout = checkpoint.get("metadata", {}).get("dropout", 0.3)

    model = BracketNet(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint

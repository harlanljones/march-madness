"""PyTorch Dataset and data preparation utilities."""

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class MarchMadnessDataset(Dataset):
    """PyTorch Dataset wrapping feature tensors and labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def augment_with_flips(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Data augmentation: add flipped matchups to enforce symmetry.

    For each (A, B) with features f and label y, add (B, A) with -f and 1-y.
    """
    flipped_features = -features
    flipped_labels = 1.0 - labels
    aug_features = np.concatenate([features, flipped_features], axis=0)
    aug_labels = np.concatenate([labels, flipped_labels], axis=0)
    return aug_features, aug_labels


def build_training_data(
    data: dict,
    train_seasons: list[int],
    val_seasons: list[int],
    augment: bool = True,
) -> tuple[MarchMadnessDataset, MarchMadnessDataset, StandardScaler]:
    """Build train and validation datasets from raw data.

    Args:
        data: Dict of DataFrames from load_all.
        train_seasons: Seasons for training.
        val_seasons: Seasons for validation.
        augment: Whether to apply flip augmentation.

    Returns:
        (train_dataset, val_dataset, scaler)
    """
    from .features import build_tournament_matchups

    stats_cache = {}

    train_features, train_labels = build_tournament_matchups(data, train_seasons, stats_cache)
    val_features, val_labels = build_tournament_matchups(data, val_seasons, stats_cache)

    # Augment training data
    if augment:
        train_features, train_labels = augment_with_flips(train_features, train_labels)

    # Fit scaler on training data only
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)

    train_ds = MarchMadnessDataset(train_features, train_labels)
    val_ds = MarchMadnessDataset(val_features, val_labels)

    return train_ds, val_ds, scaler

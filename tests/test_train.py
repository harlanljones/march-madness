"""Tests for src/train.py -- training loop, save/load."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from src.model import BracketNet
from src.dataset import MarchMadnessDataset
from src.features import N_FEATURES
from src.train import train_model, save_model, load_model


def _make_dummy_datasets(n_train=200, n_val=50):
    """Create simple linearly separable data for fast training tests."""
    rng = np.random.RandomState(42)
    X_train = rng.randn(n_train, N_FEATURES).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(np.float32)  # separable on first feature
    X_val = rng.randn(n_val, N_FEATURES).astype(np.float32)
    y_val = (X_val[:, 0] > 0).astype(np.float32)
    return MarchMadnessDataset(X_train, y_train), MarchMadnessDataset(X_val, y_val)


class TestTrainModel:
    def test_returns_model_and_history(self):
        train_ds, val_ds = _make_dummy_datasets()
        model, history = train_model(train_ds, val_ds, epochs=5, patience=100)
        assert isinstance(model, BracketNet)
        assert "train_loss" in history
        assert "val_loss" in history

    def test_loss_decreases(self):
        train_ds, val_ds = _make_dummy_datasets()
        _, history = train_model(train_ds, val_ds, epochs=30, patience=100)
        # First loss should be higher than last
        assert history["train_loss"][0] > history["train_loss"][-1]

    def test_early_stopping(self):
        """With patience=2 and easy data, training should stop before max epochs."""
        train_ds, val_ds = _make_dummy_datasets()
        _, history = train_model(train_ds, val_ds, epochs=300, patience=2)
        assert len(history["train_loss"]) < 300

    def test_history_lengths_match(self):
        train_ds, val_ds = _make_dummy_datasets()
        _, history = train_model(train_ds, val_ds, epochs=10, patience=100)
        assert len(history["train_loss"]) == len(history["val_loss"])


class TestSaveLoadModel:
    def test_roundtrip(self):
        model = BracketNet()
        model.eval()
        x = torch.randn(4, N_FEATURES)
        original_out = model(x)

        scaler = StandardScaler()
        scaler.fit(np.random.randn(10, N_FEATURES))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model.pt")
            save_model(model, path, scaler=scaler, metadata={"test": True})

            loaded, checkpoint = load_model(path)
            loaded.eval()
            loaded_out = loaded(x)

            torch.testing.assert_close(original_out, loaded_out)
            assert checkpoint["metadata"]["test"] is True
            assert "scaler_mean" in checkpoint
            assert "scaler_scale" in checkpoint

    def test_scaler_preserved(self):
        model = BracketNet()
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, N_FEATURES))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.pt")
            save_model(model, path, scaler=scaler)
            _, checkpoint = load_model(path)

            np.testing.assert_allclose(
                checkpoint["scaler_mean"], scaler.mean_, atol=1e-6
            )
            np.testing.assert_allclose(
                checkpoint["scaler_scale"], scaler.scale_, atol=1e-6
            )

    def test_creates_parent_dirs(self):
        model = BracketNet()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "sub" / "dir" / "model.pt")
            save_model(model, path)
            assert Path(path).exists()

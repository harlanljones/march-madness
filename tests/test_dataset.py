"""Tests for src/dataset.py -- PyTorch Dataset and augmentation."""

import numpy as np
import pytest
import torch

from src.dataset import MarchMadnessDataset, augment_with_flips, build_training_data


class TestMarchMadnessDataset:
    def test_len(self):
        ds = MarchMadnessDataset(np.zeros((10, 5)), np.ones(10))
        assert len(ds) == 10

    def test_getitem_types(self):
        ds = MarchMadnessDataset(np.zeros((10, 5)), np.ones(10))
        feats, label = ds[0]
        assert isinstance(feats, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert feats.dtype == torch.float32
        assert label.dtype == torch.float32

    def test_getitem_shape(self):
        ds = MarchMadnessDataset(np.zeros((10, 5)), np.ones(10))
        feats, label = ds[3]
        assert feats.shape == (5,)
        assert label.shape == ()


class TestAugmentWithFlips:
    def test_doubles_size(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([1.0, 0.0])
        aug_f, aug_l = augment_with_flips(features, labels)
        assert aug_f.shape[0] == 4
        assert aug_l.shape[0] == 4

    def test_flipped_features_negated(self):
        features = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([1.0, 0.0])
        aug_f, aug_l = augment_with_flips(features, labels)
        # Original
        np.testing.assert_array_equal(aug_f[0], [1.0, 2.0])
        # Flipped
        np.testing.assert_array_equal(aug_f[2], [-1.0, -2.0])

    def test_flipped_labels_inverted(self):
        features = np.array([[1.0, 2.0]])
        labels = np.array([1.0])
        _, aug_l = augment_with_flips(features, labels)
        assert aug_l[0] == 1.0
        assert aug_l[1] == 0.0

    def test_symmetry_property(self):
        """After augmentation, mean of all labels should be ~0.5."""
        rng = np.random.RandomState(42)
        features = rng.randn(100, 10)
        labels = rng.choice([0.0, 1.0], size=100)
        _, aug_l = augment_with_flips(features, labels)
        assert aug_l.mean() == pytest.approx(0.5)


class TestBuildTrainingData:
    def test_returns_correct_types(self, synthetic_data):
        from sklearn.preprocessing import StandardScaler
        train_ds, val_ds, scaler = build_training_data(
            synthetic_data, [2020, 2021, 2022], [2023]
        )
        assert isinstance(train_ds, MarchMadnessDataset)
        assert isinstance(val_ds, MarchMadnessDataset)
        assert isinstance(scaler, StandardScaler)

    def test_train_augmented(self, synthetic_data):
        """Training data should be ~2x the raw matchup count (flip augmentation)."""
        from src.features import build_tournament_matchups
        raw_feats, _ = build_tournament_matchups(synthetic_data, [2020, 2021, 2022])
        train_ds, _, _ = build_training_data(
            synthetic_data, [2020, 2021, 2022], [2023]
        )
        assert len(train_ds) == 2 * raw_feats.shape[0]

    def test_no_augment_option(self, synthetic_data):
        from src.features import build_tournament_matchups
        raw_feats, _ = build_tournament_matchups(synthetic_data, [2020, 2021, 2022])
        train_ds, _, _ = build_training_data(
            synthetic_data, [2020, 2021, 2022], [2023], augment=False
        )
        assert len(train_ds) == raw_feats.shape[0]

    def test_scaler_normalizes(self, synthetic_data):
        train_ds, _, scaler = build_training_data(
            synthetic_data, [2020, 2021, 2022], [2023]
        )
        # Scaled training features should have mean ~0, std ~1
        feats = train_ds.features.numpy()
        means = feats.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=0.15)

    def test_val_not_augmented(self, synthetic_data):
        """Validation set should NOT be augmented."""
        from src.features import build_tournament_matchups
        raw_val, _ = build_tournament_matchups(synthetic_data, [2023])
        _, val_ds, _ = build_training_data(
            synthetic_data, [2020, 2021, 2022], [2023]
        )
        assert len(val_ds) == raw_val.shape[0]

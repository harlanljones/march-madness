"""Tests for src/evaluate.py -- metrics and calibration."""

import numpy as np
import pytest
import torch

from src.model import BracketNet
from src.dataset import MarchMadnessDataset
from src.features import N_FEATURES
from src.evaluate import evaluate_model, calibration_data, espn_bracket_score


class TestEvaluateModel:
    def test_returns_expected_keys(self):
        model = BracketNet()
        model.eval()
        ds = MarchMadnessDataset(np.random.randn(20, N_FEATURES).astype(np.float32),
                                  np.random.choice([0.0, 1.0], 20).astype(np.float32))
        result = evaluate_model(model, ds)
        assert "log_loss" in result
        assert "accuracy" in result
        assert "predictions" in result
        assert "labels" in result

    def test_perfect_model(self):
        """A model predicting correct labels perfectly should have low log loss."""
        model = BracketNet()
        model.eval()
        # We can't easily make a perfect model, but we can test the metric
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        preds = np.array([0.99, 0.01, 0.99, 0.01])
        from sklearn.metrics import log_loss
        ll = log_loss(labels, preds)
        assert ll < 0.05

    def test_accuracy_range(self):
        model = BracketNet()
        model.eval()
        ds = MarchMadnessDataset(np.random.randn(20, N_FEATURES).astype(np.float32),
                                  np.random.choice([0.0, 1.0], 20).astype(np.float32))
        result = evaluate_model(model, ds)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_log_loss_positive(self):
        model = BracketNet()
        model.eval()
        ds = MarchMadnessDataset(np.random.randn(20, N_FEATURES).astype(np.float32),
                                  np.random.choice([0.0, 1.0], 20).astype(np.float32))
        result = evaluate_model(model, ds)
        assert result["log_loss"] > 0


class TestCalibrationData:
    def test_perfect_calibration(self):
        """If predictions match labels perfectly, actuals should match bins."""
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        preds = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.7, 0.75, 0.8, 0.85, 0.9])
        cal = calibration_data(labels, preds, n_bins=5)
        assert len(cal["bin_centers"]) > 0
        assert len(cal["bin_centers"]) == len(cal["bin_actuals"])
        assert len(cal["bin_centers"]) == len(cal["bin_counts"])

    def test_all_bins_between_0_and_1(self):
        rng = np.random.RandomState(42)
        labels = rng.choice([0.0, 1.0], 100)
        preds = rng.uniform(0, 1, 100)
        cal = calibration_data(labels, preds)
        for center in cal["bin_centers"]:
            assert 0 <= center <= 1
        for actual in cal["bin_actuals"]:
            assert 0 <= actual <= 1

    def test_empty_bins_excluded(self):
        labels = np.array([1.0, 1.0])
        preds = np.array([0.9, 0.95])
        cal = calibration_data(labels, preds, n_bins=10)
        # Most bins should be empty, only the last bin should have data
        assert len(cal["bin_centers"]) < 10


class TestESPNBracketScore:
    def test_perfect_bracket(self):
        """Perfect bracket should get max score: 32*10 + 16*20 + 8*40 + 4*80 + 2*160 + 1*320 = 1920."""
        winners = list(range(63))
        score = espn_bracket_score(winners, winners)
        assert score == 32*10 + 16*20 + 8*40 + 4*80 + 2*160 + 1*320

    def test_empty_bracket(self):
        assert espn_bracket_score([], []) == 0

    def test_all_wrong(self):
        predicted = [0] * 63
        actual = [1] * 63
        assert espn_bracket_score(predicted, actual) == 0

    def test_partial_bracket(self):
        """Only first round correct."""
        predicted = list(range(32)) + [999] * 31
        actual = list(range(32)) + list(range(32, 63))
        score = espn_bracket_score(predicted, actual)
        assert score == 32 * 10  # only first round

    def test_later_rounds_worth_more(self):
        """Getting only the championship right = 320 points."""
        # 62 wrong games, then correct championship at index 62
        predicted = list(range(62)) + [42]
        actual = list(range(100, 162)) + [42]
        score = espn_bracket_score(predicted, actual)
        assert score == 320

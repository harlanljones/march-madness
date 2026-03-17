"""Tests for src/model.py -- BracketNet architecture."""

import torch
import pytest

from src.model import BracketNet
from src.features import N_FEATURES


class TestBracketNet:
    def test_output_shape_batch(self):
        model = BracketNet()
        x = torch.randn(32, N_FEATURES)
        out = model(x)
        assert out.shape == (32,)

    def test_output_shape_single(self):
        model = BracketNet()
        model.eval()  # BatchNorm requires eval mode for batch_size=1
        x = torch.randn(1, N_FEATURES)
        out = model(x)
        assert out.shape == (1,)

    def test_output_range(self):
        """Sigmoid output should be in (0, 1)."""
        model = BracketNet()
        x = torch.randn(100, N_FEATURES)
        out = model(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_extreme_inputs(self):
        """Model shouldn't produce NaN on extreme inputs."""
        model = BracketNet()
        x = torch.ones(4, N_FEATURES) * 100
        out = model(x)
        assert not torch.any(torch.isnan(out))

    def test_custom_input_dim(self):
        model = BracketNet(input_dim=10)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8,)

    def test_custom_dropout(self):
        model = BracketNet(dropout=0.5)
        x = torch.randn(8, N_FEATURES)
        out = model(x)
        assert out.shape == (8,)

    def test_eval_vs_train_deterministic(self):
        """In eval mode, output should be deterministic."""
        model = BracketNet()
        model.eval()
        x = torch.randn(8, N_FEATURES)
        out1 = model(x)
        out2 = model(x)
        torch.testing.assert_close(out1, out2)

    def test_parameter_count_reasonable(self):
        """Model should be small enough to avoid overfitting on ~2500 games."""
        model = BracketNet()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 50000, f"Model has {n_params} params, may overfit"
        assert n_params > 100, f"Model has {n_params} params, suspiciously small"

    def test_gradient_flows(self):
        """Loss.backward() should produce non-zero gradients."""
        model = BracketNet()
        x = torch.randn(8, N_FEATURES)
        y = torch.ones(8) * 0.5
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert has_grad

    def test_symmetry_property(self):
        """Model(f) + Model(-f) should be ~1.0 for a trained symmetric model.

        For a randomly initialized model this won't be exact, but we test
        that the architecture at least accepts negated inputs correctly.
        """
        model = BracketNet()
        model.eval()
        x = torch.randn(16, N_FEATURES)
        out_pos = model(x)
        out_neg = model(-x)
        # Won't be exactly 1.0 for random weights, just check shapes work
        assert out_neg.shape == out_pos.shape

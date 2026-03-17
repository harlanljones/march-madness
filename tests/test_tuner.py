"""Tests for src/tuner.py -- Optuna hyperparameter tuning."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import optuna
import pytest
import torch

from src.model import BracketNet
from src.features import N_FEATURES
from src.tuner import suggest_config, suggest_config_from_params, run_tuning


class TestSuggestConfig:
    def test_returns_all_keys(self):
        study = optuna.create_study()
        trial = study.ask()
        config = suggest_config(trial)
        expected_keys = {
            "hidden_dims", "dropout", "lr", "weight_decay",
            "scheduler_factor", "scheduler_patience", "batch_size", "early_stop_patience",
        }
        assert set(config.keys()) == expected_keys

    def test_hidden_dims_is_tuple(self):
        study = optuna.create_study()
        trial = study.ask()
        config = suggest_config(trial)
        assert isinstance(config["hidden_dims"], tuple)
        assert len(config["hidden_dims"]) in (2, 3, 4)

    def test_values_in_range(self):
        study = optuna.create_study()
        for _ in range(5):
            trial = study.ask()
            config = suggest_config(trial)
            assert 0.1 <= config["dropout"] <= 0.5
            assert 1e-4 <= config["lr"] <= 1e-2
            assert config["batch_size"] in [32, 64, 128, 256]
            assert 10 <= config["early_stop_patience"] <= 40

    def test_suggest_config_from_params_round_trips(self):
        study = optuna.create_study()
        trial = study.ask()
        original = suggest_config(trial)
        params = trial.params
        # suggest_config_from_params needs all 4 layer sizes in params
        # ensure they're present from suggest_config call
        reconstructed = suggest_config_from_params(params)
        assert reconstructed["hidden_dims"] == original["hidden_dims"]
        assert reconstructed["dropout"] == original["dropout"]
        assert reconstructed["lr"] == original["lr"]


class TestBracketNetConfigurable:
    def test_custom_hidden_dims_output_shape(self):
        model = BracketNet(hidden_dims=(256, 128))
        x = torch.randn(8, N_FEATURES)
        out = model(x)
        assert out.shape == (8,)

    def test_single_hidden_layer(self):
        model = BracketNet(hidden_dims=(64,))
        model.eval()
        x = torch.randn(1, N_FEATURES)
        out = model(x)
        assert out.shape == (1,)
        assert 0 <= out.item() <= 1

    def test_four_hidden_layers(self):
        model = BracketNet(hidden_dims=(128, 64, 32, 16))
        x = torch.randn(4, N_FEATURES)
        out = model(x)
        assert out.shape == (4,)

    def test_load_model_infers_hidden_dims(self, tmp_path):
        from src.train import save_model, load_model

        hidden_dims = (256, 64)
        model = BracketNet(hidden_dims=hidden_dims, dropout=0.2)
        path = str(tmp_path / "test_model.pt")
        save_model(model, path, metadata={"hidden_dims": list(hidden_dims), "dropout": 0.2})

        loaded_model, checkpoint = load_model(path)
        # Verify architecture was correctly inferred from state dict
        x = torch.randn(4, N_FEATURES)
        out = loaded_model(x)
        assert out.shape == (4,)


class TestRunTuningSmoke:
    def test_two_trials(self, synthetic_data, tmp_path):
        """Smoke test: run_tuning completes 2 trials and saves best_config.json."""
        storage = str(tmp_path / "optuna.db")
        output_config = str(tmp_path / "best_config.json")

        run_tuning(
            data=synthetic_data,
            train_seasons=[2020, 2021, 2022],
            val_seasons=[2023],
            n_trials=2,
            storage_path=storage,
            study_name="test_study",
            epochs_per_trial=5,
            output_config_path=output_config,
            device="cpu",
        )

        assert Path(output_config).exists()
        with open(output_config) as f:
            config = json.load(f)

        assert "hidden_dims" in config
        assert isinstance(config["hidden_dims"], list)
        assert "_val_log_loss" in config
        assert config["_val_log_loss"] > 0

    def test_resume_adds_trials(self, synthetic_data, tmp_path):
        """Resuming a study adds more trials rather than restarting."""
        storage = str(tmp_path / "optuna.db")
        output_config = str(tmp_path / "best_config.json")

        common_kwargs = dict(
            data=synthetic_data,
            train_seasons=[2020, 2021, 2022],
            val_seasons=[2023],
            storage_path=storage,
            study_name="resume_study",
            epochs_per_trial=3,
            output_config_path=output_config,
            device="cpu",
        )

        run_tuning(**common_kwargs, n_trials=2)
        run_tuning(**common_kwargs, n_trials=2)

        import optuna as _optuna
        storage_url = f"sqlite:///{Path(storage).resolve()}"
        study = _optuna.load_study(study_name="resume_study", storage=storage_url)
        completed = [t for t in study.trials if t.state == _optuna.trial.TrialState.COMPLETE]
        assert len(completed) == 4

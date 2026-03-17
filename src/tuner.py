"""Optuna-based hyperparameter tuning for BracketNet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from .dataset import build_training_data
from .evaluate import evaluate_model
from .train import train_model

logger = logging.getLogger(__name__)


def suggest_config(trial: optuna.Trial) -> dict:
    """Sample a hyperparameter config from the trial."""
    n_layers = trial.suggest_int("n_layers", 2, 4)

    # Always suggest all 4 sizes; slice to n_layers for hidden_dims
    layer_0 = trial.suggest_categorical("layer_0_size", [64, 128, 256])
    layer_1 = trial.suggest_categorical("layer_1_size", [32, 64, 128])
    layer_2 = trial.suggest_categorical("layer_2_size", [16, 32, 64])
    layer_3 = trial.suggest_categorical("layer_3_size", [16, 32])
    hidden_dims = tuple([layer_0, layer_1, layer_2, layer_3][:n_layers])

    return {
        "hidden_dims": hidden_dims,
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "scheduler_factor": trial.suggest_float("scheduler_factor", 0.2, 0.8),
        "scheduler_patience": trial.suggest_int("scheduler_patience", 3, 15),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 10, 40),
    }


def suggest_config_from_params(params: dict) -> dict:
    """Reconstruct a config dict from Optuna's best_params flat dict."""
    n_layers = params["n_layers"]
    all_sizes = [params[f"layer_{i}_size"] for i in range(4)]
    hidden_dims = tuple(all_sizes[:n_layers])
    return {
        "hidden_dims": hidden_dims,
        "dropout": params["dropout"],
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "scheduler_factor": params["scheduler_factor"],
        "scheduler_patience": params["scheduler_patience"],
        "batch_size": params["batch_size"],
        "early_stop_patience": params["early_stop_patience"],
    }


def _make_objective(train_ds, val_ds, device: str, epochs: int):
    """Return an Optuna objective function closed over the datasets."""

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial)

        def epoch_callback(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model, _ = train_model(
            train_ds,
            val_ds,
            epochs=epochs,
            batch_size=config["batch_size"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            patience=config["early_stop_patience"],
            hidden_dims=config["hidden_dims"],
            dropout=config["dropout"],
            scheduler_factor=config["scheduler_factor"],
            scheduler_patience=config["scheduler_patience"],
            epoch_callback=epoch_callback,
            verbose=False,
            device=device,
        )

        results = evaluate_model(model, val_ds, device=device)
        return results["log_loss"]

    return objective


def run_tuning(
    data: dict,
    train_seasons: list[int],
    val_seasons: list[int],
    n_trials: int = 50,
    storage_path: str = "outputs/tuning/optuna.db",
    study_name: str = "bracketnet",
    epochs_per_trial: int = 150,
    output_config_path: str = "outputs/tuning/best_config.json",
    device: str = "cpu",
) -> dict:
    """Run Optuna hyperparameter search and save the best config.

    Supports resuming: if the SQLite study already exists, existing trials
    are loaded and the search continues from where it left off.

    Returns:
        Best config dict (also written to output_config_path as JSON).
    """
    storage_path = Path(storage_path)
    storage_path.parent.mkdir(parents=True, exist_ok=True)

    print("Building datasets (shared across all trials)...")
    train_ds, val_ds, _ = build_training_data(data, train_seasons, val_seasons)
    print(f"  Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_url = f"sqlite:///{storage_path.resolve()}"
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=30, interval_steps=5)
    sampler = TPESampler(seed=42)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Study '{study_name}': {completed} completed trials so far, running {n_trials} more.")
    print(f"Storage: {storage_path}")

    objective = _make_objective(train_ds, val_ds, device=device, epochs=epochs_per_trial)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_config = suggest_config_from_params(study.best_params)
    best_config["_val_log_loss"] = study.best_value
    best_config["_trial_number"] = study.best_trial.number

    # JSON requires lists, not tuples
    save_config = dict(best_config)
    save_config["hidden_dims"] = list(save_config["hidden_dims"])

    output_path = Path(output_config_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(save_config, f, indent=2)

    print(f"\nBest trial #{study.best_trial.number} | val log loss: {study.best_value:.4f}")
    print(f"Best config: {best_config}")
    print(f"Saved to: {output_path}")

    return best_config

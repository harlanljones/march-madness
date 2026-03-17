"""CLI entry point for March Madness bracket prediction."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_all, build_season_team_stats
from src.dataset import build_training_data
from src.train import train_model, train_ensemble, save_model, load_model
from src.evaluate import evaluate_model, plot_calibration
from src.bracket import simulate_bracket, print_bracket, save_bracket
from src.features import N_FEATURES


def _load_tuned_config(config_path: str) -> dict:
    """Load a best_config.json produced by the tune command."""
    with open(config_path) as f:
        raw = json.load(f)
    if "hidden_dims" in raw:
        raw["hidden_dims"] = tuple(raw["hidden_dims"])
    # Strip internal metadata keys (prefixed with _)
    config = {k: v for k, v in raw.items() if not k.startswith("_")}
    # Map tuner key name to train_model param name
    if "early_stop_patience" in config:
        config["patience"] = config.pop("early_stop_patience")
    return config


def cmd_train(args):
    """Train the model."""
    print("Loading data...")
    data = load_all(args.data_dir)

    train_seasons = list(range(2003, args.train_end + 1))
    val_seasons = list(range(args.val_start, args.val_end + 1))
    print(f"Training seasons: {train_seasons[0]}-{train_seasons[-1]}")
    print(f"Validation seasons: {val_seasons[0]}-{val_seasons[-1]}")

    print("Building datasets...")
    train_ds, val_ds, scaler = build_training_data(data, train_seasons, val_seasons)
    print(f"Training samples: {len(train_ds)} (with augmentation)")
    print(f"Validation samples: {len(val_ds)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tuned = {}
    if args.config:
        tuned = _load_tuned_config(args.config)
        print(f"Loaded tuned config from {args.config}: {tuned}")

    base_metadata = {
        "train_end": args.train_end,
        "val_start": args.val_start,
        "hidden_dims": list(tuned.get("hidden_dims", (128, 64, 32))),
        "dropout": tuned.get("dropout", 0.3),
    }

    # Tuned values override CLI defaults
    train_kwargs = dict(device=device, epochs=args.epochs, patience=args.patience)
    train_kwargs.update(tuned)

    if args.ensemble > 1:
        models, history = train_ensemble(
            train_ds, val_ds,
            n_models=args.ensemble,
            **train_kwargs,
        )
        # Save each model
        for i, model in enumerate(models):
            save_model(
                model,
                f"{args.output_dir}/ensemble_{i}.pt",
                scaler=scaler,
                metadata=base_metadata,
            )
        # Save first model as "best" too
        save_model(models[0], f"{args.output_dir}/best.pt", scaler=scaler,
                    metadata={**base_metadata, "ensemble_size": args.ensemble})
    else:
        model, history = train_model(train_ds, val_ds, **train_kwargs)
        save_model(model, f"{args.output_dir}/best.pt", scaler=scaler, metadata=base_metadata)

    # Evaluate on validation set
    print("\n--- Validation Results ---")
    if args.ensemble > 1:
        results = evaluate_model(models[0], val_ds, device=device)
    else:
        results = evaluate_model(model, val_ds, device=device)
    print(f"Log Loss: {results['log_loss']:.4f}")
    print(f"Accuracy: {results['accuracy']:.1%}")

    # Calibration plot
    plot_calibration(results["labels"], results["predictions"],
                     save_path=f"{args.output_dir}/calibration.png")


def cmd_evaluate(args):
    """Evaluate a trained model."""
    print("Loading model...")
    model, checkpoint = load_model(args.model)

    print("Loading data...")
    data = load_all(args.data_dir)

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.n_features_in_ = len(scaler.mean_)

    seasons = list(range(args.season_start, args.season_end + 1))
    print(f"Evaluating on seasons: {seasons}")

    from src.features import build_tournament_matchups
    features, labels = build_tournament_matchups(data, seasons)
    features = scaler.transform(features)

    from src.dataset import MarchMadnessDataset
    eval_ds = MarchMadnessDataset(features, labels)

    results = evaluate_model(model, eval_ds)
    print(f"\nLog Loss:  {results['log_loss']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.1%}")
    print(f"Games:     {len(labels)}")

    if args.calibration:
        plot_calibration(results["labels"], results["predictions"],
                         save_path=args.calibration)


def cmd_tune(args):
    """Run Optuna hyperparameter search."""
    from src.tuner import run_tuning

    print("Loading data...")
    data = load_all(args.data_dir)

    train_seasons = list(range(2003, args.train_end + 1))
    val_seasons = list(range(args.val_start, args.val_end + 1))
    print(f"Training seasons: {train_seasons[0]}-{train_seasons[-1]}")
    print(f"Validation seasons: {val_seasons[0]}-{val_seasons[-1]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    run_tuning(
        data=data,
        train_seasons=train_seasons,
        val_seasons=val_seasons,
        n_trials=args.trials,
        storage_path=args.storage,
        study_name=args.study_name,
        epochs_per_trial=args.epochs,
        output_config_path=args.output_config,
        device=device,
    )


def cmd_bracket(args):
    """Generate a tournament bracket."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model... (device: {device})")
    model, checkpoint = load_model(args.model, device=device)

    # Check for ensemble
    models = [model]
    model_dir = Path(args.model).parent
    ensemble_files = sorted(model_dir.glob("ensemble_*.pt"))
    if ensemble_files:
        print(f"Found {len(ensemble_files)} ensemble models")
        models = []
        for ef in ensemble_files:
            m, _ = load_model(str(ef), device=device)
            models.append(m)

    print("Loading data...")
    data = load_all(args.data_dir)

    # Reconstruct scaler
    scaler = StandardScaler()
    scaler.mean_ = np.array(checkpoint["scaler_mean"])
    scaler.scale_ = np.array(checkpoint["scaler_scale"])
    scaler.n_features_in_ = len(scaler.mean_)

    print(f"Generating {args.method} bracket for {args.season}...")
    bracket = simulate_bracket(
        models if len(models) > 1 else models[0],
        data,
        args.season,
        scaler,
        method=args.method,
        n_simulations=args.simulations,
        device=device,
    )

    print_bracket(bracket)

    # Save
    output_path = f"outputs/brackets/bracket_{args.season}_{args.method}.json"
    save_bracket(bracket, output_path)


def main():
    parser = argparse.ArgumentParser(description="March Madness Bracket Prediction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train the model")
    p_train.add_argument("--data-dir", default="data/raw", help="Path to Kaggle CSVs")
    p_train.add_argument("--train-end", type=int, default=2023, help="Last training season")
    p_train.add_argument("--val-start", type=int, default=2024, help="First validation season")
    p_train.add_argument("--val-end", type=int, default=2025, help="Last validation season")
    p_train.add_argument("--epochs", type=int, default=300)
    p_train.add_argument("--patience", type=int, default=20)
    p_train.add_argument("--ensemble", type=int, default=1, help="Number of ensemble models")
    p_train.add_argument("--output-dir", default="outputs/models")
    p_train.add_argument("--config", default=None, help="Path to best_config.json from tune command")

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    p_eval.add_argument("--model", required=True, help="Path to .pt checkpoint")
    p_eval.add_argument("--data-dir", default="data/raw")
    p_eval.add_argument("--season-start", type=int, default=2025)
    p_eval.add_argument("--season-end", type=int, default=2025)
    p_eval.add_argument("--calibration", default=None, help="Save calibration plot path")

    # Bracket
    p_bracket = subparsers.add_parser("bracket", help="Generate bracket")
    p_bracket.add_argument("--model", required=True, help="Path to .pt checkpoint")
    p_bracket.add_argument("--data-dir", default="data/raw")
    p_bracket.add_argument("--season", type=int, default=2026)
    p_bracket.add_argument("--method", choices=["deterministic", "probability", "monte_carlo"],
                           default="deterministic")
    p_bracket.add_argument("--simulations", type=int, default=10000, help="Monte Carlo runs")

    # Tune
    p_tune = subparsers.add_parser("tune", help="Hyperparameter search with Optuna")
    p_tune.add_argument("--data-dir", default="data/raw")
    p_tune.add_argument("--train-end", type=int, default=2023, help="Last training season")
    p_tune.add_argument("--val-start", type=int, default=2024, help="First validation season")
    p_tune.add_argument("--val-end", type=int, default=2025, help="Last validation season")
    p_tune.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    p_tune.add_argument("--epochs", type=int, default=150, help="Max epochs per trial")
    p_tune.add_argument("--storage", default="outputs/tuning/optuna.db",
                        help="SQLite path for study persistence (supports resume)")
    p_tune.add_argument("--study-name", default="bracketnet")
    p_tune.add_argument("--output-config", default="outputs/tuning/best_config.json",
                        help="Where to save the best config JSON")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "bracket":
        cmd_bracket(args)
    elif args.command == "tune":
        cmd_tune(args)


if __name__ == "__main__":
    main()

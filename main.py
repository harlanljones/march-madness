"""CLI entry point for March Madness bracket prediction."""

import argparse
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

    if args.ensemble > 1:
        models, history = train_ensemble(
            train_ds, val_ds,
            n_models=args.ensemble,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
        )
        # Save each model
        for i, model in enumerate(models):
            save_model(
                model,
                f"{args.output_dir}/ensemble_{i}.pt",
                scaler=scaler,
                metadata={"train_end": args.train_end, "val_start": args.val_start},
            )
        # Save first model as "best" too
        save_model(models[0], f"{args.output_dir}/best.pt", scaler=scaler,
                    metadata={"train_end": args.train_end, "val_start": args.val_start, "ensemble_size": args.ensemble})
    else:
        model, history = train_model(train_ds, val_ds, device=device, epochs=args.epochs, patience=args.patience)
        save_model(model, f"{args.output_dir}/best.pt", scaler=scaler,
                    metadata={"train_end": args.train_end, "val_start": args.val_start})

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

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "bracket":
        cmd_bracket(args)


if __name__ == "__main__":
    main()

"""Evaluation metrics: log loss, accuracy, calibration, bracket scoring."""

import numpy as np
import torch
from sklearn.metrics import log_loss, accuracy_score

from .model import BracketNet
from .dataset import MarchMadnessDataset


def evaluate_model(
    model: BracketNet,
    dataset: MarchMadnessDataset,
    device: str = "cpu",
) -> dict:
    """Compute evaluation metrics on a dataset.

    Returns:
        Dict with log_loss, accuracy, and predictions.
    """
    model.eval()
    with torch.no_grad():
        features = dataset.features.to(device)
        labels = dataset.labels.numpy()
        preds = model(features).cpu().numpy()

    # Clip predictions to avoid log(0)
    preds_clipped = np.clip(preds, 1e-7, 1 - 1e-7)

    logloss = log_loss(labels, preds_clipped)
    binary_preds = (preds >= 0.5).astype(float)
    acc = accuracy_score(labels, binary_preds)

    return {
        "log_loss": logloss,
        "accuracy": acc,
        "predictions": preds,
        "labels": labels,
    }


def calibration_data(labels: np.ndarray, preds: np.ndarray, n_bins: int = 10) -> dict:
    """Compute calibration curve data (binned predicted prob vs actual win rate)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_actuals = []
    bin_counts = []

    for i in range(n_bins):
        mask = (preds >= bin_edges[i]) & (preds < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = mask | (preds == bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_actuals.append(labels[mask].mean())
            bin_counts.append(mask.sum())

    return {
        "bin_centers": bin_centers,
        "bin_actuals": bin_actuals,
        "bin_counts": bin_counts,
    }


def plot_calibration(labels: np.ndarray, preds: np.ndarray, save_path: str | None = None):
    """Plot calibration curve."""
    import matplotlib.pyplot as plt

    cal = calibration_data(labels, preds)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.scatter(cal["bin_centers"], cal["bin_actuals"], s=50, zorder=5)
    ax.plot(cal["bin_centers"], cal["bin_actuals"], "o-", label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration Plot")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Calibration plot saved to {save_path}")
    else:
        plt.show()
    plt.close(fig)


def espn_bracket_score(predicted_winners: list[int], actual_winners: list[int]) -> int:
    """Score a bracket using ESPN-style scoring.

    Points per round: 10, 20, 40, 80, 160, 320
    63 games total: 32 + 16 + 8 + 4 + 2 + 1
    """
    round_points = [10, 20, 40, 80, 160, 320]
    round_sizes = [32, 16, 8, 4, 2, 1]

    total = 0
    idx = 0
    for rnd, (size, pts) in enumerate(zip(round_sizes, round_points)):
        for i in range(size):
            if idx < len(predicted_winners) and idx < len(actual_winners):
                if predicted_winners[idx] == actual_winners[idx]:
                    total += pts
            idx += 1

    return total

"""BracketNet: feedforward neural network for game outcome prediction."""

import torch
import torch.nn as nn

from .features import N_FEATURES


class BracketNet(nn.Module):
    """3-layer feedforward network with BatchNorm and Dropout.

    Architecture:
        Input(N_FEATURES) -> 128 -> 64 -> 32 -> 1 (sigmoid)
    """

    def __init__(self, input_dim: int = N_FEATURES, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

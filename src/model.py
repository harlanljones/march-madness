"""BracketNet: feedforward neural network for game outcome prediction."""

import torch
import torch.nn as nn

from .features import N_FEATURES


class BracketNet(nn.Module):
    """Feedforward network with BatchNorm and Dropout.

    Architecture:
        Input(N_FEATURES) -> hidden_dims[0] -> ... -> hidden_dims[-1] -> 1 (sigmoid)
    """

    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_size = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_size = h
        layers += [nn.Linear(in_size, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

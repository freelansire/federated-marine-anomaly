from __future__ import annotations
import torch
import torch.nn as nn

class RolaLitePredictor(nn.Module):
    """
    Lightweight predictor: windowed multivariate input -> next-step prediction.
    Anomaly score: prediction error.
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

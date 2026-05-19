from __future__ import annotations

import torch
from torch import nn


class WeatherLSTM(nn.Module):
    """LSTM architecture for daily weather time-series forecasting."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

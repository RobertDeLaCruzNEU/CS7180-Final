"""
LSTM for ego vehicle speed regression.

Input:  (batch, seq_len, 5)  — normalised [x1, y1, x2, y2, crossing_intention]
Output: (batch,)             — predicted normalised speed in [0, 1]

Multiply output by MAX_SPEED_KMH (60.0) to recover km/h.
"""

import torch
import torch.nn as nn


class SpeedLSTM(nn.Module):
    """
    LSTM encoder for scalar ego speed regression.

    Architecture mirrors IntentionLSTM but replaces the classification head
    with a single scalar output + Sigmoid to keep predictions in [0, 1],
    matching the normalised obd_speed target.

    Architecture:
        LSTM encoder -> final hidden state -> MLP head -> Sigmoid -> scalar
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, seq_len, input_dim)
        Returns:
            speed: (batch,) — predicted normalised speed in [0, 1]
        """
        _, (h_n, _) = self.lstm(features)   # h_n: (num_layers, B, hidden_dim)
        x = h_n[-1]                          # (B, hidden_dim) — last layer's final hidden
        return self.head(x).squeeze(-1)      # (B,)

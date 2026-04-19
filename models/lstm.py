"""
LSTM baseline for pedestrian crossing-intention estimation.

Input:  (batch, seq_len, input_dim)  — normalised bbox sequences
Output: (batch, 2)                   — logits for [not-crossing, crossing]
"""

import torch
import torch.nn as nn


class IntentionLSTM(nn.Module):
    """
    LSTM encoder baseline for binary intention classification.

    Architecture:
        LSTM encoder -> final hidden state -> MLP head
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
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
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: (batch, seq_len, input_dim)
        Returns:
            logits: (batch, num_classes)
        """
        _, (h_n, _) = self.lstm(bbox)   # h_n: (num_layers, B, hidden_dim)
        x = h_n[-1]                     # (B, hidden_dim) — last layer's final hidden
        return self.head(x)             # (B, num_classes)

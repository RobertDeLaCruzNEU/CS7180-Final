"""
Ego vehicle speed predictor.

Chains IntentionLSTM → SpeedLSTM:
  1. Runs IntentionLSTM on the bbox sequence to get a crossing probability
  2. Broadcasts that probability as a 5th channel alongside the bbox sequence
  3. Runs SpeedLSTM on the (seq_len, 5) feature sequence → normalised speed
  4. Returns speed in km/h

Input:  np.ndarray — normalised bbox sequence (seq_len, 4) in [x1, y1, x2, y2] format
Output: tuple(float, float) — (predicted ego speed in km/h, crossing probability in [0, 1])
"""

from __future__ import annotations

import numpy as np
import torch

from models.lstm import IntentionLSTM
from models.speed_lstm import SpeedLSTM
from data.dataset import MAX_SPEED_KMH


class SpeedPredictor:
    """
    Loads IntentionLSTM and SpeedLSTM and chains them for inference.

    IntentionLSTM predicts whether the pedestrian will cross (binary),
    which is then used as an additional input feature to SpeedLSTM.

    Usage:
        predictor = SpeedPredictor(
            intention_weights="weights/intention_lstm_best.pt",
            speed_weights="weights/speed_lstm_best.pt",
            device="cpu",
        )
        speed_kmh, crossing_prob = predictor.predict(bbox_sequence)  # (seq_len, 4) → (float, float)
    """

    def __init__(
        self,
        intention_weights: str = "weights/intention_lstm_best.pt",
        speed_weights: str = "weights/speed_lstm_best.pt",
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)

        self.intention_model = IntentionLSTM(
            input_dim=4,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            num_classes=2,
        )
        self.intention_model.load_state_dict(
            torch.load(intention_weights, map_location=self.device, weights_only=True)
        )
        self.intention_model.to(self.device).eval()

        self.speed_model = SpeedLSTM(
            input_dim=5,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
        )
        self.speed_model.load_state_dict(
            torch.load(speed_weights, map_location=self.device, weights_only=True)
        )
        self.speed_model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, bbox_sequence: np.ndarray) -> tuple[float, float]:
        """
        Predict ego vehicle speed and crossing intention from a normalised bbox sequence.

        Args:
            bbox_sequence: np.ndarray of shape (seq_len, 4),
                           normalised [x1, y1, x2, y2] in [0, 1]
        Returns:
            speed_kmh:     predicted ego speed in km/h
            crossing_prob: probability that the pedestrian will cross, in [0, 1]
        """
        seq = torch.from_numpy(bbox_sequence.astype(np.float32)).unsqueeze(0).to(self.device)
        # (1, seq_len, 4)

        # Step 1 — crossing intention probability
        logits = self.intention_model(seq)                           # (1, 2)
        crossing_prob = torch.softmax(logits, dim=-1)[0, 1].item()  # p(crossing)

        # Step 2 — broadcast intention across timesteps → (1, seq_len, 5)
        seq_len = seq.shape[1]
        intent_col = torch.full((1, seq_len, 1), crossing_prob, device=self.device)
        features = torch.cat([seq, intent_col], dim=-1)              # (1, seq_len, 5)

        # Step 3 — predict normalised speed → convert to km/h
        norm_speed = self.speed_model(features).item()               # scalar in [0, 1]
        return norm_speed * MAX_SPEED_KMH, crossing_prob

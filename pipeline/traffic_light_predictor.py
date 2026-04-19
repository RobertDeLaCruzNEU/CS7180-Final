"""
Traffic light state classifier.

Takes a cropped traffic light image and returns the predicted state:
red, yellow, or green. Uses a color-heuristic approach by default
(no model weights required), with an optional model-based override.

Input:  np.ndarray — cropped traffic light region (H, W, 3)
Output: np.ndarray — state probabilities [p_red, p_yellow, p_green]
"""

from __future__ import annotations

import numpy as np


class TrafficLightPredictor:
    """
    Predicts traffic light state from a cropped image region.

    Uses HSV color heuristics by default (top = red, middle = yellow,
    bottom = green). Can be swapped for a model-based approach.

    Usage:
        predictor = TrafficLightPredictor()
        probs = predictor.predict(crop)  # [p_red, p_yellow, p_green]
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def predict(self, crop: np.ndarray) -> np.ndarray:
        """
        Predict traffic light state probabilities for a single crop.

        Args:
            crop: np.ndarray of shape (H, W, 3) in BGR
        Returns:
            probs: np.ndarray of shape (3,) — [p_red, p_yellow, p_green]
        """
        raise NotImplementedError

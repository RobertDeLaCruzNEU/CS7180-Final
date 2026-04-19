"""
Ego vehicle speed predictor.

Takes a sequence of per-frame feature vectors (bounding box, gesture
probabilities, traffic light state, occlusion, etc.) and predicts the
appropriate ego vehicle speed via a trained LSTM.

Input:  np.ndarray — feature sequence (seq_len, input_dim)
Output: float      — predicted ego speed (normalised to [0, 1])
"""

from __future__ import annotations

import numpy as np


class SpeedPredictor:
    """
    Loads a trained LSTM and predicts ego speed from a feature sequence.

    The feature vector per timestep is expected to contain:
        [x, y, w, h, occlusion, gesture_probs..., tl_probs...]

    Usage:
        predictor = SpeedPredictor(weights_path="weights/lstm.pt", device="cpu")
        speed = predictor.predict(feature_sequence)  # scalar in [0, 1]
    """

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self.model = None  # loaded lazily or in __init__ body

    def predict(self, feature_sequence: np.ndarray) -> float:
        """
        Predict ego speed for a sequence of feature vectors.

        Args:
            feature_sequence: np.ndarray of shape (seq_len, input_dim)
        Returns:
            speed: float in [0, 1], where 0 = stopped, 1 = full speed
        """
        raise NotImplementedError

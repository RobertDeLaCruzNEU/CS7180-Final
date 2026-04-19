"""
Pedestrian gesture classifier.

Takes a cropped pedestrian image and returns a probability distribution
over gesture classes (e.g. standing, walking, raising hand, etc.).

Input:  np.ndarray — cropped pedestrian region (128, 128, 3)
Output: np.ndarray — gesture class probabilities (num_gesture_classes,)
"""

from __future__ import annotations

import numpy as np


class GesturePredictor:
    """
    Loads a trained gesture CNN and runs inference on pedestrian crops.

    Usage:
        predictor = GesturePredictor(weights_path="weights/gesture_cnn.pt", device="cpu")
        probs = predictor.predict(crop)  # shape: (num_classes,)
    """

    def __init__(self, weights_path: str, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self.model = None  # loaded lazily or in __init__ body

    def predict(self, crop: np.ndarray) -> np.ndarray:
        """
        Predict gesture probabilities for a single pedestrian crop.

        Args:
            crop: np.ndarray of shape (128, 128, 3)
        Returns:
            probs: np.ndarray of shape (num_gesture_classes,)
        """
        raise NotImplementedError

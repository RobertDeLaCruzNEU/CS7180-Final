"""
End-to-end pipeline runner.

Orchestrates the full inference pipeline:
    frame → Detector → GesturePredictor + TrafficLightPredictor → SpeedPredictor → speed

Accepts a single frame (np.ndarray) or a video file path (str) and
returns the predicted ego vehicle speed for each frame.
"""

from __future__ import annotations

import numpy as np

from pipeline.detector import Detector, Detection
from pipeline.gesture_predictor import GesturePredictor
from pipeline.traffic_light_predictor import TrafficLightPredictor
from pipeline.speed_predictor import SpeedPredictor


class PipelineRunner:
    """
    Wires together all pipeline components and runs end-to-end inference.

    Usage:
        runner = PipelineRunner(
            detector=Detector(...),
            gesture_predictor=GesturePredictor(...),
            tl_predictor=TrafficLightPredictor(),
            speed_predictor=SpeedPredictor(...),
            seq_len=16,
        )
        speed = runner.run_frame(frame)
        speeds = runner.run_video("path/to/video.mp4")
    """

    def __init__(
        self,
        detector: Detector,
        gesture_predictor: GesturePredictor,
        tl_predictor: TrafficLightPredictor,
        speed_predictor: SpeedPredictor,
        seq_len: int = 16,
    ) -> None:
        self.detector = detector
        self.gesture_predictor = gesture_predictor
        self.tl_predictor = tl_predictor
        self.speed_predictor = speed_predictor
        self.seq_len = seq_len
        self._feature_buffer: list[np.ndarray] = []

    def run_frame(self, frame: np.ndarray) -> float | None:
        """
        Process a single frame and return a predicted speed if enough
        frames have been buffered to fill the LSTM sequence window.

        Args:
            frame: np.ndarray of shape (H, W, 3) in BGR
        Returns:
            Predicted ego speed in [0, 1], or None if buffer not yet full.
        """
        raise NotImplementedError

    def run_video(self, video_path: str) -> list[float]:
        """
        Process a full video file and return predicted speeds per frame.

        Args:
            video_path: path to a video file
        Returns:
            List of predicted ego speeds (one per frame, None until buffer fills)
        """
        raise NotImplementedError

"""
End-to-end pipeline runner.

Orchestrates the full inference pipeline:
    frame → BoundingBoxEngineering → primary pedestrian bbox → SpeedPredictor → speed km/h

The runner maintains a rolling buffer of normalised bbox vectors. Once the
buffer reaches seq_len frames, it passes the sequence to SpeedPredictor and
returns a predicted ego vehicle speed. Earlier frames return None.

BoundingBoxEngineering returns (x, y, w, h) already normalised by frame dims.
The runner converts to (x1, y1, x2, y2) to match LSTM training format.
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from pipeline.detector import BoundingBoxEngineering
from pipeline.speed_predictor import SpeedPredictor

# Class IDs encoded by BoundingBoxEngineering: 1 = pedestrian, -1 = traffic light
_PEDESTRIAN_CLS = 1
_CHANNELS = 8    # 3 RGB + 5 meta (x, y, w, h, cls)
_META_START = 3  # meta begins at channel index 3 in the 224x224x8 array


class PipelineRunner:
    """
    Wires BoundingBoxEngineering and SpeedPredictor into a frame-by-frame loop.

    Usage:
        runner = PipelineRunner(
            detector=BoundingBoxEngineering(),                  # default yolo26n.pt
            speed_predictor=SpeedPredictor(),
            seq_len=15,
        )
        speed = runner.run_frame(frame)          # float | None
        speeds = runner.run_video("video.mp4")  # list[float | None]
    """

    def __init__(
        self,
        detector: BoundingBoxEngineering,
        speed_predictor: SpeedPredictor,
        seq_len: int = 15,
    ) -> None:
        self.detector = detector
        self.speed_predictor = speed_predictor
        self.seq_len = seq_len
        self._buffer: deque[np.ndarray] = deque(maxlen=seq_len)

    def _primary_pedestrian_bbox(self, frame_data: np.ndarray) -> np.ndarray | None:
        """
        Extract the normalised (x1, y1, x2, y2) bbox of the pedestrian closest
        to the image center from a single frame's output (10, 224, 224, 8).

        Meta channels at pixel (0, 0): [x, y, w, h, cls] — already normalised.
        Center proximity is used because the forward-facing camera places the
        vehicle's direct path at the frame center, making a centered pedestrian
        the highest-priority collision risk regardless of their distance.

        Returns None if no pedestrian is detected in this frame.
        """
        best_bbox = None
        best_dist = float("inf")

        for obj in frame_data:
            meta = obj[0, 0, _META_START:]         # [x, y, w, h, cls] normalised
            cls = meta[4]
            if cls != _PEDESTRIAN_CLS:
                continue
            x, y, w, h = meta[:4]
            if w == 0 and h == 0:                   # empty slot
                continue
            dist = (x - 0.5) ** 2 + (y - 0.5) ** 2  # squared distance to center
            if dist < best_dist:
                best_dist = dist
                # Convert (cx, cy, w, h) → (x1, y1, x2, y2)
                best_bbox = np.array(
                    [x - w / 2, y - h / 2, x + w / 2, y + h / 2],
                    dtype=np.float32,
                )

        return best_bbox

    def run_frame(self, frame: np.ndarray) -> float | None:
        """
        Process a single BGR frame and return a predicted speed.

        Returns None until seq_len frames have been buffered.

        Args:
            frame: np.ndarray of shape (H, W, 3) in BGR
        Returns:
            Predicted ego speed in km/h, or None if buffer not yet full.
        """
        # transform expects a list of arrays; returns (1, 10, 224, 224, 8)
        result = self.detector.transform([frame])   # (1, 10, 224, 224, 8)
        frame_data = result[0]                      # (10, 224, 224, 8)

        bbox = self._primary_pedestrian_bbox(frame_data)
        if bbox is not None:
            self._buffer.append(bbox)

        if len(self._buffer) < self.seq_len:
            return None

        bbox_seq = np.stack(list(self._buffer), axis=0)   # (seq_len, 4)
        speed_kmh, _ = self.speed_predictor.predict(bbox_seq)
        return speed_kmh

    def run_video(self, video_path: str) -> list[float | None]:
        """
        Process a full video file and return predicted speeds per frame.

        Args:
            video_path: path to a video file
        Returns:
            List of predicted ego speeds in km/h (None until buffer fills).
        """
        self._buffer.clear()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        speeds: list[float | None] = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                speeds.append(self.run_frame(frame))
        finally:
            cap.release()

        return speeds

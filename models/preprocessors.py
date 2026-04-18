from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ultralytics import YOLO
from PIL import Image

import pandas as pd
import numpy as np

class BoundingBoxEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.model = YOLO("yolo11n.pt")

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y = None, **fit_params):
        return super().fit_transform(X, y, **fit_params)
    
    def transform(self, X, y=None):
        """
        For each image, run the YOLO model to detect pedestrians and traffic 
        lights. For each detected object, crop the image to a square around 
        the bounding box, resize it to 224x224, and concatenate the bounding 
        box coordinates and class label to the pixel values. Return a list 
        of frames, where each frame is a list of cropped objects.

        Args:
            X: A list of file paths to images.
            y: Ignored.

        Returns:
            A list of frames (shape = (n, 10, 224, 224, 8)), where each frame is a 
            list of cropped objects. Each cropped object is a 224x224x8 array, 
            where the first 3 channels are the pixel values, and the last 5 
            channels are the normalized bounding box coordinates and class label.
        """
        results = self.model(X)
        frames = []
        for result in results:
            image = result.orig_img[:, :, ::-1]
            W, H =image.shape[1], image.shape[0]
            frame = np.zeros((10, 224, 224, 8), dtype=np.float32)
            xyxyl = result.boxes.data.cpu().numpy()
            i = 0
            for x1, y1, x2, y2, _, cls in xyxyl:
                if i >= 10:
                    break
                if cls not in [0, 9]:
                    continue
                cls = (1 if cls == 0 else -1)

                w, h = x2 - x1, y2 - y1
                x, y = x1 + w / 2, y1 + h / 2
                side = max(w, h)

                x1   = int(max(0, x - side / 2))
                y1   = int(max(0, y - side / 2))
                x2   = int(min(image.shape[1], x + side / 2))
                y2   = int(min(image.shape[0], y + side / 2))

                crop = image[y1:y2, x1:x2]
                crop = np.array(Image.fromarray(crop).resize((224, 224), Image.LANCZOS)).astype(np.float32) / 255.0
                meta = np.array([x, y, w, h, cls], dtype=np.float32) / np.array([W, H, W, H, 1], dtype=np.float32)
                frame[i] = np.concatenate([crop, np.tile(meta, (224, 224, 1))], axis=-1)
                i += 1
            frames.append(np.stack(frame, axis=0))

        return np.array(frames)
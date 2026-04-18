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

    @staticmethod
    def transform(self, X, y=None):
        """
        For each image, run the YOLO model to detect pedestrians and traffic 
        lights. For each detected object, crop the image to a square around 
        the bounding box, resize it to 224x224, and concatenate the bounding 
        box coordinates and class label to the pixel values. Return a list 
        of frames, where each frame is a list of cropped objects.

        X: A list of file paths to images.
        y: Ignored.
        """
        results = self.model(X)
        frames = []
        for result in results:
            image = result.orig_img[:, :, ::-1]
            frame = []
            xyxyl = result.boxes.data.cpu().numpy()
            for x1, y1, x2, y2, _, cls in xyxyl:
                if cls not in [0, 9]:
                    continue
                cls = (0 if cls == 0 else 1)

                w, h = x2 - x1, y2 - y1
                x, y = x1 + w / 2, y1 + h / 2
                side = max(w, h)

                x1   = int(max(0, x - side / 2))
                y1   = int(max(0, y - side / 2))
                x2   = int(min(image.shape[1], x + side / 2))
                y2   = int(min(image.shape[0], y + side / 2))

                crop = image[y1:y2, x1:x2]
                crop = np.array(Image.fromarray(crop).resize((224, 224), Image.LANCZOS))

                crop = np.concatenate([crop, np.tile([x, y, w, h, cls], (224, 224, 1))], axis=-1).astype(np.uint8)
                frame.append(crop)
            frames.append(np.stack(frame, axis=0))

        return frames
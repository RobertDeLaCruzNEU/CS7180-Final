from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing import List, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ConvolutionalNeuralNetwork(nn.Module, BaseEstimator, MetaEstimatorMixin):
    def __init__(
            self,
            epoch: int = 10,
            learning_rate: float = 0.1,
            layers: List[torch.nn.Module] = [],
            criterion: str = 'CrossEntropyLoss',
            optimizer: str = 'Adam',
            batch_size: int = 100,
            minimum_change: float = 0.0,
            seed: int = 0
        ):
        super(ConvolutionalNeuralNetwork, self).__init__()

        if layers is None:
            raise ValueError("Need layers bro!")

        self.losses: List[float] = []
        self.epoch: int = epoch
        self.learning_rate: float = learning_rate
        self.layers: List[torch.nn.Module] = nn.Sequential(*layers)
        self.criterion: str = criterion
        self.optimizer: str = optimizer
        self.batch_size: int = batch_size
        self.minimum_change: float = minimum_change
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.seed: int = seed
        torch.manual_seed(self.seed)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.layers(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs  = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def fit_predict(self, X: np.ndarray, y: np.array = None) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def fit(self, X: np.ndarray, y: np.array) -> np.ndarray:
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)

        criterion = getattr(torch.nn, self.criterion)()
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate, weight_decay=0.001)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in tqdm(range(self.epoch)):
            loss: float = 0.0
            for X_batch, y_batch in dataloader:
                # Forward pass
                y_batch_pred = self.forward(X_batch)
                batch_loss = criterion(y_batch_pred, y_batch)
                loss += batch_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
            self.losses.append(loss / len(dataloader))
            if len(self.losses) > 1 and abs(self.losses[-2] - self.losses[-1]) <= self.minimum_change:
                break

        print("Loss: {}".format(self.losses[-1]))
        return self
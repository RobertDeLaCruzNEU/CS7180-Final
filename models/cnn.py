from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, MetaEstimatorMixin
from sklearn.utils.validation import check_array, check_is_fitted
from typing import List, Tuple, Callable
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)
    
class Permute(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        # Result: (N*5, 4, 128, 128)
        return x.permute(*self.shape)
    
class Slicer(nn.Module):
    def __init__(self, slice_tuple: Tuple[int, int]):
        super().__init__()
        self.slice_tuple = slice_tuple

    def forward(self, x):
        # Assumes x is (Batch, Channels, H, W)
        return x[self.slice_tuple]

class ConvolutionalNeuralNetwork(nn.Module, BaseEstimator, MetaEstimatorMixin):
    def __init__(
            self,
            epoch: int = 10,
            learning_rate: float = 0.1,
            layers: List[torch.nn.Module] = [],
            criterion: str = 'MSELoss',
            optimizer: str = 'Adam',
            batch_size: int = 100,
            minimum_change: float = 0.0,
            seed: int = 0,
            passthrough: slice = slice(0, 0)
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
        self.passthrough: slice = passthrough
        self.to(self.device)
        self.seed: int = seed
        torch.manual_seed(self.seed)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.layers(X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        Xnew = torch.tensor(X, dtype=torch.float32)
        dataloader = torch.utils.data.DataLoader(Xnew, batch_size=self.batch_size, shuffle=False)
        self.eval()
        outputs = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                logits = self.forward(batch)
                outputs.append(logits.cpu().numpy())
        predictions = np.round(np.concatenate(outputs, axis=0)).astype(int)
        if self.passthrough.stop - self.passthrough.start > 0:
            return np.concatenate([X[:,:,0,0,self.passthrough], predictions], axis=-1)
        return predictions

    def fit_predict(self, X: np.ndarray, y: np.array = None) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)
    
    def fit(self, X: np.ndarray, y: np.array) -> np.ndarray:
        y = torch.tensor(y, dtype=torch.float32)
        X = torch.tensor(X, dtype=torch.float32)

        criterion = getattr(torch.nn, self.criterion)()
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.learning_rate, weight_decay=0.001)

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create the progress bar object
        progress_bar = tqdm(range(self.epoch), desc="Training CNN")
        
        for _ in progress_bar:
            loss: float = 0.0
            for X_batch, y_batch in dataloader:
                # move batch to GPU
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                y_pred = self.forward(X_batch)
                batch_loss = criterion(
                    y_pred.reshape(-1, y_pred.shape[-1]), 
                    y_batch.reshape(-1, y_batch.shape[-1])
                )
                loss += batch_loss.item()

                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            avg_loss = loss / len(dataloader)
            self.losses.append(avg_loss)
            
            # Update the progress bar with the current loss
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if len(self.losses) > 1 and abs(self.losses[-2] - self.losses[-1]) <= self.minimum_change:
                break

        self.losses = pd.Series(self.losses)
        return self
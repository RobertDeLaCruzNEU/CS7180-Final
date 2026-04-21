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

class Output(nn.Module):
    def forward(self, x):
        out, _ = x
        return out[:, -1, :]

class MinMaxScalerLayer(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.register_buffer('min_val', torch.zeros(num_features))
        self.register_buffer('max_val', torch.ones(num_features))
        self.register_buffer('initialized', torch.tensor(False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check the flag instead of hasattr
        if not self.initialized:
            # Update buffers with the min and max of the first batch
            self.min_val.copy_(x.min(dim=0)[0])
            self.max_val.copy_(x.max(dim=0)[0])
            self.initialized.fill_(True)

        range_val = self.max_val - self.min_val + 1e-7
        return (x - self.min_val) / range_val
    
class RecurrentNeuralNetwork(nn.Module, BaseEstimator, MetaEstimatorMixin):
    def __init__(
            self,
            epoch: int = 10,
            learning_rate: float = 0.1,
            layers: List[torch.nn.Module] = None,
            criterion: str = 'MSELoss',
            optimizer: str = 'Adam',
            batch_size: int = 100,
            minimum_change: float = 0.0,
            seed: int = 0,
            passthrough: slice = slice(0, 0)
        ):
        super(RecurrentNeuralNetwork, self).__init__()

        if layers is None:
            raise ValueError("Provide a list of layers.")
            
        self.losses: List[float] = []
        self.epoch: int = epoch
        self.learning_rate: float = learning_rate
        self.layers = nn.Sequential(*layers)
        self.criterion: str = criterion
        self.optimizer: str = optimizer
        self.batch_size: int = batch_size
        self.minimum_change: float = minimum_change
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.passthrough: slice = passthrough
        self.to(self.device)
        self.seed: int = seed
        torch.manual_seed(self.seed)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
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
        
        # Remove rounding for continuous variables
        predictions = np.concatenate(outputs, axis=0)
        
        if self.passthrough.stop - self.passthrough.start > 0:
            return np.concatenate([X[:, :, 0, 0, self.passthrough], predictions], axis=-1)
        return predictions

    def fit(self, X: np.ndarray, y: np.array) -> 'RecurrentNeuralNetwork':
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        criterion_func = getattr(torch.nn, self.criterion)()
        optimizer_func = getattr(torch.optim, self.optimizer)(
            self.parameters(), lr=self.learning_rate, weight_decay=0.001
        )

        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        progress_bar = tqdm(range(self.epoch), desc="Training RNN")
        
        for _ in progress_bar:
            total_loss: float = 0.0
            for X_batch, y_batch in dataloader:
                # X_batch = X_batch.to(self.device)
                # y_batch = y_batch.to(self.device)
                
                y_pred = self.forward(X_batch)
                batch_loss = criterion_func(
                    y_pred.view(-1, y_pred.shape[-1]), 
                    y_batch.view(-1, y_batch.shape[-1])
                )
                
                optimizer_func.zero_grad()
                batch_loss.backward()
                optimizer_func.step()
                total_loss += batch_loss.item()

            avg_loss = total_loss / len(dataloader)
            self.losses.append(avg_loss)
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

            if len(self.losses) > 1 and abs(self.losses[-2] - self.losses[-1]) <= self.minimum_change:
                break

        self.losses = pd.Series(self.losses)
        return self
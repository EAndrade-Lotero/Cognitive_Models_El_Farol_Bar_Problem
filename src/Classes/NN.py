# simple_mlp.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report


class SimpleMLP:
    """
    A minimal MLP for classification.

    Methods
    -------
    fit(X, y)                : train the network
    evaluate(X, y)           : print F1 / precision / recall (like sklearn.classification_report)
    predict_proba(X) -> np.ndarray
    predict(X)        -> np.ndarray
    """
    def __init__(
                self,
                categories: List[str],
                input_size: int = 4,
                hidden_size: int = 16,
                lr: float = 1e-3,
                epochs: int = 100,
                batch_size: int = 32,
                verbose: bool = True,
                device: str | None = None,
                seed: int | None = 42,
            ) -> None:
        self.input_size = input_size
        self.categories = categories
        self.num_classes = len(categories)
        self.hidden_size = hidden_size
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # --- define the network ----------------------------
        self.model = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.num_classes)
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()


    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on the given data."""
        X = self._to_tensor(X, torch.float32)
        y = self._to_tensor(y, torch.long)
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                logits = self.model(xb)
                loss = self.criterion(logits, yb)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * xb.size(0)

            if self.verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == 1):
                avg_loss = epoch_loss / len(loader.dataset)
                print(f"[{epoch:03d}/{self.epochs}] loss = {avg_loss:.4f}")

        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> str:
        """Print and return a scikit‑learn‑style classification report."""
        y_pred = self.predict(X)
        report = classification_report(y, y_pred, digits=4)
        print(report)
        return report

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities of shape (n_samples, 4)."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = self._to_tensor(X, torch.float32)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels (ints 0‑3)."""
        probs = self.predict_proba(X)
        indices = probs.argmax(axis=1)
        labels = np.vectorize(lambda x: self.categories[x])(indices)
        return labels

    def _to_tensor(self, array: np.ndarray, dtype) -> torch.Tensor:
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        try:
            return torch.tensor(array, dtype=dtype, device=self.device)
        except Exception as e:
            print('===>', type(array))
            print(array)
            raise e



class SimpleCNN(SimpleMLP):
    def __init__(
                self, 
                categories: List[str],
                lr: float = 1e-3,
                epochs: int = 100,
                batch_size: int = 32,
                verbose: bool = True,
                device: str | None = None,
                seed: int | None = 42,
            ) -> None:
        super(SimpleCNN, self).__init__(
            categories=categories, 
            lr=lr, 
            epochs=epochs, 
            batch_size=batch_size, 
            verbose=verbose, 
            device=device, 
            seed=seed
        )

        self.num_classes = len(categories)
        
        # --- define the network ----------------------------
        self.model = nn.Sequential(
            # Convolutional feature extractor
            # Input: (batch, 1, 25, 12)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (16, 15, 6)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> (32, 7, 3)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1)),  # -> (64, 3, 3)
    
            # Classifier
            nn.Flatten(),                          # -> (batch, 64*3*3 = 576)
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)            # -> (batch, 4)
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()



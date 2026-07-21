import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def build_matched_mlp(n_qubits: int, n_layers: int) -> nn.Sequential:
    """Build a small MLP whose parameter count matches the quantum model.

    Quantum model has: n_layers * n_qubits (VQC) + n_qubits + 1 (Linear head) parameters.
    We target that same count with a single hidden layer of width h where:
        h * n_qubits + h + h + 1 ≈ quantum_params
    Solved: h * (n_qubits + 2) + 1 = quantum_params
    """
    quantum_params = n_layers * n_qubits + n_qubits + 1  # VQC weights + Linear(n_qubits,1)
    # h * (n_qubits + 1) + h + 1 = quantum_params
    # h * (n_qubits + 2) = quantum_params - 1
    h = max(1, (quantum_params - 1) // (n_qubits + 2))
    return nn.Sequential(
        nn.Linear(n_qubits, h),
        nn.ReLU(),
        nn.Linear(h, 1),
        nn.Sigmoid(),
    )


class ClassicalBaselineClassifier:
    """Classical MLP with parameter count matched to HybridQuantumClassifier.

    Uses identical PCA + MinMaxScaler preprocessing so both models see
    the same input distribution.
    """

    def __init__(self, n_qubits=4, n_layers=3):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.pca      = None
        self._scaler  = None
        self.model    = build_matched_mlp(n_qubits, n_layers)

    def _preprocess(self, X, fit=False):
        if fit:
            self.pca     = PCA(n_components=self.n_qubits)
            X_pca        = self.pca.fit_transform(X)
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_out        = self._scaler.fit_transform(X_pca)
        else:
            X_out = self._scaler.transform(self.pca.transform(X))
        return X_out.astype(np.float32)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, X_train, y_train, epochs=40, lr=0.05, batch_size=32):
        X_proc = self._preprocess(X_train, fit=True)
        X_t    = torch.tensor(X_proc, dtype=torch.float32)
        y_t    = torch.tensor(y_train, dtype=torch.float32)
        n      = X_t.shape[0]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.model.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                Xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                preds = self.model(Xb).squeeze(1)
                loss  = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= n
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    all_preds = self.model(X_t).squeeze(1)
                acc = ((all_preds >= 0.5).float() == y_t).float().mean().item()
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss:.4f}  train_acc={acc:.4f}")

    def predict(self, X_test):
        self.model.eval()
        X_t = torch.tensor(self._preprocess(X_test), dtype=torch.float32)
        with torch.no_grad():
            probs = self.model(X_t).squeeze(1).numpy()
        return (probs >= 0.5).astype(np.float32)

    def predict_proba(self, X_test):
        self.model.eval()
        X_t = torch.tensor(self._preprocess(X_test), dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_t).squeeze(1).numpy()

import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class HybridQuantumClassifier(nn.Module):
    """Hybrid PCA -> quantum encoding -> VQC -> linear -> sigmoid classifier.

    Architecture:
        raw features -> PCA(n_qubits) -> scale to [0,pi] ->
        angle encoding -> BasicEntanglerLayers ->
        expval(PauliZ) on ALL qubits -> nn.Linear(n_qubits, 1) -> sigmoid

    Measuring all qubits (not just qubit 0) greatly improves expressivity
    at no extra circuit cost compared to single-qubit measurement.
    """

    def __init__(self, n_qubits=4, n_layers=3, encoding_fn=None):
        super().__init__()
        from models.encodings import angle_encoding
        self.n_qubits    = n_qubits
        self.n_layers    = n_layers
        self.encoding_fn = encoding_fn if encoding_fn is not None else angle_encoding
        self.pca         = None
        self._scaler     = None

        dev = qml.device("default.qubit", wires=n_qubits)

        # VQC weights: (n_layers, n_qubits)
        self.vqc_weights = nn.Parameter(
            torch.rand(n_layers, n_qubits, dtype=torch.float64) * np.pi
        )
        # Classical head: combine n_qubits expectation values -> logit
        self.classifier = nn.Linear(n_qubits, 1)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def _qnode(inputs, weights):
            self.encoding_fn(inputs, wires=list(range(n_qubits)))
            qml.BasicEntanglerLayers(weights, wires=list(range(n_qubits)))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._qnode = _qnode

    def _preprocess(self, X, fit=False):
        """PCA -> MinMaxScale to [0, pi] -> float32.

        If pca is already set (e.g. shared PCA from federation server), fit=True
        is silently treated as transform-only so clients don't diverge.
        """
        if fit and self.pca is None:
            self.pca     = PCA(n_components=self.n_qubits)
            X_pca        = self.pca.fit_transform(X)
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_out        = self._scaler.fit_transform(X_pca)
        else:
            X_out = self._scaler.transform(self.pca.transform(X))
        return X_out.astype(np.float32)

    def _quantum_batch(self, x):
        """Run QNode for each sample; returns (batch, n_qubits) float32 tensor."""
        rows = [
            torch.stack(self._qnode(x[i].double(), self.vqc_weights))
            for i in range(x.shape[0])
        ]
        return torch.stack(rows).float()   # (batch, n_qubits)

    def forward(self, x):
        q_out = self._quantum_batch(x)                    # (batch, n_qubits)
        logit = self.classifier(q_out).squeeze(1)         # (batch,)
        return torch.sigmoid(logit)

    def fit(self, X_train, y_train, epochs=40, lr=0.05, batch_size=32):
        """Fit preprocessor, then train with mini-batches."""
        X_proc = self._preprocess(X_train, fit=True)
        X_t    = torch.tensor(X_proc, dtype=torch.float32)
        y_t    = torch.tensor(y_train, dtype=torch.float32)
        n      = X_t.shape[0]

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        self.train()
        for epoch in range(epochs):
            perm = torch.randperm(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                idx = perm[start: start + batch_size]
                Xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                preds = self(Xb)
                loss  = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= n
            if (epoch + 1) % 5 == 0:
                with torch.no_grad():
                    all_preds = self(X_t)
                acc = ((all_preds >= 0.5).float() == y_t).float().mean().item()
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss:.4f}  train_acc={acc:.4f}")

    def predict(self, X_test):
        self.eval()
        X_t = torch.tensor(self._preprocess(X_test), dtype=torch.float32)
        with torch.no_grad():
            probs = self(X_t).numpy()
        return (probs >= 0.5).astype(np.float32)

    def predict_proba(self, X_test):
        self.eval()
        X_t = torch.tensor(self._preprocess(X_test), dtype=torch.float32)
        with torch.no_grad():
            return self(X_t).numpy()

    def get_weights(self):
        """Return all parameters as list of numpy arrays (for Flower)."""
        return [p.detach().cpu().numpy().copy() for p in self.parameters()]

    def set_weights(self, weights):
        """Load parameters from list of numpy arrays (from Flower aggregation)."""
        with torch.no_grad():
            for param, w in zip(self.parameters(), weights):
                param.copy_(torch.tensor(w, dtype=param.dtype))

    def circuit_depth(self):
        """Return gate depth of one circuit evaluation at device (primitive gate) level."""
        x_sample = torch.zeros(self.n_qubits, dtype=torch.float64)
        specs = qml.specs(self._qnode, level="device")(x_sample, self.vqc_weights)
        return specs.resources.depth

    def circuit_num_gates(self):
        """Return total primitive gate count of one circuit evaluation."""
        x_sample = torch.zeros(self.n_qubits, dtype=torch.float64)
        specs = qml.specs(self._qnode, level="device")(x_sample, self.vqc_weights)
        return specs.resources.num_gates

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

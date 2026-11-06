"""Quantum and classical models for v2. All forward() calls take q-space inputs."""
from functools import partial

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from models import encodings
from models.quantum_classifier import HybridQuantumClassifier
from v2 import config

_EMBEDDINGS = {
    "angle": encodings.angle_encoding,
    "amplitude": encodings.amplitude_encoding,
    "iqp": encodings.iqp_encoding,
    # v1 "re-uploading": both embedding blocks run before the trainable layers.
    config.VARIANT: partial(encodings.reuploading_encoding, n_layers=2),
}


class InterleavedQuantumClassifier(HybridQuantumClassifier):
    """Data re-uploading proper: [embed, layer] repeated for each trainable layer."""

    def __init__(self, n_qubits, n_layers):
        super().__init__(n_qubits=n_qubits, n_layers=n_layers,
                         encoding_fn=partial(encodings.reuploading_encoding, n_layers=1))
        device = qml.device("default.qubit", wires=n_qubits)
        wires = list(range(n_qubits))

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            for layer in range(n_layers):
                self.encoding_fn(inputs, wires=wires)
                qml.BasicEntanglerLayers(weights[layer:layer + 1], wires=wires)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._qnode = circuit


class ClassicalClassifier(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, x):
        return torch.sigmoid(self.network(x)).squeeze(1)


def build_model(encoding, n_features=4, n_layers=2, init_seed=0):
    """Build a model with initial weights determined only by init_seed and shape."""
    torch.manual_seed(init_seed)
    if encoding == "mlp13":
        return ClassicalClassifier(nn.Sequential(
            nn.Linear(n_features, 2), nn.ReLU(), nn.Linear(2, 1)))
    if encoding == "logreg":
        return ClassicalClassifier(nn.Linear(n_features, 1))
    if encoding == "mlp_control":
        return ClassicalClassifier(nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)))
    if encoding == "reuploading":
        return InterleavedQuantumClassifier(n_features, n_layers)
    # Amplitude stores up to 16 values in 4 qubits; the others use one qubit per feature.
    n_qubits = 4 if encoding == "amplitude" else n_features
    return HybridQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers,
                                   encoding_fn=_EMBEDDINGS[encoding])


def flatten(model):
    return np.concatenate([p.detach().cpu().numpy().astype(np.float64).ravel()
                           for p in model.parameters()])


def load_flat(model, vector):
    position = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            values = torch.from_numpy(vector[position:position + size].copy())
            parameter.copy_(values.reshape(parameter.shape).to(parameter.dtype))
            position += size
    if position != len(vector):
        raise ValueError(f"Expected {position} weights, got {len(vector)}")


def predict_probabilities(model, X):
    """Model output for q-space inputs, as a float64 numpy array."""
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy().astype(np.float64)


def accuracy(model, X, y):
    return float(((predict_probabilities(model, X) >= 0.5) == y).mean())

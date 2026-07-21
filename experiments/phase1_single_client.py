"""Phase 1: centralized single-client training on Wisconsin Breast Cancer.

Run from project root:
    python experiments/phase1_single_client.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from models.quantum_classifier import HybridQuantumClassifier

N_QUBITS = 4
N_LAYERS = 2
EPOCHS   = 30
LR       = 0.05
SEED     = 42

np.random.seed(SEED)

print("=== Phase 1: Single-Client Quantum Classifier ===")
print(f"  n_qubits={N_QUBITS}, n_layers={N_LAYERS}, epochs={EPOCHS}, lr={LR}")

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

model = HybridQuantumClassifier(
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    encoding_fn=angle_encoding,
)
print(f"  Trainable parameters: {model.count_parameters()}")

print("\n--- Training ---")
model.fit(X_train, y_train, epochs=EPOCHS, lr=LR)

train_preds = model.predict(X_train)
test_preds  = model.predict(X_test)
train_acc   = (train_preds == y_train).mean()
test_acc    = (test_preds  == y_test).mean()

print(f"\n=== Phase 1 Results ===")
print(f"  Train accuracy : {train_acc:.4f}")
print(f"  Test  accuracy : {test_acc:.4f}")

assert test_acc > 0.50, f"Test accuracy {test_acc:.4f} is not above 50% — something is wrong."
print("  [PASS] Test accuracy > 50%")

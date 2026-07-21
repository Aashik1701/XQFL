"""Phase 2: Classical baseline on Wisconsin Breast Cancer — same split as Phase 1.

Run from project root:
    python experiments/phase2_classical_baseline.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from models.quantum_classifier import HybridQuantumClassifier
from models.classical_baseline import ClassicalBaselineClassifier

N_QUBITS = 4
N_LAYERS = 2
EPOCHS   = 30
LR       = 0.05
SEED     = 42

np.random.seed(SEED)

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

# --- Quantum model (Phase 1 replay) ---
print("=== Quantum Model ===")
qmodel = HybridQuantumClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=angle_encoding)
print(f"  Trainable parameters: {qmodel.count_parameters()}")
qmodel.fit(X_train, y_train, epochs=EPOCHS, lr=LR)
q_train_acc = (qmodel.predict(X_train) == y_train).mean()
q_test_acc  = (qmodel.predict(X_test)  == y_test).mean()

# --- Classical baseline ---
print("\n=== Classical Baseline ===")
cmodel = ClassicalBaselineClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS)
print(f"  Trainable parameters: {cmodel.count_parameters()}")
cmodel.fit(X_train, y_train, epochs=EPOCHS, lr=LR)
c_train_acc = (cmodel.predict(X_train) == y_train).mean()
c_test_acc  = (cmodel.predict(X_test)  == y_test).mean()

# --- Side-by-side comparison ---
print("\n=== Phase 1 & 2 Results ===")
print(f"{'Model':<22} {'Params':>6}  {'Train Acc':>10}  {'Test Acc':>9}")
print("-" * 55)
print(f"{'Quantum (VQC)':<22} {qmodel.count_parameters():>6}  {q_train_acc:>10.4f}  {q_test_acc:>9.4f}")
print(f"{'Classical (MLP)':<22} {cmodel.count_parameters():>6}  {c_train_acc:>10.4f}  {c_test_acc:>9.4f}")

assert q_test_acc > 0.50, "Quantum model below 50% — check implementation."
assert c_test_acc > 0.50, "Classical model below 50% — check implementation."
print("\n[PASS] Both models above 50% test accuracy.")

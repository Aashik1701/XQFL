"""Phase 3: Compare all 4 quantum encodings on Wisconsin Breast Cancer.

Produces experiments/phase3_encoding_comparison.csv with columns:
    encoding, n_params, circuit_depth, num_gates,
    train_acc, test_acc, train_time_s

Run from project root:
    python experiments/phase3_encoding_comparison.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import functools
import numpy as np
import pandas as pd

from data.loaders import load_breast_cancer_data
from models.encodings import (
    angle_encoding,
    amplitude_encoding,
    reuploading_encoding,
    iqp_encoding,
)
from models.quantum_classifier import HybridQuantumClassifier

N_QUBITS = 4
N_LAYERS = 2
EPOCHS   = 30
LR       = 0.05
SEED     = 42

np.random.seed(SEED)

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

# reuploading_encoding's n_layers defaults to 2, matching the VQC ansatz depth.
# Bind explicitly via partial so the comparison table is self-documenting.
reuploading_2 = functools.partial(reuploading_encoding, n_layers=N_LAYERS)
reuploading_2.__name__ = "reuploading_encoding"

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_2,
    "iqp":         iqp_encoding,
}

results = []

for enc_name, enc_fn in ENCODINGS.items():
    print(f"\n{'='*55}")
    print(f"Encoding: {enc_name}")
    print(f"{'='*55}")

    model = HybridQuantumClassifier(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        encoding_fn=enc_fn,
    )

    depth     = model.circuit_depth()
    num_gates = model.circuit_num_gates()
    n_params  = model.count_parameters()
    print(f"  Circuit depth : {depth}")
    print(f"  Num gates     : {num_gates}")
    print(f"  Parameters    : {n_params}")

    t0 = time.time()
    model.fit(X_train, y_train, epochs=EPOCHS, lr=LR)
    elapsed = time.time() - t0

    train_acc = float((model.predict(X_train) == y_train).mean())
    test_acc  = float((model.predict(X_test)  == y_test).mean())

    print(f"  Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}  Time: {elapsed:.1f}s")

    results.append({
        "encoding":     enc_name,
        "n_params":     n_params,
        "circuit_depth": depth,
        "num_gates":    num_gates,
        "train_acc":    round(train_acc, 4),
        "test_acc":     round(test_acc, 4),
        "train_time_s": round(elapsed, 1),
    })

df = pd.DataFrame(results)
out_path = os.path.join(os.path.dirname(__file__), "phase3_encoding_comparison.csv")
df.to_csv(out_path, index=False)

print(f"\n{'='*55}")
print("Phase 3 Encoding Comparison — Wisconsin Breast Cancer")
print(f"  n_qubits={N_QUBITS}, n_layers={N_LAYERS}, epochs={EPOCHS}, seed={SEED}")
print(f"{'='*55}")
print(df.to_string(index=False))
print(f"\nSaved to {out_path}")

# Definition-of-done check: all encodings must train without error and converge
for _, row in df.iterrows():
    assert row["test_acc"] > 0.50, (
        f"{row['encoding']} test accuracy {row['test_acc']:.4f} is not above 50%"
    )
print("\n[PASS] All 4 encodings trained without error, test accuracy > 50%.")

# Note for amplitude circuit depth
print("\n[NOTE] amplitude circuit_depth reflects simulator-level cost (StatePrep = 1 gate).")
print("       Hardware depth for amplitude encoding is O(2^n_qubits) = O(16) for 4 qubits.")

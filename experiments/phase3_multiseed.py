"""Phase 3 (multi-seed): encoding comparison across seeds 1, 7, 42, 123.

Reports mean ± std test accuracy per encoding.
Appends results to experiments/phase3_multiseed_results.csv.

Run from project root:
    python experiments/phase3_multiseed.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time, functools
import numpy as np
import pandas as pd

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding, amplitude_encoding, reuploading_encoding, iqp_encoding
from models.quantum_classifier import HybridQuantumClassifier

N_QUBITS = 4
N_LAYERS = 2
EPOCHS   = 30
LR       = 0.05
SEEDS    = [1, 7, 42, 123]

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": functools.partial(reuploading_encoding, n_layers=N_LAYERS),
    "iqp":         iqp_encoding,
}

rows = []

for seed in SEEDS:
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)

    for enc_name, enc_fn in ENCODINGS.items():
        print(f"  seed={seed}  encoding={enc_name}", end="  ", flush=True)
        t0 = time.time()

        model = HybridQuantumClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=enc_fn)
        model.fit(X_train, y_train, epochs=EPOCHS, lr=LR)

        test_acc  = float((model.predict(X_test)  == y_test).mean())
        train_acc = float((model.predict(X_train) == y_train).mean())
        elapsed   = time.time() - t0

        print(f"test_acc={test_acc:.4f}  ({elapsed:.0f}s)")
        rows.append({"seed": seed, "encoding": enc_name,
                     "train_acc": round(train_acc, 4),
                     "test_acc":  round(test_acc,  4),
                     "train_time_s": round(elapsed, 1)})

df = pd.DataFrame(rows)
out_path = os.path.join(os.path.dirname(__file__), "phase3_multiseed_results.csv")
df.to_csv(out_path, index=False)
print(f"\nRaw results saved to {out_path}")

# Summary table: mean ± std per encoding
summary = (
    df.groupby("encoding")["test_acc"]
      .agg(["mean", "std"])
      .rename(columns={"mean": "mean_test_acc", "std": "std_test_acc"})
      .reindex(list(ENCODINGS.keys()))   # preserve encoding order
)
summary["mean±std"] = summary.apply(
    lambda r: f"{r['mean_test_acc']:.4f} ± {r['std_test_acc']:.4f}", axis=1
)

print("\n=== Phase 3 Multi-Seed Summary ===")
print(f"  Seeds: {SEEDS}  |  n_qubits={N_QUBITS}  n_layers={N_LAYERS}  epochs={EPOCHS}")
print(summary[["mean±std"]].to_string())

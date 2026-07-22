"""Phase 5: Federated learning with Flower simulation.

Runs 4-client IID federated training with FedAvg (15 rounds, angle encoding)
on the Wisconsin Breast Cancer dataset and reports per-round global accuracy.

Comparison baseline (Phase 1 centralised):
  train_acc = 80.88%   test_acc = 86.84%

Run from project root:
    python experiments/phase5_federation.py
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from federation.run_federation import run_federation

# ── reproducibility ─────────────────────────────────────────────────────────
SEED     = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── config ───────────────────────────────────────────────────────────────────
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2

# Phase 1 centralised baseline (reported from phase1_single_client.py, seed=42)
BASELINE_TRAIN = 0.8088
BASELINE_TEST  = 0.8684

# ── load data ────────────────────────────────────────────────────────────────
print("=== Phase 5: Federated Learning (Flower simulation) ===")
print(f"  Strategy       : FedAvg")
print(f"  Clients        : {N_CLIENTS}  (IID split)")
print(f"  Rounds         : {N_ROUNDS}")
print(f"  Local epochs   : {LOCAL_EPOCHS} per round")
print(f"  Encoding       : angle")
print(f"  Qubits/layers  : {N_QUBITS} / {N_LAYERS}")
print(f"  Phase 1 baseline (centralised): test={BASELINE_TEST:.4f}")
print()

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)
print(f"  Dataset: {len(X_train)} train, {len(X_test)} test")
print()

# ── partition summary ─────────────────────────────────────────────────────────
print("--- Data partitions (IID) ---")

# ── run federation ────────────────────────────────────────────────────────────
print("\n--- Federated training ---")
history, round_log = run_federation(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    encoding_fn=angle_encoding,
    n_clients=N_CLIENTS,
    n_rounds=N_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    split="iid",
    strategy_name="fedavg",
    seed=SEED,
    project_root=PROJECT_ROOT,
)

# ── results ───────────────────────────────────────────────────────────────────
print("\n=== Per-round global test accuracy ===")
print(f"  {'Round':>5}  {'Test Acc':>9}")
print(f"  {'-'*5}  {'-'*9}")
for rnd, acc in round_log:
    marker = " <-- final" if rnd == N_ROUNDS else ""
    print(f"  {rnd:>5}  {acc:>9.4f}{marker}")

if round_log:
    final_acc = round_log[-1][1]
    delta     = final_acc - BASELINE_TEST
    print(f"\n  Federated final test_acc : {final_acc:.4f}")
    print(f"  Centralised baseline     : {BASELINE_TEST:.4f}")
    print(f"  Gap (fed - centralised)  : {delta:+.4f}")

    if abs(delta) <= 0.05:
        print("\n  [OK] Federated accuracy within 5 pp of centralised baseline.")
    elif final_acc >= BASELINE_TEST:
        print("\n  [OK] Federated accuracy matches or exceeds centralised baseline.")
    else:
        print(f"\n  [NOTE] Federated accuracy {abs(delta):.2%} below centralised baseline.")
        print("         Expected for short runs; increase n_rounds or local_epochs.")

print("\n[Phase 5 complete]")

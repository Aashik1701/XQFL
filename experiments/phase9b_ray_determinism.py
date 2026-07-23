"""Item 3 — Ray aggregation determinism fix verification.

Before this fix:
    aggregate_inplace(results) processed results in Ray arrival order.
    FP non-associativity → different weights → ±2.63pp across identical reruns.

Fix applied in federation/server.py:
    DeterministicFedAvg.aggregate_fit sorts results by int(client_proxy.cid)
    before calling super(). All strategies inherit from DeterministicFedAvg.

This script reruns the three seeds that showed drift vs Phase 9a (42, 123, 999)
and compares the new results against Phase 9a values.  If sorting fixed the
non-determinism, each seed must reproduce Phase 9a within 0.5pp across runs.

As a further check: seed=42 is run 5× back-to-back.  If still non-deterministic
after the CID sort, within-seed spread is quantified.

Run from project root:
    python experiments/phase9b_ray_determinism.py
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

N_QUBITS     = 4
N_LAYERS     = 2
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3

# Phase 9a reference values (seeds 1, 42, 123, 999)
PHASE9A_REF = {
    1:   0.8421,
    7:   0.7982,
    42:  0.8421,
    123: 0.8421,
    999: 0.8421,
}

X_tr, X_te, y_tr, y_te = load_breast_cancer_data(random_state=42)

print("=" * 65)
print("=== Item 3: Ray aggregation determinism check ===")
print(f"  Fix: DeterministicFedAvg sorts results by int(cid) before aggregation")
print(f"  Test: rerun seeds 42, 123, 999 (previously showed ±2.63pp drift)\n")


# ── Pass 1: single rerun of seeds 42, 123, 999 ───────────────────────────────
SEEDS_TEST = [42, 123, 999]
rerun1 = {}

for seed in SEEDS_TEST:
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, round_log = run_federation(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        split="iid", strategy_name="fedavg", seed=seed,
        project_root=PROJECT_ROOT, n_permutations=0,
    )
    rerun1[seed] = round_log[-1][1] if round_log else float("nan")

print("  Pass 1 vs Phase 9a reference:")
print(f"  {'Seed':>5}  {'Phase9a':>8}  {'Rerun1':>8}  {'Δ':>8}  {'Status':>12}")
print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
all_match1 = True
for seed in SEEDS_TEST:
    ref = PHASE9A_REF[seed]
    r1  = rerun1[seed]
    delta = r1 - ref
    match = abs(delta) <= 0.005          # ≤ 0.5pp tolerance (= 1 sample at n=114)
    if not match:
        all_match1 = False
    status = "OK" if match else f"DRIFT {delta*100:+.2f}pp"
    print(f"  {seed:>5}  {ref:>8.4f}  {r1:>8.4f}  {delta*100:>+7.2f}pp  {status:>12}")

# ── Pass 2: rerun seeds 42, 123, 999 a second time ───────────────────────────
print(f"\n  Pass 2 (repeat) — checking within-fix stability:")
rerun2 = {}

for seed in SEEDS_TEST:
    np.random.seed(seed)
    torch.manual_seed(seed)
    _, round_log = run_federation(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        split="iid", strategy_name="fedavg", seed=seed,
        project_root=PROJECT_ROOT, n_permutations=0,
    )
    rerun2[seed] = round_log[-1][1] if round_log else float("nan")

print(f"  {'Seed':>5}  {'Rerun1':>8}  {'Rerun2':>8}  {'Δ':>8}  {'Status':>12}")
print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*12}")
all_match2 = True
for seed in SEEDS_TEST:
    r1 = rerun1[seed]
    r2 = rerun2[seed]
    delta = r2 - r1
    match = abs(delta) <= 0.005
    if not match:
        all_match2 = False
    status = "STABLE" if match else f"DRIFT {delta*100:+.2f}pp"
    print(f"  {seed:>5}  {r1:>8.4f}  {r2:>8.4f}  {delta*100:>+7.2f}pp  {status:>12}")

# ── seed=42 × 5 runs — within-seed spread ────────────────────────────────────
print(f"\n  seed=42 × 5 back-to-back runs (within-seed spread):")
seed42_runs = []
for run_i in range(5):
    np.random.seed(42)
    torch.manual_seed(42)
    _, round_log = run_federation(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        split="iid", strategy_name="fedavg", seed=42,
        project_root=PROJECT_ROOT, n_permutations=0,
    )
    acc = round_log[-1][1] if round_log else float("nan")
    seed42_runs.append(acc)
    print(f"    run {run_i+1}: {acc:.4f}")

s42_mean = float(np.mean(seed42_runs))
s42_std  = float(np.std(seed42_runs))
s42_rng  = float(np.max(seed42_runs) - np.min(seed42_runs))
print(f"  mean={s42_mean:.4f}  std={s42_std:.4f}  range={s42_rng:.4f} "
      f"({s42_rng*100:.2f}pp)")

# ── verdict ───────────────────────────────────────────────────────────────────
print(f"\n  --- Determinism verdict ---")
if s42_rng <= 0.005:
    print(f"  FIXED: seed=42 spread = {s42_rng*100:.2f}pp (≤ 0.5pp = 1 test sample).")
    print(f"  CID-sorting eliminated the Ray FP non-determinism.")
    print(f"  Federated results are now reproducible to within discrete-accuracy resolution.")
    print(f"  MN-02 Ray note should be updated: non-determinism is RESOLVED, not an")
    print(f"  ongoing concern for Phase 9b+ experiments.")
elif s42_rng <= 0.018:
    print(f"  PARTIALLY FIXED: seed=42 spread = {s42_rng*100:.2f}pp (reduced from 2.63pp).")
    print(f"  CID-sorting reduced but did not eliminate non-determinism.")
    print(f"  Residual variance may come from Ray actor scheduling or PennyLane/PyTorch")
    print(f"  non-determinism within a single round. Continue reporting ranges in Phase 9b.")
else:
    print(f"  NOT FIXED: seed=42 spread = {s42_rng*100:.2f}pp (similar to pre-fix 2.63pp).")
    print(f"  CID-sorting is insufficient. Additional source of non-determinism exists.")
    print(f"  Possible: PennyLane device, PyTorch parallel ops, or Ray worker reuse.")
    print(f"  Action: add torch.use_deterministic_algorithms(True) or run single-threaded.")

# ── Phase 9a recalibration ────────────────────────────────────────────────────
print(f"\n  Phase 9a recalibration (post-fix):")
print(f"  The following are the post-fix reference values for Wisconsin/angle/IID:")
print(f"  {'Seed':>5}  {'Phase9a':>8}  {'PostFix':>8}  {'Δ':>8}")
print(f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}")
for seed in SEEDS_TEST:
    ref    = PHASE9A_REF[seed]
    postfix = rerun1[seed]
    print(f"  {seed:>5}  {ref:>8.4f}  {postfix:>8.4f}  {(postfix-ref)*100:>+7.2f}pp")

print(f"\n[Item 3 determinism check complete]")

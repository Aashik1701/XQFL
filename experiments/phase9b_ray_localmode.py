"""Ray local_mode determinism test.

Prior result: DeterministicFedAvg (CID-sort) did not fix non-determinism.
Seed=42 x 5 back-to-back runs showed range=3.51pp.
Hypothesis: root cause is Ray VirtualClientEngineActorPool scheduling
(8 actors, 4 clients → non-deterministic order within each round).

Test: force sequential execution via ray.init(local_mode=True), which
runs all actors in a single thread with no parallelism.  If scheduling is
the cause, local_mode should collapse the within-seed spread to near-zero
(= discrete-accuracy resolution: 1 sample / 114 = 0.877pp).

Outcomes:
  A. spread ≤ 0.88pp (1 sample) → scheduling confirmed as root cause.
     Re-run Phase 9a 5-seed table under local_mode for authoritative values.
     Accept the speed cost; use local_mode for all Phase 9b experiments.
  B. spread > 0.88pp → additional source exists beyond scheduling.
     Protocol: report within-seed noise (5 reruns) alongside across-seed
     variance for every future table.  Treat any effect < 4pp as unconfirmed.

Run from project root:
    python experiments/phase9b_ray_localmode.py
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
SEED         = 42
N_RUNS       = 5

# Phase 9a reference (pre-fix) and post-CID-fix range
PHASE9A_REF   = 0.8421
POSTFIX_RANGE = 0.0351   # 3.51pp from CID-sort-only test

X_tr, X_te, y_tr, y_te = load_breast_cancer_data(random_state=42)

print("=" * 65)
print("=== Ray sequential-execution determinism test (seed=42 x 5) ===")
print(f"  local_mode=True removed in Ray 2.56.1; using num_cpus=1 instead.")
print(f"  num_cpus=1 → VCE creates 1-actor pool → clients execute sequentially.")
print(f"  Pre-fix range: {POSTFIX_RANGE*100:.2f}pp  Phase 9a ref: {PHASE9A_REF:.4f}\n")

accs = []
for run_i in range(N_RUNS):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    _, round_log = run_federation(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te,  y_test=y_te,
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        split="iid", strategy_name="fedavg", seed=SEED,
        project_root=PROJECT_ROOT, n_permutations=0,
        ray_local_mode=True,
    )
    acc = round_log[-1][1] if round_log else float("nan")
    accs.append(acc)
    print(f"  run {run_i + 1}: {acc:.4f}")

acc_arr  = np.array(accs)
mean_acc = float(acc_arr.mean())
std_acc  = float(acc_arr.std())
rng_acc  = float(acc_arr.max() - acc_arr.min())
one_sample = 1.0 / len(y_te)   # 0.877pp at n=114

print(f"\n  mean={mean_acc:.4f}  std={std_acc:.4f}  range={rng_acc:.4f} ({rng_acc*100:.2f}pp)")
print(f"  1-sample resolution = {one_sample*100:.2f}pp")

print(f"\n  --- Verdict ---")
if rng_acc <= one_sample + 1e-9:
    print(f"  FIXED: range={rng_acc*100:.2f}pp ≤ {one_sample*100:.2f}pp (1 sample).")
    print(f"  local_mode=True eliminates non-determinism to discrete-accuracy resolution.")
    print(f"  Root cause confirmed: Ray actor scheduling (not PennyLane or PyTorch ops).")
    print(f"")
    print(f"  ACTION A: Rerun the full Phase 9a 5-seed table under local_mode to get")
    print(f"  authoritative mean±std values uncontaminated by scheduling noise.")
    print(f"  The Phase 9a values may have inflated std due to ±3.5pp within-seed variance.")
    print(f"  Accept the speed cost (~{N_RUNS}x slower per run than parallel). Use local_mode")
    print(f"  for all Phase 9b experiments going forward.")
elif rng_acc <= 0.015:
    print(f"  PARTIALLY FIXED: range={rng_acc*100:.2f}pp (reduced from {POSTFIX_RANGE*100:.2f}pp).")
    print(f"  local_mode reduces but does not fully eliminate non-determinism.")
    print(f"  Residual variance from within-client PyTorch ops (Adam, mini-batch permutation).")
    print(f"  ACTION B: Report within-seed spread (5 reruns) alongside across-seed std.")
    print(f"  Treat effects < {rng_acc*100:.1f}pp as unconfirmed.")
else:
    print(f"  NOT FIXED: range={rng_acc*100:.2f}pp (similar to pre-fix {POSTFIX_RANGE*100:.2f}pp).")
    print(f"  local_mode=True does not help. Source is within-client training ops:")
    print(f"  PyTorch Adam with mini-batch shuffling (torch.randperm) is non-deterministic")
    print(f"  across runs even with the same seed when Ray actor state varies.")
    print(f"  ACTION B: Report within-seed spread alongside across-seed variance.")
    print(f"  Treat any single-cell accuracy as ±{rng_acc/2*100:.1f}pp (half-range) uncertain.")

print(f"\n[local_mode determinism test complete]")

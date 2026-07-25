"""Confound-breaking experiment: partition seed vs init seed for Pima attractor.

The same `seed` value controls both:
  - The Dirichlet partition draw (via np.random.default_rng(seed) in partition.py)
  - The VQC weight initialization (via torch.rand after torch.manual_seed(seed))

This experiment isolates the two by crossing partition_seed and init_seed:

  Test A: partition=seed42, init=seed7  → if round-0 ≈ 36% (class-1), init dominates
  Test B: partition=seed7,  init=seed42 → if round-0 ≈ 65% (class-0), init dominates

Expected from cross-alpha evidence (seeds 7/123 start at ~36% at both alpha=0.5
and alpha=0.1 with different partitions): init seed determines attractor start.

Reference originals:
  seed=42: round-0 ≈ 64.94% (class-0 attractor) — partition=42, init=42
  seed=7:  round-0 ≈ 36.36% (class-1 attractor) — partition=7,  init=7
"""

import os, sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from federation.run_federation import run_federation
from models.encodings import amplitude_encoding
from data.loaders import load_pima_data

# ── load Pima ──────────────────────────────────────────────────────────────────
X_train_p, X_test_p, y_train_p, y_test_p = load_pima_data(test_size=0.2, random_state=42)

N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2
ALPHA        = 0.1


def run_with_seeds(label, init_seed, partition_seed):
    """Run federation with independently controlled init and partition seeds."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  init_seed={init_seed}  partition_seed={partition_seed}")
    print(f"{'='*60}")

    # Seed init BEFORE calling run_federation — preserved until torch.rand in
    # HybridQuantumClassifier.__init__ (partition uses its own np.default_rng)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)

    _, round_log = run_federation(
        X_train=X_train_p, y_train=y_train_p,
        X_test=X_test_p,   y_test=y_test_p,
        encoding_fn=amplitude_encoding,
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS,  n_layers=N_LAYERS,
        split="dirichlet",  dirichlet_alpha=ALPHA,
        strategy_name="fedavg",
        seed=partition_seed,          # controls Dirichlet draw only
        project_root=PROJECT_ROOT,
        n_permutations=0,
    )

    accs = [acc for _, acc in round_log]
    print(f"\n  Round-0 acc: {accs[0]:.4f}  "
          f"({'≈class-1 attractor' if accs[0] < 0.50 else '≈class-0 attractor'})")
    print(f"  Final acc:   {accs[-1]:.4f}")
    return accs


# ── reference runs (both seeds identical — matches original sweep) ─────────────
print("\n=== REFERENCE ORIGINALS ===")
accs_42 = run_with_seeds("ORIGINAL seed=42 (partition=42, init=42)", 42, 42)
accs_7  = run_with_seeds("ORIGINAL seed=7  (partition=7,  init=7)",   7,  7)

# ── confound-breaking runs ─────────────────────────────────────────────────────
print("\n=== CONFOUND-BREAKING TESTS ===")
accs_A  = run_with_seeds("TEST A: partition=42, init=7",  init_seed=7,  partition_seed=42)
accs_B  = run_with_seeds("TEST B: partition=7,  init=42", init_seed=42, partition_seed=7)

# ── summary ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY — round-0 accuracy (pre-training, init-only diagnostic)")
print("="*60)
rows = [
    ("seed=42 (ref)",   42, 42, accs_42[0],  accs_42[-1]),
    ("seed=7  (ref)",    7,  7, accs_7[0],   accs_7[-1]),
    ("Test A (p=42,i=7)", 7, 42, accs_A[0], accs_A[-1]),
    ("Test B (p=7, i=42)", 42, 7, accs_B[0], accs_B[-1]),
]
print(f"{'Label':<24} {'Init':>4} {'Part':>4} {'R0 acc':>8} {'Final':>8}  Attractor-start")
for label, init, part, r0, fin in rows:
    att = "class-1 (~36%)" if r0 < 0.50 else "class-0 (~65%)"
    print(f"{label:<24} {init:>4} {part:>4} {r0:>8.4f} {fin:>8.4f}  {att}")

print("""
INTERPRETATION KEY:
  If R0 tracks INIT seed (not partition seed) → initialization determines attractor
  If R0 tracks PARTITION seed               → partition determines attractor
  If both match init seed                   → initialization is the dominant factor
""")

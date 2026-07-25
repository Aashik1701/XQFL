"""Confound-breaking experiment: partition seed vs init seed for Pima attractor.

The same `seed` value controls both:
  - The Dirichlet partition draw (via np.random.default_rng(seed) in partition.py)
  - The VQC weight initialization (via torch.rand after torch.manual_seed(seed))

This experiment isolates the two by crossing partition_seed and init_seed:

  Test A: partition=seed42, init=seed7  → if round-0 ≈ 36% (class-1), init dominates
  Test B: partition=seed7,  init=seed42 → if round-0 ≈ 65% (class-0), init dominates

Run each cell N_REPS times to characterise actor-RNG variance on the final accuracy.
R0 is deterministic (pre-training, init-only) and will be identical across replicates.

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
N_REPS       = 3   # replicates per cell; actor-RNG varies each run, init is fixed


def run_cell(label, init_seed, partition_seed, n_reps=N_REPS):
    """Run a 2x2 design cell n_reps times; report R0 (deterministic) and final mean±std."""
    print(f"\n{'='*60}")
    print(f"  {label}  (init={init_seed}, partition={partition_seed}, n_reps={n_reps})")
    print(f"{'='*60}")

    r0_vals, final_vals = [], []
    for rep in range(n_reps):
        # Reset torch RNG identically for each rep so VQC init is the same.
        # Actor-RNG inside Ray actors is NOT seeded and will differ each run.
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
            seed=partition_seed,
            project_root=PROJECT_ROOT,
            n_permutations=0,
        )
        accs = [acc for _, acc in round_log]
        r0_vals.append(accs[0])
        final_vals.append(accs[-1])
        print(f"  rep {rep+1}: R0={accs[0]:.4f}  final={accs[-1]:.4f}")

    r0_mean = np.mean(r0_vals)
    fin_mean = np.mean(final_vals)
    fin_std  = np.std(final_vals, ddof=1) if n_reps > 1 else 0.0
    fin_range = max(final_vals) - min(final_vals) if n_reps > 1 else 0.0
    print(f"  → R0={r0_mean:.4f} (deterministic)  final={fin_mean:.4f}±{fin_std:.2f}pp "
          f"(range={fin_range:.2f}pp)")
    return r0_mean, fin_mean, fin_std, fin_range, final_vals


# ── run all four cells ─────────────────────────────────────────────────────────
print("\n=== REFERENCE ORIGINALS ===")
r0_42, m_42, s_42, rng_42, raw_42 = run_cell("ref seed=42 (p=42,i=42)", 42, 42)
r0_7,  m_7,  s_7,  rng_7,  raw_7  = run_cell("ref seed=7  (p=7, i=7)",   7,  7)

print("\n=== CONFOUND-BREAKING TESTS ===")
r0_A, m_A, s_A, rng_A, raw_A = run_cell("Test A: p=42, i=7",  init_seed=7,  partition_seed=42)
r0_B, m_B, s_B, rng_B, raw_B = run_cell("Test B: p=7,  i=42", init_seed=42, partition_seed=7)

# ── summary table ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Label':<26} {'Init':>4} {'Part':>4} {'R0':>7}  {'Final mean':>10}  "
      f"{'±std':>6}  {'range':>6}  Attractor")
rows = [
    ("ref seed=42",    42, 42, r0_42, m_42, s_42, rng_42),
    ("ref seed=7",      7,  7, r0_7,  m_7,  s_7,  rng_7),
    ("Test A (p=42,i=7)", 7, 42, r0_A, m_A, s_A, rng_A),
    ("Test B (p=7,i=42)", 42, 7, r0_B, m_B, s_B, rng_B),
]
for lbl, ini, par, r0, m, s, rng in rows:
    att = "class-1" if r0 < 0.50 else "class-0"
    print(f"{lbl:<26} {ini:>4} {par:>4} {r0:>7.4f}  {m:>10.4f}  {s:>6.2f}pp  "
          f"{rng:>6.2f}pp  {att}")

print(f"""
FINDINGS:
  R0 is deterministic per init_seed — unaffected by partition seed or actor-RNG.
  Final accuracy variance across replicates characterises actor-RNG noise for each cell.
  Compare ref seed=7 mean ({m_7:.4f}) to original sweep value (0.3961) to assess
  whether the gap ({abs(m_7 - 0.3961)*100:.2f}pp) is within replicate noise range ({rng_7:.2f}pp).
""")

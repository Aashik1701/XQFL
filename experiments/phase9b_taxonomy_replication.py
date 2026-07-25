"""Replication check for MN-04b and MN-04d behavioral taxonomy seeds.

Runs 3 replicates each of:
  - Pima/amplitude/α=0.1/seed=1  ('transient escape → regression' pattern)
  - Pima/amplitude/α=0.5/seed=7  ('regression below baseline' pattern)

For each replicate, prints the full round-by-round trajectory so the trajectory
SHAPE (not just final value) can be assessed for reproducibility.

Note: Pima/amplitude/α=0.1/seed=7 ('anti-correlated attractor') is NOT rerun here
because the MN-04c confound experiment's 'ref seed=7' already provides 3 replicates
under identical conditions (torch.manual_seed(7), run_federation(seed=7, alpha=0.1)):
  rep 1 → 65.58%  (class-0 territory — pattern NOT reproduced)
  rep 2 → 64.29%  (class-0 territory — pattern NOT reproduced)
  rep 3 → 44.81%  (intermediate — pattern NOT reproduced)
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

X_train_p, X_test_p, y_train_p, y_test_p = load_pima_data(test_size=0.2, random_state=42)

N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2
N_REPS       = 3
MAJORITY_BASELINE = 64.94   # % class-0 in Pima test set


def run_replicate_set(label, seed, alpha, n_reps=N_REPS):
    print(f"\n{'#'*65}")
    print(f"  {label}")
    print(f"  seed={seed}  alpha={alpha}  n_reps={n_reps}")
    print(f"  majority-class baseline: {MAJORITY_BASELINE}%")
    print(f"{'#'*65}")

    all_trajectories = []
    for rep in range(n_reps):
        np.random.seed(seed)
        torch.manual_seed(seed)

        _, round_log = run_federation(
            X_train=X_train_p, y_train=y_train_p,
            X_test=X_test_p,   y_test=y_test_p,
            encoding_fn=amplitude_encoding,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,  n_layers=N_LAYERS,
            split="dirichlet",  dirichlet_alpha=alpha,
            strategy_name="fedavg",
            seed=seed,
            project_root=PROJECT_ROOT,
            n_permutations=0,
        )
        accs = [acc * 100 for _, acc in round_log]   # convert to %
        all_trajectories.append(accs)

        peak = max(accs)
        peak_r = accs.index(peak)
        below_bl = accs[-1] < MAJORITY_BASELINE
        print(f"\n  --- rep {rep+1} ---")
        print(f"  trajectory: " + "  ".join(f"R{r}={a:.1f}" for r, a in enumerate(accs)))
        print(f"  R0={accs[0]:.2f}%  peak={peak:.2f}%@R{peak_r}  R15={accs[-1]:.2f}%"
              f"  {'BELOW baseline' if below_bl else 'above baseline'}")

    # summary across reps
    finals = [t[-1] for t in all_trajectories]
    r0s    = [t[0]  for t in all_trajectories]
    peaks  = [max(t) for t in all_trajectories]
    peak_rounds = [t.index(max(t)) for t in all_trajectories]

    print(f"\n  === SUMMARY across {n_reps} reps ===")
    print(f"  R0:    {r0s}  (should be identical — deterministic)")
    print(f"  peaks: {[f'{p:.1f}%@R{r}' for p, r in zip(peaks, peak_rounds)]}")
    print(f"  R15:   {[f'{f:.2f}%' for f in finals]}")
    print(f"  R15 mean={np.mean(finals):.2f}%  std={np.std(finals, ddof=1):.1f}pp  "
          f"range={max(finals)-min(finals):.1f}pp")
    print(f"  Below baseline: {[f < MAJORITY_BASELINE for f in finals]}")
    return all_trajectories


# ── Pima / α=0.1 / seed=1 ─────────────────────────────────────────────────────
trajs_a01_s1 = run_replicate_set(
    "Pima/amplitude/α=0.1/seed=1  [MN-04b 'transient escape → regression']",
    seed=1, alpha=0.1,
)

# ── Pima / α=0.5 / seed=7 ─────────────────────────────────────────────────────
trajs_a05_s7 = run_replicate_set(
    "Pima/amplitude/α=0.5/seed=7  [MN-04d 'regression below baseline']",
    seed=7, alpha=0.5,
)

print("\n" + "="*65)
print("CROSS-CONDITION SUMMARY")
print("="*65)
for label, trajs in [
    ("α=0.1/seed=1 (transient escape)", trajs_a01_s1),
    ("α=0.5/seed=7 (regression below BL)", trajs_a05_s7),
]:
    finals = [t[-1] for t in trajs]
    print(f"\n  {label}")
    print(f"    Finals: {[f'{f:.2f}%' for f in finals]}")
    print(f"    Mean={np.mean(finals):.2f}%  Range={max(finals)-min(finals):.1f}pp")
    print(f"    Original sweep value: {'65.58%' if 'seed=1' in label else '64.29%'}")

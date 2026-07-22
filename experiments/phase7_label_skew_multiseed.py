"""Multi-seed validation of Phase 7 label-skew ordering (α=0.5 vs α=0.1).

Phase 7 seed=42 showed a non-monotonic result: α=0.5 (57.02%) < α=0.1 (63.16%) < IID (84.21%).
This is surprising — moderate label-skew beat the extreme case is expected, but α=0.5 being
*worse* than α=0.1 is counterintuitive and needs multi-seed confirmation.

Runs seeds 1, 7, 42 (load from existing CSVs), 123 at both α=0.5 and α=0.1,
with n_permutations=0 for speed (~55s/run, 6 new runs ≈ 5.5 min total).

Reports:
  - Per-client sample counts and class-1 ratios per (alpha, seed)
  - Final round-15 global test accuracy per (alpha, seed)
  - Whether the α=0.5 < α=0.1 ordering holds across seeds

Run from project root:
    python experiments/phase7_label_skew_multiseed.py
"""

import sys
import os
import csv
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.loaders import load_breast_cancer_data
from data.partition import dirichlet_split
from models.encodings import angle_encoding
from federation.run_federation import run_federation

# ── config ────────────────────────────────────────────────────────────────────
SEEDS         = [1, 7, 42, 123]
ALPHAS        = [0.5, 0.1]
N_CLIENTS     = 4
N_ROUNDS      = 15
LOCAL_EPOCHS  = 3
N_QUBITS      = 4
N_LAYERS      = 2
EXISTING_LOGS = {
    0.5: "phase7_dirichlet_05.csv",
    0.1: "phase7_dirichlet_01.csv",
}


def _load_final_acc(csv_path):
    """Extract round-15 global test accuracy from an existing phase7 CSV."""
    best_acc = None
    best_rnd = -1
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["client_id"] == "global":
                rnd = int(row["round"])
                if rnd > best_rnd:
                    best_rnd = rnd
                    best_acc = float(row["accuracy"])
    return best_acc


def _partition_stats(X_train, y_train, alpha, seed):
    """Return list of (n_samples, class1_ratio) per client."""
    partitions = dirichlet_split(X_train, y_train, N_CLIENTS, alpha=alpha, seed=seed)
    return [(len(yi), float(yi.mean())) for _, yi in partitions]


def _run_fast_federation(X_train, y_train, X_test, y_test, alpha, seed):
    """Run federation with n_permutations=0 (no Q-SHAP overhead) and return final acc."""
    _, round_log = run_federation(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS,
        n_rounds=N_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        split="dirichlet",
        dirichlet_alpha=alpha,
        strategy_name="fedavg",
        seed=seed,
        project_root=PROJECT_ROOT,
        n_permutations=0,
    )
    # round_log: list of (round_number, test_acc)
    return round_log[-1][1] if round_log else float("nan")


# ── data ──────────────────────────────────────────────────────────────────────
print("=== Phase 7: Multi-seed label-skew validation ===")
print(f"  Encoding: angle | Strategy: FedAvg | Clients: {N_CLIENTS} | "
      f"Rounds: {N_ROUNDS}\n")

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=42)

# Partition for each (alpha, seed) — done upfront so it's fast
print("── Partition statistics ─────────────────────────────────────────────────")
print(f"{'Alpha':>6}  {'Seed':>5}  {'Client':>6}  "
      f"{'n_samples':>9}  {'class1_ratio':>13}")
print(f"  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*9}  {'-'*13}")

partition_cache = {}
for alpha in ALPHAS:
    for seed in SEEDS:
        stats = _partition_stats(X_train, y_train, alpha, seed)
        partition_cache[(alpha, seed)] = stats
        for i, (n, r) in enumerate(stats):
            if i == 0:
                print(f"  {alpha:>6.1f}  {seed:>5}     {i:>4}  {n:>9}  {r:>13.3f}")
            else:
                print(f"  {'':>6}  {'':>5}     {i:>4}  {n:>9}  {r:>13.3f}")
    print()

# ── federation runs ────────────────────────────────────────────────────────────
print("── Federation accuracy per (alpha, seed) ───────────────────────────────")
print("  (seed=42 loaded from existing Phase 7 CSVs; others re-run)\n")

accuracy_table = {}   # (alpha, seed) → final_acc

for alpha in ALPHAS:
    existing_csv = os.path.join(PROJECT_ROOT, "experiments", EXISTING_LOGS[alpha])
    for seed in SEEDS:
        key = (alpha, seed)
        if seed == 42 and os.path.exists(existing_csv):
            acc = _load_final_acc(existing_csv)
            source = "loaded"
        else:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(f"  Running α={alpha}, seed={seed} ...", flush=True)
            acc = _run_fast_federation(X_train, y_train, X_test, y_test, alpha, seed)
            source = "run"
        accuracy_table[key] = acc
        print(f"  α={alpha}  seed={seed:>3}  final_acc={acc:.4f}  ({source})")

# ── comparison table ───────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Cross-seed accuracy comparison ===")
print(f"\n  {'Seed':>5}  {'α=0.5 acc':>11}  {'α=0.1 acc':>11}  {'Ordering holds?':>16}")
print(f"  {'-'*5}  {'-'*11}  {'-'*11}  {'-'*16}")

holds_count = 0
total_count = 0
for seed in SEEDS:
    acc_05 = accuracy_table.get((0.5, seed), float("nan"))
    acc_01 = accuracy_table.get((0.1, seed), float("nan"))
    if not (np.isnan(acc_05) or np.isnan(acc_01)):
        ordering = acc_05 < acc_01
        total_count += 1
        if ordering:
            holds_count += 1
        note = "YES (α=0.5 < α=0.1)" if ordering else "NO  (α=0.5 >= α=0.1)"
    else:
        note = "n/a"
    print(f"  {seed:>5}  {acc_05:>11.4f}  {acc_01:>11.4f}  {note:>16}")

print(f"\n  Ordering (α=0.5 < α=0.1) holds: {holds_count}/{total_count} seeds")

# ── per-seed partition detail table ───────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Partition detail: does poor α=0.5 performance correlate with class conflict? ===")

for seed in SEEDS:
    acc_05 = accuracy_table.get((0.5, seed), float("nan"))
    acc_01 = accuracy_table.get((0.1, seed), float("nan"))
    print(f"\n  Seed {seed}  (α=0.5 acc={acc_05:.4f}, α=0.1 acc={acc_01:.4f})")

    for alpha in ALPHAS:
        stats = partition_cache[(alpha, seed)]
        ratios = [r for _, r in stats]
        conflict = max(ratios) - min(ratios)  # range of class-1 ratios
        print(f"    α={alpha}: "
              + "  ".join(f"C{i}({n}s,{r:.2f})" for i, (n, r) in enumerate(stats))
              + f"  |conflict|={conflict:.3f}")

# ── conclusion ─────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Conclusion ===")
if total_count > 0:
    pct = holds_count / total_count * 100
    print(f"  α=0.5 < α=0.1 ordering holds in {holds_count}/{total_count} seeds ({pct:.0f}%).")
    if holds_count == total_count:
        print("  Finding is ROBUST across seeds: moderate label-skew (α=0.5) is "
              "systematically worse than extreme skew (α=0.1) for this dataset/config.")
    elif holds_count >= total_count / 2:
        print("  Finding is PARTIAL: ordering holds in most seeds but not all. "
              "Report as seed-dependent in the paper.")
    else:
        print("  Finding is FRAGILE: ordering reverses in most seeds. "
              "Seed=42 result was an outlier — do not report as a general finding.")

print("\n[Phase 7 multi-seed validation complete]")

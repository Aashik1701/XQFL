"""Phase 9a: MVP sweep — 2 datasets × 2 encodings × 5 seeds, IID FedAvg.

Design:
  - Datasets  : Wisconsin Breast Cancer (569 samples, 30 features, binary)
                Pima Indians Diabetes   (768 samples,  8 features, binary)
  - Encodings : angle_encoding, amplitude_encoding
  - Strategy  : FedAvg (IID split, 4 clients, 15 rounds, 3 local epochs)
  - Seeds     : [1, 7, 42, 123, 999]  — 5 seeds per cell as hard requirement
                (lesson from Phase 7: single-seed non-IID comparisons with n=4
                 clients are unreliable; multi-seed is mandatory for every cell)
  - n_permutations=0  — no Q-SHAP overhead, accuracy-only run

Data split is fixed (random_state=42) so train/test sets are the same across all
federation seeds.  Only the client partition and initial weights vary per seed.

Run from project root:
    python experiments/phase9a_mvp_sweep.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from itertools import product

from data.loaders import load_breast_cancer_data, load_pima_data
from models.encodings import angle_encoding, amplitude_encoding
from federation.run_federation import run_federation

# ── config ────────────────────────────────────────────────────────────────────
SEEDS        = [1, 7, 42, 123, 999]
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2

DATASETS = {
    "wisconsin": load_breast_cancer_data,
    "pima":      load_pima_data,
}
ENCODINGS = {
    "angle":     angle_encoding,
    "amplitude": amplitude_encoding,
}

# ── data loading (fixed train/test split) ─────────────────────────────────────
print("=== Phase 9a: MVP Sweep ===")
print(f"  Datasets: {list(DATASETS)}  |  Encodings: {list(ENCODINGS)}")
print(f"  Seeds: {SEEDS}  |  n_clients={N_CLIENTS}  rounds={N_ROUNDS}  "
      f"local_epochs={LOCAL_EPOCHS}\n")

loaded_data = {}
for ds_name, loader_fn in DATASETS.items():
    X_tr, X_te, y_tr, y_te = loader_fn(random_state=42)
    loaded_data[ds_name] = (X_tr, X_te, y_tr, y_te)
    pos  = y_tr.mean()
    print(f"  [{ds_name}] train={len(X_tr)} test={len(X_te)} "
          f"features={X_tr.shape[1]} class1={pos:.3f}")
print()

# ── main sweep ────────────────────────────────────────────────────────────────
# results[(ds, enc)] = list of per-seed final test accuracies
results = {(ds, enc): [] for ds, enc in product(DATASETS, ENCODINGS)}

total_runs = len(DATASETS) * len(ENCODINGS) * len(SEEDS)
run_idx    = 0

for ds_name, enc_name in product(DATASETS, ENCODINGS):
    X_train, X_test, y_train, y_test = loaded_data[ds_name]
    enc_fn = ENCODINGS[enc_name]

    for seed in SEEDS:
        run_idx += 1
        print(f"{'='*65}")
        print(f"  [{run_idx}/{total_runs}] dataset={ds_name}  "
              f"encoding={enc_name}  seed={seed}")
        np.random.seed(seed)
        torch.manual_seed(seed)

        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=enc_fn,
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            split="iid",
            strategy_name="fedavg",
            seed=seed,
            project_root=PROJECT_ROOT,
            n_permutations=0,
        )

        final_acc = round_log[-1][1] if round_log else float("nan")
        results[(ds_name, enc_name)].append(final_acc)
        print(f"  → final test acc = {final_acc:.4f}")

# ── aggregated results table ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Phase 9a: Aggregated Results (mean ± std, 5 seeds) ===\n")

# per-seed breakdown first
print("  Per-seed accuracies:")
print(f"  {'Dataset':<12}  {'Encoding':<10}  "
      + "  ".join(f"seed={s}" for s in SEEDS))
print(f"  {'-'*12}  {'-'*10}  "
      + "  ".join(["-"*6] * len(SEEDS)))
for ds_name, enc_name in product(DATASETS, ENCODINGS):
    accs = results[(ds_name, enc_name)]
    row  = "  ".join(f"{a:.4f}" for a in accs)
    print(f"  {ds_name:<12}  {enc_name:<10}  {row}")

# main aggregated table
print(f"\n  Aggregated mean ± std:")
print(f"  {'Dataset':<12}  {'Encoding':<10}  {'Mean':>7}  {'Std':>7}  "
      f"{'Min':>7}  {'Max':>7}  {'Range':>7}")
print(f"  {'-'*12}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}")

agg = {}
for ds_name, enc_name in product(DATASETS, ENCODINGS):
    accs = results[(ds_name, enc_name)]
    a    = np.array(accs)
    agg[(ds_name, enc_name)] = {
        "mean": float(a.mean()),
        "std":  float(a.std()),
        "min":  float(a.min()),
        "max":  float(a.max()),
    }
    r = agg[(ds_name, enc_name)]
    print(f"  {ds_name:<12}  {enc_name:<10}  "
          f"{r['mean']:>7.4f}  {r['std']:>7.4f}  "
          f"{r['min']:>7.4f}  {r['max']:>7.4f}  "
          f"{r['max']-r['min']:>7.4f}")

# compact mean±std grid (paper-ready)
print(f"\n  Paper-ready grid (mean% ± std%):")
header = f"  {'Dataset':<12}  " + "  ".join(f"{'enc:'+e:<18}" for e in ENCODINGS)
print(header)
print(f"  {'-'*12}  " + "  ".join(["-"*18] * len(ENCODINGS)))
for ds_name in DATASETS:
    cells = []
    for enc_name in ENCODINGS:
        r = agg[(ds_name, enc_name)]
        cells.append(f"{r['mean']*100:.1f}% ± {r['std']*100:.1f}%")
    print(f"  {ds_name:<12}  " + "  ".join(f"{c:<18}" for c in cells))

# ── findings ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Phase 9a: Findings ===\n")

# encoding comparison per dataset
for ds_name in DATASETS:
    r_ang = agg[(ds_name, "angle")]
    r_amp = agg[(ds_name, "amplitude")]
    diff  = r_ang["mean"] - r_amp["mean"]
    winner = "angle" if diff > 0 else "amplitude"
    sig    = "significant" if abs(diff) > max(r_ang["std"], r_amp["std"]) else "within noise"
    print(f"  [{ds_name}] angle={r_ang['mean']*100:.1f}%±{r_ang['std']*100:.1f}%  "
          f"amplitude={r_amp['mean']*100:.1f}%±{r_amp['std']*100:.1f}%  "
          f"→ {winner} better by {abs(diff)*100:.1f}pp ({sig})")

# dataset comparison per encoding
print()
for enc_name in ENCODINGS:
    r_wis = agg[("wisconsin", enc_name)]
    r_pim = agg[("pima",      enc_name)]
    diff  = r_wis["mean"] - r_pim["mean"]
    winner = "wisconsin" if diff > 0 else "pima"
    print(f"  [{enc_name}] wisconsin={r_wis['mean']*100:.1f}%±{r_wis['std']*100:.1f}%  "
          f"pima={r_pim['mean']*100:.1f}%±{r_pim['std']*100:.1f}%  "
          f"→ {winner} {abs(diff)*100:.1f}pp higher")

# variability flag (std > 5pp is high variance for 5 seeds)
print(f"\n  Seed variability check (high if std > 5pp):")
for ds_name, enc_name in product(DATASETS, ENCODINGS):
    r   = agg[(ds_name, enc_name)]
    tag = "HIGH" if r["std"] > 0.05 else "ok"
    print(f"    {ds_name}/{enc_name}: std={r['std']*100:.1f}pp  [{tag}]")

print(f"\n[Phase 9a complete]")

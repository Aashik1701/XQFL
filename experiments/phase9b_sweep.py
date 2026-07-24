"""Phase 9b: extended sweep.

Extends Phase 9a along three axes:
  A. Encoding sweep   — adds reuploading and IQP to the existing angle/amplitude table
                        (IID, FedAvg, both datasets, 5 seeds)
  B. Strategy sweep   — FedAvg vs TrimmedMean
                        (IID, amplitude encoding, both datasets, 5 seeds)
  C. Distribution sweep — IID vs Dirichlet α=0.5/α=0.1 vs quantity-skew
                        (amplitude encoding, FedAvg, both datasets, 5 seeds)

Hard requirements (from Phase 7 multi-seed lesson):
  - ≥5 seeds per cell, no exceptions
  - Every reported comparison must clear the ≥1.75pp reliability threshold
    (based on within-seed noise floor confirmed in Phase 9b pre-checks)
  - Effects below 3.5pp in a 5-seed mean are flagged BELOW NOISE FLOOR

Design note on non-IID seeds:
  Phase 7 multi-seed validation showed that α=0.5 vs α=0.1 ordering is
  unstable at n=4 clients because the Dirichlet partition is itself seed-dependent
  (a 'partition lottery' — one seed can give pathological allocations).  5 seeds
  mitigate this but do not eliminate it; both α conditions are reported, and
  any claimed ordering between them requires the effect to persist across ≥4/5 seeds.

Checkpointing:
  Results are written to results/phase9b_results.csv after each run so the
  script can be interrupted and restarted without losing completed work.

Run from project root:
    python experiments/phase9b_sweep.py
"""

import sys
import os
import csv
import ast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from functools import partial
from itertools import product

from data.loaders import load_breast_cancer_data, load_pima_data
from models.encodings import angle_encoding, amplitude_encoding, reuploading_encoding, iqp_encoding
from federation.run_federation import run_federation

# ── config ────────────────────────────────────────────────────────────────────
SEEDS        = [1, 7, 42, 123, 999]
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2
NOISE_FLOOR  = 0.0175   # 1.75pp half-range — effects below this flagged unconfirmed

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, "phase9b_results.csv")
CSV_FIELDS = ["part", "dataset", "encoding", "strategy", "distribution",
              "dirichlet_alpha", "seed", "final_acc"]

# reuploading needs n_layers injected; match the VQC depth
reuploading_enc = partial(reuploading_encoding, n_layers=N_LAYERS)
reuploading_enc.__name__ = "reuploading"

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_enc,
    "iqp":         iqp_encoding,
}

DATASETS = {
    "wisconsin": load_breast_cancer_data,
    "pima":      load_pima_data,
}

# Pima majority-class baseline for annotation
PIMA_BASELINE = 100 / 154   # 64.94%

# Phase 9a reference (IID, FedAvg, angle/amplitude only)
PHASE9A_REF = {
    ("wisconsin", "angle"):     {"mean": 0.8328, "std": 0.0177},
    ("wisconsin", "amplitude"): {"mean": 0.9526, "std": 0.0070},
    ("pima",      "angle"):     {"mean": 0.6000, "std": 0.0151},
    ("pima",      "amplitude"): {"mean": 0.6922, "std": 0.0151},
}

# ── checkpoint helpers ────────────────────────────────────────────────────────

def _load_done():
    """Return set of (part, dataset, encoding, strategy, distribution, alpha_str, seed) already run."""
    done = set()
    if not os.path.exists(CSV_PATH):
        return done
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["part"], row["dataset"], row["encoding"],
                   row["strategy"], row["distribution"],
                   row["dirichlet_alpha"], int(row["seed"]))
            done.add(key)
    return done


def _save_row(part, dataset, encoding, strategy, distribution, alpha, seed, acc):
    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow({
            "part": part, "dataset": dataset, "encoding": encoding,
            "strategy": strategy, "distribution": distribution,
            "dirichlet_alpha": str(alpha), "seed": seed, "final_acc": acc,
        })


# ── sweep helper ──────────────────────────────────────────────────────────────

def run_cell(part, ds_name, enc_name, strategy_name, split, alpha, seed,
             done, loaded_data, run_idx, total_runs):
    """Run a single (dataset, encoding, strategy, distribution, seed) cell."""
    alpha_str = str(alpha)
    key = (part, ds_name, enc_name, strategy_name, split, alpha_str, seed)
    if key in done:
        print(f"  [{run_idx}/{total_runs}] SKIP (cached): "
              f"{ds_name}/{enc_name}/{strategy_name}/{split}(α={alpha}) seed={seed}")
        return None   # already done

    print(f"  [{run_idx}/{total_runs}] "
          f"dataset={ds_name}  enc={enc_name}  strategy={strategy_name}  "
          f"split={split}(α={alpha})  seed={seed}")

    X_train, X_test, y_train, y_test = loaded_data[ds_name]
    np.random.seed(seed)
    torch.manual_seed(seed)

    _, round_log = run_federation(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        encoding_fn=ENCODINGS[enc_name],
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        split=split, dirichlet_alpha=float(alpha) if alpha is not None else 0.5,
        strategy_name=strategy_name,
        seed=seed, project_root=PROJECT_ROOT, n_permutations=0,
    )
    acc = round_log[-1][1] if round_log else float("nan")
    print(f"  → {acc:.4f}")
    _save_row(part, ds_name, enc_name, strategy_name, split, alpha, seed, acc)
    done.add(key)
    return acc


# ── load data ─────────────────────────────────────────────────────────────────
print("=== Phase 9b: Extended Sweep ===\n")
loaded_data = {}
for ds_name, loader_fn in DATASETS.items():
    X_tr, X_te, y_tr, y_te = loader_fn(random_state=42)
    loaded_data[ds_name] = (X_tr, X_te, y_tr, y_te)
    print(f"  [{ds_name}] train={len(X_tr)}  test={len(X_te)}  features={X_tr.shape[1]}")
print()

done = _load_done()
print(f"  Loaded {len(done)} completed runs from checkpoint.\n")

# ── Part A: Encoding sweep (IID, FedAvg, all 4 encodings) ─────────────────────
part_A_cells = list(product(DATASETS, ENCODINGS, SEEDS))
total_A = len(part_A_cells)
print(f"{'='*65}")
print(f"=== Part A: Encoding sweep (IID, FedAvg, {total_A} total) ===\n")

part_A_results = {}   # (ds, enc) → list of accs
for run_idx, (ds_name, enc_name, seed) in enumerate(part_A_cells, 1):
    acc = run_cell("A", ds_name, enc_name, "fedavg", "iid", None, seed,
                   done, loaded_data, run_idx, total_A)
    key = (ds_name, enc_name)
    if key not in part_A_results:
        part_A_results[key] = []
    # try to load from csv if None (was cached)
    if acc is None:
        # will be read from csv at summary time
        pass
    else:
        part_A_results[key].append(acc)

# ── Part B: Strategy sweep (IID, amplitude, TrimmedMean only — FedAvg from Part A) ──
# FedAvg/amplitude/IID is already covered by Part A; only TrimmedMean is new here.
part_B_cells = list(product(DATASETS, SEEDS))
total_B = len(part_B_cells)
print(f"\n{'='*65}")
print(f"=== Part B: Strategy sweep (IID, amplitude, TrimmedMean, {total_B} total) ===\n")

for run_idx, (ds_name, seed) in enumerate(part_B_cells, 1):
    run_cell("B", ds_name, "amplitude", "trimmed_mean", "iid", None, seed,
             done, loaded_data, run_idx, total_B)

# ── Part C: Distribution sweep (amplitude, FedAvg, 3 non-IID conditions) ──────
DISTRIBUTIONS = [
    ("dirichlet", 0.5),
    ("dirichlet", 0.1),
    ("quantity",  None),
]
part_C_cells = list(product(DATASETS, DISTRIBUTIONS, SEEDS))
total_C = len(part_C_cells)
print(f"\n{'='*65}")
print(f"=== Part C: Distribution sweep (amplitude, FedAvg, {total_C} total) ===\n")

for run_idx, (ds_name, (split, alpha), seed) in enumerate(part_C_cells, 1):
    run_cell("C", ds_name, "amplitude", "fedavg", split, alpha, seed,
             done, loaded_data, run_idx, total_C)

# ── Load all results from CSV for summary ─────────────────────────────────────
print(f"\n{'='*65}")
print("=== Phase 9b: Results Summary ===\n")

all_rows = []
with open(CSV_PATH, newline="") as f:
    all_rows = list(csv.DictReader(f))


def get_accs(part, ds, enc, strategy, split, alpha):
    accs = []
    for row in all_rows:
        if (row["part"] == part and row["dataset"] == ds
                and row["encoding"] == enc and row["strategy"] == strategy
                and row["distribution"] == split
                and row["dirichlet_alpha"] == str(alpha)):
            accs.append(float(row["final_acc"]))
    return accs


def cell_stat(accs):
    if not accs:
        return "  —  "
    a = np.array(accs)
    return f"{a.mean()*100:.1f}%±{a.std()*100:.1f}%"


def flag(mean, ref_mean):
    """Return noise-floor flag relative to a reference mean."""
    diff = abs(mean - ref_mean)
    if diff < NOISE_FLOOR:
        return " *NF*"   # below noise floor
    return ""


# ─── Part A summary: 2 datasets × 4 encodings ────────────────────────────────
print("Part A — Encoding comparison (IID, FedAvg, 5 seeds):\n")
print(f"  {'Dataset':<12}  {'Encoding':<12}  {'Mean%±Std%':>14}  {'vs Phase9a':>10}  {'Note'}")
print(f"  {'-'*12}  {'-'*12}  {'-'*14}  {'-'*10}  {'-'*20}")

for ds_name in DATASETS:
    for enc_name in ENCODINGS:
        accs = get_accs("A", ds_name, enc_name, "fedavg", "iid", None)
        # Phase 9a used different part label — check both A and legacy
        if not accs and (ds_name, enc_name) in [("wisconsin","angle"),("wisconsin","amplitude"),
                                                   ("pima","angle"),("pima","amplitude")]:
            # Phase 9a runs may not have been re-run under part A; re-run now was possible
            pass
        stat = cell_stat(accs)
        note = ""
        if accs:
            mean_acc = float(np.mean(accs))
            if (ds_name, enc_name) in PHASE9A_REF:
                ref = PHASE9A_REF[(ds_name, enc_name)]["mean"]
                diff = (mean_acc - ref) * 100
                vs = f"{diff:+.1f}pp"
                note = flag(mean_acc, ref)
            else:
                vs = "  new  "
            if ds_name == "pima" and mean_acc < PIMA_BASELINE:
                note += " BELOW-BASELINE"
        else:
            vs = "  —  "
        print(f"  {ds_name:<12}  {enc_name:<12}  {stat:>14}  {vs:>10}  {note}")

# ─── Part B summary: FedAvg vs TrimmedMean ───────────────────────────────────
print(f"\nPart B — Aggregation comparison (IID, amplitude, 5 seeds):\n")
print(f"  {'Dataset':<12}  {'Strategy':<14}  {'Mean%±Std%':>14}  {'Δ vs FedAvg':>12}  {'Reliable?':>10}")
print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*12}  {'-'*10}")

for ds_name in DATASETS:
    # FedAvg baseline comes from Part A (amplitude/IID) — no re-run needed
    fa_accs = get_accs("A", ds_name, "amplitude", "fedavg", "iid", None)
    tm_accs = get_accs("B", ds_name, "amplitude", "trimmed_mean", "iid", None)
    fa_mean = float(np.mean(fa_accs)) if fa_accs else float("nan")
    tm_mean = float(np.mean(tm_accs)) if tm_accs else float("nan")
    diff    = tm_mean - fa_mean
    reliable = "YES" if abs(diff) > NOISE_FLOOR * 2 else f"NO (<{NOISE_FLOOR*2*100:.1f}pp)"
    print(f"  {ds_name:<12}  {'fedavg (Part A)':<14}  {cell_stat(fa_accs):>14}  {'baseline':>12}  {'—':>10}")
    print(f"  {ds_name:<12}  {'trimmed_mean':<14}  {cell_stat(tm_accs):>14}  {diff*100:>+11.1f}pp  {reliable:>10}")

# ─── Part C summary: distribution sweep ──────────────────────────────────────
print(f"\nPart C — Distribution comparison (amplitude, FedAvg, 5 seeds):\n")
print(f"  {'Dataset':<12}  {'Distribution':<18}  {'Mean%±Std%':>14}  {'Δ vs IID':>10}  {'Reliable?':>10}")
print(f"  {'-'*12}  {'-'*18}  {'-'*14}  {'-'*10}  {'-'*10}")

for ds_name in DATASETS:
    iid_accs  = get_accs("A", ds_name, "amplitude", "fedavg", "iid", None)
    iid_mean  = float(np.mean(iid_accs)) if iid_accs else float("nan")
    print(f"  {ds_name:<12}  {'iid':<18}  {cell_stat(iid_accs):>14}  {'baseline':>10}  {'—':>10}")

    for split, alpha in DISTRIBUTIONS:
        accs = get_accs("C", ds_name, "amplitude", "fedavg", split, alpha)
        dist_label = f"dirichlet α={alpha}" if split == "dirichlet" else "quantity-skew"
        dist_mean  = float(np.mean(accs)) if accs else float("nan")
        diff       = dist_mean - iid_mean
        reliable   = "YES" if abs(diff) > NOISE_FLOOR * 2 else f"NO (<{NOISE_FLOOR*2*100:.1f}pp)"
        note = " BELOW-BASELINE" if ds_name == "pima" and accs and dist_mean < PIMA_BASELINE else ""
        print(f"  {ds_name:<12}  {dist_label:<18}  {cell_stat(accs):>14}  {diff*100:>+9.1f}pp  {reliable:>10}{note}")

# ─── cross-encoding ranking (Part A) ─────────────────────────────────────────
print(f"\nEncoding ranking per dataset (mean across 5 seeds, IID FedAvg):")
for ds_name in DATASETS:
    enc_means = {}
    for enc_name in ENCODINGS:
        accs = get_accs("A", ds_name, enc_name, "fedavg", "iid", None)
        if accs:
            enc_means[enc_name] = float(np.mean(accs))
    ranking = sorted(enc_means, key=enc_means.get, reverse=True)
    print(f"  [{ds_name}]  " + "  >  ".join(
        f"{enc}({enc_means[enc]*100:.1f}%)" for enc in ranking
    ))

# ─── noise-floor assessment ───────────────────────────────────────────────────
print(f"\nNoise-floor threshold: ±{NOISE_FLOOR*100:.2f}pp (half within-seed range).")
print(f"Reliability threshold for 5-seed mean comparison: >{NOISE_FLOOR*2*100:.2f}pp.")
print(f"Items marked *NF* are within noise floor and should be treated as UNCONFIRMED.")

print(f"\n[Phase 9b sweep complete — results in {CSV_PATH}]")

"""Phase 7: Analysis across non-IID distribution types.

Sweeps four distribution conditions using the same angle-encoding FedAvg
4-client 15-round setup validated in Phases 5 and 6:

  1. IID               — already logged to phase6_log.csv; loaded, not rerun
  2. Label-skew α=0.5  — moderate Dirichlet non-IID
  3. Label-skew α=0.1  — strong Dirichlet non-IID
  4. Quantity-skew     — heterogeneous data volume per client

For each condition, computes (per consecutive round pair):
  - Kendall's tau, Spearman rho, cosine similarity, L2 distance  (stability_report)
  - ||φ|| per round (attribution magnitude)

Then reports:
  - Direction-locking speed: first round where cosine similarity >= 0.9999
  - Mean node-global JS-divergence per distribution (local vs aggregated explanation)
  - Final global test accuracy per distribution

See analysis/methods_notes.md § MN-01 for why tau alone is insufficient at n=4.

Run from project root:
    python experiments/phase7_analysis.py
"""

import sys
import os
import json
import csv
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from experiments.config import ExperimentConfig
from experiments.logging_utils import ExperimentLogger
from explain.metrics import stability_report, js_divergence
from federation.run_federation import run_federation

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── distribution sweep config ─────────────────────────────────────────────────
DIST_CONFIGS = [
    {
        "label":          "IID",
        "split":          "iid",
        "dirichlet_alpha": 0.5,
        "distribution":   "iid",
        "log":            "phase6_log.csv",
        "rerun":          False,   # load existing Phase 6 log
    },
    {
        "label":          "Label-skew α=0.5",
        "split":          "dirichlet",
        "dirichlet_alpha": 0.5,
        "distribution":   "label_skew",
        "log":            "phase7_dirichlet_05.csv",
        "rerun":          True,
    },
    {
        "label":          "Label-skew α=0.1",
        "split":          "dirichlet",
        "dirichlet_alpha": 0.1,
        "distribution":   "label_skew",
        "log":            "phase7_dirichlet_01.csv",
        "rerun":          True,
    },
    {
        "label":          "Quantity-skew",
        "split":          "quantity",
        "dirichlet_alpha": 0.5,
        "distribution":   "quantity_skew",
        "log":            "phase7_quantity_skew.csv",
        "rerun":          True,
    },
]

N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2
COS_LOCK_THRESHOLD = 0.9999


# ── helpers ───────────────────────────────────────────────────────────────────

def load_log(path):
    """Load a phase log CSV as a list of row dicts."""
    logger = ExperimentLogger.__new__(ExperimentLogger)
    logger.csv_path = path
    return logger.load()


def compute_stability_series(rows):
    """Return list of per-round-pair stability dicts from log rows."""
    global_rows = sorted(
        [r for r in rows if r["client_id"] == "global"],
        key=lambda r: r["round"],
    )
    by_round      = {r["round"]: np.array(r["explanation_vector"]) for r in global_rows}
    acc_by_round  = {r["round"]: r["accuracy"]                      for r in global_rows}
    sorted_rounds = sorted(by_round)

    series = []
    for i in range(1, len(sorted_rounds)):
        ra = sorted_rounds[i - 1]
        rb = sorted_rounds[i]
        sm = stability_report(by_round[ra], by_round[rb])
        sm["round_a"]  = ra
        sm["round_b"]  = rb
        sm["norm_b"]   = float(np.linalg.norm(by_round[rb]))
        sm["accuracy"] = acc_by_round.get(rb, float("nan"))
        series.append(sm)
    return series


def direction_lock_round(series, threshold=COS_LOCK_THRESHOLD):
    """First round_b where cosine_similarity >= threshold; None if never."""
    for s in series:
        if s["cosine_similarity"] >= threshold:
            return s["round_b"]
    return None


def compute_jsd_by_round(rows):
    """Per-round mean JS-divergence between each client's local explanation
    and the global explanation at the same round."""
    global_vecs = {
        r["round"]: np.array(r["explanation_vector"])
        for r in rows if r["client_id"] == "global"
    }
    client_by_round = defaultdict(list)
    for r in rows:
        if r["client_id"] != "global":
            client_by_round[r["round"]].append(np.array(r["explanation_vector"]))

    jsd_by_round = {}
    for rnd, gvec in global_vecs.items():
        cvecs = client_by_round.get(rnd, [])
        if cvecs:
            jsd_by_round[rnd] = float(np.mean([js_divergence(cv, gvec) for cv in cvecs]))
    return jsd_by_round


# ── data ─────────────────────────────────────────────────────────────────────

print("=== Phase 7: Analysis Across Distribution Types ===")
print(f"  Encoding: angle | Strategy: FedAvg | Clients: {N_CLIENTS} | "
      f"Rounds: {N_ROUNDS} | Epochs/round: {LOCAL_EPOCHS}\n")

X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)
X_val = X_test[:10]


# ── run missing distributions ─────────────────────────────────────────────────

all_results = {}

for dcfg in DIST_CONFIGS:
    log_path = os.path.join(PROJECT_ROOT, "experiments", dcfg["log"])
    label    = dcfg["label"]

    print(f"{'='*60}")
    print(f"  Distribution: {label}")

    if not dcfg["rerun"] or os.path.exists(log_path):
        if os.path.exists(log_path):
            print(f"  Loading existing log: {dcfg['log']}")
            rows = load_log(log_path)
        else:
            print(f"  [SKIP] Log not found and rerun=False: {dcfg['log']}")
            continue
    else:
        print(f"  Running federation → {dcfg['log']}")
        exp_cfg = ExperimentConfig(
            dataset="wisconsin",
            encoding="angle",
            aggregator="fedavg",
            distribution=dcfg["distribution"],
            dirichlet_alpha=dcfg["dirichlet_alpha"],
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            local_epochs=LOCAL_EPOCHS,
            seed=SEED,
        )
        logger = ExperimentLogger(log_path, exp_cfg)
        _, _ = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=angle_encoding,
            n_clients=N_CLIENTS,
            n_rounds=N_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,
            n_layers=N_LAYERS,
            split=dcfg["split"],
            dirichlet_alpha=dcfg["dirichlet_alpha"],
            strategy_name="fedavg",
            seed=SEED,
            project_root=PROJECT_ROOT,
            logger=logger,
            X_val=X_val,
            n_global_explain=10,
            n_global_bg=20,
            n_explain=5,
            n_bg=10,
            n_permutations=8,
        )
        rows = load_log(log_path)

    # ── per-distribution analysis ─────────────────────────────────────────────
    stability_series = compute_stability_series(rows)
    jsd_by_round     = compute_jsd_by_round(rows)
    lock_rnd         = direction_lock_round(stability_series)

    global_rows  = sorted([r for r in rows if r["client_id"] == "global"],
                          key=lambda r: r["round"])
    final_acc    = global_rows[-1]["accuracy"] if global_rows else float("nan")
    final_norm   = float(np.linalg.norm(np.array(global_rows[-1]["explanation_vector"]))) \
                   if global_rows else float("nan")
    mean_jsd     = float(np.mean(list(jsd_by_round.values()))) if jsd_by_round else float("nan")

    # Partition sizes from client rows round 1
    client_r1 = [r for r in rows if r["client_id"] != "global" and r["round"] == 1]

    print(f"\n  --- Stability per round pair (tau | cos | L2 | acc) ---")
    print(f"  {'Rounds':>10}   {'tau':>7}   {'cos':>7}   {'L2':>8}   "
          f"{'||φ||':>7}   {'acc':>6}   {'note':}")
    for s in stability_series:
        note = ""
        if s["cosine_similarity"] >= COS_LOCK_THRESHOLD and s["round_b"] == lock_rnd:
            note = "<-- direction locks here"
        print(f"  {s['round_a']:>3} -> {s['round_b']:<3}  "
              f"{s['kendalls_tau']:>+7.4f}   "
              f"{s['cosine_similarity']:>+7.4f}   "
              f"{s['l2_distance']:>8.5f}   "
              f"{s['norm_b']:>7.5f}   "
              f"{s['accuracy']:>6.4f}   "
              f"{note}")

    print(f"\n  --- Node-global JS-divergence by round ---")
    print(f"  {'Round':>5}   {'Mean JSD':>9}")
    for rnd in sorted(jsd_by_round):
        print(f"  {rnd:>5}   {jsd_by_round[rnd]:>9.5f}")

    print(f"\n  Direction-lock round (cos >= {COS_LOCK_THRESHOLD}): "
          f"{'Round ' + str(lock_rnd) if lock_rnd is not None else 'NEVER'}")
    print(f"  Final test accuracy : {final_acc:.4f}")
    print(f"  Final ||φ||         : {final_norm:.5f}")
    print(f"  Mean node-global JSD: {mean_jsd:.5f}")

    all_results[label] = {
        "stability":  stability_series,
        "jsd":        jsd_by_round,
        "lock_round": lock_rnd,
        "final_acc":  final_acc,
        "final_norm": final_norm,
        "mean_jsd":   mean_jsd,
    }
    print()


# ── cross-distribution comparison table ───────────────────────────────────────

print(f"\n{'='*70}")
print("=== Cross-distribution comparison ===")
print(f"  {'Distribution':<22}  {'Lock Rnd':>8}  {'Final Acc':>9}  "
      f"{'Final ||φ||':>11}  {'Mean JSD':>9}")
print(f"  {'-'*22}  {'-'*8}  {'-'*9}  {'-'*11}  {'-'*9}")

for dcfg in DIST_CONFIGS:
    label = dcfg["label"]
    if label not in all_results:
        continue
    r = all_results[label]
    lock = str(r["lock_round"]) if r["lock_round"] is not None else "NEVER"
    print(f"  {label:<22}  {lock:>8}  {r['final_acc']:>9.4f}  "
          f"{r['final_norm']:>11.5f}  {r['mean_jsd']:>9.5f}")

# ── findings summary ──────────────────────────────────────────────────────────
print(f"\n=== Findings ===")

if all_results:
    # Direction-locking speed
    lock_rounds = {label: r["lock_round"] for label, r in all_results.items()
                   if r["lock_round"] is not None}
    if lock_rounds:
        fastest = min(lock_rounds, key=lock_rounds.get)
        slowest = max(lock_rounds, key=lock_rounds.get)
        print(f"  Direction-locking:")
        print(f"    Fastest: {fastest} (round {lock_rounds[fastest]})")
        if fastest != slowest:
            print(f"    Slowest: {slowest} (round {lock_rounds[slowest]})")
        never_locked = [l for l, r in all_results.items() if r["lock_round"] is None]
        if never_locked:
            print(f"    Never locked: {never_locked}")

    # JSD comparison
    jsd_vals = {label: r["mean_jsd"] for label, r in all_results.items()}
    if len(jsd_vals) > 1:
        max_jsd = max(jsd_vals, key=jsd_vals.get)
        min_jsd = min(jsd_vals, key=jsd_vals.get)
        print(f"  Node-global JS-divergence:")
        print(f"    Highest (most divergent): {max_jsd} ({jsd_vals[max_jsd]:.5f})")
        print(f"    Lowest  (most uniform)  : {min_jsd} ({jsd_vals[min_jsd]:.5f})")

    # Accuracy vs JSD trade-off
    print(f"  Final accuracy:")
    for label, r in all_results.items():
        print(f"    {label:<22}: {r['final_acc']:.4f}")

print(f"\n[Phase 7 complete]")

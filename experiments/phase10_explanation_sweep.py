"""Phase 10: Targeted explanation sweep for RQ1, RQ2, RQ3.

Sweep design
------------
Phase A (RQ1 + RQ2):
  4 encodings × IID × seed=42 — federated (15 rounds) + centralized baseline
  The centralized baseline trains for CENTRAL_EPOCHS=45 (= 15 rounds × 3 local
  epochs), matching total gradient steps but not data diversity structure. See
  caveat printed at each centralized run.

Phase B (RQ3):
  amplitude × {alpha=0.5, alpha=0.1, quantity-skew} × seed=42 — federated only
  IID amplitude (Phase A) reused for the JSD baseline condition.

Outputs
-------
  results/phase10_log.csv        — all explanation rows (all runs, all rounds)
  results/phase10_checkpoint.csv — completed run keys for restart safety

CSV schema (phase10_log.csv):
  run_type, encoding, distribution, dirichlet_alpha, seed,
  round, client_id, explanation_vector, accuracy,
  track_class, margin, norm_phi, timestamp

Track A/B classification (per round, per row):
  A  = accuracy >  majority_baseline + COLLAPSE_MARGIN  (functional model)
  B  = accuracy <= majority_baseline + COLLAPSE_MARGIN  (collapsed model)
  centralized  = always labelled "centralized" (not subject to A/B logic)
  margin = accuracy − threshold  (logged for every row; negative → Track B)
"""

import os, sys, csv, json, math
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding, amplitude_encoding
from models.encodings import reuploading_encoding, iqp_encoding
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation
from explain.q_shap import q_shap_values, make_quantum_predict_fn
from explain.metrics import (
    kendalls_tau, cosine_similarity, l2_distance, js_divergence,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS         = 4
N_LAYERS         = 2
N_CLIENTS        = 4
N_ROUNDS         = 15
LOCAL_EPOCHS     = 3
CENTRAL_EPOCHS   = 45        # 15 rounds × 3 local epochs — gradient-step matched
N_PERM           = 8
N_GLOBAL_EXPLAIN = 10
N_GLOBAL_BG      = 20
N_CLIENT_EXPLAIN = 5
N_CLIENT_BG      = 10
SWEEP_SEED       = 42
COLLAPSE_MARGIN  = 0.02      # threshold = majority_baseline + 2pp

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

DISTRIBUTIONS = {
    "iid":       {"split": "iid",       "alpha": None},
    "alpha_0.5": {"split": "dirichlet", "alpha": 0.5},
    "alpha_0.1": {"split": "dirichlet", "alpha": 0.1},
    "quantity":  {"split": "quantity",  "alpha": None},
}

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "phase10_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "phase10_checkpoint.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# ── data + thresholds ─────────────────────────────────────────────────────────

print("Loading Wisconsin (breast cancer) dataset …")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SWEEP_SEED)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Majority-class baseline : {majority_baseline:.4f}")
print(f"  Collapse threshold (A/B): {collapse_threshold:.4f}  "
      f"(baseline + {COLLAPSE_MARGIN*100:.0f}pp)")

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(run_type, encoding, distribution, seed):
    return f"{run_type}|{encoding}|{distribution}|{seed}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "done":
                done.add(row["run_key"])
    return done


def mark_done(rk):
    first = not os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_key", "status", "completed_at"])
        if first:
            w.writeheader()
        w.writerow({
            "run_key":      rk,
            "status":       "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [checkpoint] done: {rk}")

# ── results CSV helpers ───────────────────────────────────────────────────────

def init_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Created {RESULTS_CSV}")


def append_row(run_type, encoding, distribution, dirichlet_alpha, seed,
               round_num, client_id, explanation_vector, accuracy):
    margin     = float(accuracy) - collapse_threshold
    if run_type == "centralized":
        track = "centralized"
    else:
        track  = "A" if float(accuracy) > collapse_threshold else "B"
    norm_phi   = float(np.linalg.norm(explanation_vector))
    vec        = [float(v) for v in np.asarray(explanation_vector).flatten()]
    row = {
        "run_type":          run_type,
        "encoding":          encoding,
        "distribution":      distribution,
        "dirichlet_alpha":   "" if dirichlet_alpha is None else dirichlet_alpha,
        "seed":              seed,
        "round":             round_num,
        "client_id":         client_id,
        "explanation_vector": json.dumps(vec),
        "accuracy":          f"{float(accuracy):.6f}",
        "track_class":       track,
        "margin":            f"{margin:.6f}",
        "norm_phi":          f"{norm_phi:.6f}",
        "timestamp":         datetime.now(timezone.utc).isoformat(),
    }
    with open(RESULTS_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=RESULTS_COLS).writerow(row)

# ── Phase10Logger — duck-typed for run_federation's logger param ──────────────

class Phase10Logger:
    """Minimal interface compatible with LoggingFedAvg and run_federation.

    run_federation calls logger.log(round, client_id, vector, accuracy).
    No ExperimentConfig needed — we carry encoding/distribution directly.
    """
    def __init__(self, run_type, encoding, distribution, dirichlet_alpha, seed):
        self.run_type       = run_type
        self.encoding       = encoding
        self.distribution   = distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.seed           = seed

    def log(self, round_num, client_id, explanation_vector, accuracy):
        append_row(
            self.run_type, self.encoding, self.distribution, self.dirichlet_alpha,
            self.seed, round_num, client_id, explanation_vector, float(accuracy),
        )

# ── federated run ─────────────────────────────────────────────────────────────

def run_federated(encoding_name, distribution_name, seed):
    rk = make_run_key("federated", encoding_name, distribution_name, seed)
    if rk in load_checkpoint():
        print(f"  [skip] {rk}")
        return

    dist = DISTRIBUTIONS[distribution_name]
    print(f"\n{'='*70}")
    print(f"  FEDERATED  {encoding_name} / {distribution_name} / seed={seed}")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Phase10Logger(
        "federated", encoding_name, distribution_name, dist["alpha"], seed
    )

    run_federation(
        X_train=X_train, y_train=y_train,
        X_test=X_test,   y_test=y_test,
        encoding_fn=ENCODINGS[encoding_name],
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS,   n_layers=N_LAYERS,
        split=dist["split"],
        dirichlet_alpha=dist["alpha"] if dist["alpha"] is not None else 0.5,
        strategy_name="fedavg",
        seed=seed,
        project_root=PROJECT_ROOT,
        logger=logger,
        n_global_explain=N_GLOBAL_EXPLAIN,
        n_global_bg=N_GLOBAL_BG,
        n_explain=N_CLIENT_EXPLAIN,
        n_bg=N_CLIENT_BG,
        n_permutations=N_PERM,
    )

    mark_done(rk)

# ── centralized baseline ──────────────────────────────────────────────────────

def run_centralized(encoding_name, seed):
    rk = make_run_key("centralized", encoding_name, "iid", seed)
    if rk in load_checkpoint():
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  CENTRALIZED  {encoding_name} / seed={seed}")
    print(f"  {CENTRAL_EPOCHS} epochs = {N_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs")
    print(f"  Caveat (MN-07 §3): controls total gradient steps, not data-diversity")
    print(f"  structure. Centralized sees full dataset each epoch; federated clients")
    print(f"  see per-shard data with periodic aggregation. The comparison isolates")
    print(f"  federation's effect on explanations as cleanly as is reasonably")
    print(f"  possible, but is not a perfect ceteris-paribus control.")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Fit the same PCA + scaler as run_federation would use
    pca    = PCA(n_components=N_QUBITS, random_state=seed)
    X_pca  = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_pca)

    model         = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ENCODINGS[encoding_name]
    )
    model.pca     = pca
    model._scaler = scaler
    model.fit(X_train, y_train, epochs=CENTRAL_EPOCHS, lr=0.05, batch_size=32)

    acc = float((model.predict(X_test) == y_test).mean())
    print(f"  Centralized test accuracy: {acc:.4f}")

    X_test_q  = model._preprocess(X_test[:N_GLOBAL_EXPLAIN])
    X_bg_q    = model._preprocess(X_train[:N_GLOBAL_BG])
    q_predict = make_quantum_predict_fn(model)
    qshap     = q_shap_values(
        q_predict, X_test_q, X_bg_q, n_permutations=N_PERM, seed=seed
    )
    global_qshap = qshap.mean(axis=0)

    append_row("centralized", encoding_name, "iid", None, seed,
               15, "global", global_qshap, acc)
    mark_done(rk)

# ── analysis ──────────────────────────────────────────────────────────────────

def load_results():
    """Load results CSV; skip rows from runs not yet marked done."""
    done = load_checkpoint()
    rows = []
    with open(RESULTS_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            rk = make_run_key(
                row["run_type"], row["encoding"], row["distribution"],
                int(row["seed"])
            )
            if rk not in done:
                continue
            row["round"]            = int(row["round"])
            row["accuracy"]         = float(row["accuracy"])
            row["margin"]           = float(row["margin"])
            row["norm_phi"]         = float(row["norm_phi"])
            row["seed"]             = int(row["seed"])
            row["dirichlet_alpha"]  = (float(row["dirichlet_alpha"])
                                       if row["dirichlet_alpha"] else None)
            row["explanation_vector"] = json.loads(row["explanation_vector"])
            try:
                row["client_id"] = int(row["client_id"])
            except (ValueError, TypeError):
                pass
            rows.append(row)
    return rows


def print_rq1(rows):
    print("\n" + "="*70)
    print("RQ1  FEDERATED vs CENTRALIZED Q-SHAP  (Wisconsin / IID / seed=42)")
    print("  τ, cosine, L2 compare fed round-15 global vs centralized Q-SHAP vector")
    print("  Positive τ/cosine = same feature ranking; small L2 = similar magnitude")
    print("="*70)
    print(f"  {'Encoding':<13} {'FedAcc':>7} {'CentAcc':>8} {'τ':>7} "
          f"{'cosine':>8} {'L2':>8}")
    print(f"  {'-'*13} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")

    for enc in ENCODINGS:
        fed_r15 = [r for r in rows
                   if r["run_type"] == "federated" and r["encoding"] == enc
                   and r["distribution"] == "iid" and r["round"] == 15
                   and r["client_id"] == "global"]
        cent = [r for r in rows
                if r["run_type"] == "centralized" and r["encoding"] == enc]

        if not fed_r15 or not cent:
            print(f"  {enc:<13}  (data missing — run not yet complete)")
            continue

        f = fed_r15[-1]   # take last if there are duplicates from restart
        c = cent[-1]
        tau     = kendalls_tau(f["explanation_vector"], c["explanation_vector"])
        cos_sim = cosine_similarity(f["explanation_vector"], c["explanation_vector"])
        l2      = l2_distance(f["explanation_vector"], c["explanation_vector"])
        print(f"  {enc:<13} {f['accuracy']:>7.4f} {c['accuracy']:>8.4f} "
              f"{tau:>7.3f} {cos_sim:>8.4f} {l2:>8.4f}")


def print_rq2(rows):
    print("\n" + "="*70)
    print("RQ2  EXPLANATION STABILITY PER ENCODING  (Wisconsin / IID / seed=42)")
    print("  Consecutive-round metrics on global Q-SHAP vector")
    print("="*70)

    for enc in ENCODINGS:
        global_rows = sorted(
            [r for r in rows
             if r["run_type"] == "federated" and r["encoding"] == enc
             and r["distribution"] == "iid" and r["client_id"] == "global"],
            key=lambda x: x["round"],
        )
        if len(global_rows) < 2:
            print(f"\n  {enc.upper()}: (insufficient data)")
            continue

        print(f"\n  {enc.upper()}")
        print(f"    {'Rnd→Rnd':>8} {'τ':>7} {'cosine':>8} {'L2':>8} {'||φ||(→)':>10}")

        taus, coss, l2s = [], [], []
        for i in range(1, len(global_rows)):
            a = global_rows[i - 1]
            b = global_rows[i]
            tau     = kendalls_tau(a["explanation_vector"], b["explanation_vector"])
            cos_sim = cosine_similarity(a["explanation_vector"], b["explanation_vector"])
            l2      = l2_distance(a["explanation_vector"], b["explanation_vector"])
            taus.append(tau); coss.append(cos_sim); l2s.append(l2)
            print(f"    {a['round']:>3}→{b['round']:<3}  {tau:>7.3f} "
                  f"{cos_sim:>8.4f} {l2:>8.4f} {b['norm_phi']:>10.5f}")

        print(f"    {'MEAN':>8}  {np.mean(taus):>7.3f} {np.mean(coss):>8.4f} "
              f"{np.mean(l2s):>8.4f}")
        print(f"    Final ||φ|| = {global_rows[-1]['norm_phi']:.5f}  |  "
              f"R0 acc = {global_rows[0]['accuracy']:.4f}  "
              f"R15 acc = {global_rows[-1]['accuracy']:.4f}")


def print_rq3(rows):
    print("\n" + "="*70)
    print("RQ3  LOCAL-GLOBAL EXPLANATION DIVERGENCE vs NON-IID LEVEL")
    print("  Encoding: amplitude  |  seed=42")
    print(f"  Collapse threshold = {collapse_threshold:.4f}  "
          f"(majority {majority_baseline:.4f} + {COLLAPSE_MARGIN*100:.0f}pp)")
    print("  Track A: functional (margin > 0)  |  Track B: collapsed (margin ≤ 0)")
    print("  JSD = Jensen-Shannon divergence between mean local and global Q-SHAP")
    print("="*70)

    for dist_name in DISTRIBUTIONS:
        fed_rows = [r for r in rows
                    if r["run_type"] == "federated" and r["encoding"] == "amplitude"
                    and r["distribution"] == dist_name]
        if not fed_rows:
            print(f"\n  {dist_name}: (no data)")
            continue

        global_rows = sorted(
            [r for r in fed_rows if r["client_id"] == "global"],
            key=lambda x: x["round"],
        )
        client_by_round = {}
        for r in fed_rows:
            if r["client_id"] != "global":
                client_by_round.setdefault(r["round"], []).append(r)

        print(f"\n  DISTRIBUTION: {dist_name}")
        print(f"    {'Rnd':>4} {'Acc':>7} {'Trk':>4} {'Margin':>+8} "
              f"{'||φ_g||':>9} {'JSD(loc,glb)':>14}")

        track_a_jsds  = []
        track_b_norms = []

        for gr in global_rows:
            rnd = gr["round"]
            trk = gr["track_class"]
            mar = gr["margin"]

            if trk == "A" and rnd in client_by_round:
                jsds = [
                    js_divergence(cr["explanation_vector"], gr["explanation_vector"])
                    for cr in client_by_round[rnd]
                ]
                jsds = [v for v in jsds if not math.isnan(v)]
                mjsd = np.mean(jsds) if jsds else float("nan")
                if not math.isnan(mjsd):
                    track_a_jsds.append(mjsd)
                jsd_str = f"{mjsd:.4f}" if not math.isnan(mjsd) else "nan"
            elif trk == "B":
                track_b_norms.append(gr["norm_phi"])
                jsd_str = "n/a (B)"
            else:
                jsd_str = "n/a"

            print(f"    {rnd:>4} {gr['accuracy']:>7.4f} {trk:>4} {mar:>+8.4f} "
                  f"{gr['norm_phi']:>9.5f} {jsd_str:>14}")

        n_a = sum(1 for g in global_rows if g["track_class"] == "A")
        n_b = sum(1 for g in global_rows if g["track_class"] == "B")
        print(f"\n    Track A: {n_a}/{len(global_rows)} rounds  |  "
              f"Track B: {n_b}/{len(global_rows)} rounds")
        if track_a_jsds:
            print(f"    Mean JSD (Track A): {np.mean(track_a_jsds):.4f}  "
                  f"(n={len(track_a_jsds)} rounds)")
        if track_b_norms:
            print(f"    Track B ||φ_global|| range: "
                  f"{min(track_b_norms):.5f} – {max(track_b_norms):.5f}")
            print(f"    Interpretation: near-zero norms confirm explanation degeneration"
                  f" under collapse.")

# ── main ──────────────────────────────────────────────────────────────────────

init_results_csv()
print(f"\n{'#'*70}")
print(f"  Phase 10 Explanation Sweep  |  Wisconsin / FedAvg / seed={SWEEP_SEED}")
print(f"  Results → {RESULTS_CSV}")
print(f"  Checkpoint → {CHECKPOINT_CSV}")
already_done = load_checkpoint()
print(f"  Already completed: {len(already_done)} run(s) of "
      f"{len(ENCODINGS)*2 + 3} total (A: {len(ENCODINGS)} fed + {len(ENCODINGS)} cent, "
      f"B: 3 fed)")
print(f"{'#'*70}")

# Phase A: RQ1 + RQ2
print("\n[PHASE A]  4 encodings × IID — federated + centralized")
for enc in ENCODINGS:
    run_federated(enc, "iid", SWEEP_SEED)
    run_centralized(enc, SWEEP_SEED)

# Phase B: RQ3
print("\n[PHASE B]  amplitude × non-IID distributions — federated")
for dist in ["alpha_0.5", "alpha_0.1", "quantity"]:
    run_federated("amplitude", dist, SWEEP_SEED)

# Analysis
rows = load_results()
print(f"\nTotal rows loaded (from completed runs): {len(rows)}")

print_rq1(rows)
print_rq2(rows)
print_rq3(rows)

print(f"\n{'='*70}")
print("[Phase 10 complete]")
print(f"  Results : {RESULTS_CSV}")
print(f"  Checkpoint: {CHECKPOINT_CSV}")

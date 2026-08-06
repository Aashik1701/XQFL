"""Phase 16: Pima explanation sweep — all encodings × all distributions × 5 seeds.

Wires Pima into the RQ1/2/3 explanation pipeline (Q-SHAP, Track A/B, dual JSD
reference, per-round logging).  Mirrors Phase 10/11 Wisconsin sweep structure:

  Phase A — IID baseline: 4 encodings × 5 seeds (federated + centralized)
             → per-encoding live IID_std for Trigger S calibration
  Phase B — non-IID sweep: 4 encodings × 3 distributions × 5 seeds
             → MN-08 G1/G2 replication per (encoding, distribution) block

Ghost-client guard is active in federation/client.py:
  α=0.1/seed=1  (C1=0): skipped in fit(), returns zero-weight update
  α=0.1/seed=999 (C0=0, C2=0): same

Outputs:
  results/phase16_pima_log.csv         — per-round explanation rows
  results/phase16_pima_checkpoint.csv  — run-key checkpoint (resume-safe)
"""

import os, sys, csv, json, signal, subprocess, time
import numpy as np
import torch
from datetime import datetime, timezone
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_pima_data
from data.partition import dirichlet_split, iid_split, quantity_skew_split
from models.encodings import (angle_encoding, amplitude_encoding,
                               reuploading_encoding, iqp_encoding)
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation
from explain.q_shap import q_shap_values, make_quantum_predict_fn

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS           = 4
N_LAYERS           = 2
N_CLIENTS          = 4
N_ROUNDS           = 15
LOCAL_EPOCHS       = 3
CENTRAL_EPOCHS     = 45
N_PERM             = 8
N_GLOBAL_EXPLAIN   = 10
N_GLOBAL_BG        = 20
N_CLIENT_EXPLAIN   = 5
N_CLIENT_BG        = 10
N_REPLICATION_REPS = 2
COLLAPSE_MARGIN    = 0.02
GHOST_THRESHOLD    = 20
RUN_TIMEOUT        = 600

ENCODINGS = ["angle", "amplitude", "reuploading", "iqp"]
ENCODING_FNS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

SEEDS = [42, 1, 7, 123, 999]

# split and dirichlet_alpha are passed directly to run_federation via **kw.
# For qty_skew, dirichlet_alpha is ignored by run_federation (quantity split
# uses skew_factor=0.5 internally); it is NOT logged to the results CSV.
DIST_KWARGS = {
    "iid":       dict(split="iid",       dirichlet_alpha=0.5),
    "alpha_0.5": dict(split="dirichlet", dirichlet_alpha=0.5),
    "alpha_0.1": dict(split="dirichlet", dirichlet_alpha=0.1),
    "qty_skew":  dict(split="quantity",  dirichlet_alpha=0.5),
}
NON_IID_DISTS = ["alpha_0.5", "alpha_0.1", "qty_skew"]

# Fallback stds used ONLY if Phase A produces < 2 seeds for an encoding.
# amplitude: Phase 9b smoke-test live Pima value; others: Phase 11 Wisconsin.
BOOTSTRAP_IID_STD = {
    "angle":       0.0306,
    "amplitude":   0.0092,
    "reuploading": 0.0088,
    "iqp":         0.0394,
}

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "phase16_pima_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "phase16_pima_checkpoint.csv")

RESULTS_COLS = [
    "run_type", "encoding", "distribution", "dirichlet_alpha", "seed",
    "round", "client_id", "explanation_vector", "accuracy",
    "track_class", "margin", "norm_phi", "timestamp",
]

# ── data loading ──────────────────────────────────────────────────────────────

print("Loading Pima dataset …")
X_train, X_test, y_train, y_test = load_pima_data(random_state=42)
majority_baseline  = float(np.bincount(y_test.astype(int)).max()) / len(y_test)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN

print(f"  Train: {len(X_train)}  Test: {len(y_test)}")
print(f"  Features: {X_train.shape[1]}  →  {N_QUBITS} via PCA")
print(f"  Majority-class baseline: {majority_baseline:.4f}  ({majority_baseline*100:.2f}%)")
print(f"  Collapse threshold:      {collapse_threshold:.4f}  ({collapse_threshold*100:.2f}%)")
print()

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(run_type, encoding, distribution, seed, rep=None):
    if rep is None:
        return f"{run_type}|{encoding}|{distribution}|{seed}"
    return f"{run_type}|{encoding}|{distribution}|{seed}|rep{rep}"


def make_replcheck_key(encoding, distribution):
    return f"replcheck|{encoding}|{distribution}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") in ("done", "timeout"):
                done.add(row["run_key"])
    return done


def mark_done(rk, status="done"):
    first = not os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run_key", "status", "completed_at"])
        if first:
            w.writeheader()
        w.writerow({
            "run_key":      rk,
            "status":       status,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"  [checkpoint] {status}: {rk}")


def timed_out_seeds(encoding, distribution):
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    timed = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "timeout":
                parts = row["run_key"].split("|")
                if (len(parts) >= 4
                        and parts[0] == "federated"
                        and parts[1] == encoding
                        and parts[2] == distribution):
                    try:
                        timed.add(int(parts[3]))
                    except ValueError:
                        pass
    return timed


# ── results CSV helpers ───────────────────────────────────────────────────────

def init_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=RESULTS_COLS).writeheader()
        print(f"  Created {RESULTS_CSV}")


def append_row(run_type, encoding, distribution, dirichlet_alpha, seed,
               round_num, client_id, explanation_vector, accuracy):
    margin   = float(accuracy) - collapse_threshold
    if run_type == "centralized":
        track = "centralized"
    else:
        track = "A" if float(accuracy) > collapse_threshold else "B"
    norm_phi = float(np.linalg.norm(explanation_vector))
    vec      = [float(v) for v in np.asarray(explanation_vector).flatten()]
    row = {
        "run_type":           run_type,
        "encoding":           encoding,
        "distribution":       distribution,
        "dirichlet_alpha":    "" if dirichlet_alpha is None else dirichlet_alpha,
        "seed":               seed,
        "round":              round_num,
        "client_id":          client_id,
        "explanation_vector": json.dumps(vec),
        "accuracy":           f"{float(accuracy):.6f}",
        "track_class":        track,
        "margin":             f"{margin:.6f}",
        "norm_phi":           f"{norm_phi:.6f}",
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }
    with open(RESULTS_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=RESULTS_COLS).writerow(row)


# ── logger ────────────────────────────────────────────────────────────────────

class Phase16Logger:
    def __init__(self, run_type, encoding, distribution, dirichlet_alpha, seed):
        self.run_type        = run_type
        self.encoding        = encoding
        self.distribution    = distribution
        self.dirichlet_alpha = dirichlet_alpha
        self.seed            = seed

    def log(self, round_num, client_id, explanation_vector, accuracy):
        append_row(
            self.run_type, self.encoding, self.distribution, self.dirichlet_alpha,
            self.seed, round_num, client_id, explanation_vector, float(accuracy),
        )


# ── partition geometry (MN-08 Trigger G) ─────────────────────────────────────

def partition_stats(partition):
    total_c1 = int((y_train == 1).sum())
    total_c0 = int((y_train == 0).sum())
    n_total  = sum(len(yc) for _, yc in partition)
    stats = []
    for i, (_, yc) in enumerate(partition):
        n_c    = len(yc)
        c1     = int((yc == 1).sum())
        c0     = int((yc == 0).sum())
        weight = n_c / n_total if n_total > 0 else 0.0
        c1_rat = c1 / n_c if n_c > 0 else float("nan")
        purity = max(c1_rat, 1 - c1_rat) if n_c > 0 else 0.0
        stats.append({
            "client":   i,
            "n":        n_c,
            "c1":       c1,
            "c0":       c0,
            "weight":   weight,
            "c1_ratio": c1_rat,
            "purity":   purity,
            "c1_share": c1 / total_c1 if total_c1 > 0 else 0.0,
            "c0_share": c0 / total_c0 if total_c0 > 0 else 0.0,
            "ghost":    n_c < GHOST_THRESHOLD,
        })
    return stats


def geometry_flag(partition):
    stats   = partition_stats(partition)
    reasons = []
    for s in stats:
        if s["ghost"]:
            continue
        if s["weight"] >= 0.60 and s["purity"] >= 0.90:
            reasons.append(
                f"G1: C{s['client']} weight={s['weight']:.1%} purity={s['purity']:.1%}"
                f" (c1_ratio={s['c1_ratio']:.2f})"
            )
        if s["c1_share"] >= 0.90:
            reasons.append(
                f"G2: C{s['client']} holds {s['c1_share']:.1%} of class-1"
                f" (n={s['n']})"
            )
        if s["c0_share"] >= 0.90:
            reasons.append(
                f"G2: C{s['client']} holds {s['c0_share']:.1%} of class-0"
                f" (n={s['n']})"
            )
    return bool(reasons), reasons


# ── pre-compute G1/G2 flags for all (non-IID dist × seed) combos ─────────────

print("Pre-computing G1/G2 geometry flags …")
GEOMETRY_FLAGS = {}  # (dist, seed) → (flagged, reasons)

for dist in NON_IID_DISTS:
    kw    = DIST_KWARGS[dist]
    alpha = kw.get("dirichlet_alpha", 0.5)
    print(f"  {dist}:")
    for seed in SEEDS:
        if kw["split"] == "dirichlet":
            part = dirichlet_split(X_train, y_train, N_CLIENTS, alpha=alpha, seed=seed)
        elif kw["split"] == "quantity":
            part = quantity_skew_split(X_train, y_train, N_CLIENTS,
                                       skew_factor=0.5, seed=seed)
        else:
            part = iid_split(X_train, y_train, N_CLIENTS, seed=seed)

        flagged, reasons = geometry_flag(part)
        GEOMETRY_FLAGS[(dist, seed)] = (flagged, reasons)

        counts  = "/".join(str(s["n"]) for s in partition_stats(part))
        primary = reasons[0] if reasons else "—"
        flag_tag = "[FLAG]" if flagged else "[    ]"
        print(f"    seed={seed:>3}  ({counts})  {flag_tag}  {primary}")
print()


# ── Ray cleanup ────────────────────────────────────────────────────────────────

def cleanup_ray():
    subprocess.run(["pkill", "-f", r"ray.*worker"],      capture_output=True)
    subprocess.run(["pkill", "-f", r"ray/core.*raylet"], capture_output=True)
    subprocess.run(["pkill", "-f", r"ray.*log_monitor"], capture_output=True)
    time.sleep(3)


# ── federated run ─────────────────────────────────────────────────────────────

def run_federated(encoding_name, distribution_name, seed,
                  run_type="federated", rep=None):
    rk   = make_run_key(run_type, encoding_name, distribution_name, seed, rep)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    kw         = DIST_KWARGS[distribution_name]
    logger_alpha = kw["dirichlet_alpha"] if kw["split"] == "dirichlet" else None
    rep_tag    = f" (rep {rep})" if rep is not None else ""
    geo_flag   = GEOMETRY_FLAGS.get((distribution_name, seed), (False, []))

    print(f"\n{'='*70}")
    print(f"  {run_type.upper()}  {encoding_name} / {distribution_name} / seed={seed}{rep_tag}")
    if run_type in ("federated", "replication") and distribution_name != "iid" and geo_flag[0]:
        print(f"  [MN-08 GeoFlag] {geo_flag[1]}")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = Phase16Logger(run_type, encoding_name, distribution_name,
                           logger_alpha, seed)

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"Run exceeded {RUN_TIMEOUT}s")

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=ENCODING_FNS[encoding_name],
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS,   n_layers=N_LAYERS,
            strategy_name="fedavg",
            seed=seed,
            project_root=PROJECT_ROOT,
            logger=logger,
            n_global_explain=N_GLOBAL_EXPLAIN,
            n_global_bg=N_GLOBAL_BG,
            n_explain=N_CLIENT_EXPLAIN,
            n_bg=N_CLIENT_BG,
            n_permutations=N_PERM,
            **kw,
        )
        signal.alarm(0)
        mark_done(rk)
    except TimeoutError as e:
        signal.alarm(0)
        print(f"\n  [TIMEOUT] {rk}: {e}")
        mark_done(rk, status="timeout")
        cleanup_ray()
    except Exception as e:
        signal.alarm(0)
        print(f"\n  [ERROR] {rk}: {e}")
        raise


def run_centralized(encoding_name, seed):
    rk   = make_run_key("centralized", encoding_name, "iid", seed)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  CENTRALIZED  {encoding_name} / seed={seed}  ({CENTRAL_EPOCHS} epochs)")
    print(f"{'='*70}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    pca    = PCA(n_components=N_QUBITS, random_state=seed)
    X_pca  = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit(X_pca)

    model         = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        encoding_fn=ENCODING_FNS[encoding_name],
    )
    model.pca     = pca
    model._scaler = scaler
    model.fit(X_train, y_train, epochs=CENTRAL_EPOCHS, lr=0.05, batch_size=32)

    acc   = float((model.predict(X_test) == y_test).mean())
    track = "Track A" if acc > collapse_threshold else "Track B"
    print(f"  Centralized test accuracy: {acc:.4f}  ({track})")

    X_test_q     = model._preprocess(X_test[:N_GLOBAL_EXPLAIN])
    X_bg_q       = model._preprocess(X_train[:N_GLOBAL_BG])
    q_predict    = make_quantum_predict_fn(model)
    qshap        = q_shap_values(q_predict, X_test_q, X_bg_q,
                                 n_permutations=N_PERM, seed=seed)
    global_qshap = qshap.mean(axis=0)
    print(f"  Centralized ‖φ‖: {np.linalg.norm(global_qshap):.4f}")

    append_row("centralized", encoding_name, "iid", None, seed,
               N_ROUNDS, "global", global_qshap, acc)
    mark_done(rk)


# ── compute per-encoding IID_std from Phase A outputs ────────────────────────

def compute_live_iid_stds():
    done        = load_checkpoint()
    iid_accs    = {enc: {} for enc in ENCODINGS}

    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                enc = row.get("encoding", "")
                if enc not in ENCODINGS:
                    continue
                rk = make_run_key("federated", enc, "iid", int(row["seed"]))
                if (rk in done
                        and row["run_type"] == "federated"
                        and row["distribution"] == "iid"
                        and row["client_id"] == "global"
                        and int(row["round"]) == N_ROUNDS):
                    iid_accs[enc][int(row["seed"])] = float(row["accuracy"])

    stds = {}
    print("Per-encoding Phase A IID summary:")
    for enc in ENCODINGS:
        accs = iid_accs[enc]
        if len(accs) >= 2:
            vals      = list(accs.values())
            mean      = float(np.mean(vals))
            std       = float(np.std(vals, ddof=1))
            stds[enc] = std
            source    = "LIVE"
        else:
            std       = BOOTSTRAP_IID_STD[enc]
            stds[enc] = std
            source    = f"BOOTSTRAP (only {len(accs)} Phase A seed(s))"
            mean      = float(np.mean(list(accs.values()))) if accs else 0.0

        acc_str = "  ".join(f"s{s}={a:.4f}" for s, a in sorted(accs.items()))
        print(f"  {enc:<12}  mean={mean:.4f}  std={std:.4f}  thr2σ={2*std:.4f}  "
              f"[{source}]")
        if acc_str:
            print(f"               {acc_str}")
    print()
    return stds


# ── MN-08 replication block per (encoding, distribution) ─────────────────────

def check_and_replicate(enc, dist, encoding_iid_std):
    replcheck_key = make_replcheck_key(enc, dist)
    done          = load_checkpoint()
    if replcheck_key in done:
        print(f"  [skip replcheck] {replcheck_key}")
        return

    # Gather Phase B R15 global accuracies
    phase_b_accs = {}
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, newline="") as fh:
            for row in csv.DictReader(fh):
                rk = make_run_key("federated", enc, dist, int(row["seed"]))
                if (rk in done
                        and row["run_type"] == "federated"
                        and row["encoding"] == enc
                        and row["distribution"] == dist
                        and row["client_id"] == "global"
                        and int(row["round"]) == N_ROUNDS):
                    phase_b_accs[int(row["seed"])] = float(row["accuracy"])

    print(f"\n  Phase B {enc}/{dist} final accuracies:")
    for s, a in sorted(phase_b_accs.items()):
        t = "A" if a > collapse_threshold else "B"
        print(f"    seed={s}  acc={a:.4f}  {t}")

    # Trigger G: geometry-flagged seeds (shared across encodings for same dist)
    geo_seeds = {s for s in SEEDS if GEOMETRY_FLAGS.get((dist, s), (False,))[0]}

    # Trigger S: seeds deviating > 2σ from the encoding's condition mean
    trig_s_thr = 2.0 * encoding_iid_std
    stat_seeds  = set()
    if len(phase_b_accs) >= 2:
        b_mean = float(np.mean(list(phase_b_accs.values())))
        for seed, acc in phase_b_accs.items():
            dev = abs(acc - b_mean)
            if dev > trig_s_thr:
                print(f"  [MN-08 StatFlag] {enc}/{dist}/seed={seed}: "
                      f"acc={acc:.4f}  b_mean={b_mean:.4f}  "
                      f"dev={dev*100:.2f}pp > thr={trig_s_thr*100:.2f}pp")
                stat_seeds.add(seed)

    to_replicate = (geo_seeds | stat_seeds) - timed_out_seeds(enc, dist)

    print(f"\n  MN-08 summary  {enc}/{dist}:")
    print(f"    Geometry-flagged: {sorted(geo_seeds) or 'none'}")
    print(f"    Statistical:      {sorted(stat_seeds) or 'none'}")
    print(f"    To replicate:     {sorted(to_replicate) or 'none'}")

    if to_replicate:
        alpha = DIST_KWARGS[dist].get("dirichlet_alpha") if DIST_KWARGS[dist]["split"] == "dirichlet" else None
        print(f"    Running {N_REPLICATION_REPS} reps per flagged seed …")
        for seed in sorted(to_replicate):
            for rep in range(1, N_REPLICATION_REPS + 1):
                run_federated(enc, dist, seed, run_type="replication", rep=rep)
    else:
        print(f"    No replication triggered.")

    mark_done(replcheck_key)


# ── Phase A: IID baseline ─────────────────────────────────────────────────────

print("=" * 70)
print("PHASE A: IID baseline — 4 encodings × 5 seeds (federated + centralized)")
print("=" * 70)
init_results_csv()

for enc in ENCODINGS:
    print(f"\n  ── {enc} ──")
    for seed in SEEDS:
        run_federated(enc, "iid", seed)
        run_centralized(enc, seed)

# ── compute IID_stds after Phase A ───────────────────────────────────────────

print("=" * 70)
print("COMPUTING PER-ENCODING IID_STD FROM PHASE A")
print("=" * 70)
ENCODING_IID_STD = compute_live_iid_stds()

# ── Phase B: non-IID sweep ────────────────────────────────────────────────────

print("=" * 70)
print("PHASE B: non-IID sweep — 4 encodings × 3 distributions × 5 seeds")
print("=" * 70)

for enc in ENCODINGS:
    for dist in NON_IID_DISTS:
        alpha = (DIST_KWARGS[dist]["dirichlet_alpha"]
                 if DIST_KWARGS[dist]["split"] == "dirichlet" else None)
        print(f"\n  ══ {enc} / {dist} ══")
        for seed in SEEDS:
            run_federated(enc, dist, seed)
        check_and_replicate(enc, dist, ENCODING_IID_STD[enc])

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'='*70}")
print("PHASE 16 SUMMARY")
print(f"{'='*70}")

done_final = load_checkpoint()

total_rows = 0
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline="") as fh:
        total_rows = sum(1 for _ in csv.DictReader(fh))

N_ENC  = len(ENCODINGS)
N_SEED = len(SEEDS)
N_DIST = len(NON_IID_DISTS)

phase_a_fed  = sum(make_run_key("federated",   e, "iid", s) in done_final
                   for e in ENCODINGS for s in SEEDS)
phase_a_cent = sum(make_run_key("centralized", e, "iid", s) in done_final
                   for e in ENCODINGS for s in SEEDS)
phase_b_prim = sum(make_run_key("federated",   e, d, s) in done_final
                   for e in ENCODINGS for d in NON_IID_DISTS for s in SEEDS)
replchecks   = sum(make_replcheck_key(e, d) in done_final
                   for e in ENCODINGS for d in NON_IID_DISTS)
n_reps       = sum(1 for rk in done_final if rk.startswith("replication|"))

print(f"  Phase A federated:   {phase_a_fed:3d} / {N_ENC * N_SEED}")
print(f"  Phase A centralized: {phase_a_cent:3d} / {N_ENC * N_SEED}")
print(f"  Phase B primaries:   {phase_b_prim:3d} / {N_ENC * N_DIST * N_SEED}")
print(f"  Replication runs:    {n_reps}")
print(f"  Replcheck blocks:    {replchecks:3d} / {N_ENC * N_DIST}")
print(f"  Total CSV rows:      {total_rows}")

all_done = (phase_a_fed  == N_ENC * N_SEED
            and phase_a_cent == N_ENC * N_SEED
            and phase_b_prim == N_ENC * N_DIST * N_SEED
            and replchecks   == N_ENC * N_DIST)

print(f"\n  Sweep status: {'COMPLETE' if all_done else 'INCOMPLETE — see counts above'}")
print()

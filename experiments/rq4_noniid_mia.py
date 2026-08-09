"""RQ4 non-IID MIA: does collapse affect membership leakage?

Phase 13 tested MIA only under IID conditions. This script tests whether
Track-B (collapsed) models leak more, less, or the same as Track-A (functional)
models under non-IID label skew.

Scientific question (from project Limitations section):
  "Testing whether collapsed models leak more or less than functional ones is
   scientifically interesting and would round out RQ4."

Hypothesis space:
  - Track B leaks MORE: collapsed constant-output may expose membership signal
    through the explanation of why a specific sample gets the default output
  - Track B leaks LESS: near-zero ‖φ‖ means the explanation carries no
    information → MIA has nothing to attack
  - Track B leaks SIMILARLY: leakage comes from model memorisation regardless
    of collapse state

Scope
-----
Distributions selected to maximise Track A/B contrast:
  α=0.1   — highest Track B rate; angle 4B/5, iqp 3B/5, reuploading 1B/5,
             amplitude 0B/5 (all A)
  qty_skew — all Track A across all encodings; provides functional non-IID baseline

Encodings: all 4 (amplitude, angle, reuploading, iqp)
Seeds: original 5 [1, 7, 42, 123, 999]

Protocol per (encoding, distribution, seed):
  1. Retrain federated model (same split/alpha/seed as Phase 10/11)
  2. Record actual Track A/B from retrained model (actor-RNG → may differ from log)
  3. Compute per-sample Q-SHAP for 50 members + 50 non-members
  4. Run direct MIA (cv=5 LR)

Note: retrain may produce different Track A/B than Phase 10/11 due to actor-RNG.
The logged track_class reflects the RETRAINED model, not Phase 10/11.

Outputs:
  results/rq4_noniid_mia_log.csv
  results/rq4_noniid_mia_checkpoint.csv
"""

import os, sys, csv, json, signal, time, statistics
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import (angle_encoding, amplitude_encoding,
                               reuploading_encoding, iqp_encoding)
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import run_mia_attack, _qshap_for_samples

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS       = 4
N_LAYERS       = 2
N_CLIENTS      = 4
N_ROUNDS       = 15
LOCAL_EPOCHS   = 3
N_MIA_SAMPLES  = 50
N_BG           = 20
N_PERMUTATIONS = 8
COLLAPSE_MARGIN = 0.02
RUN_TIMEOUT    = 600

SEEDS = [1, 7, 42, 123, 999]

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

# Distributions chosen to give Track A/B contrast
DISTRIBUTIONS = {
    "alpha_0.1": {"split": "dirichlet", "alpha": 0.1},
    "qty_skew":  {"split": "quantity",  "alpha": None},
}

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq4_noniid_mia_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq4_noniid_mia_checkpoint.csv")
PHASE13_CSV    = os.path.join(PROJECT_ROOT, "results", "phase13_mia_log.csv")

RESULTS_COLS = [
    "encoding", "distribution", "seed",
    "target_acc", "track_class", "margin",
    "direct_mia_auc",
    "n_members", "n_nonmembers", "wall_time_s", "timestamp",
]

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_run_key(enc, dist, seed):
    return f"{enc}|{dist}|{seed}"


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


def append_result(enc, dist, seed, target_acc, track, margin, auc, n_mem, n_nonmem, wall_time):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if first:
            w.writeheader()
        w.writerow({
            "encoding":       enc,
            "distribution":   dist,
            "seed":           seed,
            "target_acc":     f"{target_acc:.6f}",
            "track_class":    track,
            "margin":         f"{margin:.6f}",
            "direct_mia_auc": f"{auc:.6f}",
            "n_members":      n_mem,
            "n_nonmembers":   n_nonmem,
            "wall_time_s":    f"{wall_time:.1f}",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })


# ── timeout / Ray cleanup ─────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("Federated training timed out")


def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


# ── majority baseline (compute once from data) ────────────────────────────────

_X_train, _X_test, _y_train, _y_test = load_breast_cancer_data(random_state=42)
_majority_baseline  = float(np.bincount(_y_test.astype(int)).max()) / len(_y_test)
_collapse_threshold = _majority_baseline + COLLAPSE_MARGIN
print(f"Wisconsin collapse threshold: {_collapse_threshold:.4f}")


# ── run one (encoding, distribution, seed) ────────────────────────────────────

def run_one(enc_name, dist_name, seed):
    rk   = make_run_key(enc_name, dist_name, seed)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    dist_cfg = DISTRIBUTIONS[dist_name]

    print(f"\n{'='*70}")
    print(f"  NONIID-MIA  {enc_name} / {dist_name} / seed={seed}")
    print(f"{'='*70}")

    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)

    # 1. Retrain federated model (same parameters as Phase 10/11)
    model_out = []
    kw = {}
    if dist_cfg["split"] == "dirichlet":
        kw["dirichlet_alpha"] = dist_cfg["alpha"]
    else:
        kw["dirichlet_alpha"] = 0.5   # ignored for quantity split

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=ENCODINGS[enc_name],
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS, n_layers=N_LAYERS,
            split=dist_cfg["split"],
            strategy_name="fedavg",
            seed=seed,
            project_root=PROJECT_ROOT,
            n_permutations=0,
            model_out=model_out,
            **kw,
        )
        signal.alarm(0)
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  [TIMEOUT] {e}")
        cleanup_ray()
        mark_done(rk, status="timeout")
        return
    except Exception as e:
        signal.alarm(0)
        print(f"  [ERROR] {e}")
        cleanup_ray()
        raise

    final_acc = round_log[-1][1] if round_log else float("nan")
    track  = "A" if final_acc > _collapse_threshold else "B"
    margin = final_acc - _collapse_threshold
    print(f"  Federated R15 acc={final_acc:.4f}  Track {track}  "
          f"(margin={margin:+.4f})  ({time.time()-t0:.0f}s)")

    # 2. Reconstruct target model
    shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
    target_model = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ENCODINGS[enc_name]
    )
    target_model.pca      = shared_pca
    target_model._scaler  = shared_scaler
    target_model.set_weights(model_out[0])
    target_acc = float((target_model.predict(X_test) == y_test).mean())

    # 3. Q-SHAP for members + non-members
    print(f"  Q-SHAP members ({N_MIA_SAMPLES}) …")
    exps_in  = _qshap_for_samples(target_model, X_train,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=seed)
    print(f"  Q-SHAP non-members ({N_MIA_SAMPLES}) …")
    exps_out = _qshap_for_samples(target_model, X_test,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=seed)

    # 4. Direct MIA
    auc = run_mia_attack(exps_in, exps_out, seed=seed, cv=5)
    print(f"  Direct MIA AUC: {auc:.4f}  (Track {track})")

    wall_time = time.time() - t0
    append_result(enc_name, dist_name, seed, target_acc, track, margin, auc,
                  len(exps_in), len(exps_out), wall_time)
    mark_done(rk)
    print(f"  [DONE]  {wall_time:.0f}s  ({wall_time/60:.1f} min)")


# ── analysis: Track A vs B MIA comparison ────────────────────────────────────

def print_analysis():
    print("\n" + "=" * 72)
    print("  RQ4 NON-IID MIA — Track A vs Track B Leakage Comparison")
    print("=" * 72)

    if not os.path.exists(RESULTS_CSV):
        print("  No results yet.")
        return

    rows = list(csv.DictReader(open(RESULTS_CSV)))

    # IID baseline from Phase 13
    iid_aucs = {}
    if os.path.exists(PHASE13_CSV):
        p13 = list(csv.DictReader(open(PHASE13_CSV)))
        for enc in ENCODINGS:
            aucs = [float(r["direct_mia_auc"]) for r in p13 if r["encoding"]==enc]
            iid_aucs[enc] = statistics.mean(aucs) if aucs else float("nan")

    print(f"\n{'Encoding':<12}  {'IID (ref)':>10}  {'α=0.1 TrackA':>14}  "
          f"{'α=0.1 TrackB':>14}  {'qty TrackA':>12}")
    print("-" * 70)

    for enc in ENCODINGS:
        iid_mean = iid_aucs.get(enc, float("nan"))

        def get_aucs(dist, track):
            return [float(r["direct_mia_auc"]) for r in rows
                    if r["encoding"]==enc and r["distribution"]==dist
                    and r["track_class"]==track]

        def fmt(aucs):
            if not aucs:
                return "      —"
            mean = statistics.mean(aucs)
            std  = statistics.stdev(aucs) if len(aucs)>1 else 0
            return f"{mean:.3f}±{std:.3f}(n={len(aucs)})"

        a01_a = get_aucs("alpha_0.1", "A")
        a01_b = get_aucs("alpha_0.1", "B")
        qty_a = get_aucs("qty_skew",  "A")

        print(f"  {enc:<12}  {iid_mean:>10.3f}  {fmt(a01_a):>14}  "
              f"{fmt(a01_b):>14}  {fmt(qty_a):>12}")

    # Summary: Track A vs B direction per encoding × distribution
    print("\nTrack A vs Track B MIA comparison (positive = Track B leaks MORE):")
    for dist in DISTRIBUTIONS:
        for enc in ENCODINGS:
            track_a = [float(r["direct_mia_auc"]) for r in rows
                       if r["encoding"]==enc and r["distribution"]==dist
                       and r["track_class"]=="A"]
            track_b = [float(r["direct_mia_auc"]) for r in rows
                       if r["encoding"]==enc and r["distribution"]==dist
                       and r["track_class"]=="B"]
            if track_a and track_b:
                diff = statistics.mean(track_b) - statistics.mean(track_a)
                direction = "B>A (B leaks more)" if diff > 0 else "A>B (B leaks less)"
                print(f"  {enc}/{dist}:  Track A mean={statistics.mean(track_a):.3f}  "
                      f"Track B mean={statistics.mean(track_b):.3f}  "
                      f"Δ={diff:+.3f}  → {direction}")


# ── main ──────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

print("\n" + "=" * 72)
print("  RQ4 NON-IID MIA SWEEP")
print(f"  Encodings: {list(ENCODINGS.keys())}")
print(f"  Distributions: {list(DISTRIBUTIONS.keys())}")
print(f"  Seeds: {SEEDS}")
print(f"  Total runs: {len(ENCODINGS) * len(DISTRIBUTIONS) * len(SEEDS)}")
print("=" * 72 + "\n")

for enc_name in ENCODINGS:
    for dist_name in DISTRIBUTIONS:
        for seed in SEEDS:
            run_one(enc_name, dist_name, seed)

print_analysis()

"""MN-08 replication protocol for two seed=7 MIA anomalies flagged in RQ4 nonIID sweep.

Flagged conditions:
  A. IQP / α=0.1    / seed=7  — primary draw AUC=0.964 (Track A, acc=0.693)
     Concern: seed=7 is MN-09 bimodally unstable for IQP across all distributions.
     Single-draw AUC=0.964 may not reproduce.

  B. reuploading / qty_skew / seed=7  — primary draw AUC=0.922 (Track A, acc=0.877)
     Concern: seed=7 extreme for reuploading/qty_skew; reuploading/seed=7 also had
     AUC=0.930 under IID (Phase 13), but qty_skew is a different model retrain.

Protocol: 2 additional replication draws per condition (total 3 draws including primary).
Same protocol as rq4_noniid_mia.py — federated retrain + Q-SHAP 50 mem/50 non-mem +
direct MIA cv=5 LR.

Outputs:
  results/rq4_seed7_replication_log.csv
  results/rq4_seed7_replication_checkpoint.csv
"""

import os, sys, csv, signal, time
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import iqp_encoding, reuploading_encoding
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import run_mia_attack, _qshap_for_samples

# ── constants ────────────────────────────────────────────────────────────────

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
SEED           = 7
N_REPS         = 2   # primary already done; 2 more reps → 3 total

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq4_seed7_replication_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq4_seed7_replication_checkpoint.csv")

RESULTS_COLS = [
    "anomaly_id", "encoding", "distribution", "seed", "draw_num", "run_type",
    "target_acc", "track_class", "margin",
    "direct_mia_auc", "n_members", "n_nonmembers", "wall_time_s", "timestamp",
]

# Primary draw results (from rq4_noniid_mia_log.csv)
PRIMARIES = {
    "A": {"encoding": "iqp",        "dist": "alpha_0.1", "enc_fn": iqp_encoding,
          "split": "dirichlet", "alpha": 0.1,
          "primary_acc": 0.692982, "primary_auc": 0.964000, "primary_track": "A"},
    "B": {"encoding": "reuploading", "dist": "qty_skew",  "enc_fn": reuploading_encoding,
          "split": "quantity",  "alpha": None,
          "primary_acc": 0.877193, "primary_auc": 0.922000, "primary_track": "A"},
}

# ── helpers ──────────────────────────────────────────────────────────────────

def make_key(anomaly_id, rep):
    return f"rep|{anomaly_id}|rep{rep}"


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
        w.writerow({"run_key": rk, "status": "done",
                    "completed_at": datetime.now(timezone.utc).isoformat()})
    print(f"  [checkpoint] done: {rk}")


def append_result(anomaly_id, enc, dist, draw_num, run_type,
                  target_acc, track, margin, auc, n_mem, n_nonmem, wall_time):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if first:
            w.writeheader()
        w.writerow({
            "anomaly_id":     anomaly_id,
            "encoding":       enc,
            "distribution":   dist,
            "seed":           SEED,
            "draw_num":       draw_num,
            "run_type":       run_type,
            "target_acc":     f"{target_acc:.6f}",
            "track_class":    track,
            "margin":         f"{margin:.6f}",
            "direct_mia_auc": f"{auc:.6f}",
            "n_members":      n_mem,
            "n_nonmembers":   n_nonmem,
            "wall_time_s":    f"{wall_time:.1f}",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })


def _timeout_handler(signum, frame):
    raise TimeoutError("Federated training timed out")


def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


# ── majority class threshold ─────────────────────────────────────────────────

_X_train, _X_test, _y_train, _y_test = load_breast_cancer_data(random_state=42)
_majority_baseline  = float(np.bincount(_y_test.astype(int)).max()) / len(_y_test)
_collapse_threshold = _majority_baseline + COLLAPSE_MARGIN
print(f"Wisconsin collapse threshold: {_collapse_threshold:.4f}")


# ── run one replication draw ─────────────────────────────────────────────────

def run_rep(anomaly_id, rep_num):
    cfg = PRIMARIES[anomaly_id]
    rk  = make_key(anomaly_id, rep_num)
    done = load_checkpoint()
    if rk in done:
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  SEED=7 REP  anomaly={anomaly_id}  {cfg['encoding']}/{cfg['dist']}  draw={rep_num+1}")
    print(f"  Primary: acc={cfg['primary_acc']:.4f}  AUC={cfg['primary_auc']:.4f}  Track {cfg['primary_track']}")
    print(f"{'='*70}")

    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

    model_out = []
    kw = {}
    if cfg["split"] == "dirichlet":
        kw["dirichlet_alpha"] = cfg["alpha"]
    else:
        kw["dirichlet_alpha"] = 0.5  # ignored for quantity

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=cfg["enc_fn"],
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS, n_layers=N_LAYERS,
            split=cfg["split"],
            strategy_name="fedavg",
            seed=SEED,
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
        return
    except Exception as e:
        signal.alarm(0)
        print(f"  [ERROR] {e}")
        cleanup_ray()
        raise

    final_acc = round_log[-1][1] if round_log else float("nan")
    track  = "A" if final_acc > _collapse_threshold else "B"
    margin = final_acc - _collapse_threshold
    print(f"  R15 acc={final_acc:.4f}  Track {track}  margin={margin:+.4f}  ({time.time()-t0:.0f}s)")

    shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
    target_model = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=cfg["enc_fn"]
    )
    target_model.pca      = shared_pca
    target_model._scaler  = shared_scaler
    target_model.set_weights(model_out[0])
    target_acc = float((target_model.predict(X_test) == y_test).mean())

    print(f"  Q-SHAP members ({N_MIA_SAMPLES}) …")
    exps_in  = _qshap_for_samples(target_model, X_train,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=SEED)
    print(f"  Q-SHAP non-members ({N_MIA_SAMPLES}) …")
    exps_out = _qshap_for_samples(target_model, X_test,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=SEED)

    auc = run_mia_attack(exps_in, exps_out, seed=SEED, cv=5)
    wall_time = time.time() - t0
    print(f"  Direct MIA AUC: {auc:.4f}  ({wall_time/60:.1f} min)")

    append_result(anomaly_id, cfg["encoding"], cfg["dist"], rep_num + 1, "replication",
                  target_acc, track, margin, auc, len(exps_in), len(exps_out), wall_time)
    mark_done(rk)


# ── main ─────────────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("  RQ4 SEED=7 MIA ANOMALY REPLICATION — MN-08 protocol")
print("  Flagged: IQP/α=0.1 (AUC=0.964) and reuploading/qty_skew (AUC=0.922)")
print("  2 replication draws per condition → 3 draws total including primary")
print("="*72 + "\n")

# Write primary draws to output first (draw_num=1 = the original sweep run)
done = load_checkpoint()
for aid, cfg in PRIMARIES.items():
    pk = f"primary|{aid}"
    if pk not in done:
        append_result(aid, cfg["encoding"], cfg["dist"], 1, "primary",
                      cfg["primary_acc"], cfg["primary_track"],
                      cfg["primary_acc"] - _collapse_threshold,
                      cfg["primary_auc"], N_MIA_SAMPLES, N_MIA_SAMPLES, 0.0)
        mark_done(pk)

# Run 2 replication draws for each anomaly
for aid in ["A", "B"]:
    for rep in [2, 3]:
        run_rep(aid, rep)

# ── verdict summary ───────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  VERDICTS")
print("="*72)

rows = list(csv.DictReader(open(RESULTS_CSV))) if os.path.exists(RESULTS_CSV) else []
for aid in ["A", "B"]:
    cfg = PRIMARIES[aid]
    enc = cfg["encoding"]; dist = cfg["dist"]
    draws = [r for r in rows if r["anomaly_id"]==aid]
    if not draws:
        continue
    aucs = [float(r["direct_mia_auc"]) for r in draws]
    tracks = [r["track_class"] for r in draws]
    accs = [float(r["target_acc"]) for r in draws]
    print(f"\nAnomaly {aid} — {enc}/{dist}/seed={SEED}")
    print(f"  Primary flag: AUC={cfg['primary_auc']:.3f}  Track {cfg['primary_track']}")
    print(f"  {'Draw':<6}  {'Type':<12}  {'Acc':>8}  {'Track':>6}  {'AUC':>7}")
    for r in draws:
        print(f"  {r['draw_num']:<6}  {r['run_type']:<12}  "
              f"{float(r['target_acc']):>8.4f}  {r['track_class']:>6}  "
              f"{float(r['direct_mia_auc']):>7.4f}")
    high = sum(1 for a in aucs if a > 0.80)
    import statistics as _st
    print(f"  Draws with AUC>0.80: {high}/{len(aucs)}  Mean={_st.mean(aucs):.3f}")
    if high >= 2:
        print(f"  VERDICT: Elevation REPRODUCES — {high}/{len(aucs)} draws >0.80. Genuine.")
    elif high == 1:
        print(f"  VERDICT: Elevation does NOT consistently reproduce — 1/{len(aucs)} draws >0.80.")
    else:
        print(f"  VERDICT: Elevation does NOT reproduce — 0/{len(aucs)} draws >0.80. Single-draw fluke.")

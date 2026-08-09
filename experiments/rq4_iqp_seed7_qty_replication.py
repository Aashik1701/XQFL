"""MN-08 replication: IQP/seed=7/qty_skew single-draw AUC=0.864.

Primary draw (rq4_noniid_mia_log.csv): acc=0.825, Track A, AUC=0.864.
IQP/seed=7 is MN-09 bimodally unstable — confirmed under α=0.1 (Track A: 0.87-0.96,
Track B: 0.71 across 3 draws). No basis for assuming qty_skew is stable from a single
draw. Need 2 replication draws → 3 total.

Protocol: identical to rq4_noniid_mia.py (federated retrain qty_skew + Q-SHAP 50/50
members/non-members + direct MIA cv=5 LR).

Outputs:
  results/rq4_iqp_seed7_qty_replication_log.csv
  results/rq4_iqp_seed7_qty_replication_checkpoint.csv
"""

import os, sys, csv, signal, time, statistics
import numpy as np
import torch
from datetime import datetime, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import iqp_encoding
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import run_mia_attack, _qshap_for_samples

N_QUBITS        = 4
N_LAYERS        = 2
N_CLIENTS       = 4
N_ROUNDS        = 15
LOCAL_EPOCHS    = 3
N_MIA_SAMPLES   = 50
N_BG            = 20
N_PERMUTATIONS  = 8
COLLAPSE_MARGIN = 0.02
RUN_TIMEOUT     = 600
SEED            = 7

PRIMARY_ACC     = 0.824561
PRIMARY_AUC     = 0.864000
PRIMARY_TRACK   = "A"

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq4_iqp_seed7_qty_replication_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq4_iqp_seed7_qty_replication_checkpoint.csv")

COLS = ["draw_num", "run_type", "target_acc", "track_class", "margin",
        "direct_mia_auc", "n_members", "n_nonmembers", "wall_time_s", "timestamp"]

_X_train, _X_test, _y_train, _y_test = load_breast_cancer_data(random_state=42)
_majority_baseline  = float(np.bincount(_y_test.astype(int)).max()) / len(_y_test)
_collapse_threshold = _majority_baseline + COLLAPSE_MARGIN
print(f"Wisconsin collapse threshold: {_collapse_threshold:.4f}")


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


def append_result(draw_num, run_type, target_acc, track, margin, auc, wall_time):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if first:
            w.writeheader()
        w.writerow({
            "draw_num":       draw_num,
            "run_type":       run_type,
            "target_acc":     f"{target_acc:.6f}",
            "track_class":    track,
            "margin":         f"{margin:.6f}",
            "direct_mia_auc": f"{auc:.6f}",
            "n_members":      N_MIA_SAMPLES,
            "n_nonmembers":   N_MIA_SAMPLES,
            "wall_time_s":    f"{wall_time:.1f}",
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        })


def _timeout_handler(signum, frame):
    raise TimeoutError("timed out")


def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass


def run_rep(draw_num):
    rk = f"rep|iqp|qty_skew|7|draw{draw_num}"
    if rk in load_checkpoint():
        print(f"  [skip] {rk}")
        return

    print(f"\n{'='*70}")
    print(f"  IQP / qty_skew / seed=7  draw={draw_num}")
    print(f"  Primary: acc={PRIMARY_ACC:.4f}  AUC={PRIMARY_AUC:.4f}  Track {PRIMARY_TRACK}")
    print(f"{'='*70}")

    t0 = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

    model_out = []
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=iqp_encoding,
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS, n_layers=N_LAYERS,
            split="quantity",
            strategy_name="fedavg",
            seed=SEED,
            project_root=PROJECT_ROOT,
            n_permutations=0,
            model_out=model_out,
            dirichlet_alpha=0.5,   # ignored for quantity split
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
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=iqp_encoding
    )
    target_model.pca     = shared_pca
    target_model._scaler = shared_scaler
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

    append_result(draw_num, "replication", target_acc, track, margin, auc, wall_time)
    mark_done(rk)


# ── main ─────────────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("  IQP/seed=7/qty_skew — MN-08 REPLICATION")
print(f"  Primary: AUC={PRIMARY_AUC}  Track {PRIMARY_TRACK}  acc={PRIMARY_ACC:.4f}")
print("  Running 2 replication draws (→ 3 total including primary)")
print("="*72 + "\n")

# Write primary as draw 1
done = load_checkpoint()
if "primary|iqp|qty_skew|7" not in done:
    append_result(1, "primary", PRIMARY_ACC, PRIMARY_TRACK,
                  PRIMARY_ACC - _collapse_threshold, PRIMARY_AUC, 0.0)
    mark_done("primary|iqp|qty_skew|7")

for draw in [2, 3]:
    run_rep(draw)

# ── verdict ───────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  VERDICT — IQP/seed=7/qty_skew")
print("="*72)

rows = list(csv.DictReader(open(RESULTS_CSV))) if os.path.exists(RESULTS_CSV) else []
aucs   = [float(r["direct_mia_auc"]) for r in rows]
tracks = [r["track_class"] for r in rows]
accs   = [float(r["target_acc"]) for r in rows]

print(f"\n  {'Draw':<6}  {'Type':<12}  {'Acc':>8}  {'Track':>6}  {'AUC':>7}")
for r in rows:
    print(f"  {r['draw_num']:<6}  {r['run_type']:<12}  "
          f"{float(r['target_acc']):>8.4f}  {r['track_class']:>6}  "
          f"{float(r['direct_mia_auc']):>7.4f}")

high = sum(1 for a in aucs if a > 0.80)
mean_auc = statistics.mean(aucs)
print(f"\n  Draws with AUC>0.80: {high}/{len(aucs)}")
print(f"  Mean AUC: {mean_auc:.3f}  (primary was {PRIMARY_AUC:.3f})")

if high >= 2:
    print(f"  VERDICT: Elevation REPRODUCES ({high}/{len(aucs)} draws >0.80).")
    print(f"  Use 3-draw mean {mean_auc:.3f} for IQP/qty_skew/seed=7.")
    print(f"  IQP qty_skew 5-seed corrected mean: "
          f"{(0.806 + mean_auc + 0.852 + 0.414 + 0.680)/5:.3f}")
elif high == 1:
    print(f"  VERDICT: Elevation partially reproduces (1/{len(aucs)} draws >0.80). Bimodal.")
    print(f"  Use 3-draw mean {mean_auc:.3f} for IQP/qty_skew/seed=7.")
    print(f"  IQP qty_skew 5-seed corrected mean: "
          f"{(0.806 + mean_auc + 0.852 + 0.414 + 0.680)/5:.3f}")
else:
    print(f"  VERDICT: Does NOT reproduce (0/{len(aucs)} draws >0.80). Single-draw fluke.")
    print(f"  Use 3-draw mean {mean_auc:.3f} for IQP/qty_skew/seed=7.")
    print(f"  IQP qty_skew 5-seed corrected mean: "
          f"{(0.806 + mean_auc + 0.852 + 0.414 + 0.680)/5:.3f}")

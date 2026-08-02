"""Phase 13: Cross-encoding MIA sweep.

Extends Phase 8's membership inference attack across all 4 quantum encodings,
5 seeds each (IID, Wisconsin). Reports per-encoding AUC mean±std.

Architecture note: Phase 10 IID vectors are global background-averaged Q-SHAP
(one vector per round per client), not the per-sample Q-SHAP needed for MIA.
Phase 10 model weights were not saved. This script retrains each IID model
from scratch using model_out=[] to capture weights, then computes per-sample
Q-SHAP post-training.

Protocol per (encoding, seed):
  1. Retrain IID FedAvg model (4 clients, 15 rounds, n_permutations=0)
     using model_out=[] to capture final weights without saving to disk.
  2. Reconstruct target model; verify accuracy matches federated final.
  3. Q-SHAP for 50 members (training) and 50 non-members (test).
  4. Direct MIA: cross-validated LR on target model explanations (cv=5).
  5. Shadow model MIA: n_shadow=3, 15 local epochs, shadow-CV sanity check.
  6. Log per-seed AUCs; print per-encoding mean±std at end.

Run from project root:
    python experiments/phase13_mia_sweep.py

Runtime estimate: ~2-2.5 hours for all 20 runs (amplitude slowest ~9 min/seed).
Supports resume via checkpoint file (results/phase13_mia_checkpoint.csv).
"""

import sys
import os
import signal
import time
import csv
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.loaders import load_breast_cancer_data
from models.encodings import (
    angle_encoding, amplitude_encoding, reuploading_encoding, iqp_encoding
)
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import ShadowModelMIA, run_mia_attack, _qshap_for_samples

# ── config ────────────────────────────────────────────────────────────────────
N_QUBITS       = 4
N_LAYERS       = 2
N_CLIENTS      = 4
N_ROUNDS       = 15
LOCAL_EPOCHS   = 3
N_MIA_SAMPLES  = 50      # members and non-members each
N_BG           = 20      # Q-SHAP background samples
N_PERMUTATIONS = 8       # Q-SHAP permutations
N_SHADOW       = 3       # shadow classifiers
SHADOW_EPOCHS  = 15      # shadow model training epochs
RUN_TIMEOUT    = 480     # seconds — guards federated training only; IID unlikely to hang

SEEDS = [1, 7, 42, 123, 999]   # matches Phase 10 IID seed matrix

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

RESULTS_CSV    = "results/phase13_mia_log.csv"
CHECKPOINT_CSV = "results/phase13_mia_checkpoint.csv"

RESULTS_COLS = [
    "encoding", "seed",
    "target_acc", "direct_mia_auc", "shadow_mia_auc", "shadow_cv_auc",
    "n_members", "n_nonmembers", "wall_time_s", "timestamp",
]

# ── checkpoint helpers ─────────────────────────────────────────────────────────

def _run_key(enc, seed):
    return f"{enc}|{seed}"


def load_done():
    done = set()
    if not os.path.exists(CHECKPOINT_CSV):
        return done
    with open(CHECKPOINT_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("status") == "done":
                done.add(row["run_key"])
    return done


def mark_done(rk, status="done"):
    write_header = not os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["run_key", "status", "timestamp"])
        w.writerow([rk, status, datetime.utcnow().isoformat()])


def append_result(enc, seed, target_acc, direct_auc,
                  shadow_mia_auc, shadow_cv_auc,
                  n_members, n_nonmembers, wall_time_s):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(RESULTS_COLS)
        w.writerow([
            enc, seed,
            f"{target_acc:.6f}",
            f"{direct_auc:.6f}",
            f"{shadow_mia_auc:.6f}",
            f"{shadow_cv_auc:.6f}",
            n_members, n_nonmembers,
            f"{wall_time_s:.1f}",
            datetime.utcnow().isoformat(),
        ])

# ── timeout handling ───────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("Federated training timed out")


def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

# ── main sweep ────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    done = load_done()

    total_runs = len(ENCODINGS) * len(SEEDS)
    print("=== Phase 13: Cross-Encoding MIA Sweep ===")
    print(f"  Encodings : {list(ENCODINGS.keys())}")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Total runs: {total_runs}  |  Already done: {len(done)}")

    for enc_name, enc_fn in ENCODINGS.items():
        for seed in SEEDS:
            rk = _run_key(enc_name, seed)
            if rk in done:
                print(f"\n[SKIP] {rk}")
                continue

            print(f"\n{'='*60}")
            print(f"[RUN] encoding={enc_name}  seed={seed}")
            t0 = time.time()

            np.random.seed(seed)
            torch.manual_seed(seed)

            X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)
            print(f"  Data: {len(X_train)} train / {len(X_test)} test")

            # ── Step 1: Retrain IID federated model ──────────────────────────
            print(f"  [1/4] Federated training (IID, {N_ROUNDS} rounds) ...", flush=True)
            model_out = []
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(RUN_TIMEOUT)
            try:
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
                    n_permutations=0,    # skip Q-SHAP during training
                    model_out=model_out,
                )
                signal.alarm(0)
            except TimeoutError as exc:
                print(f"  TIMEOUT: {exc}")
                cleanup_ray()
                mark_done(rk, status="timeout")
                continue
            except Exception as exc:
                signal.alarm(0)
                print(f"  ERROR during training: {type(exc).__name__}: {exc}")
                cleanup_ray()
                mark_done(rk, status=f"error:{type(exc).__name__}")
                continue

            final_fed_acc = round_log[-1][1] if round_log else float("nan")
            print(f"  Federated accuracy: {final_fed_acc:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")

            # ── Step 2: Reconstruct target model ─────────────────────────────
            shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
            target_model = HybridQuantumClassifier(
                n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=enc_fn
            )
            target_model.pca      = shared_pca
            target_model._scaler  = shared_scaler
            target_model.set_weights(model_out[0])
            target_acc = float((target_model.predict(X_test) == y_test).mean())
            print(f"  Reconstructed accuracy: {target_acc:.4f}")
            if abs(target_acc - final_fed_acc) > 0.02:
                print(f"  WARNING: accuracy mismatch ({target_acc:.4f} vs {final_fed_acc:.4f})")

            # ── Step 3: Q-SHAP for members + non-members ─────────────────────
            print(f"  [2/4] Q-SHAP members ({N_MIA_SAMPLES} samples) ...", flush=True)
            exps_in = _qshap_for_samples(
                target_model, X_train,
                n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                n_permutations=N_PERMUTATIONS, seed=seed,
            )
            print(f"  [3/4] Q-SHAP non-members ({N_MIA_SAMPLES} samples) ...", flush=True)
            exps_out = _qshap_for_samples(
                target_model, X_test,
                n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                n_permutations=N_PERMUTATIONS, seed=seed,
            )
            n_mem    = len(exps_in)
            n_nonmem = len(exps_out)
            print(f"  exps_in={exps_in.shape}  exps_out={exps_out.shape}  "
                  f"elapsed={time.time()-t0:.0f}s")

            # ── Step 4: Direct MIA ────────────────────────────────────────────
            direct_auc = run_mia_attack(exps_in, exps_out, seed=seed, cv=5)
            print(f"  Direct MIA AUC (cv=5): {direct_auc:.4f}")

            # ── Step 5: Shadow model MIA ──────────────────────────────────────
            print(f"  [4/4] Shadow model MIA (n_shadow={N_SHADOW}) ...", flush=True)

            # Capture loop variables in default args to avoid late-binding closure
            def make_factory(ef=enc_fn, pca=shared_pca, scaler=shared_scaler):
                def factory():
                    m = HybridQuantumClassifier(
                        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ef
                    )
                    m.pca     = pca
                    m._scaler = scaler
                    return m
                return factory

            mia = ShadowModelMIA(
                model_factory=make_factory(),
                n_shadow=N_SHADOW,
                local_epochs=SHADOW_EPOCHS,
                shadow_split=0.5,
                n_explain=20,
                n_bg=N_BG,
                n_permutations=N_PERMUTATIONS,
                seed=seed,
            )
            mia.generate_shadow_data(X_train, y_train)
            mia.fit_attack()
            shadow_cv_auc  = mia.shadow_cv_auc(cv=3)
            shadow_mia_auc = mia.attack_auc(exps_in, exps_out)
            print(f"  Shadow CV AUC: {shadow_cv_auc:.4f}  "
                  f"Shadow→Target AUC: {shadow_mia_auc:.4f}")

            # ── Log and checkpoint ────────────────────────────────────────────
            wall_time = time.time() - t0
            append_result(
                enc_name, seed, target_acc,
                direct_auc, shadow_mia_auc, shadow_cv_auc,
                n_mem, n_nonmem, wall_time,
            )
            mark_done(rk)
            print(f"  [DONE]  wall_time={wall_time:.0f}s  ({wall_time/60:.1f} min)")

    # ── Summary table ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== Phase 13: Per-Encoding AUC Summary ===\n")

    if not os.path.exists(RESULTS_CSV):
        print("  No results found.")
        return

    rows = []
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))

    print(f"  {'Encoding':<12}  {'Seeds':>5}  "
          f"{'Direct AUC':>14}  {'Shadow AUC':>14}  {'Shadow CV':>12}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*14}  {'-'*14}  {'-'*12}")
    for enc_name in ENCODINGS:
        enc_rows = [r for r in rows if r["encoding"] == enc_name]
        if not enc_rows:
            print(f"  {enc_name:<12}  (no results)")
            continue
        d_aucs  = [float(r["direct_mia_auc"])  for r in enc_rows]
        s_aucs  = [float(r["shadow_mia_auc"])   for r in enc_rows]
        sv_aucs = [float(r["shadow_cv_auc"])     for r in enc_rows]
        print(
            f"  {enc_name:<12}  {len(enc_rows):>5}  "
            f"{np.mean(d_aucs):.3f}±{np.std(d_aucs):.3f}       "
            f"{np.mean(s_aucs):.3f}±{np.std(s_aucs):.3f}     "
            f"{np.mean(sv_aucs):.3f}±{np.std(sv_aucs):.3f}"
        )

    # Per-seed breakdown for reference
    print(f"\n  Per-seed detail:")
    print(f"  {'Encoding':<12}  {'Seed':>5}  {'Acc':>6}  {'Direct':>7}  {'Shadow':>7}  {'ShCV':>7}  {'Time(s)':>8}")
    print(f"  {'-'*12}  {'-'*5}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}")
    for enc_name in ENCODINGS:
        for row in [r for r in rows if r["encoding"] == enc_name]:
            print(
                f"  {enc_name:<12}  {int(row['seed']):>5}  "
                f"{float(row['target_acc']):.4f}  "
                f"{float(row['direct_mia_auc']):.4f}  "
                f"{float(row['shadow_mia_auc']):.4f}  "
                f"{float(row['shadow_cv_auc']):.4f}  "
                f"{float(row['wall_time_s']):>8.0f}"
            )

    print("\n[Phase 13 complete]")


if __name__ == "__main__":
    main()

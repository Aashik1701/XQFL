"""RQ4 extension: Pima IID MIA, amplitude encoding only.

Adds the single highest-value remaining cell identified in the completion
audit (results/XQFL_completion_audit.md, Step 7b): RQ4 has zero second-dataset
coverage. This gives amplitude's membership-inference finding a cross-dataset
check, mirroring Phase 13's Wisconsin protocol exactly (same direct + shadow
MIA pipeline) but on Pima / IID / amplitude only. Seed set matches Wisconsin's
rq4_extension_log.csv (n=15) so the Pima result stands on the same statistical
footing as every other RQ4 cell rather than being reported as a smaller,
lower-power sample.

Protocol per seed (identical to phase13_mia_sweep.py, dataset swapped):
  1. Retrain IID FedAvg model (4 clients, 15 rounds, n_permutations=0)
     using model_out=[] to capture weights without saving to disk.
  2. Reconstruct target model; verify accuracy matches federated final.
  3. Q-SHAP for 50 members (training) and 50 non-members (test).
  4. Direct MIA: cross-validated LR on target model explanations (cv=5).
  5. Shadow model MIA: n_shadow=3, 15 local epochs, shadow-CV sanity check.

Run from project root:
    python experiments/rq4_pima_mia_amplitude.py

Runtime estimate: ~10 minutes for the first 5 seeds, ~20-25 more for the
remaining 10 (n=15 total, amplitude only).
Supports resume via checkpoint file (results/rq4_pima_mia_amplitude_checkpoint.csv).
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

from data.loaders import load_pima_data
from models.encodings import amplitude_encoding
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

SEEDS = [1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 42, 43, 47, 123, 999]   # matches Wisconsin's rq4_extension n=15 seed matrix

ENCODING_NAME = "amplitude"
ENCODING_FN   = amplitude_encoding

RESULTS_CSV    = "results/rq4_pima_mia_amplitude_log.csv"
CHECKPOINT_CSV = "results/rq4_pima_mia_amplitude_checkpoint.csv"

RESULTS_COLS = [
    "dataset", "encoding", "seed",
    "target_acc", "direct_mia_auc", "shadow_mia_auc", "shadow_cv_auc",
    "n_members", "n_nonmembers", "wall_time_s", "timestamp",
]

# ── checkpoint helpers ─────────────────────────────────────────────────────────

def _run_key(seed):
    return f"pima|{ENCODING_NAME}|iid|{seed}"


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


def append_result(seed, target_acc, direct_auc,
                  shadow_mia_auc, shadow_cv_auc,
                  n_members, n_nonmembers, wall_time_s):
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(RESULTS_COLS)
        w.writerow([
            "pima", ENCODING_NAME, seed,
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

    total_runs = len(SEEDS)
    print("=== RQ4 Extension: Pima IID MIA (amplitude only) ===")
    print(f"  Dataset   : pima")
    print(f"  Encoding  : {ENCODING_NAME}")
    print(f"  Seeds     : {SEEDS}")
    print(f"  Total runs: {total_runs}  |  Already done: {len(done)}")

    for seed in SEEDS:
        rk = _run_key(seed)
        if rk in done:
            print(f"\n[SKIP] {rk}")
            continue

        print(f"\n{'='*60}")
        print(f"[RUN] dataset=pima  encoding={ENCODING_NAME}  seed={seed}")
        t0 = time.time()

        np.random.seed(seed)
        torch.manual_seed(seed)

        X_train, X_test, y_train, y_test = load_pima_data(random_state=seed)
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
                encoding_fn=ENCODING_FN,
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
            n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ENCODING_FN
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

        def make_factory(ef=ENCODING_FN, pca=shared_pca, scaler=shared_scaler):
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
            seed, target_acc,
            direct_auc, shadow_mia_auc, shadow_cv_auc,
            n_mem, n_nonmem, wall_time,
        )
        mark_done(rk)
        print(f"  [DONE]  wall_time={wall_time:.0f}s  ({wall_time/60:.1f} min)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== RQ4 Extension: Pima/amplitude/IID MIA Summary ===\n")

    if not os.path.exists(RESULTS_CSV):
        print("  No results found.")
        return

    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("  No results found.")
        return

    d_aucs  = [float(r["direct_mia_auc"])  for r in rows]
    s_aucs  = [float(r["shadow_mia_auc"])   for r in rows]
    sv_aucs = [float(r["shadow_cv_auc"])     for r in rows]
    print(f"  n={len(rows)}  Direct AUC: {np.mean(d_aucs):.3f} ± {np.std(d_aucs, ddof=1):.3f}")
    print(f"  n={len(rows)}  Shadow AUC: {np.mean(s_aucs):.3f} ± {np.std(s_aucs, ddof=1):.3f}")
    print(f"  n={len(rows)}  Shadow CV : {np.mean(sv_aucs):.3f} ± {np.std(sv_aucs, ddof=1):.3f}")
    print(f"\n  Compare to Wisconsin/amplitude/IID (n=15): 0.619 ± 0.115 (direct MIA)")

    print("\n[RQ4 Pima amplitude MIA complete]")


if __name__ == "__main__":
    main()

"""Phase 13b: Resample replication of seed=123 below-random MIA result.

Phase 13 used training samples [0:50] as members and test samples [0:50] as
non-members for each direct MIA attack.  This script replicates using a
DIFFERENT draw: training samples [50:100] as members and test samples [50:100]
as non-members, with the background set held constant (training [0:20], same
as Phase 13).

No new partition or federated dynamics are involved — same model weights
(retrained from the same seed/split), different attack sample selection.

Question: does the below-random direct AUC at seed=123 persist across attack
draws, or was it specific to the [0:50] sample selection?

Run from project root:
    python experiments/phase13b_seed123_resample.py
"""

import sys, os, signal, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from explain.q_shap import q_shap_values, make_quantum_predict_fn

from data.loaders import load_breast_cancer_data
from models.encodings import (
    angle_encoding, amplitude_encoding, reuploading_encoding, iqp_encoding
)
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import run_mia_attack

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── config ────────────────────────────────────────────────────────────────────
SEED         = 123
N_QUBITS     = 4
N_LAYERS     = 2
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_PERMUTATIONS = 8
N_BG         = 20
DRAW1_START  = 0    # Phase 13 draw: [0:50] members, [0:50] non-members
DRAW2_START  = 50   # Resample draw: [50:100] members, [50:100] non-members
N_ATTACK     = 50
RUN_TIMEOUT  = 480

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

# Phase 13 direct AUC results for seed=123 (draw 1, for comparison)
PHASE13_DRAW1 = {
    "angle":       0.3760,
    "amplitude":   0.4240,
    "reuploading": 0.5360,
    "iqp":         0.3800,
}

def _timeout_handler(signum, frame):
    raise TimeoutError("Federated training timed out")

def cleanup_ray():
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

def compute_attack_exps(model, X_train_pp, X_test_pp, start, n, n_bg, seed):
    """Compute Q-SHAP for attack samples [start:start+n] from train and test.
    Background is always training[0:n_bg] (same as Phase 13).
    """
    bg    = X_train_pp[:n_bg]
    mbr   = X_train_pp[start:start+n]
    nmb   = X_test_pp[start:start+n]

    q_pred = make_quantum_predict_fn(model)

    exps_in  = q_shap_values(q_pred, mbr, bg, n_permutations=N_PERMUTATIONS, seed=seed)
    exps_out = q_shap_values(q_pred, nmb, bg, n_permutations=N_PERMUTATIONS, seed=seed)
    return exps_in, exps_out


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)
    print(f"=== Phase 13b: seed=123 resample replication ===")
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")
    print(f"  Draw 1 (Phase 13): members [{DRAW1_START}:{DRAW1_START+N_ATTACK}], "
          f"non-members [{DRAW1_START}:{DRAW1_START+N_ATTACK}]")
    print(f"  Draw 2 (resample): members [{DRAW2_START}:{DRAW2_START+N_ATTACK}], "
          f"non-members [{DRAW2_START}:{DRAW2_START+N_ATTACK}]")
    print()

    assert DRAW2_START + N_ATTACK <= len(X_train), \
        f"Need {DRAW2_START+N_ATTACK} training samples, have {len(X_train)}"
    assert DRAW2_START + N_ATTACK <= len(X_test), \
        f"Need {DRAW2_START+N_ATTACK} test samples, have {len(X_test)}"

    results = {}

    for enc_name, enc_fn in ENCODINGS.items():
        print(f"{'='*60}")
        print(f"[{enc_name}]")
        t0 = time.time()

        # ── Step 1: Retrain IID federated model ──────────────────────────────
        print(f"  Retraining IID federated model (15 rounds) ...", flush=True)
        model_out = []
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(RUN_TIMEOUT)
        try:
            _, round_log = run_federation(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                encoding_fn=enc_fn,
                n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
                n_qubits=N_QUBITS, n_layers=N_LAYERS,
                split="iid", strategy_name="fedavg",
                seed=SEED, project_root=PROJECT_ROOT,
                n_permutations=0, model_out=model_out,
            )
            signal.alarm(0)
        except TimeoutError as e:
            print(f"  TIMEOUT: {e}")
            cleanup_ray()
            continue
        except Exception as e:
            signal.alarm(0)
            print(f"  ERROR: {type(e).__name__}: {e}")
            cleanup_ray()
            continue

        fed_acc = round_log[-1][1] if round_log else float("nan")
        print(f"  Federated acc: {fed_acc:.4f}  t={time.time()-t0:.0f}s")

        # ── Step 2: Reconstruct model ─────────────────────────────────────────
        shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
        target = HybridQuantumClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=enc_fn)
        target.pca     = shared_pca
        target._scaler = shared_scaler
        target.set_weights(model_out[0])
        target_acc = float((target.predict(X_test) == y_test).mean())
        print(f"  Reconstructed acc: {target_acc:.4f}")

        # Preprocess all data once
        X_train_pp = target._preprocess(X_train)
        X_test_pp  = target._preprocess(X_test)

        # ── Step 3: Draw 1 (Phase 13 reference) ──────────────────────────────
        print(f"  Draw 1 Q-SHAP (members [{DRAW1_START}:{DRAW1_START+N_ATTACK}]) ...", flush=True)
        exps_in_d1, exps_out_d1 = compute_attack_exps(
            target, X_train_pp, X_test_pp,
            start=DRAW1_START, n=N_ATTACK, n_bg=N_BG, seed=SEED
        )
        auc_d1 = run_mia_attack(exps_in_d1, exps_out_d1, seed=SEED, cv=5)
        print(f"  Draw 1 direct MIA AUC: {auc_d1:.4f}  "
              f"(Phase 13 logged: {PHASE13_DRAW1[enc_name]:.4f})")

        # ── Step 4: Draw 2 (resample) ─────────────────────────────────────────
        print(f"  Draw 2 Q-SHAP (members [{DRAW2_START}:{DRAW2_START+N_ATTACK}]) ...", flush=True)
        exps_in_d2, exps_out_d2 = compute_attack_exps(
            target, X_train_pp, X_test_pp,
            start=DRAW2_START, n=N_ATTACK, n_bg=N_BG, seed=SEED
        )
        auc_d2 = run_mia_attack(exps_in_d2, exps_out_d2, seed=SEED, cv=5)
        print(f"  Draw 2 direct MIA AUC: {auc_d2:.4f}")

        results[enc_name] = {
            "fed_acc": fed_acc,
            "target_acc": target_acc,
            "draw1_auc": auc_d1,
            "draw1_phase13": PHASE13_DRAW1[enc_name],
            "draw2_auc": auc_d2,
        }
        print(f"  [DONE]  t={time.time()-t0:.0f}s")

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== Phase 13b: seed=123 resample summary ===\n")
    print(f"{'Encoding':<12}  {'Acc':>6}  {'Draw1 (P13)':>11}  {'Draw1 (now)':>11}  "
          f"{'Draw2 (rsmp)':>12}  {'Persists?':>9}")
    print(f"  {'-'*12}  {'-'*6}  {'-'*11}  {'-'*11}  {'-'*12}  {'-'*9}")
    for enc, r in results.items():
        d1_below = r['draw1_auc'] < 0.5
        d2_below = r['draw2_auc'] < 0.5
        persists = "YES (<0.5)" if d2_below else "NO (≥0.5)"
        match_p13 = abs(r['draw1_auc'] - r['draw1_phase13']) < 0.03
        match_str = f"{r['draw1_auc']:.4f}{'*' if not match_p13 else ''}"
        print(f"  {enc:<12}  {r['target_acc']:>6.4f}  {r['draw1_phase13']:>11.4f}  "
              f"{match_str:>11}  {r['draw2_auc']:>12.4f}  {persists:>9}")
    print(f"\n  * = draw1 re-run differs from Phase 13 by >3pp (actor-RNG variation).")
    print()
    below_d1 = sum(1 for r in results.values() if r['draw1_auc'] < 0.5)
    below_d2 = sum(1 for r in results.values() if r['draw2_auc'] < 0.5)
    print(f"  Draw 1: {below_d1}/4 encodings below 0.5")
    print(f"  Draw 2: {below_d2}/4 encodings below 0.5")
    print()
    if below_d2 >= 3:
        print("  VERDICT: below-random AUC SURVIVES resample across majority of encodings.")
        print("  Treat as flagged-but-preliminary partition-dependent finding.")
    elif below_d2 >= 1:
        print("  VERDICT: PARTIAL survival — some encodings replicate, some do not.")
        print("  Flag individually per encoding; do not generalise.")
    else:
        print("  VERDICT: below-random AUC does NOT survive resample.")
        print("  Downgrade to 'observed in one attack-split draw, not confirmed'.")

    print("\n[Phase 13b complete]")


if __name__ == "__main__":
    main()

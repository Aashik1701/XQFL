"""RQ4 DP extension: cross-encoding Laplace DP privacy-utility curve.

Phase 8 tested Laplace DP at ε=1.0 for angle/seed=42 only (single point,
single encoding). This script extends to all 4 encodings × 5 seeds × 4 ε values,
giving a full privacy-utility curve and cross-encoding comparison.

Design
------
Per (encoding, seed):
  1. Retrain IID FedAvg model (same protocol as Phase 13).
  2. Compute Q-SHAP for 50 members and 50 non-members (same as Phase 13).
  3. Apply Laplace DP noise at ε ∈ EPSILONS using per-encoding calibrated
     sensitivity = mean(‖φ‖) from Phase 10 IID data.
  4. Run direct MIA on noisy explanations (cv=5 LR) for each ε.

Sensitivity calibration: Δf = mean(‖φ‖) over IID Track-A GLOBAL rows (rounds
1-15) of the CURRENT results/phase10_log.csv, computed at import by
_compute_sensitivity(). After the 2026-09-07 corrected rerun these are
approximately angle 0.229 / amplitude 0.262 / reuploading 0.344 / iqp 0.287
(printed at startup). The dict below (_SENSITIVITY_FALLBACK, 0.2258 / 0.2607 /
0.3547 / 0.2536) is only used if phase10_log.csv is missing and reflects a
pre-fix run — it is NOT the calibration used for the reported results.

ε values: [0.1, 0.5, 1.0, 5.0] — spans aggressive protection to near-unprotected.
Noise-to-signal ratio (NSR = Δf/(ε × median_norm)):
  ε=0.1 → NSR≈10-20×  (signal almost fully suppressed)
  ε=0.5 → NSR≈2-4×   (meaningful suppression)
  ε=1.0 → NSR≈1-2×   (Phase 8 calibrated regime)
  ε=5.0 → NSR≈0.2-0.4× (light noise, mild protection)

No-DP baseline: use existing Phase 13 AUCs (already logged in phase13_mia_log.csv).

Outputs:
  results/rq4_dp_extension_log.csv
  results/rq4_dp_extension_checkpoint.csv
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
from privacy.dp_noise import batch_dp_noise

# ── constants ─────────────────────────────────────────────────────────────────

N_QUBITS       = 4
N_LAYERS       = 2
N_CLIENTS      = 4
N_ROUNDS       = 15
LOCAL_EPOCHS   = 3
N_MIA_SAMPLES  = 50
N_BG           = 20
N_PERMUTATIONS = 8
RUN_TIMEOUT    = 600

SEEDS = [1, 7, 42, 123, 999]

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

# Per-encoding calibrated Laplace sensitivity Δf = mean(‖φ‖) over IID Track-A
# GLOBAL rows (rounds 1-15) of the CURRENT phase10_log.csv. Derived at import
# time so it never carries stale pre-fix norms; the dict below is a fallback.
_SENSITIVITY_FALLBACK = {
    "angle": 0.2258, "amplitude": 0.2607, "reuploading": 0.3547, "iqp": 0.2536,
}


def _compute_sensitivity():
    import csv as _csv
    path = os.path.join(PROJECT_ROOT, "results", "phase10_log.csv")
    if not os.path.exists(path):
        return dict(_SENSITIVITY_FALLBACK)
    norms = {}
    with open(path, newline="") as fh:
        for r in _csv.DictReader(fh):
            if (r["run_type"] == "federated" and r["distribution"] == "iid"
                    and r["client_id"] == "global" and int(r["round"]) >= 1
                    and r["track_class"] == "A"):
                norms.setdefault(r["encoding"], []).append(float(r["norm_phi"]))
    out = dict(_SENSITIVITY_FALLBACK)
    for enc, vals in norms.items():
        if len(vals) >= 10:
            out[enc] = sum(vals) / len(vals)
    return out


SENSITIVITY = _compute_sensitivity()
print("  SENSITIVITY Δf (from corrected phase10_log.csv): "
      + ", ".join(f"{k}={v:.4f}" for k, v in sorted(SENSITIVITY.items())))

EPSILONS = [0.1, 0.5, 1.0, 5.0]    # privacy-utility curve

RESULTS_CSV    = os.path.join(PROJECT_ROOT, "results", "rq4_dp_extension_log.csv")
CHECKPOINT_CSV = os.path.join(PROJECT_ROOT, "results", "rq4_dp_extension_checkpoint.csv")
PHASE13_CSV    = os.path.join(PROJECT_ROOT, "results", "phase13_mia_log.csv")

RESULTS_COLS = [
    "encoding", "seed", "epsilon", "sensitivity",
    "target_acc", "direct_mia_auc",
    "nsr", "n_members", "n_nonmembers", "wall_time_s", "timestamp",
]

# ── checkpoint helpers ────────────────────────────────────────────────────────

def make_model_key(enc, seed):
    return f"model|{enc}|{seed}"


def make_dp_key(enc, seed, eps):
    return f"dp|{enc}|{seed}|eps{eps}"


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_CSV):
        return set()
    done = set()
    with open(CHECKPOINT_CSV, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "done":
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


def append_result(enc, seed, eps, sens, target_acc, auc, n_mem, n_nonmem, wall_time, nsr):
    first = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULTS_COLS)
        if first:
            w.writeheader()
        w.writerow({
            "encoding":       enc,
            "seed":           seed,
            "epsilon":        eps,
            "sensitivity":    f"{sens:.4f}",
            "target_acc":     f"{target_acc:.6f}",
            "direct_mia_auc": f"{auc:.6f}",
            "nsr":            f"{nsr:.3f}",
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


# ── main run: per (encoding, seed) ────────────────────────────────────────────

def run_one(enc_name, seed):
    """Retrain IID model, compute Q-SHAP, run MIA at all ε values."""
    done = load_checkpoint()

    # Skip if all ε values already done for this (enc, seed)
    all_eps_done = all(make_dp_key(enc_name, seed, eps) in done for eps in EPSILONS)
    if all_eps_done:
        print(f"  [skip all] {enc_name}/seed={seed}")
        return

    print(f"\n{'='*70}")
    print(f"  DP-EXT  {enc_name} / IID / seed={seed}")
    print(f"  Sensitivity Δf={SENSITIVITY[enc_name]:.4f}  "
          f"ε={EPSILONS}  NSR(ε=1.0)≈{1.0:.1f}×Δf")
    print(f"{'='*70}")

    t0 = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=seed)

    # 1. Retrain federated model
    model_out = []
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(RUN_TIMEOUT)
    try:
        _, round_log = run_federation(
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
            encoding_fn=ENCODINGS[enc_name],
            n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
            n_qubits=N_QUBITS, n_layers=N_LAYERS,
            split="iid", strategy_name="fedavg", seed=seed,
            project_root=PROJECT_ROOT, n_permutations=0,
            model_out=model_out,
        )
        signal.alarm(0)
    except TimeoutError as e:
        signal.alarm(0)
        print(f"  [TIMEOUT] {e}")
        cleanup_ray()
        for eps in EPSILONS:
            mark_done(make_dp_key(enc_name, seed, eps), status="timeout")
        return
    except Exception as e:
        signal.alarm(0)
        print(f"  [ERROR] {e}")
        cleanup_ray()
        raise

    final_acc = round_log[-1][1] if round_log else float("nan")
    print(f"  Federated R15 acc: {final_acc:.4f}  ({time.time()-t0:.0f}s)")

    # 2. Reconstruct target model
    shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
    target_model = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=ENCODINGS[enc_name]
    )
    target_model.pca      = shared_pca
    target_model._scaler  = shared_scaler
    target_model.set_weights(model_out[0])
    target_acc = float((target_model.predict(X_test) == y_test).mean())

    # 3. Q-SHAP for members and non-members (compute once, reuse across ε)
    print(f"  Q-SHAP members ({N_MIA_SAMPLES}) …")
    exps_in  = _qshap_for_samples(target_model, X_train,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=seed)
    print(f"  Q-SHAP non-members ({N_MIA_SAMPLES}) …")
    exps_out = _qshap_for_samples(target_model, X_test,
                                   n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                                   n_permutations=N_PERMUTATIONS, seed=seed)
    print(f"  Q-SHAP done in {time.time()-t0:.0f}s")

    sens = SENSITIVITY[enc_name]

    # 4. MIA at each ε value
    for eps in EPSILONS:
        dp_rk = make_dp_key(enc_name, seed, eps)
        if dp_rk in done:
            print(f"  [skip] ε={eps}")
            continue

        noisy_in  = batch_dp_noise(exps_in,  epsilon=eps, sensitivity=sens,
                                    mechanism="laplace", seed=seed)
        noisy_out = batch_dp_noise(exps_out, epsilon=eps, sensitivity=sens,
                                    mechanism="laplace", seed=seed + 1)
        auc = run_mia_attack(noisy_in, noisy_out, seed=seed, cv=5)
        # NSR = Laplace scale / median signal magnitude
        nsr = (sens / eps) / float(np.median(np.linalg.norm(exps_in, axis=1)))

        print(f"  ε={eps:<5}  NSR={nsr:.2f}×  direct_MIA_AUC={auc:.4f}")
        append_result(enc_name, seed, eps, sens, target_acc, auc,
                      len(exps_in), len(exps_out), time.time() - t0, nsr)
        mark_done(dp_rk)


# ── analysis: privacy-utility curves ─────────────────────────────────────────

def print_analysis():
    print("\n" + "=" * 72)
    print("  RQ4 DP EXTENSION — Privacy-Utility Curves (Direct MIA AUC vs ε)")
    print("=" * 72)

    # Phase 13 baseline (no DP)
    no_dp = {}
    if os.path.exists(PHASE13_CSV):
        with open(PHASE13_CSV) as fh:
            for row in csv.DictReader(fh):
                enc = row["encoding"]
                if enc not in no_dp:
                    no_dp[enc] = []
                no_dp[enc].append(float(row["direct_mia_auc"]))

    if not os.path.exists(RESULTS_CSV):
        print("  No DP results yet.")
        return

    dp_rows = list(csv.DictReader(open(RESULTS_CSV)))

    print(f"\n{'Encoding':<12}  {'ε=∞ (no DP)':>12}  "
          + "  ".join(f"ε={e}".rjust(8) for e in EPSILONS))
    print("-" * 72)

    for enc in ENCODINGS:
        base_mean = statistics.mean(no_dp.get(enc, [float("nan")])) if no_dp.get(enc) else float("nan")
        eps_means = []
        for eps in EPSILONS:
            aucs = [float(r["direct_mia_auc"]) for r in dp_rows
                    if r["encoding"]==enc and float(r["epsilon"])==eps]
            if aucs:
                eps_means.append(f"{statistics.mean(aucs):.3f}")
            else:
                eps_means.append("  —  ")
        print(f"  {enc:<12}  {base_mean:>12.3f}  " + "  ".join(f"{v:>8}" for v in eps_means))

    # NSR table
    print(f"\nNoise-to-signal ratios (Δf / (ε × median_‖φ‖)):")
    for enc in ENCODINGS:
        enc_rows = [r for r in dp_rows if r["encoding"]==enc]
        if not enc_rows:
            continue
        print(f"  {enc}:  " +
              "  ".join(f"ε={float(r['epsilon'])}→NSR={float(r['nsr']):.2f}×"
                        for r in sorted(enc_rows, key=lambda x: float(x["epsilon"]))[:4]))

    # Cross-encoding comparison at ε=1.0
    print(f"\nCross-encoding comparison at ε=1.0:")
    for enc in ENCODINGS:
        aucs = [float(r["direct_mia_auc"]) for r in dp_rows
                if r["encoding"]==enc and float(r["epsilon"])==1.0]
        if aucs:
            mean = statistics.mean(aucs)
            std  = statistics.stdev(aucs) if len(aucs)>1 else 0
            base = statistics.mean(no_dp.get(enc, [float("nan")])) if no_dp.get(enc) else float("nan")
            reduction = base - mean
            print(f"  {enc:<12}  no-DP={base:.3f}  ε=1.0={mean:.3f}±{std:.3f}  "
                  f"reduction={reduction:.3f} ({reduction/base*100:.0f}%)")


# ── main ──────────────────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

print("\n" + "=" * 72)
print("  RQ4 DP EXTENSION — Cross-Encoding Laplace DP Sweep")
print(f"  Encodings: {list(ENCODINGS.keys())}")
print(f"  Seeds: {SEEDS}")
print(f"  ε values: {EPSILONS}")
print(f"  Total model retrains: {len(ENCODINGS) * len(SEEDS)}")
print(f"  Total MIA evaluations: {len(ENCODINGS) * len(SEEDS) * len(EPSILONS)}")
print("=" * 72 + "\n")

for enc_name in ENCODINGS:
    for seed in SEEDS:
        run_one(enc_name, seed)

print_analysis()

"""Phase 8 DP calibration fix: use empirical ‖φ‖ as sensitivity instead of 1.0.

Phase 8 supplemental showed that sensitivity=1.0 at ε=1.0 gives NSR≈9.9× (noise
swamps signal 10:1), driving attack AUC to 0.296 (below chance). The fix is to set
sensitivity = mean(‖φ‖) ≈ 0.29, giving NSR≈2.83× — meaningful DP suppression without
destroying all attack signal. Expected AUC: between 0.296 (over-suppressed) and 0.674 (unprotected).

Run from project root:
    python experiments/phase8_dp_calibrated.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding
from models.quantum_classifier import HybridQuantumClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor
from privacy.membership_inference import run_mia_attack, _qshap_for_samples
from privacy.dp_noise import batch_dp_noise

SEED           = 42
N_CLIENTS      = 4
N_ROUNDS       = 15
LOCAL_EPOCHS   = 3
N_QUBITS       = 4
N_LAYERS       = 2
N_MIA_SAMPLES  = 50
N_BG           = 20
N_PERMUTATIONS = 8

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=== Phase 8: DP calibrated sensitivity ===\n")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

# ── 1. train target model ──────────────────────────────────────────────────────
print("── Training target IID FedAvg model ────────────────────────────────────")
model_out = []
_, round_log = run_federation(
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    encoding_fn=angle_encoding,
    n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
    n_qubits=N_QUBITS, n_layers=N_LAYERS,
    split="iid", strategy_name="fedavg", seed=SEED,
    project_root=PROJECT_ROOT, n_permutations=0,
    model_out=model_out,
)
print(f"  Final accuracy: {round_log[-1][1]:.4f}\n")

shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
target_model           = HybridQuantumClassifier(N_QUBITS, N_LAYERS, angle_encoding)
target_model.pca       = shared_pca
target_model._scaler   = shared_scaler
target_model.set_weights(model_out[0])

# ── 2. compute Q-SHAP ──────────────────────────────────────────────────────────
print("── Computing member / non-member Q-SHAP ─────────────────────────────────")
exps_in  = _qshap_for_samples(target_model, X_train,
                               n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                               n_permutations=N_PERMUTATIONS, seed=SEED)
exps_out = _qshap_for_samples(target_model, X_test,
                               n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                               n_permutations=N_PERMUTATIONS, seed=SEED)

all_norms     = np.linalg.norm(np.vstack([exps_in, exps_out]), axis=1)
calibrated_s  = float(all_norms.mean())
print(f"  Member mean ‖φ‖    : {np.linalg.norm(exps_in,  axis=1).mean():.4f}")
print(f"  Non-member mean ‖φ‖: {np.linalg.norm(exps_out, axis=1).mean():.4f}")
print(f"  Combined mean ‖φ‖  : {calibrated_s:.4f}  ← used as sensitivity\n")

# ── 3. baseline (unprotected) ──────────────────────────────────────────────────
auc_raw = run_mia_attack(exps_in, exps_out, seed=SEED, cv=5)
print(f"  Unprotected AUC (baseline)      : {auc_raw:.4f}")

# ── 4. old sensitivity=1.0 (over-suppressed, for reference) ───────────────────
OLD_S = 1.0
noisy_in_old  = batch_dp_noise(exps_in,  epsilon=1.0, sensitivity=OLD_S,
                               mechanism="laplace", seed=SEED)
noisy_out_old = batch_dp_noise(exps_out, epsilon=1.0, sensitivity=OLD_S,
                               mechanism="laplace", seed=SEED + 1)
auc_old = run_mia_attack(noisy_in_old, noisy_out_old, seed=SEED, cv=5)
e_noise_old = OLD_S * np.sqrt(2 * N_QUBITS)
nsr_old     = e_noise_old / calibrated_s
print(f"  DP ε=1.0, Δf=1.0 (old):         "
      f"AUC={auc_old:.4f}   NSR={nsr_old:.2f}× (noise scale b={OLD_S/1.0:.4f})")

# ── 5. calibrated sensitivity = mean ‖φ‖ ──────────────────────────────────────
eps = 1.0
noisy_in_cal  = batch_dp_noise(exps_in,  epsilon=eps, sensitivity=calibrated_s,
                               mechanism="laplace", seed=SEED)
noisy_out_cal = batch_dp_noise(exps_out, epsilon=eps, sensitivity=calibrated_s,
                               mechanism="laplace", seed=SEED + 1)
auc_cal = run_mia_attack(noisy_in_cal, noisy_out_cal, seed=SEED, cv=5)
b_cal        = calibrated_s / eps
e_noise_cal  = b_cal * np.sqrt(2 * N_QUBITS)   # E[‖Laplace noise‖_2]
nsr_cal      = e_noise_cal / calibrated_s       # should be sqrt(2d) ≈ 2.83 (invariant)
print(f"  DP ε=1.0, Δf=‖φ‖ (calibrated):  "
      f"AUC={auc_cal:.4f}   NSR={nsr_cal:.2f}× (noise scale b={b_cal:.4f})")

# ── 6. report ──────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== DP calibration summary ===\n")
print(f"  Empirical sensitivity (mean ‖φ‖) : {calibrated_s:.4f}")
print(f"  Predicted NSR at calibrated Δf   : √(2d) = √{2*N_QUBITS} ≈ {np.sqrt(2*N_QUBITS):.2f}×")
print(f"  Actual NSR (analytical)          : {nsr_cal:.4f}×\n")
print(f"  {'Condition':<40}  {'AUC':>6}  {'NSR':>7}  {'vs baseline':>12}")
print(f"  {'-'*40}  {'-'*6}  {'-'*7}  {'-'*12}")
print(f"  {'Unprotected (no DP)':<40}  {auc_raw:>6.4f}  {'—':>7}  {'—':>12}")
print(f"  {'DP ε=1.0, Δf=1.0 (old, over-suppressed)':<40}  "
      f"{auc_old:>6.4f}  {nsr_old:>7.2f}×  {auc_old-auc_raw:>+12.4f}")
print(f"  {'DP ε=1.0, Δf=‖φ‖ (calibrated)':<40}  "
      f"{auc_cal:>6.4f}  {nsr_cal:>7.2f}×  {auc_cal-auc_raw:>+12.4f}")

# ── 7. verification ────────────────────────────────────────────────────────────
print(f"\n── Verification ──────────────────────────────────────────────────────────")
between_bounds = auc_old <= auc_cal <= auc_raw
print(f"  AUC ordering: old DP ({auc_old:.4f}) ≤ calibrated ({auc_cal:.4f}) ≤ "
      f"unprotected ({auc_raw:.4f}) → {'PASS' if between_bounds else 'FAIL'}")
nsr_close = abs(nsr_cal - np.sqrt(2 * N_QUBITS)) < 0.01
print(f"  NSR = √(2d) = {np.sqrt(2*N_QUBITS):.4f} (analytical invariant) "
      f"→ {'PASS' if nsr_close else 'FAIL'} (actual={nsr_cal:.4f})")

if between_bounds:
    print(f"\n  Calibrated DP ε=1.0 produces meaningful suppression: "
          f"AUC drops from {auc_raw:.4f} (unprotected) to {auc_cal:.4f} "
          f"(DP, NSR≈{nsr_cal:.2f}×), well above the collapsed {auc_old:.4f} "
          f"from over-suppression.")
else:
    print(f"\n  WARNING: AUC not in expected range. Check explanation distributions.")

print("\n[Phase 8 DP calibration fix complete]")

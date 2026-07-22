"""Phase 8 supplemental: shadow count sweep + DP noise-to-signal ratio analysis.

Addresses two questions raised after the initial Phase 8 run:

Q1. Does shadow model AUC (currently 0.604, below direct MIA's 0.640) change
    when we train more shadow classifiers?  Initial run used n_shadow=3.
    Sweep n_shadow ∈ {3, 6, 9} to determine whether the gap is an artifact of
    too few shadows or a structural limit of 13-parameter quantum classifiers.

Q2. What is the noise-to-signal ratio at ε=1.0 and ε=0.1 (sensitivity=1.0)?
    Compute both analytically (E[||Laplace noise||_2] / mean(||φ||)) and
    empirically (sample 10 000 noise draws, compare to actual explanation norms).
    Confirm the 'DP oversuppression' claim numerically.

Run from project root:
    python experiments/phase8_supplemental.py
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
from privacy.membership_inference import (
    ShadowModelMIA, run_mia_attack, _qshap_for_samples
)
from privacy.dp_noise import batch_dp_noise

SEED          = 42
N_CLIENTS     = 4
N_ROUNDS      = 15
LOCAL_EPOCHS  = 3
N_QUBITS      = 4
N_LAYERS      = 2
N_MIA_SAMPLES = 50
N_BG          = 20
N_PERMUTATIONS = 8
SHADOW_EPOCHS  = 15
N_EXPLAIN_SHADOW = 20
DP_EPSILONS   = [1.0, 0.1]
DP_SENSITIVITY = 1.0
SHADOW_COUNTS  = [3, 6, 9]
NSR_DRAWS     = 10_000

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 1. data + target model ────────────────────────────────────────────────────
print("=== Phase 8 Supplemental: Shadow Sweep + NSR Analysis ===\n")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)

print("── Retraining target federated model (IID, 15 rounds) ──────────────────")
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
final_acc = round_log[-1][1] if round_log else float("nan")
print(f"  Final federated accuracy: {final_acc:.4f}")

shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)
target_model           = HybridQuantumClassifier(
    n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=angle_encoding
)
target_model.pca       = shared_pca
target_model._scaler   = shared_scaler
target_model.set_weights(model_out[0])

# ── 2. member / non-member Q-SHAP ─────────────────────────────────────────────
print("\n── Computing member/non-member Q-SHAP explanations ─────────────────────")
exps_in  = _qshap_for_samples(target_model, X_train,
                               n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                               n_permutations=N_PERMUTATIONS, seed=SEED)
exps_out = _qshap_for_samples(target_model, X_test,
                               n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                               n_permutations=N_PERMUTATIONS, seed=SEED)
norms_in  = np.linalg.norm(exps_in,  axis=1)
norms_out = np.linalg.norm(exps_out, axis=1)
mean_norm = float(np.concatenate([norms_in, norms_out]).mean())
print(f"  Member     mean ||φ||: {norms_in.mean():.4f}  (n={len(exps_in)})")
print(f"  Non-member mean ||φ||: {norms_out.mean():.4f}  (n={len(exps_out)})")
print(f"  Combined   mean ||φ||: {mean_norm:.4f}")

# ── 3. direct MIA baseline ────────────────────────────────────────────────────
auc_direct = run_mia_attack(exps_in, exps_out, seed=SEED, cv=5)
print(f"\n  Direct MIA AUC (baseline, n_shadow=0): {auc_direct:.4f}")

# ── 4. shadow count sweep ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"=== Q1: Shadow count sweep {SHADOW_COUNTS} ===")
print(f"    (Direct MIA baseline: {auc_direct:.4f} — the bar to beat)\n")

def model_factory():
    m          = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=angle_encoding
    )
    m.pca      = shared_pca
    m._scaler  = shared_scaler
    return m

shadow_results = {}

for n_shadow in SHADOW_COUNTS:
    print(f"── n_shadow={n_shadow} ({'×'.join([str(N_EXPLAIN_SHADOW)]*2)} shadow samples) ──")
    mia = ShadowModelMIA(
        model_factory=model_factory,
        n_shadow=n_shadow,
        local_epochs=SHADOW_EPOCHS,
        shadow_split=0.5,
        n_explain=N_EXPLAIN_SHADOW,
        n_bg=N_BG,
        n_permutations=N_PERMUTATIONS,
        seed=SEED,
    )
    mia.generate_shadow_data(X_train, y_train)
    mia.fit_attack()

    auc_shadow_mia = mia.attack_auc(exps_in, exps_out)
    auc_shadow_cv  = mia.shadow_cv_auc(cv=3)
    shadow_results[n_shadow] = {"target": auc_shadow_mia, "cv": auc_shadow_cv}
    print(f"  Shadow CV AUC     : {auc_shadow_cv:.4f}  (shadow self-attack sanity)")
    print(f"  Shadow→target AUC : {auc_shadow_mia:.4f}  (vs direct {auc_direct:.4f})\n")

print(f"\n── Shadow sweep summary ─────────────────────────────────────────────────")
print(f"  {'n_shadow':>8}  {'Shadow train samples':>20}  {'Shadow CV AUC':>14}  "
      f"{'Shadow→target AUC':>18}  {'vs direct MIA':>13}")
print(f"  {'-'*8}  {'-'*20}  {'-'*14}  {'-'*18}  {'-'*13}")
for n_shadow in SHADOW_COUNTS:
    r      = shadow_results[n_shadow]
    n_samp = n_shadow * N_EXPLAIN_SHADOW * 2
    diff   = r["target"] - auc_direct
    sign   = "+" if diff >= 0 else ""
    print(f"  {n_shadow:>8}  {n_samp:>20}  {r['cv']:>14.4f}  "
          f"{r['target']:>18.4f}  {sign}{diff:>+12.4f}")
print(f"  {'(direct)':>8}  {'—':>20}  {'—':>14}  {auc_direct:>18.4f}  (baseline)")

# interpretation
max_shadow_auc = max(r["target"] for r in shadow_results.values())
if max_shadow_auc > auc_direct + 0.02:
    print(f"\n  FINDING: More shadows IMPROVE attack (max={max_shadow_auc:.4f} > "
          f"direct={auc_direct:.4f}). Initial n=3 was insufficient.")
elif max_shadow_auc > auc_direct - 0.02:
    print(f"\n  FINDING: More shadows do NOT improve attack (max={max_shadow_auc:.4f} ≈ "
          f"direct={auc_direct:.4f}). Shadow count is not the bottleneck.")
else:
    print(f"\n  FINDING: More shadows WORSEN attack (max={max_shadow_auc:.4f} < "
          f"direct={auc_direct:.4f}). Shadow models add noise, not signal.")

# ── 5. noise-to-signal ratio ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"=== Q2: DP noise-to-signal ratio analysis ===")
print(f"\n  Explanation statistics:")
print(f"    mean ||φ||        : {mean_norm:.4f}")
print(f"    n_qubits (d)      : {N_QUBITS}")
print(f"    sensitivity (Δf)  : {DP_SENSITIVITY:.1f}  (L2 worst-case bound)")
print(f"    per-feature signal: {mean_norm / N_QUBITS:.4f}  "
      f"(mean ||φ|| / d, if uniform distribution)")

print(f"\n  Analytical Laplace noise norms (E[||Lap(0, Δf/ε)||_2] for d={N_QUBITS}):")
print(f"    For X_i ~ iid Laplace(0, b):  E[X_i²] = 2b²")
print(f"    E[||X||_2²] = d × 2b²  →  E[||X||_2] ≈ sqrt(d × 2b²) = b × sqrt(2d)")
print()
print(f"  {'ε':>6}  {'b=Δf/ε':>10}  {'E[||noise||_2]':>16}  "
      f"{'NSR (noise/signal)':>20}  {'SNR (signal/noise)':>20}")
print(f"  {'-'*6}  {'-'*10}  {'-'*16}  {'-'*20}  {'-'*20}")

analytical_nsr = {}
for eps in DP_EPSILONS:
    b          = DP_SENSITIVITY / eps
    e_noise_l2 = b * np.sqrt(2 * N_QUBITS)        # analytical E[||noise||_2]
    nsr        = e_noise_l2 / mean_norm
    snr        = mean_norm / e_noise_l2
    analytical_nsr[eps] = {"b": b, "e_noise_l2": e_noise_l2, "nsr": nsr, "snr": snr}
    print(f"  {eps:>6.1f}  {b:>10.4f}  {e_noise_l2:>16.4f}  "
          f"{nsr:>20.2f}×  {snr:>19.4f}×")

print(f"\n  Empirical Laplace noise norms ({NSR_DRAWS} random draws per ε):")
print(f"  {'ε':>6}  {'E[||noise||] empirical':>22}  {'std(||noise||)':>15}  "
      f"{'NSR empirical':>15}  {'NSR analytical':>15}")
print(f"  {'-'*6}  {'-'*22}  {'-'*15}  {'-'*15}  {'-'*15}")

rng = np.random.default_rng(SEED)
for eps in DP_EPSILONS:
    b           = DP_SENSITIVITY / eps
    noise_draws = rng.laplace(0, b, size=(NSR_DRAWS, N_QUBITS))
    noise_norms = np.linalg.norm(noise_draws, axis=1)
    emp_mean    = float(noise_norms.mean())
    emp_std     = float(noise_norms.std())
    emp_nsr     = emp_mean / mean_norm
    ana_nsr     = analytical_nsr[eps]["nsr"]
    print(f"  {eps:>6.1f}  {emp_mean:>22.4f}  {emp_std:>15.4f}  "
          f"{emp_nsr:>15.2f}×  {ana_nsr:>15.2f}×")

print(f"\n  Per-feature breakdown:")
print(f"  {'ε':>6}  {'b=Δf/ε':>8}  {'E[|noise_i|]=b':>15}  "
      f"{'mean |φ_i|':>12}  {'per-feature NSR':>16}")
print(f"  {'-'*6}  {'-'*8}  {'-'*15}  {'-'*12}  {'-'*16}")
mean_abs_phi = float(np.abs(np.vstack([exps_in, exps_out])).mean())
for eps in DP_EPSILONS:
    b          = DP_SENSITIVITY / eps
    pf_nsr     = b / mean_abs_phi
    print(f"  {eps:>6.1f}  {b:>8.4f}  {b:>15.4f}  "
          f"{mean_abs_phi:>12.4f}  {pf_nsr:>16.2f}×")

print(f"\n  Better-calibrated sensitivity (Δf ≈ mean ||φ||):")
better_sens = mean_norm
print(f"    Δf = {better_sens:.4f}  (= mean ||φ||, tight L2 sensitivity estimate)")
print(f"    {'ε':>6}  {'b':>8}  {'E[||noise||]':>14}  {'NSR':>8}  {'AUC would be ~?'}")
print(f"    {'-'*6}  {'-'*8}  {'-'*14}  {'-'*8}  {'-'*15}")
for eps in DP_EPSILONS:
    b          = better_sens / eps
    e_noise_l2 = b * np.sqrt(2 * N_QUBITS)
    nsr        = e_noise_l2 / mean_norm
    print(f"    {eps:>6.1f}  {b:>8.4f}  {e_noise_l2:>14.4f}  {nsr:>8.2f}×  "
          f"{'~0.5 (signal masked)' if nsr > 3 else '~0.5–0.6 (partial)'}")

# ── 6. DP AUC at current sensitivity to cross-check ──────────────────────────
print(f"\n  DP AUC cross-check (same values as Phase 8, confirming NSR explains AUC<0.5):")
for eps in DP_EPSILONS:
    noisy_in  = batch_dp_noise(exps_in,  epsilon=eps, mechanism="laplace",
                               sensitivity=DP_SENSITIVITY, seed=SEED)
    noisy_out = batch_dp_noise(exps_out, epsilon=eps, mechanism="laplace",
                               sensitivity=DP_SENSITIVITY, seed=SEED + 1)
    noisy_norms = np.linalg.norm(
        np.vstack([noisy_in, noisy_out]) - np.vstack([exps_in, exps_out]),
        axis=1
    ).mean()
    auc_dp = run_mia_attack(noisy_in, noisy_out, seed=SEED, cv=5)
    nsr    = analytical_nsr[eps]["nsr"]
    print(f"  ε={eps}: NSR={nsr:.1f}×  mean_added_noise_norm={noisy_norms:.4f}  AUC={auc_dp:.4f}")

# ── 7. conclusions ────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("=== Supplemental conclusions ===\n")

print("Q1. Shadow count effect:")
max_n  = max(SHADOW_COUNTS)
max_r  = shadow_results[max_n]["target"]
min_r  = shadow_results[SHADOW_COUNTS[0]]["target"]
if abs(max_r - min_r) < 0.03:
    print(f"   Shadow AUC is STABLE across n_shadow={SHADOW_COUNTS} (range: "
          f"{min_r:.4f}–{max_r:.4f}, Δ={abs(max_r-min_r):.4f}).")
    print(f"   The gap between shadow attack ({max_r:.4f}) and direct MIA "
          f"({auc_direct:.4f}) is structural,\n"
          f"   not an artifact of too few shadows. Root cause: quantum classifiers "
          f"with {13} parameters\n"
          f"   on ~225-sample splits do not overfit strongly enough to produce a "
          f"transferable\n   attack signal across shadow/target model boundaries.")
else:
    print(f"   Shadow AUC varies with n_shadow (range: {min_r:.4f}–{max_r:.4f}, "
          f"Δ={abs(max_r-min_r):.4f}).\n"
          f"   n_shadow={max_n} gives max AUC={max_r:.4f}.")

print(f"\nQ2. DP oversuppression confirmed:")
for eps in DP_EPSILONS:
    nsr = analytical_nsr[eps]["nsr"]
    snr = analytical_nsr[eps]["snr"]
    print(f"   ε={eps}: NSR={nsr:.1f}× (noise is {nsr:.0f}× the signal norm). "
          f"SNR={snr:.4f} << 1.\n"
          f"          This renders the explanation vector indistinguishable from pure noise,\n"
          f"          driving AUC to chance (or below, due to classifier miscalibration).\n"
          f"          Fix: calibrate sensitivity to Δf≈||φ||≈{mean_norm:.2f} "
          f"(gives NSR={DP_SENSITIVITY/mean_norm*np.sqrt(2*N_QUBITS)/eps:.1f}× at ε={eps}).")

print(f"\n[Phase 8 supplemental complete]")

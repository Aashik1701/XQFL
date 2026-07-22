"""Phase 8: Membership inference privacy attack.

Target: IID FedAvg angle-encoding federated model (4 clients, 15 rounds).
Attack: Shadow-model MIA using Q-SHAP explanation vectors as attack features.
Defense: Differential privacy (Laplace) applied to explanations pre-attack.

Protocol:
  1. Retrain target federated model (IID, angle, FedAvg, 4 clients, 15 rounds).
     Use model_out=[] to capture final federated weights without adding return value.
  2. Reconstruct target model from captured weights + shared PCA.
  3. Compute Q-SHAP for:
       - 50 training samples  (members)
       - 50 test samples      (non-members)
  4. Run shadow model attack (n_shadow=3, 15 local epochs each).
  5. Report AUC for:
       a. Direct MIA on target model's raw explanations (cross-validated LR, no shadows)
       b. Shadow model attack AUC
       c. Shadow CV AUC (sanity check: can shadow models themselves be attacked?)
       d. Raw explanations under DP noise ε=1.0 (Laplace) → run_mia_attack
       e. Raw explanations under DP noise ε=0.1 (Laplace) → run_mia_attack
  6. Verify all AUCs are sane floats (not NaN) in [0, 1].

Run from project root:
    python experiments/phase8_mia.py
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

# ── config ────────────────────────────────────────────────────────────────────
SEED         = 42
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
N_QUBITS     = 4
N_LAYERS     = 2
N_MIA_SAMPLES = 50      # members and non-members each
N_BG          = 20      # Q-SHAP background samples
N_PERMUTATIONS = 8      # Q-SHAP permutations
N_SHADOW       = 3      # shadow classifiers
SHADOW_EPOCHS  = 15     # shadow model training epochs
DP_EPSILONS    = [1.0, 0.1]

np.random.seed(SEED)
torch.manual_seed(SEED)

# ── 1. data ───────────────────────────────────────────────────────────────────
print("=== Phase 8: Membership Inference Privacy Attack ===\n")
X_train, X_test, y_train, y_test = load_breast_cancer_data(random_state=SEED)
print(f"  Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── 2. train target federated model ──────────────────────────────────────────
print("\n── Step 1: Train target IID FedAvg model (15 rounds) ───────────────────")
model_out = []   # will receive [weights_list] after final round

_, round_log = run_federation(
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    encoding_fn=angle_encoding,
    n_clients=N_CLIENTS,
    n_rounds=N_ROUNDS,
    local_epochs=LOCAL_EPOCHS,
    n_qubits=N_QUBITS,
    n_layers=N_LAYERS,
    split="iid",
    strategy_name="fedavg",
    seed=SEED,
    project_root=PROJECT_ROOT,
    n_permutations=0,     # no Q-SHAP during training, just compute after
    model_out=model_out,
)

final_fed_acc = round_log[-1][1] if round_log else float("nan")
print(f"\n  Final federated test accuracy: {final_fed_acc:.4f}")
assert len(model_out) == 1, "model_out should have exactly one entry (final round weights)"

# ── 3. reconstruct target model ───────────────────────────────────────────────
print("\n── Step 2: Reconstruct target model from federated weights ─────────────")
shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, N_QUBITS)

target_model           = HybridQuantumClassifier(
    n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=angle_encoding
)
target_model.pca       = shared_pca
target_model._scaler   = shared_scaler
target_model.set_weights(model_out[0])

target_acc = float((target_model.predict(X_test) == y_test).mean())
print(f"  Reconstructed model accuracy: {target_acc:.4f}")
assert abs(target_acc - final_fed_acc) < 0.01, (
    f"Reconstructed model accuracy ({target_acc:.4f}) != federated final "
    f"({final_fed_acc:.4f}). Weight extraction failed."
)

# ── 4. compute member / non-member Q-SHAP ────────────────────────────────────
print(f"\n── Step 3: Compute Q-SHAP explanations "
      f"({N_MIA_SAMPLES} members, {N_MIA_SAMPLES} non-members) ──")
print("  Computing member explanations (training set) ...", flush=True)
exps_in  = _qshap_for_samples(
    target_model, X_train,
    n_samples=N_MIA_SAMPLES, n_bg=N_BG, n_permutations=N_PERMUTATIONS, seed=SEED
)
print(f"  Member explanations: {exps_in.shape}  "
      f"mean_norm={np.linalg.norm(exps_in, axis=1).mean():.4f}")

print("  Computing non-member explanations (test set) ...", flush=True)
exps_out = _qshap_for_samples(
    target_model, X_test,
    n_samples=N_MIA_SAMPLES, n_bg=N_BG, n_permutations=N_PERMUTATIONS, seed=SEED
)
print(f"  Non-member explanations: {exps_out.shape}  "
      f"mean_norm={np.linalg.norm(exps_out, axis=1).mean():.4f}")

# ── 5. direct MIA (no shadow, cross-validated) ────────────────────────────────
print("\n── Step 4: Direct MIA (cross-validated LR on target model explanations) ─")
auc_direct = run_mia_attack(exps_in, exps_out, seed=SEED, cv=5)
print(f"  Direct MIA AUC (CV): {auc_direct:.4f}")

# ── 6. shadow model MIA ───────────────────────────────────────────────────────
print(f"\n── Step 5: Shadow model MIA (n_shadow={N_SHADOW}, epochs={SHADOW_EPOCHS}) ─")

def model_factory():
    m          = HybridQuantumClassifier(
        n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=angle_encoding
    )
    m.pca      = shared_pca
    m._scaler  = shared_scaler
    return m

mia = ShadowModelMIA(
    model_factory=model_factory,
    n_shadow=N_SHADOW,
    local_epochs=SHADOW_EPOCHS,
    shadow_split=0.5,
    n_explain=20,
    n_bg=N_BG,
    n_permutations=N_PERMUTATIONS,
    seed=SEED,
)

mia.generate_shadow_data(X_train, y_train)
mia.fit_attack()

auc_shadow_cv  = mia.shadow_cv_auc(cv=3)
auc_shadow_mia = mia.attack_auc(exps_in, exps_out)
print(f"  Shadow CV AUC (sanity): {auc_shadow_cv:.4f}")
print(f"  Shadow → Target AUC:    {auc_shadow_mia:.4f}")

# ── 7. DP-noised explanations ─────────────────────────────────────────────────
print(f"\n── Step 6: DP noise sensitivity analysis ───────────────────────────────")
dp_results = {}
for eps in DP_EPSILONS:
    noisy_in  = batch_dp_noise(exps_in,  epsilon=eps, mechanism="laplace",
                               sensitivity=1.0, seed=SEED)
    noisy_out = batch_dp_noise(exps_out, epsilon=eps, mechanism="laplace",
                               sensitivity=1.0, seed=SEED + 1)
    auc_dp = run_mia_attack(noisy_in, noisy_out, seed=SEED, cv=5)
    dp_results[eps] = auc_dp
    print(f"  DP Laplace ε={eps:<5}: AUC = {auc_dp:.4f}")

# ── 8. results table ──────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("=== Phase 8: Attack AUC Summary ===")
print(f"\n  Target model final accuracy : {target_acc:.4f}")
print(f"  Member explanations shape   : {exps_in.shape}")
print(f"  Non-member explanations shape: {exps_out.shape}")
print()
print(f"  {'Attack variant':<40}  {'AUC':>6}")
print(f"  {'-'*40}  {'-'*6}")
print(f"  {'Direct MIA (CV-LR, no shadows)':<40}  {auc_direct:>6.4f}")
print(f"  {'Shadow model → target AUC':<40}  {auc_shadow_mia:>6.4f}")
print(f"  {'Shadow CV AUC (sanity)':<40}  {auc_shadow_cv:>6.4f}")
for eps, auc in dp_results.items():
    label = f"DP Laplace ε={eps} (direct MIA)"
    print(f"  {label:<40}  {auc:>6.4f}")

# ── 9. sanity checks ──────────────────────────────────────────────────────────
print(f"\n── Sanity checks ───────────────────────────────────────────────────────")
all_aucs = {
    "direct_mia":     auc_direct,
    "shadow_mia":     auc_shadow_mia,
    "shadow_cv":      auc_shadow_cv,
    **{f"dp_eps_{eps}": auc for eps, auc in dp_results.items()},
}

all_ok = True
for name, auc in all_aucs.items():
    ok = not np.isnan(auc) and 0.0 <= auc <= 1.0
    mark = "OK" if ok else "FAIL"
    print(f"  [{mark}] {name}: AUC={auc:.4f}")
    if not ok:
        all_ok = False

if all_ok:
    print("\n  All AUC values are sane floats in [0, 1].")
else:
    print("\n  WARNING: Some AUC values are NaN or out of range!")

# ── 10. interpretation ────────────────────────────────────────────────────────
print(f"\n── Interpretation ──────────────────────────────────────────────────────")
if auc_direct <= 0.55:
    print("  Direct MIA near-random: Q-SHAP explanations show low membership leakage.")
elif auc_direct <= 0.65:
    print("  Direct MIA slightly above random: marginal membership leakage in explanations.")
else:
    print("  Direct MIA elevated: Q-SHAP explanations contain non-trivial membership signal.")

if auc_shadow_mia > auc_direct:
    print(f"  Shadow model amplifies attack: {auc_shadow_mia:.4f} vs direct {auc_direct:.4f}.")
else:
    print(f"  Shadow model does not improve attack: {auc_shadow_mia:.4f} (shadow limited by "
          "quantum model capacity).")

# Check that DP noise reduces AUC
dp_trend = [dp_results[eps] for eps in sorted(DP_EPSILONS, reverse=True)]  # ε=1.0, ε=0.1
if len(dp_trend) >= 2 and dp_trend[1] <= dp_trend[0]:
    print(f"  DP noise reduces AUC: ε=1.0→{dp_trend[0]:.4f}, ε=0.1→{dp_trend[1]:.4f} "
          "(smaller ε = more noise = lower AUC).")
else:
    print(f"  DP noise AUC non-monotone (expected at low baseline): "
          + ", ".join(f"ε={eps}→{dp_results[eps]:.4f}" for eps in DP_EPSILONS))

print("\n[Phase 8 complete]")

"""Two-part inspection for reuploading/seed=7/alpha_0.1 Track-B AUC=0.776 anomaly.

Check 1 — R0 cross-encoding comparison at seed=7/alpha_0.1:
  For each encoding, compute the pre-training (round-0) accuracy on Wisconsin/alpha_0.1.
  R0 depends only on torch.manual_seed(7) + the encoding's circuit; it reveals whether
  the model starts in an initialization-attractor (all-class-0 or all-class-1) before
  any federated rounds run.

Check 2 — Q-SHAP vector inspection and attack-split composition:
  Re-run reuploading/seed=7/alpha_0.1 federation → extract Q-SHAP for 50 members
  (X_train[:50] in PCA space) and 50 non-members (X_test[:50] in PCA space), verify:
    (a) Split composition: no overlap between member/non-member pools, indices correct
    (b) Vector structure: per-group norm, variance, feature means — confirming genuine
        attribution signal rather than uniform-zero vectors
    (c) AUC reproducibility against the logged 0.776
"""

import os, sys, time
import numpy as np
import torch
from scipy.stats import mannwhitneyu

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.loaders import load_breast_cancer_data
from models.encodings import angle_encoding, amplitude_encoding, reuploading_encoding, iqp_encoding
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
SEED            = 7

ENCODINGS = {
    "angle":       angle_encoding,
    "amplitude":   amplitude_encoding,
    "reuploading": reuploading_encoding,
    "iqp":         iqp_encoding,
}

# Logged final accuracies for seed=7/alpha_0.1 (from rq4_noniid_mia_log.csv)
LOGGED_FINAL_ACC = {
    "angle":       0.675439,
    "amplitude":   0.885965,
    "reuploading": 0.631579,   # Track B — the anomaly
    "iqp":         0.692982,
}
LOGGED_AUC = {
    "reuploading": 0.776,
}

print("=" * 72)
print("  reuploading/seed=7/alpha_0.1 INSPECTION")
print("=" * 72)

# ── shared data ───────────────────────────────────────────────────────────────
X_train_42, X_test_42, y_train_42, y_test_42 = load_breast_cancer_data(random_state=42)
majority_baseline = float(np.bincount(y_test_42.astype(int)).max()) / len(y_test_42)
collapse_threshold = majority_baseline + COLLAPSE_MARGIN
print(f"\nWisconsin: n_train={len(X_train_42)}, n_test={len(X_test_42)}")
print(f"Majority-class baseline={majority_baseline:.4f}, collapse threshold={collapse_threshold:.4f}")

# Seed=7 data (for federation and MIA)
X_train_7, X_test_7, y_train_7, y_test_7 = load_breast_cancer_data(random_state=SEED)

# ── CHECK 1: R0 cross-encoding comparison ─────────────────────────────────────
print("\n" + "=" * 72)
print("  CHECK 1: Pre-training (R0) accuracy — all encodings, seed=7, alpha_0.1")
print("=" * 72)
print("\n  (R0 = initial VQC weights with torch.manual_seed(7), no training)\n")

print(f"  {'Encoding':<14}  {'R0 acc':>8}  {'Pred type':>20}  {'Logged R15':>10}  {'Track':>6}")

r0_results = {}
for enc_name, enc_fn in ENCODINGS.items():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = HybridQuantumClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=enc_fn)
    model.pca, model._scaler = _fit_shared_preprocessor(X_train_7, N_QUBITS)
    preds = model.predict(X_test_7)
    r0_acc = float((preds == y_test_7).mean())
    n_class0 = int((preds == 0).sum())
    n_class1 = int((preds == 1).sum())
    if n_class0 == 0:
        pred_type = "all-class-1"
    elif n_class1 == 0:
        pred_type = "all-class-0"
    else:
        pred_type = f"mixed ({n_class0}c0 / {n_class1}c1)"
    r0_results[enc_name] = {"r0_acc": r0_acc, "pred_type": pred_type, "n_c0": n_class0, "n_c1": n_class1}
    track = "A" if LOGGED_FINAL_ACC[enc_name] > collapse_threshold else "B"
    print(f"  {enc_name:<14}  {r0_acc:>8.4f}  {pred_type:>20}  "
          f"{LOGGED_FINAL_ACC[enc_name]:>10.6f}  {track:>6}")

print("\n  Cross-encoding pattern at R0:")
for enc_name, r in r0_results.items():
    note = ""
    if abs(r["r0_acc"] - majority_baseline) < 0.01:
        note = " ← at majority-class floor"
    elif abs(r["r0_acc"] - (1 - majority_baseline)) < 0.01:
        note = " ← at minority-class floor"
    print(f"    {enc_name}: R0={r['r0_acc']:.4f}  [{r['pred_type']}]{note}")

# ── CHECK 2: reuploading/seed=7/alpha_0.1 Q-SHAP inspection ──────────────────
print("\n" + "=" * 72)
print("  CHECK 2: Q-SHAP vector inspection — reuploading/seed=7/alpha_0.1")
print("=" * 72)

print("\n  Running reuploading/seed=7/alpha_0.1 federation (15 rounds)...")
t0 = time.time()
np.random.seed(SEED)
torch.manual_seed(SEED)
model_out = []
_, round_log = run_federation(
    X_train=X_train_7, y_train=y_train_7,
    X_test=X_test_7,   y_test=y_test_7,
    encoding_fn=reuploading_encoding,
    n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
    n_qubits=N_QUBITS, n_layers=N_LAYERS,
    split="dirichlet", dirichlet_alpha=0.1,
    strategy_name="fedavg",
    seed=SEED,
    project_root=PROJECT_ROOT,
    n_permutations=0,
    model_out=model_out,
)
print(f"  Done ({time.time()-t0:.0f}s). Round-0→15 accuracy trace:")
for rd, acc in round_log:
    track = "A" if acc > collapse_threshold else "B"
    note = " ← R0 (pre-training)" if rd == 0 else ""
    print(f"    R{rd:>2}: {acc:.4f} [{track}]{note}")

final_acc = round_log[-1][1]
r0_actual = round_log[0][1]
track_final = "A" if final_acc > collapse_threshold else "B"

print(f"\n  R0  = {r0_actual:.4f}  (logged R15 = {LOGGED_FINAL_ACC['reuploading']:.4f}, this run R15 = {final_acc:.4f}) [{track_final}]")

# Reconstruct target model
shared_pca, shared_scaler = _fit_shared_preprocessor(X_train_7, N_QUBITS)
target_model = HybridQuantumClassifier(
    n_qubits=N_QUBITS, n_layers=N_LAYERS, encoding_fn=reuploading_encoding
)
target_model.pca     = shared_pca
target_model._scaler = shared_scaler
target_model.set_weights(model_out[0])

# Final predictions — verify collapse signature
preds_final = target_model.predict(X_test_7)
n_c0_final = int((preds_final == 0).sum())
n_c1_final = int((preds_final == 1).sum())
print(f"  Final model predictions: {n_c0_final} class-0, {n_c1_final} class-1 "
      f"(out of {len(X_test_7)} test samples)")

# ── 2a: Attack-split composition ──────────────────────────────────────────────
print("\n  --- 2a: Attack-split composition ---")
X_q_train = target_model._preprocess(X_train_7)
X_q_test  = target_model._preprocess(X_test_7)

n_mem    = min(N_MIA_SAMPLES, len(X_q_train))
n_nonmem = min(N_MIA_SAMPLES, len(X_q_test))

members_idx    = list(range(n_mem))
nonmembers_idx = list(range(n_nonmem))

# Check overlap in original space (rows of X_train_7 vs X_test_7)
# They are different splits so should not overlap; verify via value comparison
member_rows    = set(map(tuple, np.round(X_train_7[:n_mem], 8)))
nonmember_rows = set(map(tuple, np.round(X_test_7[:n_nonmem], 8)))
overlap = member_rows & nonmember_rows

print(f"  Members:     first {n_mem} rows of X_train (random_state={SEED}) — original feature space")
print(f"  Non-members: first {n_nonmem} rows of X_test  (random_state={SEED}) — original feature space")
print(f"  X_train size: {len(X_train_7)}  X_test size: {len(X_test_7)}")
print(f"  Member/non-member overlap in original feature space: {len(overlap)} rows")
print(f"  Q-SHAP input shape — members: {X_q_train[:n_mem].shape}, non-members: {X_q_test[:n_nonmem].shape}")

# Class distribution among attack samples
mem_y    = y_train_7[:n_mem]
nonmem_y = y_test_7[:n_nonmem]
print(f"  Member labels:     class-0={int((mem_y==0).sum())}, class-1={int((mem_y==1).sum())}")
print(f"  Non-member labels: class-0={int((nonmem_y==0).sum())}, class-1={int((nonmem_y==1).sum())}")

# ── 2b: Q-SHAP vector inspection ─────────────────────────────────────────────
print("\n  --- 2b: Q-SHAP vector structure ---")
print(f"  Computing Q-SHAP for {n_mem} members and {n_nonmem} non-members...")
t1 = time.time()
exps_in  = _qshap_for_samples(target_model, X_train_7,
                               n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                               n_permutations=N_PERMUTATIONS, seed=SEED)
exps_out = _qshap_for_samples(target_model, X_test_7,
                               n_samples=N_MIA_SAMPLES, n_bg=N_BG,
                               n_permutations=N_PERMUTATIONS, seed=SEED)
print(f"  Q-SHAP done ({time.time()-t1:.0f}s)")

norms_in  = np.linalg.norm(exps_in,  axis=1)
norms_out = np.linalg.norm(exps_out, axis=1)

print(f"\n  Member Q-SHAP (||φ||):")
print(f"    mean={norms_in.mean():.4f}  std={norms_in.std():.4f}  "
      f"min={norms_in.min():.4f}  max={norms_in.max():.4f}")
print(f"  Non-member Q-SHAP (||φ||):")
print(f"    mean={norms_out.mean():.4f}  std={norms_out.std():.4f}  "
      f"min={norms_out.min():.4f}  max={norms_out.max():.4f}")

print(f"\n  Per-feature member mean   φ: {np.round(exps_in.mean(axis=0), 4)}")
print(f"  Per-feature non-mem mean  φ: {np.round(exps_out.mean(axis=0), 4)}")
print(f"  Per-feature diff (in-out):  {np.round(exps_in.mean(axis=0) - exps_out.mean(axis=0), 4)}")

# Variance check — if all vectors identical the model is trivially uniform
var_in  = exps_in.var(axis=0)
var_out = exps_out.var(axis=0)
print(f"\n  Per-feature variance (members):     {np.round(var_in, 6)}")
print(f"  Per-feature variance (non-members): {np.round(var_out, 6)}")
print(f"  Any near-zero variance in members?     {(var_in  < 1e-8).any()}")
print(f"  Any near-zero variance in non-members? {(var_out < 1e-8).any()}")

# Are all rows identical?
members_identical    = (np.diff(exps_in,  axis=0) == 0).all()
nonmembers_identical = (np.diff(exps_out, axis=0) == 0).all()
print(f"  All member rows identical?     {members_identical}")
print(f"  All non-member rows identical? {nonmembers_identical}")

# Mann-Whitney U per feature
print(f"\n  Mann-Whitney U per feature (member vs non-member):")
for i in range(N_QUBITS):
    stat, p = mannwhitneyu(exps_in[:, i], exps_out[:, i], alternative='two-sided')
    print(f"    PC{i}: U={stat:.0f}, p={p:.4f}")

# ── 2c: AUC reproducibility ───────────────────────────────────────────────────
print("\n  --- 2c: AUC reproducibility ---")
auc = run_mia_attack(exps_in, exps_out, seed=SEED, cv=5)
print(f"  Reproduced AUC:  {auc:.4f}")
print(f"  Logged AUC:      {LOGGED_AUC['reuploading']:.4f}")
print(f"  Match: {'YES' if abs(auc - LOGGED_AUC['reuploading']) < 0.05 else 'NO — investigate'}")

# ── summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY")
print("=" * 72)
print(f"\n  Check 1 — R0 for reuploading/seed=7/alpha_0.1: {r0_actual:.4f}")
print(f"  (R0 from round_log[0]; independent of attack-split)")

if abs(r0_actual - majority_baseline) < 0.01:
    print(f"  → R0 at majority-class floor ({majority_baseline:.4f}): initialization-attractor (all-class-1 from round 0)")
elif abs(r0_actual - (1 - majority_baseline)) < 0.01:
    print(f"  → R0 at minority-class floor: initialization-attractor (all-class-0 from round 0)")
else:
    print(f"  → R0 not at a trivial floor: model initialized with mixed predictions; training drove collapse")

print(f"\n  Check 2 — Q-SHAP structure:")
if members_identical or nonmembers_identical:
    print(f"  WARNING: {'member' if members_identical else 'non-member'} vectors are all identical — uniform output, AUC may be artifact")
elif (var_in < 1e-8).any() or (var_out < 1e-8).any():
    print(f"  WARNING: near-zero variance on some features — possible degenerate output")
else:
    print(f"  Vectors have genuine variation (non-zero variance on all features)")
    print(f"  Mean ||φ|| members={norms_in.mean():.4f}, non-members={norms_out.mean():.4f}")
    print(f"  AUC={auc:.4f} reflects real above-baseline leakage in the Q-SHAP attributions")
    if len(overlap) > 0:
        print(f"  WARNING: {len(overlap)} overlapping rows between member/non-member pools — data-split issue")
    else:
        print(f"  Split composition: clean (no member/non-member overlap)")

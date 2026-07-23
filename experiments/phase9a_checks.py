"""Phase 9a pre-checks before Phase 9b.

Check 1 — Pima classical baseline:
    Run ClassicalBaselineClassifier (same matched-parameter MLP as Phase 2)
    on Pima Indians Diabetes with identical PCA(4) + MinMaxScaler([0,π])
    preprocessing, 5 seeds (1, 7, 42, 123, 999).
    Question: does the 13-parameter classical model also fail to beat the
    65.1% majority-class baseline on Pima?  If yes, the weakness is the
    parameter / feature budget, not the quantum encoding.

Check 2 — Wisconsin/angle discrete-accuracy analysis:
    Seeds 1, 42, 123, 999 all hit 84.2% (96/114 correct) in the Phase 9a
    federation sweep.  Retrain those 4 seeds with model_out to recover
    final weights, then compare per-sample misclassification patterns.
    Question: same 18 samples misclassified by all 4 seeds (genuine ceiling)?
    Or same COUNT but different samples (discrete-accuracy artifact)?

Run from project root:
    python experiments/phase9a_checks.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch

from data.loaders import load_breast_cancer_data, load_pima_data
from models.encodings import angle_encoding
from models.quantum_classifier import HybridQuantumClassifier
from models.classical_baseline import ClassicalBaselineClassifier
from federation.run_federation import run_federation, _fit_shared_preprocessor

SEEDS_5 = [1, 7, 42, 123, 999]
SEEDS_84 = [1, 42, 123, 999]     # the 4 Wisconsin/angle seeds that tied at 84.2%

N_QUBITS     = 4
N_LAYERS     = 2
N_CLIENTS    = 4
N_ROUNDS     = 15
LOCAL_EPOCHS = 3
# Classical: 45 epochs mirrors 15 rounds × 3 local epochs total training
CLASSICAL_EPOCHS = 45
LR = 0.05

# ── data (fixed train/test splits) ────────────────────────────────────────────
X_wis_tr, X_wis_te, y_wis_tr, y_wis_te = load_breast_cancer_data(random_state=42)
X_pim_tr, X_pim_te, y_pim_tr, y_pim_te = load_pima_data(random_state=42)

majority_pima = float(max((y_pim_te == 0).mean(), (y_pim_te == 1).mean()))
majority_class = int((y_pim_te == 0).mean() > 0.5)   # 0 = non-diabetic is majority

print("=== Phase 9a pre-checks ===\n")
print(f"  Pima test set: {len(y_pim_te)} samples  "
      f"majority-class ({majority_class}) rate = {majority_pima:.4f} ({majority_pima*100:.1f}%)")
print(f"  Wisconsin test set: {len(y_wis_te)} samples\n")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 1: Pima classical MLP baseline (5 seeds)
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("=== Check 1: Pima classical MLP baseline (5 seeds) ===\n")

# Count parameters of the matched MLP for the same n_qubits/n_layers as Phase 9a
_probe = ClassicalBaselineClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS)
n_classical_params = _probe.count_parameters()
n_quantum_params   = N_LAYERS * N_QUBITS + N_QUBITS + 1   # VQC weights + Linear(4,1)
print(f"  Quantum model params : {n_quantum_params}")
print(f"  Classical MLP params : {n_classical_params}")
print(f"  Majority-class baseline : {majority_pima*100:.1f}%\n")

classical_accs = []
for seed in SEEDS_5:
    np.random.seed(seed)
    torch.manual_seed(seed)
    clf = ClassicalBaselineClassifier(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    clf.fit(X_pim_tr, y_pim_tr, epochs=CLASSICAL_EPOCHS, lr=LR)
    acc = float((clf.predict(X_pim_te) == y_pim_te).mean())
    classical_accs.append(acc)
    beats = "beats baseline" if acc > majority_pima else "BELOW BASELINE"
    print(f"  seed={seed:>4}: classical acc={acc:.4f}  ({beats})")

cl_mean = float(np.mean(classical_accs))
cl_std  = float(np.std(classical_accs))
print(f"\n  Classical MLP on Pima: mean={cl_mean:.4f}  std={cl_std:.4f}")
print(f"  Majority-class baseline : {majority_pima:.4f}")
print(f"  Gap (classical - baseline): {cl_mean - majority_pima:+.4f}")

if cl_mean <= majority_pima + 0.02:
    print(f"\n  FINDING: Classical matched-parameter MLP also fails to clearly "
          f"beat the\n  majority-class baseline on Pima ({cl_mean*100:.1f}% vs "
          f"{majority_pima*100:.1f}%).\n  Correct framing: the weakness is the "
          f"parameter / feature budget\n  (4 PCA components, {n_quantum_params} trainable params), "
          f"not the quantum encoding.")
else:
    print(f"\n  FINDING: Classical MLP clears the majority-class baseline by "
          f"{(cl_mean - majority_pima)*100:.1f}pp\n  while quantum angle encoding does not. "
          f"Quantum encoding is the bottleneck on Pima, not model capacity.")

# ══════════════════════════════════════════════════════════════════════════════
# CHECK 2: Wisconsin/angle discrete-accuracy analysis
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("=== Check 2: Wisconsin/angle misclassification overlap ===\n")

n_test     = len(y_wis_te)
n_correct  = round(0.8421052631578947 * n_test)   # 96/114
n_wrong    = n_test - n_correct
print(f"  Test set size: {n_test}  (96/114 = 84.21% for seeds 1, 42, 123, 999)")
print(f"  Misclassified per seed: {n_wrong} samples\n")

error_masks = {}   # seed → boolean array, True = wrong prediction

for seed in SEEDS_84:
    print(f"  Retraining Wisconsin/angle seed={seed} (federation) ...", flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_out = []
    _, round_log = run_federation(
        X_train=X_wis_tr, y_train=y_wis_tr,
        X_test=X_wis_te,  y_test=y_wis_te,
        encoding_fn=angle_encoding,
        n_clients=N_CLIENTS, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS,
        n_qubits=N_QUBITS, n_layers=N_LAYERS,
        split="iid", strategy_name="fedavg", seed=seed,
        project_root=PROJECT_ROOT, n_permutations=0,
        model_out=model_out,
    )

    final_acc = round_log[-1][1]
    print(f"    final_acc={final_acc:.4f}  (expected 0.8421)")

    shared_pca, shared_scaler = _fit_shared_preprocessor(X_wis_tr, N_QUBITS)
    model           = HybridQuantumClassifier(N_QUBITS, N_LAYERS, angle_encoding)
    model.pca       = shared_pca
    model._scaler   = shared_scaler
    model.set_weights(model_out[0])

    preds = model.predict(X_wis_te)
    errors = (preds != y_wis_te)   # True where wrong
    error_masks[seed] = errors
    wrong_idx = np.where(errors)[0]
    print(f"    wrong indices: {wrong_idx.tolist()}")

# pairwise overlap
print(f"\n  Pairwise misclassification overlap ({n_wrong} misclassified per seed):")
print(f"  {'Seeds':>12}  {'Shared errors':>13}  {'Distinct A':>10}  {'Distinct B':>10}  "
      f"{'Jaccard':>8}  {'Same samples?':>13}")
print(f"  {'-'*12}  {'-'*13}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*13}")

seeds_list = SEEDS_84
pair_overlaps = []
for i in range(len(seeds_list)):
    for j in range(i + 1, len(seeds_list)):
        s_a, s_b = seeds_list[i], seeds_list[j]
        m_a, m_b = error_masks[s_a], error_masks[s_b]
        shared   = int((m_a & m_b).sum())
        only_a   = int((m_a & ~m_b).sum())
        only_b   = int((~m_a & m_b).sum())
        union    = int((m_a | m_b).sum())
        jaccard  = shared / union if union > 0 else 1.0
        same     = shared == n_wrong == int(m_a.sum()) == int(m_b.sum())
        pair_overlaps.append(jaccard)
        label    = "YES (identical)" if same else f"NO  ({shared}/{n_wrong} shared)"
        print(f"  {s_a:>5}↔{s_b:<5}  {shared:>13}  {only_a:>10}  {only_b:>10}  "
              f"{jaccard:>8.4f}  {label:>13}")

# find all-four overlap
all_wrong_idx = np.where(
    error_masks[seeds_list[0]] &
    error_masks[seeds_list[1]] &
    error_masks[seeds_list[2]] &
    error_masks[seeds_list[3]]
)[0]
any_wrong_idx = np.where(
    error_masks[seeds_list[0]] |
    error_masks[seeds_list[1]] |
    error_masks[seeds_list[2]] |
    error_masks[seeds_list[3]]
)[0]

print(f"\n  Samples wrong in ALL 4 seeds  : {len(all_wrong_idx):2d}  {all_wrong_idx.tolist()}")
print(f"  Samples wrong in ANY seed     : {len(any_wrong_idx):2d}")
print(f"  Per-seed wrong count          : {[int(error_masks[s].sum()) for s in SEEDS_84]}")

# interpret
mean_jaccard = float(np.mean(pair_overlaps))
print(f"\n  Mean pairwise Jaccard similarity: {mean_jaccard:.4f}")

if mean_jaccard > 0.95:
    verdict = ("IDENTICAL FAILURE SET: all seeds misclassify the same samples.\n"
               "  Interpretation: the model has a genuine accuracy ceiling — those\n"
               "  18 samples are structurally hard at this PCA dimensionality / param count.")
elif mean_jaccard > 0.5:
    verdict = ("PARTIAL OVERLAP: seeds agree on a core hard subset but not all.\n"
               "  Interpretation: a hard core exists, plus some sample-level variance.")
else:
    verdict = ("LOW OVERLAP (DISCRETE-ACCURACY ARTIFACT): seeds hit the same COUNT\n"
               "  (96/114) but misclassify different samples. The 84.2% is a\n"
               "  discretisation artefact — 96 is simply the nearest integer to\n"
               "  whichever test accuracy this config converges to (~84%),\n"
               "  not evidence that the model has 'saturated' or plateaued.\n"
               "  Analogous to tau=+1 saturation at n=4 features in Phase 6:\n"
               "  the metric resolution masks real variation in the underlying model.")
print(f"\n  VERDICT: {verdict}")

# ── methods note ─────────────────────────────────────────────────────────────
notes_path = os.path.join(PROJECT_ROOT, "analysis", "methods_notes.md")
mn02_tag   = "## MN-02"
with open(notes_path) as f:
    existing = f.read()

if mn02_tag not in existing:
    artifact_label = "LOW OVERLAP" if mean_jaccard < 0.5 else "HIGH OVERLAP"
    mn02 = f"""
---

## MN-02: Discrete-Accuracy Artifact at Small Test Sets

**Status:** Confirmed (Phase 9a checks, 2026-07-22)
**Affects:** Accuracy comparisons when test set n is small enough that 1 sample = ~1 pp.

### Finding

In Phase 9a, Wisconsin/angle encoding hit 84.21% (96/114 correct) across 4 of 5 seeds
(seeds 1, 42, 123, 999). With 114 test samples, each sample is worth 0.877pp.  The
apparent "tie" at 84.21% may reflect:

- **(a) Genuine ceiling:** all 4 seeds misclassify the same {n_wrong} samples — same hard
  subset, real convergence.
- **(b) Discrete-accuracy artifact:** each seed misclassifies {n_wrong} different samples,
  and 96/114 is simply the nearest integer to where the angle-encoding model converges
  (~84%), not evidence of saturation.

### Result (Phase 9a check)

Mean pairwise Jaccard between misclassification sets: {mean_jaccard:.4f}
Samples wrong in ALL 4 seeds: {len(all_wrong_idx)} / {n_wrong}
Verdict: {artifact_label}

"""
    if mean_jaccard < 0.5:
        mn02 += """The evidence supports interpretation (b): discrete-accuracy artifact.
The 84.21% ceiling is a resolution artefact of a 114-sample test set combined with a
model that converges to ~84% accuracy regardless of random seed. Do NOT describe this
as the model 'saturating' at a performance ceiling. Report as: all seeds converge to
the same mean accuracy level; count-level ties arise from integer rounding at n=114.

"""
    else:
        mn02 += f"""The evidence supports interpretation (a): genuine hard subset.
{len(all_wrong_idx)} samples are misclassified by every seed — these are structurally
difficult for this architecture (4 PCA components, angle encoding).

"""
    mn02 += """### Protocol

When reporting tied accuracies across seeds at small test sets (n < 200):
1. Always check Jaccard overlap of misclassified samples before attributing to "ceiling".
2. If mean Jaccard < 0.5: report as "discrete-accuracy artifact" (resolution limit).
3. If mean Jaccard > 0.9: report as "hard subset" (genuine failure mode).
4. Increase test set size or use k-fold CV to resolve ambiguity.

See MN-01 for the analogous metric-resolution issue in Kendall's tau at n=4 features.
"""
    with open(notes_path, "a") as f:
        f.write(mn02)
    print(f"\n  [methods_notes.md] MN-02 appended.")
else:
    print(f"\n  [methods_notes.md] MN-02 already present, skipping.")

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print("=== Summary ===\n")
print(f"  Check 1: Pima classical baseline")
print(f"    Classical MLP ({n_classical_params} params, 4 PCA features): "
      f"{cl_mean*100:.1f}% ± {cl_std*100:.1f}%")
print(f"    Majority-class baseline: {majority_pima*100:.1f}%")
print(f"    Quantum angle (Phase 9a): 60.0% ± 1.5%")
if cl_mean <= majority_pima + 0.02:
    print(f"    → Both classical and quantum fail to beat baseline. "
          f"Capacity/feature bottleneck, not encoding.")
else:
    print(f"    → Classical beats baseline, quantum does not. Encoding bottleneck.")

print(f"\n  Check 2: Wisconsin/angle misclassification overlap")
print(f"    Test set n={n_test}, n_wrong={n_wrong} per seed")
print(f"    All-4-seed common errors: {len(all_wrong_idx)}/{n_wrong}")
print(f"    Mean pairwise Jaccard: {mean_jaccard:.4f}")
if mean_jaccard < 0.5:
    print(f"    → DISCRETE-ACCURACY ARTIFACT. Document as MN-02.")
else:
    print(f"    → GENUINE HARD SUBSET. {len(all_wrong_idx)} structurally hard samples.")

print(f"\n[Phase 9a checks complete]")

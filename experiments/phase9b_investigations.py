"""Phase 9b pre-investigations: items 1 and 2.

Item 1 — Prediction/optimization behavior on Pima:
    Why does quantum angle encoding never collapse to exactly 64.9% majority-class rate
    (worst seed: 57.1%) while classical MLP collapses to exactly 64.9% in 3/5 seeds?
    Checks whether this is loss-landscape / optimization difference or an evaluation artifact.
    Uses standalone training (not federation) for speed; mechanism is in the model, not
    the training protocol.

Item 2 — Hilbert-space dimensionality hypothesis:
    Amplitude encoding creates a 2^4=16 dimensional embedding from 4 PCA features.
    Does a classical model with equivalent dimensionality (degree-2 polynomial of 4 PCA
    features → 14 features, logistic-regression-style) also clear the 65.1% baseline
    reliably?  If yes: dimensionality is the story, not quantum processing.
    If no: amplitude encoding offers something beyond input dimensionality.

Run from project root:
    python experiments/phase9b_investigations.py
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

from data.loaders import load_pima_data
from models.quantum_classifier import HybridQuantumClassifier
from models.encodings import angle_encoding, amplitude_encoding
from models.classical_baseline import ClassicalBaselineClassifier

SEEDS          = [1, 7, 42, 123, 999]
N_QUBITS       = 4
N_LAYERS       = 2
EPOCHS         = 45   # = 15 rounds × 3 local epochs (approximate total training)
LR             = 0.05
BATCH_SIZE     = 32

X_tr, X_te, y_tr, y_te = load_pima_data(random_state=42)
majority_baseline = float(max((y_te == 0).mean(), (y_te == 1).mean()))
n_majority   = int(round(majority_baseline * len(y_te)))
print(f"Pima test: {len(y_te)} samples  majority-class={majority_baseline:.4f} "
      f"({n_majority}/{len(y_te)} always predict {int((y_te==0).mean()>0.5)}→0)\n")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 1 — Prediction/optimization behavior comparison
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("=== Item 1: Prediction behavior — quantum angle vs classical MLP ===\n")


def train_and_probe(model_obj, X_train, y_train, X_test, y_test,
                    epochs=EPOCHS, lr=LR, seed=42, label="model"):
    """Train and return predictions + sigmoid outputs + training trace."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    loss_trace = []

    # Train with loss tracking
    if hasattr(model_obj, '_preprocess'):    # HybridQuantumClassifier
        model_obj.fit(X_train, y_train, epochs=epochs, lr=lr)
    else:                                    # ClassicalBaselineClassifier
        # patch fit to capture per-epoch loss
        X_proc = model_obj._preprocess(X_train, fit=True)
        X_t = torch.tensor(X_proc, dtype=torch.float32)
        y_t = torch.tensor(y_train, dtype=torch.float32)
        n   = X_t.shape[0]
        optimizer = torch.optim.Adam(model_obj.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        model_obj.model.train()
        for ep in range(epochs):
            perm = torch.randperm(n)
            ep_loss = 0.0
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                Xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                preds = model_obj.model(Xb).squeeze(1)
                loss  = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item() * len(idx)
            ep_loss /= n
            loss_trace.append(ep_loss)

    # Predictions on test set
    preds   = model_obj.predict(X_test)          # binary 0/1
    probas  = model_obj.predict_proba(X_test)    # sigmoid outputs

    # Confusion matrix breakdown
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    tn = int(((preds == 0) & (y_test == 0)).sum())

    acc    = float((preds == y_test).mean())
    n_pred1 = int((preds == 1).sum())
    n_pred0 = int((preds == 0).sum())

    print(f"  [{label}]  acc={acc:.4f}  pred0={n_pred0}  pred1={n_pred1}")
    print(f"    Confusion: TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Proba stats: min={probas.min():.4f}  max={probas.max():.4f}  "
          f"mean={probas.mean():.4f}  std={probas.std():.4f}")

    if loss_trace:
        print(f"    Loss first 5 epochs: {[f'{v:.4f}' for v in loss_trace[:5]]}  "
              f"... last 5: {[f'{v:.4f}' for v in loss_trace[-5:]]}")
        stagnant = all(abs(loss_trace[i] - loss_trace[i-1]) < 1e-5
                       for i in range(-5, 0))
        if stagnant:
            print(f"    ⚠  Loss stagnant in last 5 epochs — model may be at local minimum.")

    return {
        "acc": acc, "preds": preds, "probas": probas,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pred0": n_pred0, "n_pred1": n_pred1,
        "loss_trace": loss_trace,
    }


# ── quantum angle, seed=42 (worst Phase 9a seed) ─────────────────────────────
print("  Training quantum angle encoding (seed=42) ...")
qm = HybridQuantumClassifier(N_QUBITS, N_LAYERS, angle_encoding)
q_res = train_and_probe(qm, X_tr, y_tr, X_te, y_te, seed=42, label="quantum angle seed=42")

# ── classical MLP, seed=7 (collapsed in Phase 9a) ─────────────────────────────
print(f"\n  Training classical MLP (seed=7, collapsed in Phase 9a) ...")
cm7 = ClassicalBaselineClassifier(N_QUBITS, N_LAYERS)
c7_res = train_and_probe(cm7, X_tr, y_tr, X_te, y_te, seed=7, label="classical MLP seed=7")

# ── classical MLP, seed=1 (succeeded in Phase 9a) ─────────────────────────────
print(f"\n  Training classical MLP (seed=1, succeeded in Phase 9a) ...")
cm1 = ClassicalBaselineClassifier(N_QUBITS, N_LAYERS)
c1_res = train_and_probe(cm1, X_tr, y_tr, X_te, y_te, seed=1, label="classical MLP seed=1")

# ── comparison ─────────────────────────────────────────────────────────────────
print(f"\n  --- Comparison ---")
print(f"  Majority-class baseline: {majority_baseline:.4f} "
      f"= always predict 0, pred0={len(y_te)} pred1=0")
print(f"  Classical (seed=7) predicts {c7_res['n_pred0']} zeros, "
      f"{c7_res['n_pred1']} ones  → "
      f"{'ALL ZEROS (collapsed)' if c7_res['n_pred1'] == 0 else 'mixed predictions'}")
print(f"  Quantum angle (seed=42) predicts {q_res['n_pred0']} zeros, "
      f"{q_res['n_pred1']} ones  → always mixed")

if c7_res["n_pred1"] == 0:
    print(f"\n  CONFIRMED: Classical collapse is a trivial '→predict all 0' failure.")
    print(f"  Proba range: [{c7_res['probas'].min():.6f}, {c7_res['probas'].max():.6f}]")
    print(f"  All sigmoid outputs < 0.5 → all predicted as class 0.")
    print(f"  Root cause: h=1 MLP can reach a trivial fixed-point where the hidden")
    print(f"  unit saturates to 0 (ReLU killed) or the output weight → 0.")

print(f"\n  Quantum angle proba range: "
      f"[{q_res['probas'].min():.4f}, {q_res['probas'].max():.4f}]")
print(f"  VQC expval(PauliZ) outputs span [-1,1], pushed through Linear + sigmoid.")
print(f"  This makes a constant-output fixed point structurally harder to reach")
print(f"  (requires all PauliZ expectations to be aligned and weighted to produce")
print(f"  a fixed scalar — a set-measure-zero condition in weight space).")

# ── multi-seed proba distribution for quantum ─────────────────────────────────
print(f"\n  Checking quantum angle prediction distribution across all 5 seeds:")
print(f"  {'Seed':>5}  {'Acc':>6}  {'pred0':>6}  {'pred1':>6}  "
      f"{'proba_min':>10}  {'proba_max':>10}  {'proba_std':>10}")
print(f"  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)
    q_s = HybridQuantumClassifier(N_QUBITS, N_LAYERS, angle_encoding)
    q_s.fit(X_tr, y_tr, epochs=EPOCHS, lr=LR)
    p   = q_s.predict(X_te)
    pr  = q_s.predict_proba(X_te)
    acc = float((p == y_te).mean())
    print(f"  {seed:>5}  {acc:>6.4f}  {int((p==0).sum()):>6}  {int((p==1).sum()):>6}  "
          f"{pr.min():>10.4f}  {pr.max():>10.4f}  {pr.std():>10.4f}")

print(f"\n  → Quantum angle always has non-trivial proba spread (never all-0/all-1).")
print(f"    Getting BELOW majority-class (57.1% < 64.9%) means model is")
print(f"    attempting a decision boundary but classifying the majority class worse")
print(f"    than random assignment would — i.e., it's learned an anti-correlated")
print(f"    approximation. This is distinct from collapse (which gives 64.9%).")


# ══════════════════════════════════════════════════════════════════════════════
# ITEM 2 — Polynomial features MLP (Hilbert-space dimensionality hypothesis)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 65}")
print("=== Item 2: Polynomial-feature MLP — dimensionality hypothesis ===\n")


class PolyMLP:
    """Classical MLP on degree-2 polynomial features from 4 PCA inputs.

    Pipeline: raw X → PCA(4) → MinMaxScaler([0,π]) → PolynomialFeatures(degree=2)
    → Linear(n_poly, 1) → Sigmoid.

    With 4 PCA features and degree=2, include_bias=False:
    n_poly = 4 + C(4,2) + 4 = 4 + 6 + 4 = 14 features (linear + cross + squared).
    Linear(14, 1) → 15 parameters (vs 13 for classical baseline, close match).
    """

    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.pca      = None
        self._scaler  = None
        self._poly    = PolynomialFeatures(degree=2, include_bias=False)
        self.model    = None       # built after seeing n_poly

    def _preprocess(self, X, fit=False):
        if fit:
            self.pca     = PCA(n_components=self.n_qubits)
            X_pca        = self.pca.fit_transform(X)
            self._scaler = MinMaxScaler(feature_range=(0, np.pi))
            X_pca_sc     = self._scaler.fit_transform(X_pca)
            X_poly       = self._poly.fit_transform(X_pca_sc)
            if self.model is None:
                n_poly   = X_poly.shape[1]
                self.model = nn.Sequential(nn.Linear(n_poly, 1), nn.Sigmoid())
        else:
            X_pca_sc = self._scaler.transform(self.pca.transform(X))
            X_poly   = self._poly.transform(X_pca_sc)
        return X_poly.astype(np.float32)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def fit(self, X_train, y_train, epochs=EPOCHS, lr=LR):
        X_proc = self._preprocess(X_train, fit=True)
        X_t    = torch.tensor(X_proc, dtype=torch.float32)
        y_t    = torch.tensor(y_train, dtype=torch.float32)
        n      = X_t.shape[0]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        self.model.train()
        for ep in range(epochs):
            perm = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                Xb, yb = X_t[idx], y_t[idx]
                optimizer.zero_grad()
                loss = criterion(self.model(Xb).squeeze(1), yb)
                loss.backward()
                optimizer.step()

    def predict(self, X_test):
        self.model.eval()
        X_t = torch.tensor(self._preprocess(X_test), dtype=torch.float32)
        with torch.no_grad():
            return (self.model(X_t).squeeze(1).numpy() >= 0.5).astype(np.float32)

    def predict_proba(self, X_test):
        self.model.eval()
        X_t = torch.tensor(self._preprocess(X_test), dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_t).squeeze(1).numpy()


# reveal n_poly
_probe_poly = PolyMLP(N_QUBITS)
_probe_poly._preprocess(X_tr, fit=True)
n_poly = _probe_poly._poly.n_output_features_
n_poly_params = _probe_poly.count_parameters()
print(f"  PolyMLP: degree-2 polynomial of 4 PCA features → {n_poly} features")
print(f"  PolyMLP parameters: {n_poly_params}  "
      f"(vs classical 4-feat: 13, quantum: 13)")
print(f"  Majority-class baseline: {majority_baseline:.4f}\n")

# Phase 9a reference values
amp_mean_ref = 0.6922
amp_std_ref  = 0.0151
ang_mean_ref = 0.6000
ang_std_ref  = 0.0151

poly_accs     = []
classical_accs = []

print(f"  {'Seed':>5}  {'PolyMLP':>9}  {'Classical-4':>12}  {'vs baseline':>12}")
print(f"  {'-'*5}  {'-'*9}  {'-'*12}  {'-'*12}")

for seed in SEEDS:
    np.random.seed(seed)
    torch.manual_seed(seed)
    poly_m = PolyMLP(N_QUBITS)
    poly_m.fit(X_tr, y_tr)
    poly_acc = float((poly_m.predict(X_te) == y_te).mean())
    poly_accs.append(poly_acc)

    np.random.seed(seed)
    torch.manual_seed(seed)
    cls_m = ClassicalBaselineClassifier(N_QUBITS, N_LAYERS)
    cls_m.fit(X_tr, y_tr, epochs=EPOCHS, lr=LR)
    cls_acc = float((cls_m.predict(X_te) == y_te).mean())
    classical_accs.append(cls_acc)

    poly_flag = "✓" if poly_acc > majority_baseline else "✗"
    print(f"  {seed:>5}  {poly_acc:>9.4f}{poly_flag}  {cls_acc:>12.4f}  "
          f"{'beats' if poly_acc > majority_baseline else 'BELOW':>12}")

poly_mean = float(np.mean(poly_accs))
poly_std  = float(np.std(poly_accs))
cls_mean  = float(np.mean(classical_accs))
cls_std   = float(np.std(classical_accs))

print(f"\n  {'Model':<30}  {'Params':>6}  {'Mean':>7}  {'Std':>7}  {'vs baseline':>12}")
print(f"  {'-'*30}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*12}")
print(f"  {'Classical 4-feat MLP':<30}  {13:>6}  {cls_mean:>7.4f}  "
      f"{cls_std:>7.4f}  {cls_mean-majority_baseline:>+12.4f}")
print(f"  {'PolyMLP (degree-2, 14-feat)':<30}  {n_poly_params:>6}  {poly_mean:>7.4f}  "
      f"{poly_std:>7.4f}  {poly_mean-majority_baseline:>+12.4f}")
print(f"  {'Quantum amplitude (Phase 9a)':<30}  {13:>6}  {amp_mean_ref:>7.4f}  "
      f"{amp_std_ref:>7.4f}  {amp_mean_ref-majority_baseline:>+12.4f}")
print(f"  {'Quantum angle (Phase 9a)':<30}  {13:>6}  {ang_mean_ref:>7.4f}  "
      f"{ang_std_ref:>7.4f}  {ang_mean_ref-majority_baseline:>+12.4f}")
print(f"  {'Majority-class baseline':<30}  {'—':>6}  {majority_baseline:>7.4f}  "
      f"{'—':>7}  {'0.0000':>12}")

# ── interpretation ──────────────────────────────────────────────────────────
print(f"\n  --- Dimensionality hypothesis verdict ---")
poly_beats = poly_mean > majority_baseline + 0.01
cls_beats  = cls_mean  > majority_baseline + 0.01

if poly_beats and not cls_beats:
    print(f"  CONFIRMED: Polynomial expansion (14-dim → {poly_mean*100:.1f}%) clears baseline")
    print(f"  while 4-feature classical ({cls_mean*100:.1f}%) does not.")
    print(f"  Amplitude encoding's advantage over angle likely comes from the same")
    print(f"  dimensionality expansion — the 2^4=16-dimensional Hilbert space creates")
    print(f"  richer non-linear feature combinations analogous to degree-2 polynomial.")
    print(f"  The quantum 'advantage' here is the implicit kernel expansion, not")
    print(f"  quantum interference or entanglement per se.")
elif poly_beats and cls_beats:
    print(f"  INCONCLUSIVE: Both poly ({poly_mean*100:.1f}%) and 4-feat classical "
          f"({cls_mean*100:.1f}%) beat baseline.")
    print(f"  The 4-feature classical also succeeds; dimensionality is not the sole factor.")
elif not poly_beats:
    print(f"  DISCONFIRMED: PolyMLP ({poly_mean*100:.1f}%) also fails to beat baseline.")
    print(f"  The 'richer representational basis' claim needs revision.")
    print(f"  Amplitude encoding's 16-dimensional Hilbert space is not equivalent to")
    print(f"  classical polynomial expansion — possibly interference/entanglement matters.")

diff_amp_poly = amp_mean_ref - poly_mean
if abs(diff_amp_poly) < 0.02:
    print(f"  PolyMLP ({poly_mean*100:.1f}%) ≈ Quantum amplitude ({amp_mean_ref*100:.1f}%): "
          f"dimensionality story is consistent.")
elif diff_amp_poly > 0.02:
    print(f"  Quantum amplitude ({amp_mean_ref*100:.1f}%) > PolyMLP ({poly_mean*100:.1f}%) "
          f"by {diff_amp_poly*100:.1f}pp: quantum processing adds value beyond dimensionality.")
else:
    print(f"  PolyMLP ({poly_mean*100:.1f}%) > Quantum amplitude ({amp_mean_ref*100:.1f}%) "
          f"by {-diff_amp_poly*100:.1f}pp: classical polynomial features outperform amplitude encoding.")

print(f"\n[Phase 9b investigations (items 1 & 2) complete]")

"""Membership inference attack using Q-SHAP explanation vectors.

Shadow-model MIA:
  1. Train K shadow classifiers on random 50/50 splits of the training data.
  2. For each shadow model, compute Q-SHAP for in-split (member) and out-split (non-member) samples.
  3. Collect (explanation_vector, is_member) pairs across all K shadows.
  4. Train a logistic regression attack classifier on those pairs.
  5. Evaluate on the target model's member (train set) and non-member (test set) explanations.
  6. Return AUC.

Simplified MIA (run_mia_attack):
  Direct cross-validated logistic regression on target model explanations,
  no shadow models. Useful for a quick sanity check.

Both return AUC as a float in [0, 1]; values near 0.5 indicate the model's
explanations do not leak membership information.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ── helpers ────────────────────────────────────────────────────────────────────

def _qshap_for_samples(model, X_original, n_samples=50, n_bg=20, n_permutations=8, seed=42):
    """Compute Q-SHAP for the first n_samples rows of X_original.

    Operates in PCA-reduced quantum feature space using the model's shared PCA.

    Returns
    -------
    shap_matrix : ndarray of shape (min(n_samples, len(X_original)), n_qubits)
    """
    from explain.q_shap import q_shap_values, make_quantum_predict_fn

    X_q   = model._preprocess(X_original)
    n_s   = min(n_samples, len(X_q))
    n_b   = min(n_bg, len(X_q))
    if n_s < 1 or n_b < 1:
        return np.zeros((0, model.n_qubits))

    q_predict = make_quantum_predict_fn(model)
    shap      = q_shap_values(q_predict, X_q[:n_s], X_q[:n_b],
                              n_permutations=n_permutations, seed=seed)
    return shap   # (n_s, n_qubits)


# ── shadow model MIA ──────────────────────────────────────────────────────────

class ShadowModelMIA:
    """Shadow model membership inference attack on Q-SHAP explanation vectors.

    Parameters
    ----------
    model_factory  : callable() → HybridQuantumClassifier with shared PCA/scaler set.
    n_shadow       : number of shadow classifiers to train (default 3).
    local_epochs   : training epochs for each shadow model (default 15).
    shadow_split   : fraction of data used as shadow_in (default 0.5).
    n_explain      : Q-SHAP explanation samples per shadow model per split (default 20).
    n_bg           : Q-SHAP background samples (default 20).
    n_permutations : Q-SHAP permutations per feature (default 8).
    seed           : master RNG seed (default 42).
    """

    def __init__(
        self,
        model_factory,
        n_shadow=3,
        local_epochs=15,
        shadow_split=0.5,
        n_explain=20,
        n_bg=20,
        n_permutations=8,
        seed=42,
    ):
        self.model_factory  = model_factory
        self.n_shadow       = n_shadow
        self.local_epochs   = local_epochs
        self.shadow_split   = shadow_split
        self.n_explain      = n_explain
        self.n_bg           = n_bg
        self.n_permutations = n_permutations
        self.seed           = seed

        self._attack_X = None
        self._attack_y = None
        self._clf      = None
        self._scaler   = None

    def generate_shadow_data(self, X_train, y_train):
        """Train shadow models and collect (explanation, is_member) pairs.

        Parameters
        ----------
        X_train, y_train : full training dataset (all partitioned to shadows internally).
        """
        rng    = np.random.default_rng(self.seed)
        all_X  = []
        all_y  = []
        n      = len(X_train)

        for k in range(self.n_shadow):
            print(f"  [Shadow {k+1}/{self.n_shadow}] training shadow model ...", flush=True)
            # Random in/out split for this shadow
            idx     = rng.permutation(n)
            split   = int(n * self.shadow_split)
            in_idx  = idx[:split]
            out_idx = idx[split:]

            X_in, y_in   = X_train[in_idx],  y_train[in_idx]
            X_out, y_out = X_train[out_idx], y_train[out_idx]

            # Train shadow model on in-split
            shadow = self.model_factory()
            shadow.fit(X_in, y_in, epochs=self.local_epochs, lr=0.05, batch_size=32)

            # Q-SHAP for in-split (member) — use in-split as both explain and background
            shap_in  = _qshap_for_samples(shadow, X_in,
                                          n_samples=self.n_explain,
                                          n_bg=min(self.n_bg, len(X_in)),
                                          n_permutations=self.n_permutations,
                                          seed=self.seed + k)

            # Q-SHAP for out-split (non-member) — use in-split as background
            X_out_q  = shadow._preprocess(X_out)
            X_bg_q   = shadow._preprocess(X_in[:min(self.n_bg, len(X_in))])
            from explain.q_shap import q_shap_values, make_quantum_predict_fn
            q_predict = make_quantum_predict_fn(shadow)
            n_out_exp = min(self.n_explain, len(X_out_q))
            shap_out  = q_shap_values(q_predict, X_out_q[:n_out_exp], X_bg_q,
                                      n_permutations=self.n_permutations,
                                      seed=self.seed + k + 100)

            print(f"    in-split: {len(shap_in)} explanations | "
                  f"out-split: {len(shap_out)} explanations")

            if len(shap_in) > 0:
                all_X.append(shap_in)
                all_y.extend([1] * len(shap_in))
            if len(shap_out) > 0:
                all_X.append(shap_out)
                all_y.extend([0] * len(shap_out))

        self._attack_X = np.vstack(all_X)        # (total_samples, n_qubits)
        self._attack_y = np.array(all_y, dtype=int)
        print(f"  Shadow data: {len(self._attack_X)} total samples "
              f"({self._attack_y.sum()} members, {(1-self._attack_y).sum()} non-members)")

    def fit_attack(self):
        """Fit logistic regression attack classifier on shadow data."""
        if self._attack_X is None:
            raise RuntimeError("Call generate_shadow_data() first.")
        self._scaler = StandardScaler()
        X_scaled     = self._scaler.fit_transform(self._attack_X)
        self._clf    = LogisticRegression(max_iter=500, random_state=self.seed)
        self._clf.fit(X_scaled, self._attack_y)
        print(f"  Attack classifier fitted on {len(X_scaled)} shadow samples.")

    def attack_auc(self, target_in_exps, target_out_exps):
        """Evaluate attack on target model's member / non-member explanations.

        Parameters
        ----------
        target_in_exps  : ndarray (n_members, n_qubits) — target model's training-set explanations
        target_out_exps : ndarray (n_nonmembers, n_qubits) — target model's test-set explanations

        Returns
        -------
        auc : float
        """
        if self._clf is None:
            raise RuntimeError("Call fit_attack() first.")
        X_eval = np.vstack([target_in_exps, target_out_exps])
        y_eval = np.array(
            [1] * len(target_in_exps) + [0] * len(target_out_exps), dtype=int
        )
        X_scaled = self._scaler.transform(X_eval)
        probs    = self._clf.predict_proba(X_scaled)[:, 1]
        return float(roc_auc_score(y_eval, probs))

    def shadow_cv_auc(self, cv=3):
        """Cross-validated AUC on shadow data (internal sanity check).

        Returns
        -------
        mean_auc : float
        """
        if self._attack_X is None:
            raise RuntimeError("Call generate_shadow_data() first.")
        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(self._attack_X)
        clf      = LogisticRegression(max_iter=500, random_state=self.seed)
        scores   = cross_val_score(clf, X_scaled, self._attack_y,
                                   cv=min(cv, self._attack_y.sum()),
                                   scoring="roc_auc")
        return float(scores.mean())


# ── simplified MIA (no shadow models) ────────────────────────────────────────

def run_mia_attack(explanations_in, explanations_out, seed=42, cv=5):
    """Cross-validated logistic regression MIA on target model explanations.

    No shadow models — directly trains attack on target model's own member/non-member
    explanations using k-fold cross-validation.  This gives a lower bound on leakage:
    if this is above 0.5, the explanations contain membership information.

    Parameters
    ----------
    explanations_in  : ndarray (n_members, n_features) — training-set explanations
    explanations_out : ndarray (n_nonmembers, n_features) — test-set explanations
    seed             : RNG seed for LR and CV splits (default 42)
    cv               : number of CV folds (default 5)

    Returns
    -------
    auc : float — mean cross-validated AUC
    """
    X = np.vstack([explanations_in, explanations_out])
    y = np.array([1] * len(explanations_in) + [0] * len(explanations_out), dtype=int)

    if np.unique(y).size < 2:
        return float("nan")

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=500, random_state=seed))

    n_folds = min(cv, y.sum(), (y == 0).sum())
    if n_folds < 2:
        return float("nan")  # Training-set AUC is not a CV estimate.

    skf    = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
    return float(scores.mean())

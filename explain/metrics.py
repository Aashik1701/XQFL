"""Explanation quality metrics for Phase 4 (and later phases).

Implements:
  faithfulness   — deletion AUC: does removing high-attribution features hurt accuracy?
  infidelity     — expected squared mismatch between explanation and actual model change
  kendalls_tau   — rank correlation between two explanation vectors (round-to-round stability)
  spearman_corr  — Spearman rank correlation (same use)
  js_divergence  — Jensen-Shannon divergence between local and global explanation distributions
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr
from scipy.spatial.distance import jensenshannon


# ──────────────────────────────────────────────
# faithfulness
# ──────────────────────────────────────────────

def faithfulness(
    predict_fn,
    explanations: np.ndarray,
    X: np.ndarray,
    background_mean: np.ndarray = None,
    metric: str = "auc",
) -> float:
    """Measure explanation faithfulness via absolute-deviation deletion.

    For each sample, rank features by descending |attribution|.  At step k,
    replace the top-k features with background_mean and re-evaluate the model.
    The score measures how much the model output *changes* (in absolute terms)
    as the most important features are removed.

    Why absolute deviation, not signed drop:
        A faithful explanation may have negative attributions (features that push
        the prediction toward class 0).  Removing such a feature increases the
        predicted class-1 probability — a signed-drop metric would penalise this
        as "unfaithful" even though the model clearly responds to the feature.
        Absolute deviation is direction-agnostic and handles both cases correctly.

    Score = mean over samples of (AUC of |f(x) - f(x_masked_k)| vs k),
            normalised by n_features so it lives in [0, 1].
            Higher is better (greater response to removing attributed features).

    Parameters
    ----------
    predict_fn : callable mapping (n_samples, n_features) -> (n_samples,) probabilities.
    explanations : ndarray of shape (n_samples, n_features)
    X : ndarray of shape (n_samples, n_features) — same feature space as predict_fn.
    background_mean : ndarray of shape (n_features,) — replacement for masked features.
        Defaults to column means of X.
    metric : 'auc' (default) | 'max_dev'
        'auc'     → normalised AUC of the mean absolute-deviation curve.
        'max_dev' → mean absolute deviation after removing ALL features.

    Returns
    -------
    score : float in [0, 1]  (higher = more faithful)
    """
    if background_mean is None:
        background_mean = X.mean(axis=0)

    n_samples, n_features = X.shape
    steps  = list(range(n_features + 1))           # [0, 1, ..., n_features]
    f_full = predict_fn(X)                          # (n_samples,) baseline predictions
    # dev_curve[k] = mean |f(x) - f(x with top-k features ablated)|
    dev_curve = np.zeros(len(steps))               # dev_curve[0] = 0 by construction

    for step_idx, k in enumerate(steps):
        if k == 0:
            dev_curve[0] = 0.0
            continue
        X_masked = X.copy()
        for s in range(n_samples):
            ranked = np.argsort(-np.abs(explanations[s]))  # descending |attribution|
            X_masked[s, ranked[:k]] = background_mean[ranked[:k]]
        f_masked          = predict_fn(X_masked)
        dev_curve[step_idx] = np.abs(f_full - f_masked).mean()

    if metric == "max_dev":
        return float(dev_curve[-1])

    # Normalise AUC: divide by n_features (maximum possible area if max_dev=1 everywhere)
    auc   = np.trapezoid(dev_curve, steps)
    score = auc / max(n_features, 1)
    return float(np.clip(score, 0.0, 1.0))


# ──────────────────────────────────────────────
# infidelity
# ──────────────────────────────────────────────

def infidelity(
    predict_fn,
    explanations: np.ndarray,
    X: np.ndarray,
    n_perturbations: int = 100,
    noise_std: float = 0.1,
    seed: int = 42,
) -> float:
    """Compute explanation infidelity (Yeh et al., NeurIPS 2019).

    INFD(φ, f, x) = E_{δx}[(φ(x)·δx - (f(x) - f(x - δx)))^2]

    A small infidelity means the linear approximation φ·δx closely tracks
    the actual change in model output under perturbation δx.

    Parameters
    ----------
    predict_fn : callable mapping (n_samples, n_features) -> (n_samples,) probabilities.
    explanations : ndarray of shape (n_samples, n_features)
    X : ndarray of shape (n_samples, n_features)
    n_perturbations : int — number of random perturbations per sample.
    noise_std : float — std of Gaussian perturbation noise.
    seed : int

    Returns
    -------
    score : float  (lower is better — smaller mismatch)
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X.shape
    f_x = predict_fn(X)          # baseline predictions: (n_samples,)
    total_infd = 0.0

    for _ in range(n_perturbations):
        delta = rng.normal(0.0, noise_std, size=X.shape).astype(np.float32)
        f_perturbed = predict_fn(X - delta)      # (n_samples,)

        # Linear approximation: φ(x) · δx
        linear_approx = (explanations * delta).sum(axis=1)

        # Actual change
        actual_change = f_x - f_perturbed

        total_infd += float(np.mean((linear_approx - actual_change) ** 2))

    return total_infd / n_perturbations


# ──────────────────────────────────────────────
# Rank-correlation metrics (round-to-round stability)
# ──────────────────────────────────────────────

def _flatten_mean(exp):
    """If 2D (n_samples, n_features), return mean across samples; else return as-is."""
    exp = np.asarray(exp, dtype=float)
    return exp.mean(axis=0) if exp.ndim == 2 else exp


def kendalls_tau(
    explanation_round_a,
    explanation_round_b,
) -> float:
    """Kendall's Tau rank correlation between two explanation vectors.

    Both inputs may be:
      - 1D arrays of shape (n_features,): explanation for a single sample or
        the mean explanation at a given round.
      - 2D arrays of shape (n_samples, n_features): averaged to 1D first.

    Returns
    -------
    tau : float in [-1, 1].  1 = identical ranking, -1 = fully reversed.
    """
    a = _flatten_mean(explanation_round_a)
    b = _flatten_mean(explanation_round_b)
    tau, _ = kendalltau(a, b)
    return float(tau)


def spearman_corr(
    explanation_round_a,
    explanation_round_b,
) -> float:
    """Spearman rank correlation between two explanation vectors.

    Same input convention as kendalls_tau.

    Returns
    -------
    rho : float in [-1, 1].  1 = identical ranking.
    """
    a = _flatten_mean(explanation_round_a)
    b = _flatten_mean(explanation_round_b)
    rho, _ = spearmanr(a, b)
    return float(rho)


# ──────────────────────────────────────────────
# Jensen-Shannon divergence
# ──────────────────────────────────────────────

def js_divergence(
    explanation_local,
    explanation_global,
    epsilon: float = 1e-10,
) -> float:
    """Jensen-Shannon divergence between local and global explanation distributions.

    Both inputs may be 1D (single explanation) or 2D (averaged to 1D).
    Explanations are converted to probability distributions by taking absolute
    values and normalising to sum=1.

    Returns
    -------
    jsd : float in [0, 1] (using base-2 logarithm in scipy.jensenshannon).
        0 = identical distributions; 1 = maximally different.
    """
    a = _flatten_mean(explanation_local)
    b = _flatten_mean(explanation_global)

    # Convert to non-negative probability distributions
    a_prob = np.abs(a) + epsilon
    b_prob = np.abs(b) + epsilon
    a_prob /= a_prob.sum()
    b_prob /= b_prob.sum()

    return float(jensenshannon(a_prob, b_prob, base=2))


# ──────────────────────────────────────────────
# Convenience wrapper: compute all metrics at once
# ──────────────────────────────────────────────

def explanation_report(
    predict_fn,
    explanations: np.ndarray,
    X: np.ndarray,
    background_mean: np.ndarray = None,
) -> dict:
    """Compute faithfulness and infidelity and return as a dict."""
    return {
        "faithfulness": faithfulness(predict_fn, explanations, X, background_mean),
        "infidelity":   infidelity(predict_fn, explanations, X),
    }

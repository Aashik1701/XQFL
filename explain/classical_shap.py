"""Classical SHAP wrapper for any black-box predict function.

Uses shap.KernelExplainer, which works on any model via predict calls only —
no gradient access required.  This makes it directly applicable to both the
classical baseline and the quantum model (treated as a black box).

Key distinction from q_shap.py:
  - classical_shap operates in the ORIGINAL feature space (e.g. 30 features
    for Wisconsin BC), letting the model handle PCA internally.
  - q_shap operates in the PCA-reduced quantum feature space (n_qubits features).
"""

import numpy as np
import shap


def compute_shap_values(
    predict_fn,
    X_explain: np.ndarray,
    background: np.ndarray,
    n_background: int = 50,
    nsamples: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Compute SHAP values for X_explain using KernelExplainer.

    Parameters
    ----------
    predict_fn : callable
        Function mapping (n_samples, n_features) -> (n_samples,) probabilities.
        For HybridQuantumClassifier use model.predict_proba.
    X_explain : ndarray of shape (n_samples, n_features)
        Samples to explain.  May be in the original or PCA feature space —
        must match the input domain that predict_fn expects.
    background : ndarray of shape (n_bg, n_features)
        Background dataset used by KernelExplainer to marginalise features.
        Typically a representative subsample of the training set.
    n_background : int
        How many background samples to use (shap.kmeans summary if large).
    nsamples : int
        Number of coalition samples passed to KernelExplainer.shap_values.
        Default 256 for research runs; reduce to ~64 for fast sanity checks.
        (Auto default in SHAP is 2*n_features+2048, which is far too slow for
        quantum models where each predict call runs a circuit.)
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    shap_values : ndarray of shape (n_samples, n_features)
        SHAP attribution value for each feature of each sample.
        Positive value → feature pushes prediction toward class 1.
    """
    np.random.seed(seed)

    if len(background) > n_background:
        background_summary = shap.kmeans(background, n_background)
    else:
        background_summary = background

    explainer   = shap.KernelExplainer(predict_fn, background_summary)
    shap_values = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)

    # KernelExplainer on a binary classifier returns a 2D array (n_samples, n_features)
    return np.array(shap_values)


def mean_abs_shap(shap_values: np.ndarray) -> np.ndarray:
    """Global feature importance: mean absolute SHAP value across samples."""
    return np.abs(shap_values).mean(axis=0)

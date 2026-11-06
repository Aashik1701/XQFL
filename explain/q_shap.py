"""Q-SHAP: feature-level Shapley attribution in the quantum (PCA) feature space.

IMPORTANT — no pip package exists for Q-SHAP.
This is a hand-rolled implementation inspired by the QuXAI / Q-MEDLEY line of work
(Heese et al., 2023; Pira & Roch, 2023).  The simplification vs. those papers:
  - We use the standard permutation-sampling Shapley approximation (same maths as
    KernelSHAP) rather than a quantum-specific kernel.
  - Attribution is computed in the PCA-reduced quantum input space (n_qubits
    features), NOT the original feature space.  This is the natural space for
    quantum circuit analysis because each feature maps to one qubit.
  - Feature masking replaces the masked feature with its mean value from a
    background dataset (same as KernelSHAP's baseline strategy).

Distinction from classical_shap.py:
  - classical_shap.py attributes to ORIGINAL features (e.g. 30 for Wisconsin).
  - q_shap.py attributes to QUANTUM features (n_qubits PCA components).
    These can be projected back to original space via model.pca.components_.
"""

import numpy as np
from math import factorial
from itertools import permutations
from typing import Callable


def _subset_predict(predict_fn, x, mask, background_mean):
    """Evaluate predict_fn on x with features in mask=False replaced by mean."""
    x_masked = x.copy()
    x_masked[~mask] = background_mean[~mask]
    return float(predict_fn(x_masked[np.newaxis, :])[0])


def q_shap_values(
    predict_fn: Callable,
    X_quantum: np.ndarray,
    background: np.ndarray,
    n_permutations: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Compute Q-SHAP values via permutation sampling in the quantum feature space.

    Algorithm (permutation Shapley, Strumbelj & Kononenko 2014):
      For each sample x and each feature i:
        1. Sample a random permutation of all features.
        2. Let S = features appearing before i in the permutation.
        3. Marginal contribution of i = f(S ∪ {i}) - f(S).
        4. Average M such estimates → φ_i(x).

    Parameters
    ----------
    predict_fn : callable
        Maps (n_samples, n_qubits) -> (n_samples,) probabilities.
        Must operate in the quantum feature space (after PCA + scaling).
        Use model.predict_proba_quantum (see below) or wrap model._preprocess
        + model.predict_proba appropriately.
    X_quantum : ndarray of shape (n_samples, n_qubits)
        Samples to explain, already in the quantum (PCA-reduced) feature space.
    background : ndarray of shape (n_bg, n_qubits)
        Background dataset in the quantum feature space; mean is used as baseline.
    n_permutations : int
        Number of random permutations per sample per feature.  64 is sufficient
        for n_qubits=4; increase for larger feature sets.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    shap_values : ndarray of shape (n_samples, n_qubits)
        Q-SHAP attribution in the quantum feature space.
    """
    rng = np.random.default_rng(seed)
    n_samples, n_features = X_quantum.shape
    background_mean = background.mean(axis=0)

    shap_values = np.zeros((n_samples, n_features), dtype=float)

    for s_idx in range(n_samples):
        x = X_quantum[s_idx].copy()
        phi = np.zeros(n_features)

        for _ in range(n_permutations):
            perm = rng.permutation(n_features)
            prev_mask = np.zeros(n_features, dtype=bool)
            prev_val  = _subset_predict(predict_fn, x, prev_mask, background_mean)

            for i in perm:
                new_mask = prev_mask.copy()
                new_mask[i] = True
                new_val = _subset_predict(predict_fn, x, new_mask, background_mean)
                phi[i] += new_val - prev_val
                prev_mask = new_mask
                prev_val  = new_val

        shap_values[s_idx] = phi / n_permutations

    return shap_values


def exact_q_shap_values(
    predict_fn: Callable,
    X_quantum: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """Exact Shapley values by evaluating all 2^d coalitions of each sample.

    Uses the same value function as q_shap_values (absent features take the
    background mean), but no sampling: the result is deterministic. For d=4 it
    costs 16 model evaluations per sample, against 40 for q_shap_values with
    n_permutations=8 (which spends d+1 evaluations per permutation).

    The cost grows as 2^d, so use q_shap_values for much larger feature sets.

    Parameters
    ----------
    predict_fn : callable mapping (n, d) inputs to (n,) outputs.
    X_quantum : ndarray of shape (n_samples, d), samples to explain.
    background : ndarray of shape (n_bg, d); its mean is the reference.

    Returns
    -------
    shap_values : ndarray of shape (n_samples, d).
    """
    n_samples, d = X_quantum.shape
    background_mean = background.mean(axis=0)
    n_coalitions = 2 ** d

    # masks[c, i] is True when feature i belongs to coalition c (bits of c).
    masks = ((np.arange(n_coalitions)[:, None] >> np.arange(d)) & 1).astype(bool)
    sizes = masks.sum(axis=1)

    inputs = np.where(masks[None, :, :], X_quantum[:, None, :], background_mean)
    values = np.asarray(predict_fn(inputs.reshape(-1, d)), dtype=float)
    values = values.reshape(n_samples, n_coalitions)

    # Shapley kernel weight |S|! (d-|S|-1)! / d! for each coalition size.
    fact = [factorial(k) for k in range(d + 1)]
    weights = np.array(
        [fact[k] * fact[d - k - 1] / fact[d] for k in range(d)], dtype=float
    )

    shap_values = np.zeros((n_samples, d), dtype=float)
    for i in range(d):
        bit = 1 << i
        without_i = np.flatnonzero(~masks[:, i])
        marginal = values[:, without_i + bit] - values[:, without_i]
        shap_values[:, i] = marginal @ weights[sizes[without_i]]
    return shap_values


def make_quantum_predict_fn(model):
    """Return a predict function that operates in the quantum feature space.

    The returned callable accepts (n_samples, n_qubits) inputs that are already
    PCA-reduced and scaled, bypassing model._preprocess.  This is needed because
    q_shap works by masking features in the quantum space and re-evaluating —
    re-running PCA on masked inputs would change the subspace, breaking the
    Shapley semantics.
    """
    import torch

    def _predict_quantum(X_q: np.ndarray) -> np.ndarray:
        model.eval()
        import torch
        X_t = torch.tensor(X_q.astype(np.float32))
        with torch.no_grad():
            return model(X_t).numpy()

    return _predict_quantum


def mean_abs_q_shap(shap_values: np.ndarray) -> np.ndarray:
    """Global quantum-feature importance: mean absolute Q-SHAP value."""
    return np.abs(shap_values).mean(axis=0)

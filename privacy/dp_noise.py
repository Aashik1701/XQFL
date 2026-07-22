"""Differential privacy noise mechanisms for Q-SHAP explanation vectors.

Two mechanisms:
  Laplace  — noise ~ Laplace(0, sensitivity/epsilon)
  Gaussian — noise ~ N(0, sigma^2), sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon

For Q-SHAP over n_qubits=4 PCA features, a conservative sensitivity bound is 1.0
(the sigmoid output is bounded [0,1], so each Shapley value is bounded by ±1).
"""

import numpy as np


def add_dp_noise(
    explanation_vector,
    epsilon,
    mechanism="laplace",
    sensitivity=None,
    delta=1e-5,
    seed=None,
):
    """Add DP noise to a single Q-SHAP explanation vector.

    Parameters
    ----------
    explanation_vector : array-like of shape (n_features,)
    epsilon            : privacy budget (smaller = more noise)
    mechanism          : 'laplace' or 'gaussian'
    sensitivity        : L1 (Laplace) or L2 (Gaussian) sensitivity.
                         Defaults to 1.0 if None.
    delta              : failure probability for Gaussian mechanism (default 1e-5)
    seed               : RNG seed (None = random)

    Returns
    -------
    noisy : ndarray of shape (n_features,)
    """
    vec = np.asarray(explanation_vector, dtype=np.float64)
    rng = np.random.default_rng(seed)
    s   = float(sensitivity) if sensitivity is not None else 1.0

    if mechanism == "laplace":
        scale = s / epsilon
        noise = rng.laplace(loc=0.0, scale=scale, size=vec.shape)
    elif mechanism == "gaussian":
        sigma = s * np.sqrt(2.0 * np.log(1.25 / delta)) / epsilon
        noise = rng.normal(loc=0.0, scale=sigma, size=vec.shape)
    else:
        raise ValueError(f"Unknown mechanism: {mechanism!r}. Use 'laplace' or 'gaussian'.")

    return vec + noise


def batch_dp_noise(
    explanation_matrix,
    epsilon,
    mechanism="laplace",
    sensitivity=None,
    delta=1e-5,
    seed=None,
):
    """Add independent DP noise to each row of an explanation matrix.

    Parameters
    ----------
    explanation_matrix : array-like of shape (n_samples, n_features)
    epsilon            : privacy budget
    mechanism          : 'laplace' or 'gaussian'
    sensitivity        : sensitivity bound (default 1.0)
    delta              : Gaussian failure probability
    seed               : RNG seed for reproducibility (each row gets seed+i)

    Returns
    -------
    noisy_matrix : ndarray of shape (n_samples, n_features)
    """
    mat  = np.asarray(explanation_matrix, dtype=np.float64)
    rows = []
    for i, row in enumerate(mat):
        row_seed = None if seed is None else seed + i
        rows.append(
            add_dp_noise(row, epsilon, mechanism=mechanism,
                         sensitivity=sensitivity, delta=delta, seed=row_seed)
        )
    return np.vstack(rows)

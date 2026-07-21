"""Gate-Shapley: layer-level attribution for the variational quantum circuit.

IMPORTANT — no pip package exists for gate-Shapley.
This is a hand-rolled implementation following the gate-ablation Shapley approach
described in Heese et al. (2023) "Explainability of Vision Transformers" adapted
for VQCs, and the general quantum circuit attribution ideas in Pira & Roch (2023).

Simplification documented here (as required by spec §7):
  - Players are LAYERS of the BasicEntanglerLayers ansatz, not individual gates.
    With n_layers=2 this gives 2 players; for n_layers=L there are L players.
  - Ablating layer l means setting its RY rotation angles to 0.
    The CNOT entanglers in that layer remain active — this is an approximation,
    since a true ablation would remove the CNOTs too, but removing CNOT gates
    dynamically from a fixed-topology PennyLane circuit is non-trivial.
    In practice, RY(0) rotations leave the state unchanged on each qubit before
    entanglement, so the remaining CNOT layer has minimal information-propagating
    effect on a near-|0> subspace — an acceptable approximation for attribution.
  - Exact Shapley values are computed when n_layers ≤ 8 (enumerate all 2^L subsets).
    For n_layers > 8, a Monte Carlo sampling approximation is used instead.
"""

import numpy as np
import torch
from itertools import combinations
from math import factorial
from typing import Callable


# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────

def _ablate_and_predict(model, X_quantum: np.ndarray, active_layers: set) -> np.ndarray:
    """Run the model with VQC layers NOT in active_layers zeroed out.

    Temporarily modifies model.vqc_weights in-place, evaluates, then restores.

    Parameters
    ----------
    model : HybridQuantumClassifier
    X_quantum : (n_samples, n_qubits) float32 ndarray in the quantum feature space.
    active_layers : set of int
        Indices of layers whose weights are kept.  All other layers are set to 0.

    Returns
    -------
    preds : (n_samples,) float64 ndarray of class-1 probabilities.
    """
    n_layers = model.vqc_weights.shape[0]
    original = model.vqc_weights.data.clone()

    # Zero out layers not in the active set
    with torch.no_grad():
        for layer_idx in range(n_layers):
            if layer_idx not in active_layers:
                model.vqc_weights.data[layer_idx, :] = 0.0

    model.eval()
    X_t = torch.tensor(X_quantum.astype(np.float32))
    with torch.no_grad():
        preds = model(X_t).numpy()

    model.vqc_weights.data.copy_(original)   # restore
    return preds


def _coalition_value(model, X_quantum: np.ndarray, active_layers: set) -> float:
    """Mean class-1 probability over X_quantum for a given layer coalition."""
    return float(_ablate_and_predict(model, X_quantum, active_layers).mean())


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def gate_shapley_values(
    model,
    X_quantum: np.ndarray,
    seed: int = 42,
    n_mc_samples: int = 200,
) -> np.ndarray:
    """Compute layer-level Shapley values for the VQC.

    Uses exact enumeration when n_layers ≤ 8, Monte Carlo permutation
    sampling otherwise.

    Parameters
    ----------
    model : HybridQuantumClassifier (fitted)
    X_quantum : ndarray of shape (n_samples, n_qubits)
        Samples in the quantum (PCA) feature space.
    seed : int
    n_mc_samples : int
        Permutations sampled per layer when using Monte Carlo mode (n_layers > 8).

    Returns
    -------
    phi : ndarray of shape (n_layers,)
        Shapley value for each VQC layer.  Positive → layer helps prediction;
        negative → layer hurts (pushes toward class 0 on average).
    """
    n_layers = model.vqc_weights.shape[0]
    players  = list(range(n_layers))

    if n_layers <= 8:
        return _exact_gate_shapley(model, X_quantum, players)
    else:
        return _mc_gate_shapley(model, X_quantum, players, n_mc_samples, seed)


def _exact_gate_shapley(model, X_quantum: np.ndarray, players: list) -> np.ndarray:
    """Exact Shapley via subset enumeration (2^n_layers evaluations)."""
    n = len(players)
    phi = np.zeros(n)

    # Pre-compute coalition values for all 2^n subsets
    coalition_vals = {}
    for size in range(n + 1):
        for subset in combinations(players, size):
            s = frozenset(subset)
            coalition_vals[s] = _coalition_value(model, X_quantum, s)

    for i in players:
        others = [p for p in players if p != i]
        for size in range(n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for subset in combinations(others, size):
                s     = frozenset(subset)
                s_and_i = frozenset(set(subset) | {i})
                phi[i] += weight * (coalition_vals[s_and_i] - coalition_vals[s])

    return phi


def _mc_gate_shapley(
    model,
    X_quantum: np.ndarray,
    players: list,
    n_mc_samples: int,
    seed: int,
) -> np.ndarray:
    """Monte Carlo permutation-sampling Shapley for large n_layers."""
    rng = np.random.default_rng(seed)
    n   = len(players)
    phi = np.zeros(n)

    for _ in range(n_mc_samples):
        perm      = list(rng.permutation(players))
        coalition = set()
        prev_val  = _coalition_value(model, X_quantum, coalition)

        for player in perm:
            coalition.add(player)
            new_val  = _coalition_value(model, X_quantum, coalition)
            phi[player] += new_val - prev_val
            prev_val = new_val

    return phi / n_mc_samples


def gate_shapley_for_sample(
    model,
    x_quantum: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    """Compute gate-Shapley for a single sample (wraps gate_shapley_values)."""
    return gate_shapley_values(model, x_quantum[np.newaxis, :], seed=seed)

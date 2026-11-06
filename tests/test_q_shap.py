"""Tests for the exact and permutation-sampled Q-SHAP estimators."""
import sys
import unittest
from itertools import permutations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explain.q_shap import exact_q_shap_values, q_shap_values


def nonlinear(X):
    """Non-additive test function with interactions, so orderings matter."""
    return np.sin(X[:, 0]) * X[:, 1] + X[:, 2] ** 2 - X[:, 0] * X[:, 3]


def brute_force(fn, x, reference):
    """Shapley values by walking all d! orderings."""
    d = len(x)
    phi = np.zeros(d)
    orderings = list(permutations(range(d)))
    for order in orderings:
        current = reference.copy()
        previous = fn(current[None, :])[0]
        for i in order:
            current[i] = x[i]
            value = fn(current[None, :])[0]
            phi[i] += value - previous
            previous = value
    return phi / len(orderings)


class ExactQShapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.uniform(0, np.pi, size=(6, 4))
        self.background = rng.uniform(0, np.pi, size=(20, 4))
        self.reference = self.background.mean(axis=0)

    def test_matches_all_orderings(self):
        exact = exact_q_shap_values(nonlinear, self.X, self.background)
        for x, phi in zip(self.X, exact):
            np.testing.assert_allclose(phi, brute_force(nonlinear, x, self.reference))

    def test_efficiency(self):
        exact = exact_q_shap_values(nonlinear, self.X, self.background)
        expected = nonlinear(self.X) - nonlinear(self.reference[None, :])[0]
        np.testing.assert_allclose(exact.sum(axis=1), expected)

    def test_null_feature_gets_zero(self):
        fn = lambda X: X[:, 0] * X[:, 1] + X[:, 2]
        exact = exact_q_shap_values(fn, self.X, self.background)
        np.testing.assert_allclose(exact[:, 3], 0.0, atol=1e-12)

    def test_permutation_estimator_converges_to_exact(self):
        exact = exact_q_shap_values(nonlinear, self.X, self.background)
        sampled = q_shap_values(nonlinear, self.X, self.background,
                                n_permutations=4000, seed=1)
        np.testing.assert_allclose(sampled, exact, atol=0.05)

    def test_uses_fewer_evaluations_than_eight_permutations(self):
        calls = {"n": 0}

        def counting(X):
            calls["n"] += len(X)
            return nonlinear(X)

        exact_q_shap_values(counting, self.X, self.background)
        exact_calls = calls["n"]
        calls["n"] = 0
        q_shap_values(counting, self.X, self.background, n_permutations=8)
        self.assertEqual(exact_calls, 16 * len(self.X))
        self.assertEqual(calls["n"], 40 * len(self.X))


if __name__ == "__main__":
    unittest.main()

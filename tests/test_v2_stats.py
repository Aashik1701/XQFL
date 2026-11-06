"""Statistics and RQ4 helpers, checked on synthetic values only."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2 import stats
from v2.mia import attack_features, interpret, laplace_release, sample_loss


class StatsTests(unittest.TestCase):
    def test_holm_known_values(self):
        np.testing.assert_allclose(stats.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_holm_ignores_nan(self):
        out = stats.holm([0.01, np.nan, 0.04])
        self.assertTrue(np.isnan(out[1]))
        np.testing.assert_allclose(out[[0, 2]], [0.02, 0.04])

    def test_holm_is_monotone_and_capped(self):
        out = stats.holm([0.9, 0.5, 0.6, 0.7])
        self.assertTrue((out <= 1).all())
        self.assertEqual(out[1], 1.0)

    def test_hodges_lehmann(self):
        self.assertEqual(stats.hodges_lehmann([1, 2, 3]), 2.0)
        self.assertEqual(stats.hodges_lehmann([0, 0, 10]), 2.5)

    def test_rank_biserial_extremes(self):
        self.assertEqual(stats.rank_biserial([1, 2, 3]), 1.0)
        self.assertEqual(stats.rank_biserial([-1, -2, -3]), -1.0)
        self.assertEqual(stats.rank_biserial([1, -1]), 0.0)

    def test_paired_matches_scipy_and_interval_covers(self):
        rng = np.random.default_rng(0)
        x = rng.normal(1, 1, 30)
        y = rng.normal(0, 1, 30)
        result = stats.paired_test(x, y, n_boot=2000)
        self.assertAlmostEqual(result["p"], wilcoxon(x - y).pvalue)
        self.assertLess(result["ci_low"], result["median_diff"])
        self.assertGreater(result["ci_high"], result["median_diff"])

    def test_paired_refuses_small_samples(self):
        self.assertEqual(stats.paired_test(np.arange(9.0))["status"], "not_testable")

    def test_all_zero_differences(self):
        result = stats.paired_test(np.ones(12), np.ones(12), n_boot=200)
        self.assertEqual(result["p"], 1.0)

    def test_false_positive_rate_is_controlled(self):
        rng = np.random.default_rng(1)
        p = [stats.paired_test(rng.normal(0, 1, 30), n_boot=10)["p"] for _ in range(400)]
        self.assertAlmostEqual(np.mean(np.array(p) < 0.05), 0.05, delta=0.03)

    def test_mcnemar(self):
        a = np.array([1] * 9 + [0] * 1 + [1] * 10, bool)
        b = np.array([0] * 9 + [1] * 1 + [1] * 10, bool)
        result = stats.mcnemar_exact(a, b)
        self.assertEqual((result["only_first"], result["only_second"]), (9, 1))
        self.assertAlmostEqual(result["p"], 0.021484375)

    def test_wilson(self):
        low, high = stats.wilson_interval(15, 30)
        self.assertAlmostEqual(low, 0.3315, places=3)
        self.assertAlmostEqual(high, 0.6685, places=3)
        self.assertEqual(stats.wilson_interval(0, 30)[0], 0.0)

    def test_hierarchical_interval_covers_true_difference(self):
        rng = np.random.default_rng(2)
        a = {s: rng.normal(1, 0.2, 4) for s in range(30)}
        b = {s: rng.normal(0, 0.2, 4) for s in range(30)}
        low, high = stats.hierarchical_interval(a, b, n_boot=500)
        self.assertLess(low, 1)
        self.assertGreater(high, 1)
        self.assertGreater(low, 0.8)


class AttackHelperTests(unittest.TestCase):
    def test_features_shape(self):
        self.assertEqual(attack_features(np.ones((3, 4))).shape, (3, 9))

    def test_loss_is_low_for_correct_confident_output(self):
        self.assertLess(sample_loss(np.array([0.99]), np.array([1.0]))[0], 0.02)
        self.assertGreater(sample_loss(np.array([0.01]), np.array([1.0]))[0], 4)

    def test_noise_scale_follows_per_sample_norm(self):
        phi = np.array([[3.0, 4.0, 0, 0], [6.0, 8.0, 0, 0]])
        noisy, scale = laplace_release(phi, 2.0, np.random.default_rng(0))
        self.assertAlmostEqual(scale, 15.0)
        self.assertFalse(np.allclose(noisy, phi))

    def test_reading_table(self):
        self.assertIn("unvalidated", interpret(False, True, True))
        self.assertIn("leakage", interpret(True, False, True))
        self.assertIn("not proof", interpret(True, True, False))
        self.assertIn("attack-specific", interpret(True, False, False))


if __name__ == "__main__":
    unittest.main()

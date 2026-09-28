import unittest
import numpy as np
from privacy.dp_noise import batch_dp_noise
from privacy.release_evaluation import evaluate_release, utility


class ReleaseTests(unittest.TestCase):
    def test_no_offset_noise_reuse(self):
        z = np.zeros((50, 4))
        a = batch_dp_noise(z, .1, seed=42)
        b = batch_dp_noise(z, .1, seed=43)
        self.assertFalse(np.array_equal(a[1:], b[:-1]))
        np.testing.assert_array_equal(a, batch_dp_noise(z, .1, seed=42))
        rng = np.random.default_rng(42)
        c = batch_dp_noise(np.zeros((100, 4)), .1, seed=rng)
        self.assertEqual(len(np.unique(c, axis=0)), 100)

    def test_held_out_scores_and_signal(self):
        y = np.r_[np.ones(50, int), np.zeros(50, int)]
        X = np.random.default_rng(1).normal(size=(100,4)) + y[:,None]*6
        metrics, scores, folds = evaluate_release(X, y, 42)
        self.assertGreater(metrics['auc_mean_fold'], .95)
        self.assertGreater(metrics['attack_advantage'], .8)
        seen = []
        for f in folds:
            self.assertFalse(set(f['train_indices']) & set(f['test_indices']))
            seen += f['test_indices']
        self.assertEqual(sorted(seen), list(range(100)))
        self.assertTrue(np.isfinite(scores['score_raw']).all())

    def test_constant_no_signal(self):
        y = np.r_[np.ones(50,int), np.zeros(50,int)]
        metrics, _, _ = evaluate_release(np.zeros((100,4)), y)
        self.assertEqual(metrics['auc_mean_fold'], .5)
        self.assertEqual(metrics['attack_advantage'], 0)

    def test_utility_zero_handling(self):
        a = np.array([[1.,2,3,4], [0,0,0,0]])
        u = utility(a,a)
        self.assertAlmostEqual(u['cosine_mean'], 1)
        self.assertEqual(u['cosine_undefined_n'], 1)
        self.assertEqual(u['l2_mean'], 0)


if __name__ == '__main__':
    unittest.main()

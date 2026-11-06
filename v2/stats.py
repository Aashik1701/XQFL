"""Statistics fixed by protocol v2, section 7."""
import numpy as np
from scipy.stats import binomtest, rankdata, wilcoxon

BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261105
MIN_PAIRS = 10


def hodges_lehmann(differences):
    d = np.asarray(differences, float)
    i, j = np.triu_indices(len(d))
    return float(np.median((d[i] + d[j]) / 2))


def rank_biserial(differences):
    """Matched-pairs rank-biserial correlation, in [-1, 1]. Zero differences are dropped."""
    d = np.asarray(differences, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def bootstrap_hl_interval(differences, n_boot=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    d = np.asarray(differences, float)
    rng = np.random.default_rng(seed)
    i, j = np.triu_indices(len(d))
    estimates = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        chosen = d[rng.integers(0, len(d), size=(min(1000, n_boot - start), len(d)))]
        estimates[start:start + 1000] = np.median((chosen[:, i] + chosen[:, j]) / 2, axis=1)
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def paired_test(x, y=None, n_boot=BOOTSTRAPS):
    """Two-sided Wilcoxon on x - y (or on x vs 0), with effect sizes and interval."""
    x = np.asarray(x, float)
    d = x if y is None else x - np.asarray(y, float)
    d = d[np.isfinite(d)]
    result = dict(n=len(d))
    if len(d) < MIN_PAIRS:
        return dict(result, status="not_testable", p=np.nan, median_diff=np.nan,
                    rank_biserial=np.nan, ci_low=np.nan, ci_high=np.nan)
    p = 1.0 if np.all(d == 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
    low, high = bootstrap_hl_interval(d, n_boot)
    return dict(result, status="tested", p=p, median_diff=hodges_lehmann(d),
                rank_biserial=rank_biserial(d), ci_low=low, ci_high=high)


def hierarchical_interval(arm_a, arm_b, n_boot=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """Interval for the mean per-seed difference: resample seeds, then clients in seeds.

    arm_a and arm_b map a seed to an array of client values. A scalar arm can pass
    one-element arrays.
    """
    seeds = sorted(set(arm_a) & set(arm_b))
    if len(seeds) < MIN_PAIRS:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)

    def resampled_mean(values):
        values = np.asarray(values, float)
        return values[rng.integers(0, len(values), len(values))].mean()

    estimates = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, len(seeds), len(seeds))
        estimates[b] = np.mean([resampled_mean(arm_a[seeds[c]]) - resampled_mean(arm_b[seeds[c]])
                                for c in chosen])
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def mcnemar_exact(a, b):
    """Exact McNemar test on paired booleans, with the difference in rates."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    discordant = only_a + only_b
    p = 1.0 if discordant == 0 else float(binomtest(only_a, discordant, 0.5).pvalue)
    return dict(n=len(a), status="tested", p=p, median_diff=float(a.mean() - b.mean()),
                rank_biserial=np.nan, ci_low=np.nan, ci_high=np.nan,
                only_first=only_a, only_second=only_b)


def holm(p_values):
    """Holm-adjusted p-values. NaN entries stay NaN and do not count toward m."""
    p = np.asarray(p_values, float)
    adjusted = np.full(len(p), np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid])]
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(valid) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def wilson_interval(successes, n, z=1.959963984540054):
    if n == 0:
        return np.nan, np.nan
    rate = successes / n
    denominator = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return float(max(0, centre - half)), float(min(1, centre + half))

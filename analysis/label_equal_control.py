"""Post-hoc control for RQ3: label skew with equal client sizes.

The Dirichlet partition changes class proportions and client sizes together (Table C1). This partition
keeps every client at N/K records (within one) and draws each client's class proportions from
Dirichlet(alpha = 0.5), then fits the client-by-class count matrix to the class totals with Sinkhorn
scaling and largest-remainder rounding. Runs use the frozen v2 training and explanation code; only the
partition function is replaced, for this regime only. Not part of the frozen protocol.

    PYTHONPATH=. OMP_NUM_THREADS=1 xqfl-env/bin/python analysis/label_equal_control.py
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from v2 import config
from v2.config import RunSpec
from v2.data import _rng

OUT = Path("results/protocol_v2/posthoc_label_equal")
REGIME = "label_equal_0.5"
ALPHA = 0.5


def _round_to_margins(M, rows, cols):
    """Integer matrix with the given row and column sums, close to M (largest remainder, then repair)."""
    out = np.floor(M).astype(int)
    for i in range(len(rows)):
        short = rows[i] - out[i].sum()
        order = np.argsort(-(M[i] - out[i]))
        for j in order[:max(short, 0)]:
            out[i, j] += 1
    # Repair column sums by moving records between clients that can swap classes.
    for _ in range(10_000):
        diff = out.sum(axis=0) - cols
        if not diff.any():
            return out
        over, under = int(np.argmax(diff)), int(np.argmin(diff))
        candidates = np.flatnonzero(out[:, over] > 0)
        i = candidates[np.argmax(M[candidates, under] - out[candidates, under])]
        out[i, over] -= 1
        out[i, under] += 1
    raise RuntimeError("could not match class totals")


def label_equal_split(y, n_clients, seed, alpha=ALPHA):
    y = np.asarray(y)
    classes = np.unique(y)
    rng = _rng(seed, 77)
    totals = np.array([(y == c).sum() for c in classes])
    sizes = np.full(n_clients, len(y) // n_clients)
    sizes[: len(y) - sizes.sum()] += 1
    M = rng.dirichlet(np.full(len(classes), alpha), size=n_clients) * sizes[:, None] + 1e-9
    for _ in range(500):  # Sinkhorn: rows to client sizes, columns to class totals
        M *= (sizes / M.sum(axis=1))[:, None]
        M *= totals / M.sum(axis=0)
    counts = _round_to_margins(M * (sizes / M.sum(axis=1))[:, None], sizes, totals)
    pools = [rng.permutation(np.flatnonzero(y == c)) for c in classes]
    used = np.zeros(len(classes), int)
    parts = []
    for i in range(n_clients):
        chosen = []
        for j in range(len(classes)):
            chosen.append(pools[j][used[j]:used[j] + counts[i, j]])
            used[j] += counts[i, j]
        parts.append(np.sort(np.concatenate(chosen)))
    return parts


def _patch():
    import v2.train as train
    original = train.partition_indices

    def partition_indices(y_train, n_clients, regime, seed):
        if regime == REGIME:
            return label_equal_split(y_train, n_clients, seed)
        return original(y_train, n_clients, regime, seed)

    train.partition_indices = partition_indices


def run_one(spec):
    import torch
    from v2.explain import explain_task
    from v2.train import train_task
    torch.set_num_threads(1)
    _patch()
    (OUT / spec.key).mkdir(parents=True, exist_ok=True)
    train_task(spec, OUT)
    explain_task(spec, OUT)
    return spec.key


def main():
    specs = [RunSpec("fed", d, e, s, REGIME, group="posthoc") for d in config.DATASETS for e in config.ENCODINGS
             for s in config.SEEDS]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(7) as pool:
        for i, key in enumerate(pool.imap_unordered(run_one, specs), 1):
            if i % 20 == 0:
                print(f"{i}/{len(specs)}", flush=True)


if __name__ == "__main__":
    main()

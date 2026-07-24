"""Federated data partitioning strategies.

Three splits:
  iid_split          — balanced random partition (IID baseline)
  dirichlet_split    — label-skew via Dirichlet(alpha); alpha→0 = extreme non-IID
  quantity_skew_split — some clients get more data than others (quantity heterogeneity)

All functions return a list of (X_i, y_i) tuples, one per client.
"""

import numpy as np


def iid_split(X, y, n_clients, seed=42):
    """Randomly shuffle and split X, y evenly across n_clients (IID).

    Each client receives roughly len(X) // n_clients samples drawn uniformly
    at random, so the class distribution on every client mirrors the global one.

    Returns
    -------
    partitions : list of (X_i, y_i) — n_clients tuples of numpy arrays.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)
    chunks = np.array_split(indices, n_clients)
    return [(X[chunk], y[chunk]) for chunk in chunks]


def dirichlet_split(X, y, n_clients, alpha=0.5, seed=42):
    """Label-skew partition using a Dirichlet distribution (non-IID).

    For each class c, the fraction of class-c samples that go to each client
    is drawn from Dir(alpha).  Small alpha -> extreme concentration on few
    clients; alpha -> infinity recovers IID.

    Parameters
    ----------
    alpha : float — Dirichlet concentration parameter (default 0.5).

    Returns
    -------
    partitions : list of (X_i, y_i) — n_clients tuples.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]

    for c in classes:
        class_idx = np.where(y == c)[0]
        rng.shuffle(class_idx)
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        splits = (np.cumsum(props[:-1]) * len(class_idx)).astype(int)
        chunks = np.split(class_idx, splits)
        for i, chunk in enumerate(chunks):
            client_indices[i].extend(chunk.tolist())

    partitions = []
    for idxs in client_indices:
        idxs = np.array(idxs, dtype=int)
        rng.shuffle(idxs)
        partitions.append((X[idxs], y[idxs]))
    return partitions


def quantity_skew_split(X, y, n_clients, skew_factor=0.5, seed=42):
    """Quantity-heterogeneous partition: some clients receive much more data.

    Client sizes follow an exponential ramp:
        size_i proportional to exp(skew_factor * i)  for i = 0 ... n_clients-1

    skew_factor=0 -> equal sizes.
    Higher skew_factor -> later clients dominate the data volume.

    Returns
    -------
    partitions : list of (X_i, y_i) — n_clients tuples.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    indices = rng.permutation(n)

    raw_props = np.exp(skew_factor * np.arange(n_clients, dtype=float))
    props = raw_props / raw_props.sum()
    sizes = (props * n).astype(int)
    sizes[-1] = n - sizes[:-1].sum()   # absorb rounding remainder

    partitions = []
    start = 0
    for s in sizes:
        chunk = indices[start: start + s]
        partitions.append((X[chunk], y[chunk]))
        start += s
    return partitions

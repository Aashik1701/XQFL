"""Per-seed data splits, preprocessing, partitions and explanation record sets."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data.partition import dirichlet_split, iid_split, quantity_skew_split
from v2 import config

PIMA_MISSING_AS_ZERO = ("plas", "pres", "skin", "insu", "mass")
RECORD_CAP = {"spambase": 1500, "phoneme": 1500}
OPENML_ID = {"pima": 37, "spambase": 44, "phoneme": 1489}
POSITIVE_LABEL = {"pima": "tested_positive", "spambase": "1", "phoneme": "2"}


@lru_cache(maxsize=None)
def load_raw(dataset):
    """Return raw features, binary labels and column names."""
    if dataset == "wisconsin":
        source = load_breast_cancer()
        return (source.data.astype(np.float64), source.target.astype(np.float32),
                tuple(source.feature_names))
    source = fetch_openml(data_id=OPENML_ID[dataset], as_frame=True, parser="auto")
    labels = source.target.astype(str).to_numpy()
    y = (labels == POSITIVE_LABEL[dataset]).astype(np.float32)
    return (source.data.to_numpy(dtype=np.float64), y, tuple(source.data.columns))


@dataclass
class SeedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    meta: dict


def make_data(dataset, seed, n_features=4, zero_handling="missing"):
    """Split 70/10/20 and fit all preprocessing on the training split only."""
    X, y, columns = load_raw(dataset)
    if n_features > X.shape[1]:
        raise ValueError(f"{dataset} has {X.shape[1]} features; asked for {n_features}")
    indices = np.arange(len(y))
    if dataset in RECORD_CAP:
        indices, _ = train_test_split(indices, train_size=RECORD_CAP[dataset],
                                      stratify=y, random_state=seed)
    rest, test = train_test_split(indices, test_size=0.2, stratify=y[indices],
                                  random_state=seed)
    train, val = train_test_split(rest, test_size=0.125, stratify=y[rest],
                                  random_state=seed)
    X = X.copy()
    medians = {}
    if dataset == "pima" and zero_handling == "missing":
        for name in PIMA_MISSING_AS_ZERO:
            col = columns.index(name)
            X[X[:, col] == 0, col] = np.nan
            medians[name] = float(np.nanmedian(X[train, col]))
            X[np.isnan(X[:, col]), col] = medians[name]
    elif zero_handling not in ("missing", "as_values"):
        raise ValueError(zero_handling)

    standard = StandardScaler().fit(X[train])
    pca = PCA(n_components=n_features, svd_solver="full").fit(standard.transform(X[train]))
    scale = MinMaxScaler(feature_range=(0, np.pi)).fit(pca.transform(standard.transform(X[train])))

    def transform(rows):
        return scale.transform(pca.transform(standard.transform(X[rows]))).astype(np.float32)

    X_val, X_test = transform(val), transform(test)
    held_out = np.concatenate([X_val, X_test])
    meta = dict(
        train_indices=train.tolist(), val_indices=val.tolist(), test_indices=test.tolist(),
        medians=medians, explained_variance=float(pca.explained_variance_ratio_.sum()),
        fraction_outside_range=float(((held_out < 0) | (held_out > np.pi)).any(axis=1).mean()),
    )
    return SeedData(transform(train), y[train], X_val, y[val], X_test, y[test], meta)


def partition_indices(y_train, n_clients, regime, seed):
    """Client index arrays, from the same partition functions v1 used."""
    index = np.arange(len(y_train))
    if regime == "iid":
        parts = iid_split(index, y_train, n_clients, seed=seed)
    elif regime == "qty_skew":
        parts = quantity_skew_split(index, y_train, n_clients, skew_factor=0.5, seed=seed)
    elif regime.startswith("alpha_"):
        parts = dirichlet_split(index, y_train, n_clients, alpha=float(regime[6:]), seed=seed)
    else:
        raise ValueError(regime)
    return [np.asarray(part, dtype=int) for part, _ in parts]


def _rng(*parts):
    return np.random.default_rng(np.random.SeedSequence(list(parts)))


def stratified_pick(y, n_first, n_second, rng):
    """Pick two disjoint position sets, each keeping class shares where possible."""
    y = np.asarray(y)
    classes = np.unique(y)
    shares = np.array([(y == c).mean() for c in classes])

    def allocate(n, available):
        raw = shares * n
        counts = np.minimum(np.floor(raw).astype(int), available)
        order = np.argsort(-(raw - np.floor(raw)))
        while counts.sum() < n:
            progressed = False
            for i in order:
                if counts.sum() < n and counts[i] < available[i]:
                    counts[i] += 1
                    progressed = True
            if not progressed:
                break
        return counts

    pools = [rng.permutation(np.flatnonzero(y == c)) for c in classes]
    available = np.array([len(p) for p in pools])
    first_counts = allocate(n_first, available)
    second_counts = allocate(n_second, available - first_counts)
    first = [p[:k] for p, k in zip(pools, first_counts)]
    second = [p[k:k + m] for p, k, m in zip(pools, first_counts, second_counts)]
    return np.concatenate(first), np.concatenate(second)


@dataclass
class ExplanationSets:
    E_global: np.ndarray  # positions in the test split
    B_global: np.ndarray  # positions in the training split
    half_a: np.ndarray  # positions within E_global
    half_b: np.ndarray
    E_local: list  # per client, positions in the training split; None if sparse
    B_local: list


def explanation_sets(data, partitions, seed):
    """Build the fixed record sets. E_global and B_global depend only on the seed split."""
    E_global, _ = stratified_pick(data.y_test, config.N_GLOBAL_EXPLAINED, 0, _rng(seed, 1))
    B_global, _ = stratified_pick(data.y_train, config.N_GLOBAL_BACKGROUND, 0, _rng(seed, 2))
    half = len(E_global) // 2
    half_a, half_b = stratified_pick(data.y_test[E_global], half, len(E_global) - half,
                                     _rng(seed, 3))
    E_local, B_local = [], []
    for k, part in enumerate(partitions):
        if len(part) < config.SPARSE_CLIENT:
            E_local.append(None)
            B_local.append(None)
            continue
        n_explained = min(config.N_LOCAL_EXPLAINED, len(part) // 2)
        n_background = min(config.N_LOCAL_BACKGROUND, len(part) - n_explained)
        first, second = stratified_pick(data.y_train[part], n_explained, n_background,
                                        _rng(seed, 4, k))
        E_local.append(part[first])
        B_local.append(part[second])
    return ExplanationSets(E_global, B_global, half_a, half_b, E_local, B_local)

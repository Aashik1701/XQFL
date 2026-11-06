"""Deterministic FedAvg and centralized training for v2.

Client RNGs depend only on (seed, round, partition), so a run's weights do not
depend on the order or the process in which clients execute. The acceptance
script checks this loop against the same client logic run through Flower and Ray.
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from v2 import config
from v2.data import make_data, partition_indices
from v2.models import accuracy, build_model, flatten, load_flat


def child_seed(*parts):
    state = np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0]
    return int(state)


def seed_rngs(value):
    np.random.seed(value)
    torch.manual_seed(value)


def initial_seed(seed, replicate=1):
    return child_seed(seed, 0, replicate)


def local_train(model, X, y, epochs, lr=config.LEARNING_RATE, batch_size=config.BATCH_SIZE):
    """Same steps as HybridQuantumClassifier.fit, on inputs already in q-space."""
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(X_t))
        for start in range(0, len(X_t), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            criterion(model(X_t[batch]), y_t[batch]).backward()
            optimizer.step()


def client_update(model, global_weights, X_k, y_k, seed, round_number, partition_id):
    """One client's round. Used by both this loop and the Flower check."""
    load_flat(model, global_weights)
    seed_rngs(child_seed(seed, round_number, partition_id + 1))
    local_train(model, X_k, y_k, config.LOCAL_EPOCHS)
    return flatten(model)


def fedavg(updates):
    """Sample-weighted mean of (weights, n) pairs, in partition order."""
    total = sum(n for _, n in updates)
    return sum(weights * (n / total) for weights, n in updates if n > 0)


def _majority(y):
    return float(max(y.mean(), 1 - y.mean()))


def _data_for(spec):
    return make_data(spec.dataset, spec.seed, spec.n_features, spec.zero_handling)


def run_federated(spec, data=None):
    data = data or _data_for(spec)
    parts = partition_indices(data.y_train, spec.n_clients, spec.regime, spec.seed)
    assigned = np.sort(np.concatenate(parts))
    if not np.array_equal(assigned, np.arange(len(data.y_train))):
        raise RuntimeError("Partitions do not cover each training record exactly once")

    model = build_model(spec.encoding, spec.n_features, spec.n_layers, initial_seed(spec.seed))
    global_weights = flatten(model)
    P = len(global_weights)
    history = np.zeros((spec.n_rounds + 1, P))
    clients = np.full((spec.n_rounds, spec.n_clients, P), np.nan)
    history[0] = global_weights
    val_acc, test_acc = [accuracy(model, data.X_val, data.y_val)], [accuracy(model, data.X_test, data.y_test)]

    for t in range(1, spec.n_rounds + 1):
        updates = []
        for k, part in enumerate(parts):
            if len(part) == 0:
                continue
            weights = client_update(model, global_weights, data.X_train[part],
                                    data.y_train[part], spec.seed, t, k)
            clients[t - 1, k] = weights
            updates.append((weights, len(part)))
        global_weights = fedavg(updates)
        # Round through the model so history holds exactly the stored float precision.
        load_flat(model, global_weights)
        global_weights = flatten(model)
        history[t] = global_weights
        val_acc.append(accuracy(model, data.X_val, data.y_val))
        test_acc.append(accuracy(model, data.X_test, data.y_test))

    meta = dict(
        val_acc=val_acc, test_acc=test_acc,
        train_acc_final=accuracy(model, data.X_train, data.y_train),
        partition_sizes=[len(p) for p in parts],
        partition_positive=[int(data.y_train[p].sum()) for p in parts],
    )
    arrays = dict(global_weights=history, client_weights=clients)
    arrays.update({f"partition_{k}": p for k, p in enumerate(parts)})
    return arrays, meta, data


def run_centralized(spec, data=None):
    data = data or _data_for(spec)
    model = build_model(spec.encoding, spec.n_features, spec.n_layers,
                        initial_seed(spec.seed, spec.replicate))
    seed_rngs(child_seed(spec.seed, 0, 1000 + spec.replicate))
    local_train(model, data.X_train, data.y_train, spec.n_rounds * config.LOCAL_EPOCHS)
    meta = dict(val_acc=[accuracy(model, data.X_val, data.y_val)],
                test_acc=[accuracy(model, data.X_test, data.y_test)],
                train_acc_final=accuracy(model, data.X_train, data.y_train))
    return dict(global_weights=flatten(model)[None, :]), meta, data


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=1, allow_nan=True))
    os.replace(temporary, path)


def train_task(spec, out_dir):
    """Train one run and save weights. Skips runs that already completed."""
    folder = out_dir / spec.key
    done = folder / "train.json"
    if done.exists():
        return "skipped"
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runner = run_federated if spec.kind == "fed" else run_centralized
    arrays, meta, data = runner(spec)
    temporary = folder / "weights.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, folder / "weights.npz")
    meta.update(spec=spec.as_dict(), code=config.code_fingerprint(),
                seconds=time.monotonic() - started, data=data.meta,
                val_majority=_majority(data.y_val), test_majority=_majority(data.y_test))
    write_json(done, meta)
    return "trained"

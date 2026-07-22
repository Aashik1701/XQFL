"""Federation entry point for XQFL simulation.

run_federation() orchestrates the full federated pipeline:
  1. Fit a shared PCA + scaler on the full training set (centralised preprocessing).
  2. Partition data across n_clients using the requested split strategy.
  3. Build a per-round server-side evaluate_fn that logs global test accuracy.
  4. Launch Flower simulation (Ray backend) with the chosen FL strategy.
  5. Return (history, round_log) where round_log = [(round, test_acc), ...].

Design note on shared PCA:
  All clients project into the same 4-dimensional quantum feature space by using
  a single PCA fitted on the full training set.  Only VQC weights and the linear
  classification head are federated.  Fitting separate PCAs per client would make
  the weight spaces incommensurable and aggregation meaningless.
"""

import os
import sys
import numpy as np

from flwr.simulation import start_simulation
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters, Context

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from data.partition import iid_split, dirichlet_split, quantity_skew_split
from federation.server import make_fedavg_strategy, TrimmedMeanStrategy


# ── helpers ──────────────────────────────────────────────────────────────────

def _fit_shared_preprocessor(X_train, n_qubits):
    """Fit PCA(n_qubits) + MinMaxScaler([0, pi]) on the full training set.

    Returns (pca, scaler) — sklearn objects that can be attached to any
    HybridQuantumClassifier instance to skip re-fitting.
    """
    pca = PCA(n_components=n_qubits)
    X_pca = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit_transform(X_pca)
    return pca, scaler


def _partition_data(X_train, y_train, n_clients, split, seed):
    if split == "iid":
        return iid_split(X_train, y_train, n_clients, seed=seed)
    elif split == "dirichlet":
        return dirichlet_split(X_train, y_train, n_clients, alpha=0.5, seed=seed)
    elif split == "quantity":
        return quantity_skew_split(X_train, y_train, n_clients, skew_factor=0.5, seed=seed)
    else:
        raise ValueError(f"Unknown split strategy: {split!r}")


# ── main entry point ──────────────────────────────────────────────────────────

def run_federation(
    X_train, y_train, X_test, y_test,
    encoding_fn,
    n_clients=4,
    n_rounds=15,
    local_epochs=3,
    n_qubits=4,
    n_layers=2,
    split="iid",
    strategy_name="fedavg",
    proportion_to_cut=0.1,
    seed=42,
    project_root=None,
):
    """Run federated learning simulation.

    Parameters
    ----------
    X_train, y_train : full training set (all clients' data before partitioning).
    X_test, y_test   : held-out test set for server-side global evaluation.
    encoding_fn      : quantum encoding function (e.g. angle_encoding).
    n_clients        : number of virtual FL clients (default 4).
    n_rounds         : number of federated rounds (default 15).
    local_epochs     : gradient epochs per client per round (default 3).
    n_qubits         : number of qubits / PCA components (default 4).
    n_layers         : VQC layers in BasicEntanglerLayers (default 2).
    split            : 'iid' | 'dirichlet' | 'quantity' (default 'iid').
    strategy_name    : 'fedavg' | 'trimmed_mean' (default 'fedavg').
    proportion_to_cut: trim fraction for TrimmedMeanStrategy (default 0.1).
    seed             : random seed for data partitioning (default 42).
    project_root     : absolute path to the XQFL project root; added to
                       PYTHONPATH in Ray workers so they can import project modules.

    Returns
    -------
    history   : flwr.server.history.History object
    round_log : list of (round_number, global_test_accuracy) tuples
    """
    # ── 1. shared preprocessor ──────────────────────────────────────────────
    shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, n_qubits)

    # ── 2. data partitioning ────────────────────────────────────────────────
    partitions = _partition_data(X_train, y_train, n_clients, split, seed)
    for i, (Xi, yi) in enumerate(partitions):
        print(f"  Client {i}: {len(Xi)} samples  "
              f"(class 1 ratio: {yi.mean():.2f})")

    # ── 3. server-side evaluation model ────────────────────────────────────
    # Imported here so this function is self-contained (Ray workers do their own imports)
    from models.quantum_classifier import HybridQuantumClassifier
    from federation.client import XQFLClient

    eval_model = HybridQuantumClassifier(
        n_qubits=n_qubits, n_layers=n_layers, encoding_fn=encoding_fn
    )
    eval_model.pca     = shared_pca
    eval_model._scaler = shared_scaler

    round_log = []

    def evaluate_fn(server_round, parameters, eval_config):
        eval_model.set_weights(parameters)
        acc  = float((eval_model.predict(X_test) == y_test).mean())
        loss = 1.0 - acc
        round_log.append((server_round, acc))
        print(f"  Round {server_round:3d} | global_test_acc = {acc:.4f}")
        return loss, {"accuracy": acc}

    # ── 4. build strategy ───────────────────────────────────────────────────
    initial_params = ndarrays_to_parameters(eval_model.get_weights())

    strategy_kwargs = dict(
        initial_params=initial_params,
        evaluate_fn=evaluate_fn,
        n_clients=n_clients,
        n_rounds=n_rounds,
    )

    if strategy_name == "fedavg":
        strategy = make_fedavg_strategy(**strategy_kwargs)
    elif strategy_name == "trimmed_mean":
        strategy = TrimmedMeanStrategy(
            proportion_to_cut=proportion_to_cut,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=n_clients,
            min_evaluate_clients=0,
            min_available_clients=n_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_params,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")

    # ── 5. client factory ───────────────────────────────────────────────────
    # Capture all closure variables as plain Python objects so Ray can pickle them.
    _partitions    = partitions
    _shared_pca    = shared_pca
    _shared_scaler = shared_scaler
    _n_clients     = n_clients
    _n_qubits      = n_qubits
    _n_layers      = n_layers
    _local_epochs  = local_epochs
    _project_root  = project_root
    _encoding_fn   = encoding_fn

    def client_fn(context: Context):
        # Ray worker process: ensure project root is importable.
        if _project_root and _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from models.quantum_classifier import HybridQuantumClassifier
        from federation.client import XQFLClient

        partition_id = int(context.node_id) % _n_clients
        X_i, y_i    = _partitions[partition_id]

        model          = HybridQuantumClassifier(
            n_qubits=_n_qubits, n_layers=_n_layers, encoding_fn=_encoding_fn
        )
        model.pca      = _shared_pca
        model._scaler  = _shared_scaler

        return XQFLClient(model, X_i, y_i, local_epochs=_local_epochs).to_client()

    # ── 6. launch simulation ─────────────────────────────────────────────────
    ray_init_args = {"include_dashboard": False, "log_to_driver": False}
    if project_root:
        ray_init_args["runtime_env"] = {
            "env_vars": {
                "PYTHONPATH": project_root + ":" + os.environ.get("PYTHONPATH", "")
            }
        }

    history = start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
        ray_init_args=ray_init_args,
    )

    return history, round_log

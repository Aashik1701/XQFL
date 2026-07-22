"""Federation entry point for XQFL simulation.

run_federation() orchestrates the full federated pipeline:
  1. Fit a shared PCA + scaler on the full training set (centralised preprocessing).
  2. Partition data across n_clients using the requested split strategy.
  3. Build a per-round server-side evaluate_fn that logs global test accuracy.
     If a logger is provided, also computes Q-SHAP on a shared validation set
     and appends the global explanation row after each round.
  4. Launch Flower simulation (Ray backend) with the chosen FL strategy.
  5. Return (history, round_log).

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
from federation.server import make_fedavg_strategy, TrimmedMeanStrategy, LoggingFedAvg


# ── helpers ──────────────────────────────────────────────────────────────────

def _fit_shared_preprocessor(X_train, n_qubits):
    """Fit PCA(n_qubits) + MinMaxScaler([0, pi]) on the full training set."""
    pca    = PCA(n_components=n_qubits)
    X_pca  = pca.fit_transform(X_train)
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    scaler.fit_transform(X_pca)
    return pca, scaler


def _partition_data(X_train, y_train, n_clients, split, seed, dirichlet_alpha=0.5):
    if split == "iid":
        return iid_split(X_train, y_train, n_clients, seed=seed)
    elif split == "dirichlet":
        return dirichlet_split(X_train, y_train, n_clients, alpha=dirichlet_alpha, seed=seed)
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
    dirichlet_alpha=0.5,
    strategy_name="fedavg",
    proportion_to_cut=0.1,
    seed=42,
    project_root=None,
    # Phase 6 logging parameters (all optional — omit for Phase 5 behaviour)
    logger=None,
    X_val=None,          # shared validation set for global Q-SHAP (defaults to X_test)
    n_global_explain=10, # samples for global Q-SHAP
    n_global_bg=20,      # background samples for global Q-SHAP
    n_explain=5,         # samples for local Q-SHAP per client (0 = skip)
    n_bg=10,             # background for local Q-SHAP
    n_permutations=0,    # Q-SHAP permutations; 0 = skip (Phase 5 default)
):
    """Run federated learning simulation.

    Parameters
    ----------
    X_train, y_train : full training set (all clients' data before partitioning).
    X_test, y_test   : held-out test set for server-side global accuracy.
    encoding_fn      : quantum encoding function (e.g. angle_encoding).
    n_clients        : number of virtual FL clients (default 4).
    n_rounds         : number of federated rounds (default 15).
    local_epochs     : gradient epochs per client per round (default 3).
    n_qubits         : qubits / PCA components (default 4).
    n_layers         : VQC layers (default 2).
    split            : 'iid' | 'dirichlet' | 'quantity' (default 'iid').
    strategy_name    : 'fedavg' | 'trimmed_mean' (default 'fedavg').
    proportion_to_cut: trim fraction for TrimmedMeanStrategy (default 0.1).
    seed             : RNG seed for data partitioning (default 42).
    project_root     : absolute path to XQFL project root (for Ray workers).
    logger           : ExperimentLogger — if provided, per-client and global
                       Q-SHAP rows are written after every round.
    X_val            : optional shared validation set for global explanation.
                       Defaults to X_test when None.
    n_global_explain : explanation samples for global Q-SHAP (default 10).
    n_global_bg      : background samples for global Q-SHAP (default 20).
    n_explain        : explanation samples per client per round (default 5).
    n_bg             : background samples per client per round (default 10).
    n_permutations   : Q-SHAP permutations (0 = skip; default 0 preserves Phase 5).

    Returns
    -------
    history   : flwr.server.history.History object
    round_log : list of (round_number, global_test_accuracy) tuples
    """
    # ── 1. shared preprocessor ──────────────────────────────────────────────
    shared_pca, shared_scaler = _fit_shared_preprocessor(X_train, n_qubits)

    # ── 2. data partitioning ────────────────────────────────────────────────
    partitions = _partition_data(X_train, y_train, n_clients, split, seed,
                                  dirichlet_alpha=dirichlet_alpha)
    for i, (Xi, yi) in enumerate(partitions):
        print(f"  Client {i}: {len(Xi)} samples  "
              f"(class 1 ratio: {yi.mean():.2f})")

    # ── 3. server-side evaluation model ─────────────────────────────────────
    from models.quantum_classifier import HybridQuantumClassifier

    eval_model            = HybridQuantumClassifier(
        n_qubits=n_qubits, n_layers=n_layers, encoding_fn=encoding_fn
    )
    eval_model.pca        = shared_pca
    eval_model._scaler    = shared_scaler

    # Shared validation set for global explanation (default: X_test)
    _X_val   = X_test if X_val is None else X_val
    _y_val   = y_test if X_val is None else None    # accuracy only logged when X_test
    round_log = []

    def evaluate_fn(server_round, parameters, eval_config):
        eval_model.set_weights(parameters)
        test_acc = float((eval_model.predict(X_test) == y_test).mean())
        round_log.append((server_round, test_acc))
        print(f"  Round {server_round:3d} | global_test_acc = {test_acc:.4f}")

        # Global Q-SHAP on shared validation set
        if logger is not None and n_permutations > 0:
            from explain.q_shap import q_shap_values, make_quantum_predict_fn
            X_val_q  = eval_model._preprocess(_X_val)
            X_bg_q   = eval_model._preprocess(X_train[:min(n_global_bg, len(X_train))])
            q_predict = make_quantum_predict_fn(eval_model)
            n_exp     = min(n_global_explain, len(X_val_q))
            qshap     = q_shap_values(
                q_predict, X_val_q[:n_exp], X_bg_q,
                n_permutations=n_permutations, seed=42
            )
            global_qshap = qshap.mean(axis=0)
            val_acc      = test_acc   # using test set as validation set here
            logger.log(server_round, "global", global_qshap, val_acc)

        return 1.0 - test_acc, {"accuracy": test_acc}

    # ── 4. build strategy ────────────────────────────────────────────────────
    initial_params = ndarrays_to_parameters(eval_model.get_weights())

    _common = dict(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=0,
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_params,
    )

    if strategy_name == "fedavg":
        if logger is not None:
            strategy = LoggingFedAvg(logger=logger, n_qubits=n_qubits, **_common)
        else:
            strategy = make_fedavg_strategy(
                initial_params, evaluate_fn, n_clients, n_rounds
            )
    elif strategy_name == "trimmed_mean":
        strategy = TrimmedMeanStrategy(
            proportion_to_cut=proportion_to_cut, **_common
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy_name!r}")

    # ── 5. client factory ────────────────────────────────────────────────────
    _partitions     = partitions
    _shared_pca     = shared_pca
    _shared_scaler  = shared_scaler
    _n_clients      = n_clients
    _n_qubits       = n_qubits
    _n_layers       = n_layers
    _local_epochs   = local_epochs
    _project_root   = project_root
    _encoding_fn    = encoding_fn
    _n_permutations = n_permutations
    _n_explain      = n_explain
    _n_bg           = n_bg

    def client_fn(context: Context):
        if _project_root and _project_root not in sys.path:
            sys.path.insert(0, _project_root)

        from models.quantum_classifier import HybridQuantumClassifier
        from federation.client import XQFLClient

        partition_id   = int(context.node_id) % _n_clients
        X_i, y_i      = _partitions[partition_id]

        model          = HybridQuantumClassifier(
            n_qubits=_n_qubits, n_layers=_n_layers, encoding_fn=_encoding_fn
        )
        model.pca      = _shared_pca
        model._scaler  = _shared_scaler

        return XQFLClient(
            model, X_i, y_i,
            client_id=partition_id,
            local_epochs=_local_epochs,
            n_permutations=_n_permutations,
            n_explain=_n_explain,
            n_bg=_n_bg,
        ).to_client()

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

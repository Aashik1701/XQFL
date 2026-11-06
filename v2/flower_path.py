"""The same client logic run through Flower and Ray, for the acceptance check only."""
import numpy as np
from flwr.client import NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import start_simulation

from v2 import config
from v2.data import make_data, partition_indices
from v2.models import build_model, flatten, load_flat
from v2.train import client_update, initial_seed


def _to_arrays(model, vector):
    load_flat(model, vector)
    return [p.detach().numpy().copy() for p in model.parameters()]


def _from_arrays(arrays):
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


class _Client(NumPyClient):
    def __init__(self, spec, data, part, partition_id):
        self.spec, self.data, self.part, self.partition_id = spec, data, part, partition_id
        self.model = build_model(spec.encoding, spec.n_features, spec.n_layers, initial_seed(spec.seed))

    def fit(self, parameters, fit_config):
        import torch

        torch.set_num_threads(1)
        round_number = int(fit_config["round"])
        weights = client_update(self.model, _from_arrays(parameters), self.data.X_train[self.part],
                                self.data.y_train[self.part], self.spec.seed, round_number,
                                self.partition_id)
        return _to_arrays(self.model, weights), len(self.part), {"partition_id": self.partition_id}


class _Recording(FedAvg):
    def __init__(self, expected, **kwargs):
        super().__init__(**kwargs)
        self.expected = expected
        self.history = []

    def aggregate_fit(self, server_round, results, failures):
        ids = sorted(int(r.metrics["partition_id"]) for _, r in results)
        if ids != list(range(self.expected)):
            raise RuntimeError(f"Round {server_round}: partitions {ids}")
        ordered = sorted(results, key=lambda item: int(item[1].metrics["partition_id"]))
        parameters, metrics = super().aggregate_fit(server_round, ordered, failures)
        self.history.append(_from_arrays(parameters_to_ndarrays(parameters)))
        return parameters, metrics


def run_flower(spec):
    """Return global weights per round, shape (rounds + 1, P)."""
    data = make_data(spec.dataset, spec.seed, spec.n_features, spec.zero_handling)
    parts = partition_indices(data.y_train, spec.n_clients, spec.regime, spec.seed)
    model = build_model(spec.encoding, spec.n_features, spec.n_layers, initial_seed(spec.seed))
    start = flatten(model)
    strategy = _Recording(
        spec.n_clients, fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=spec.n_clients,
        min_evaluate_clients=0, min_available_clients=spec.n_clients,
        initial_parameters=ndarrays_to_parameters(_to_arrays(model, start)),
        on_fit_config_fn=lambda r: {"round": r})

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        return _Client(spec, data, parts[partition_id], partition_id).to_client()

    start_simulation(
        client_fn=client_fn, num_clients=spec.n_clients,
        config=ServerConfig(num_rounds=spec.n_rounds), strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0},
        ray_init_args={"include_dashboard": False, "log_to_driver": False,
                       "runtime_env": {"env_vars": {"PYTHONPATH": str(config.ROOT)}}},
    )
    import ray

    ray.shutdown()
    return np.vstack([start] + strategy.history)

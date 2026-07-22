"""Server-side federation strategies.

Provides:
  make_fedavg_strategy   — build a standard FedAvg strategy with server-side eval.
  TrimmedMeanStrategy    — custom FedAvg subclass with trimmed-mean aggregation.

Spec note (section 7): Flower does NOT ship a Trimmed-Mean strategy.  The helper
aggregate_trimmed_avg exists in flwr.server.strategy.aggregate but is not exposed
as a Strategy class.  TrimmedMeanStrategy implements it as a proper Strategy
subclass, overriding aggregate_fit.
"""

from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_trimmed_avg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


def make_fedavg_strategy(initial_params, evaluate_fn, n_clients, n_rounds):
    """Return a FedAvg strategy configured for XQFL simulation.

    Parameters
    ----------
    initial_params : Parameters — starting weights broadcast to all clients in round 1.
    evaluate_fn    : callable(server_round, NDArrays, config) -> (loss, metrics)
                     Called after every round to compute global test accuracy.
    n_clients      : int — total number of virtual clients.
    n_rounds       : int — total number of federated rounds (informational only here).

    Returns
    -------
    strategy : FedAvg instance
    """
    return FedAvg(
        fraction_fit=1.0,            # sample all clients every round
        fraction_evaluate=0.0,       # skip client-side evaluate rounds; use evaluate_fn
        min_fit_clients=n_clients,
        min_evaluate_clients=0,
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_params,
    )


class TrimmedMeanStrategy(FedAvg):
    """FedAvg with trimmed-mean aggregation instead of weighted average.

    Overrides aggregate_fit to discard the top and bottom `proportion_to_cut`
    fraction of parameter updates (sorted by L2 norm) before averaging.
    This provides robustness against a small number of Byzantine or outlier clients.

    All other FedAvg behaviour (client selection, evaluation, metrics) is inherited.

    Parameters
    ----------
    proportion_to_cut : float in (0, 0.5) — fraction to trim from each tail.
                        E.g. 0.1 trims the 10% lowest and 10% highest updates.
    **kwargs          : passed through to FedAvg.__init__ (fraction_fit, evaluate_fn, etc.)
    """

    def __init__(self, proportion_to_cut=0.1, **kwargs):
        super().__init__(**kwargs)
        self.proportion_to_cut = proportion_to_cut

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client parameters using trimmed mean.

        Parameters
        ----------
        server_round : int
        results      : list of (ClientProxy, FitRes)
        failures     : list of BaseException

        Returns
        -------
        (aggregated_parameters, metrics_dict)
        """
        if not results:
            return None, {}

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate_trimmed_avg(
            weights_results, self.proportion_to_cut
        )

        return ndarrays_to_parameters(aggregated_ndarrays), {}

"""Server-side federation strategies.

Provides:
  make_fedavg_strategy   — build a standard FedAvg strategy with server-side eval.
  TrimmedMeanStrategy    — custom FedAvg subclass with trimmed-mean aggregation.
  LoggingFedAvg          — FedAvg subclass that extracts per-client Q-SHAP from
                           fit metrics and writes them to an ExperimentLogger.

Spec note (section 7): Flower does NOT ship a Trimmed-Mean strategy.  The helper
aggregate_trimmed_avg exists in flwr.server.strategy.aggregate but is not exposed
as a Strategy class.  TrimmedMeanStrategy implements it as a proper Strategy
subclass, overriding aggregate_fit.
"""

from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate_trimmed_avg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


# ── Deterministic base ───────────────────────────────────────────────────────

class DeterministicFedAvg(FedAvg):
    """FedAvg with results sorted by integer client CID before aggregation.

    Flower's aggregate_inplace accumulates weights in list order:
        params = results[0] * scale[0]
        for i in 1..: params += results[i] * scale[i]
    Floating-point addition is non-associative, so arrival order (which Ray does
    not guarantee) produces different aggregated weights across runs even with the
    same Python seed.  Sorting by CID makes aggregation order deterministic.
    """

    def aggregate_fit(self, server_round, results, failures):
        # Correctness guard for the node_id%n_clients partition-assignment bug
        # (fixed in run_federation.client_fn): every round MUST train each data
        # partition exactly once. The observed partition ids are logged here and
        # asserted to be exactly {0, 1, ..., n-1}; anything else (a dropped or
        # doubled partition) halts the run rather than silently corrupting it.
        pids = sorted(int(fr.metrics["client_id"]) for _, fr in results
                      if fr.metrics and "client_id" in fr.metrics)
        if pids:
            expected = list(range(self.min_fit_clients))
            print(f"[round {server_round}] partitions trained: {pids}")
            if pids != expected:
                raise RuntimeError(
                    f"[round {server_round}] partition ids {pids} != expected "
                    f"{expected} — a partition was dropped, double-counted, or "
                    "failed; client-to-partition assignment is broken."
                )
        sorted_results = sorted(results, key=lambda x: int(x[0].cid))
        return super().aggregate_fit(server_round, sorted_results, failures)


# ── FedAvg factory ───────────────────────────────────────────────────────────

def make_fedavg_strategy(initial_params, evaluate_fn, n_clients, n_rounds):
    """Return a DeterministicFedAvg strategy configured for XQFL simulation.

    Parameters
    ----------
    initial_params : Parameters — starting weights broadcast in round 1.
    evaluate_fn    : callable(server_round, NDArrays, config) -> (loss, metrics)
                     Called after every round to compute global test accuracy.
    n_clients      : int — total virtual clients.
    n_rounds       : int — informational; not used directly here.

    Returns
    -------
    strategy : DeterministicFedAvg instance
    """
    return DeterministicFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=0,
        min_available_clients=n_clients,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_params,
    )


# ── Trimmed-Mean strategy (custom subclass) ──────────────────────────────────

class TrimmedMeanStrategy(DeterministicFedAvg):
    """FedAvg with trimmed-mean aggregation instead of weighted average.

    Overrides aggregate_fit to discard the top and bottom `proportion_to_cut`
    fraction of parameter updates (sorted by L2 norm) before averaging.
    This provides robustness against a small number of Byzantine or outlier clients.

    All other FedAvg behaviour (client selection, evaluation, metrics) is inherited.

    Parameters
    ----------
    proportion_to_cut : float in (0, 0.5) — fraction to trim from each tail.
                        E.g. 0.1 trims the 10% lowest and 10% highest updates.
    **kwargs          : passed through to FedAvg.__init__.
    """

    def __init__(self, proportion_to_cut=0.1, **kwargs):
        super().__init__(**kwargs)
        self.proportion_to_cut = proportion_to_cut

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        results = sorted(results, key=lambda x: int(x[0].cid))

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        aggregated_ndarrays = aggregate_trimmed_avg(
            weights_results, self.proportion_to_cut
        )

        return ndarrays_to_parameters(aggregated_ndarrays), {}


# ── Logging FedAvg (Phase 6) ─────────────────────────────────────────────────

class LoggingFedAvg(DeterministicFedAvg):
    """FedAvg that extracts per-client Q-SHAP attributions and logs them.

    After each round's aggregate_fit, this class reads the Q-SHAP values that
    each XQFLClient attached to its FitRes.metrics dict and writes one log row
    per client to the provided ExperimentLogger.

    Parameters
    ----------
    logger   : ExperimentLogger | None — if None, behaves like plain FedAvg.
    n_qubits : int — number of Q-SHAP features expected (default 4).
    **kwargs : passed through to FedAvg.__init__.
    """

    def __init__(self, logger=None, n_qubits=4, **kwargs):
        super().__init__(**kwargs)
        self._logger   = logger
        self._n_qubits = n_qubits

    def aggregate_fit(self, server_round, results, failures):
        # Extract and log per-client explanations BEFORE aggregating so the
        # values represent locally-trained (pre-aggregation) model behaviour.
        if self._logger is not None:
            for _, fit_res in results:
                m = fit_res.metrics
                if not m:
                    continue
                client_id = m.get("client_id", -1)
                qshap = [m.get(f"qshap_{i}", None) for i in range(self._n_qubits)]
                if any(v is not None for v in qshap):
                    qshap_clean = [v if v is not None else 0.0 for v in qshap]
                    acc = m.get("accuracy", 0.0)
                    self._logger.log(server_round, client_id, qshap_clean, acc)

        return super().aggregate_fit(server_round, results, failures)

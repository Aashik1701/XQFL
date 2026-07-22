"""Experiment configuration dataclass for XQFL.

A single ExperimentConfig fully specifies one federated run so that any row
in the log file is reproducible from the config alone.
"""

from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    dataset: str         = "wisconsin"   # "wisconsin", "pima", "mnist", ...
    encoding: str        = "angle"       # "angle", "amplitude", "reuploading", "iqp"
    aggregator: str      = "fedavg"      # "fedavg", "trimmed_mean"
    distribution: str    = "iid"         # "iid", "label_skew", "quantity_skew"
    dirichlet_alpha: float = 0.5         # only used when distribution == "label_skew"
    n_clients: int       = 4
    n_rounds: int        = 15
    n_qubits: int        = 4
    n_layers: int        = 2
    local_epochs: int    = 3
    seed: int            = 42

    def as_dict(self):
        return {
            "dataset":         self.dataset,
            "encoding":        self.encoding,
            "aggregator":      self.aggregator,
            "distribution":    self.distribution,
            "dirichlet_alpha": self.dirichlet_alpha,
            "n_clients":       self.n_clients,
            "n_rounds":        self.n_rounds,
            "n_qubits":        self.n_qubits,
            "n_layers":        self.n_layers,
            "local_epochs":    self.local_epochs,
            "seed":            self.seed,
        }

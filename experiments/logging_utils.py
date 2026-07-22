"""Flat CSV experiment logger for XQFL.

Writes one row per (round, client_id) pair with the schema required by Phase 6:
  round, client_id, encoding, aggregation_method, distribution_type,
  explanation_vector, accuracy, timestamp

client_id is an integer for federated clients or the string "global" for the
server's centralized explanation on the shared validation set.

Usage
-----
    from experiments.config import ExperimentConfig
    from experiments.logging_utils import ExperimentLogger

    cfg = ExperimentConfig(encoding="angle", aggregator="fedavg", distribution="iid")
    logger = ExperimentLogger("experiments/phase6_run.csv", cfg)
    logger.log(round_num=1, client_id=0, explanation_vector=[0.1, -0.2, 0.05, 0.3], accuracy=0.82)
    logger.log(round_num=1, client_id="global", explanation_vector=[...], accuracy=0.85)
"""

import csv
import json
import os
from datetime import datetime, timezone
import numpy as np


class ExperimentLogger:
    COLUMNS = [
        "round",
        "client_id",
        "encoding",
        "aggregation_method",
        "distribution_type",
        "explanation_vector",
        "accuracy",
        "timestamp",
    ]

    def __init__(self, csv_path, config):
        """Initialise logger and write CSV header.

        Parameters
        ----------
        csv_path : str — absolute or project-relative path to the output CSV.
        config   : ExperimentConfig
        """
        self.csv_path = csv_path
        self.config   = config
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            writer.writeheader()

    def log(self, round_num, client_id, explanation_vector, accuracy):
        """Append one row to the log CSV.

        Parameters
        ----------
        round_num          : int — federated round (0 = pre-training global eval).
        client_id          : int | str — integer for clients, "global" for server eval.
        explanation_vector : array-like of floats (n_qubits,) — Q-SHAP attribution.
        accuracy           : float — model accuracy on the relevant dataset.
        """
        vec = [float(v) for v in np.asarray(explanation_vector).flatten()]
        row = {
            "round":               round_num,
            "client_id":           client_id,
            "encoding":            self.config.encoding,
            "aggregation_method":  self.config.aggregator,
            "distribution_type":   self.config.distribution,
            "explanation_vector":  json.dumps(vec),
            "accuracy":            f"{float(accuracy):.6f}",
            "timestamp":           datetime.now(timezone.utc).isoformat(),
        }
        with open(self.csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.COLUMNS)
            writer.writerow(row)

    def load(self):
        """Load the log as a list of dicts (for post-hoc analysis).

        explanation_vector is parsed back from JSON to a Python list of floats.
        """
        rows = []
        with open(self.csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                row["round"]       = int(row["round"])
                row["accuracy"]    = float(row["accuracy"])
                row["explanation_vector"] = json.loads(row["explanation_vector"])
                # client_id may be int or "global"
                try:
                    row["client_id"] = int(row["client_id"])
                except ValueError:
                    pass
                rows.append(row)
        return rows

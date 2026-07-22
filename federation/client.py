"""Flower federated client wrapping HybridQuantumClassifier.

XQFLClient implements the NumPyClient interface:
  get_parameters — return current VQC + classifier weights as NDArrays
  fit            — receive aggregated weights, do local training, return updated weights
                   (optionally computes local Q-SHAP and returns it in the metrics dict)
  evaluate       — report local train accuracy

Shared-PCA guard
----------------
All clients MUST share the same PCA fitted on the full training set before
federation starts.  The __init__ assertion catches misconfiguration immediately
rather than allowing silent divergence later.  Without a shared preprocessor,
each client projects into a different 4-D quantum feature space, making weight
aggregation mathematically meaningless (apples-to-oranges in hidden-layer space).
"""

import numpy as np
from flwr.client import NumPyClient


class XQFLClient(NumPyClient):
    """Flower client wrapping a HybridQuantumClassifier.

    Parameters
    ----------
    model         : HybridQuantumClassifier with pca and _scaler already set to
                    the server-fitted shared preprocessor.
    X_train       : local training data in the original feature space.
    y_train       : local training labels.
    client_id     : int — partition index, included in fit() metrics for logging.
    local_epochs  : gradient epochs per federated round (default 3).
    n_permutations: Q-SHAP permutations per feature per sample.  0 = skip Q-SHAP
                    (Phase 5 behaviour).  8–16 is sufficient for 4 qubits (default 8).
    n_explain     : number of local samples to explain per round (default 5).
    n_bg          : number of background samples for Q-SHAP baseline (default 10).
    """

    def __init__(
        self,
        model,
        X_train,
        y_train,
        client_id=0,
        local_epochs=3,
        n_permutations=0,
        n_explain=5,
        n_bg=10,
    ):
        # ── shared-PCA guard ────────────────────────────────────────────────
        if model.pca is None or model._scaler is None:
            raise AssertionError(
                f"[Client {client_id}] model.pca and model._scaler must be set "
                "to the server's shared pre-fitted PCA before creating XQFLClient. "
                "Without a shared preprocessor all clients project into different "
                "quantum feature spaces; weight aggregation is then meaningless."
            )
        print(
            f"  [Client {client_id}] Using shared pre-fitted PCA from server "
            f"(n_components={model.n_qubits}, "
            f"explained_var={model.pca.explained_variance_ratio_.sum():.3f})"
        )
        # ── store state ──────────────────────────────────────────────────────
        self.model         = model
        self.X_train       = X_train
        self.y_train       = y_train
        self.client_id     = int(client_id)
        self.local_epochs  = local_epochs
        self.n_permutations = n_permutations
        self.n_explain     = n_explain
        self.n_bg          = n_bg

    # ── Flower NumPyClient interface ─────────────────────────────────────────

    def get_parameters(self, config):
        """Return current model weights as a list of numpy arrays."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Receive global weights, run local training, return updated weights.

        If n_permutations > 0, also computes a local Q-SHAP explanation on a
        small subset of X_train and returns the per-feature mean attributions
        in the metrics dict under keys qshap_0 … qshap_{n_qubits-1}.

        Returns
        -------
        (weights, n_samples, metrics)
        """
        self.model.set_weights(parameters)
        self.model.fit(
            self.X_train, self.y_train,
            epochs=self.local_epochs,
            lr=0.05,
            batch_size=32,
        )

        acc = float((self.model.predict(self.X_train) == self.y_train).mean())

        metrics = {
            "client_id": self.client_id,
            "accuracy":  acc,
        }

        if self.n_permutations > 0:
            qshap_mean = self._compute_local_qshap()
            for i, v in enumerate(qshap_mean):
                metrics[f"qshap_{i}"] = float(v)

        return self.model.get_weights(), len(self.X_train), metrics

    def evaluate(self, parameters, config):
        """Evaluate received weights on local training data.

        Returns
        -------
        (loss, n_samples, metrics)  — loss = 1 - train_accuracy
        """
        self.model.set_weights(parameters)
        acc  = float((self.model.predict(self.X_train) == self.y_train).mean())
        loss = 1.0 - acc
        return loss, len(self.X_train), {"accuracy": acc}

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_local_qshap(self):
        """Compute mean |Q-SHAP| over a small local subset.

        Operates in the quantum (PCA-reduced) feature space, using the shared PCA
        already attached to self.model.  Background is built from the first n_bg
        preprocessed local samples.

        Returns
        -------
        mean_qshap : ndarray of shape (n_qubits,)
        """
        from explain.q_shap import q_shap_values, make_quantum_predict_fn

        X_q   = self.model._preprocess(self.X_train)          # (n_local, n_qubits)
        n_exp = min(self.n_explain, len(X_q))
        n_bg  = min(self.n_bg, len(X_q))

        # Guard for severely imbalanced partitions (can occur with Dirichlet alpha<<1)
        if n_exp < 1 or n_bg < 1:
            return np.zeros(self.model.n_qubits)

        q_predict = make_quantum_predict_fn(self.model)
        qshap     = q_shap_values(
            q_predict,
            X_q[:n_exp],
            X_q[:n_bg],
            n_permutations=self.n_permutations,
            seed=42,
        )                                                      # (n_exp, n_qubits)
        return qshap.mean(axis=0)                              # (n_qubits,)

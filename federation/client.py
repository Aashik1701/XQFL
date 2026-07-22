"""Flower federated client wrapping HybridQuantumClassifier.

XQFLClient implements the NumPyClient interface:
  get_parameters — return current VQC + classifier weights as NDArrays
  fit            — receive aggregated weights, do local training, return updated weights
  evaluate       — report local train accuracy (server-side evaluate_fn handles test set)

The model's PCA/scaler must be pre-fitted (shared from the federation server) before
passing into this client, so all clients project into the same quantum feature space.
"""

from flwr.client import NumPyClient


class XQFLClient(NumPyClient):
    """Flower client wrapping a HybridQuantumClassifier.

    Parameters
    ----------
    model       : HybridQuantumClassifier with pca and _scaler already set.
    X_train     : local training data in the original feature space.
    y_train     : local training labels.
    local_epochs: number of gradient steps per federated round.
    """

    def __init__(self, model, X_train, y_train, local_epochs=3):
        self.model        = model
        self.X_train      = X_train
        self.y_train      = y_train
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        """Return current model weights as a list of numpy arrays."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Receive global weights, run local training, return updated weights.

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
        return self.model.get_weights(), len(self.X_train), {}

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

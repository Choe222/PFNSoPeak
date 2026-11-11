import numpy as np
from sklearn.neural_network import MLPClassifier


class MLPModel:
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-2,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=True,
        random_state=42,
    ):
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state,
        )

    def forward(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        return self.model.predict(X_arr)

    def __call__(self, X):
        return self.forward(X)

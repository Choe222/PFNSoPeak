import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class RFMLPStackingModel:
    def __init__(
        self,
        n_splits=5,
        random_state=42,
        n_jobs=-1,
        rf_params=None,
        mlp_params=None,
        meta_lr_params=None,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs

        rf_cfg = {
            "n_estimators": 300,
            "max_depth": None,
            "random_state": random_state,
            "n_jobs": n_jobs,
        }
        if rf_params:
            rf_cfg.update(rf_params)

        mlp_cfg = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-2,
            "batch_size": 64,
            "learning_rate_init": 1e-3,
            "max_iter": 300,
            "early_stopping": True,
            "random_state": random_state,
        }
        if mlp_params:
            mlp_cfg.update(mlp_params)
        if isinstance(mlp_cfg["hidden_layer_sizes"], list):
            mlp_cfg["hidden_layer_sizes"] = tuple(mlp_cfg["hidden_layer_sizes"])

        meta_cfg = {
            "C": 1.0,
            "max_iter": 1000,
            "random_state": random_state,
        }
        if meta_lr_params:
            meta_cfg.update(meta_lr_params)

        self.rf = RandomForestClassifier(**rf_cfg)
        self.mlp = MLPClassifier(**mlp_cfg)
        self.meta_lr = LogisticRegression(**meta_cfg)

    def forward(self, X):
        X_arr = np.asarray(X, dtype=np.float32)
        rf_proba = self.rf.predict_proba(X_arr)
        mlp_proba = self.mlp.predict_proba(X_arr)
        meta_features = np.hstack([rf_proba, mlp_proba])
        return self.meta_lr.predict(meta_features)

    def __call__(self, X):
        return self.forward(X)

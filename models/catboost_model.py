import os
from pathlib import Path
from catboost import CatBoostClassifier


class CatBoostModel:
    def __init__(
        self,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
        train_dir=None,
        allow_writing_files=False,
    ):
        if train_dir is None:
            log_dir = os.environ.get("LOG_DIR")
            if log_dir:
                train_dir = str(Path(log_dir) / "catboost_info")
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=verbose,
            train_dir=train_dir,
            allow_writing_files=allow_writing_files,
        )

    def forward(self, X):
        return self.model.predict(X).ravel()

    def __call__(self, X):
        return self.forward(X)


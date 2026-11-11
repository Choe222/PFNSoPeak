import sys
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.rf_mlp_stacking import RFMLPStackingModel
from utils import load_config, load_train_data, save_joblib


def train_rf_mlp_stacking(X, y, config: dict = None, save: bool = True) -> RFMLPStackingModel:
    cfg = config or load_config("rf_mlp_stacking")
    model_cfg = cfg.get("model", {})

    model = RFMLPStackingModel(
        n_splits=model_cfg.get("n_splits", 5),
        random_state=model_cfg.get("random_state", 42),
        n_jobs=model_cfg.get("n_jobs", -1),
        rf_params=model_cfg.get("rf"),
        mlp_params=model_cfg.get("mlp"),
        meta_lr_params=model_cfg.get("meta_lr"),
    )

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y).ravel()
    n_classes = len(np.unique(y_arr))
    oof_features = np.zeros((len(X_arr), n_classes * 2), dtype=np.float64)

    skf = StratifiedKFold(
        n_splits=model.n_splits,
        shuffle=True,
        random_state=model.random_state,
    )
    for train_idx, val_idx in skf.split(X_arr, y_arr):
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_va = X_arr[val_idx]

        rf_fold = clone(model.rf).fit(X_tr, y_tr)
        mlp_fold = clone(model.mlp).fit(X_tr, y_tr)

        rf_proba = rf_fold.predict_proba(X_va)
        mlp_proba = mlp_fold.predict_proba(X_va)
        oof_features[val_idx] = np.hstack([rf_proba, mlp_proba])

    model.meta_lr.fit(oof_features, y_arr)
    model.rf.fit(X_arr, y_arr)
    model.mlp.fit(X_arr, y_arr)

    save_path = cfg.get("train", {}).get("save_path")
    if save and save_path:
        save_joblib(
            {
                "rf": model.rf,
                "mlp": model.mlp,
                "meta_lr": model.meta_lr,
            },
            save_path,
        )

    return model


def main():
    cfg = load_config("rf_mlp_stacking")
    X_train, y_train = load_train_data(cfg.get("data"))
    train_rf_mlp_stacking(X_train, y_train, cfg, save=True)
    print("RF + MLP Stacking training completed successfully.")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.mlp_model import MLPModel
from utils import load_config, load_train_data, save_joblib


def train_mlp(X, y, config: dict = None, save: bool = True) -> MLPModel:
    cfg = config or load_config("mlp")
    model = MLPModel(**cfg.get("model", {}))

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y).ravel()
    model.model.fit(X_arr, y_arr)

    save_path = cfg.get("train", {}).get("save_path")
    if save and save_path:
        save_joblib(model.model, save_path)

    return model


def main():
    cfg = load_config("mlp")
    X_train, y_train = load_train_data(cfg.get("data"))
    train_mlp(X_train, y_train, cfg, save=True)
    print("MLP training completed successfully.")


if __name__ == "__main__":
    main()

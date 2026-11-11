import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.xgboost_model import XGBoostModel
from utils import load_config, load_train_data, save_joblib


def train_xgboost(X, y, config: dict = None, save: bool = True) -> XGBoostModel:
    cfg = config or load_config("xgboost")
    model = XGBoostModel(**cfg.get("model", {}))
    model.model.fit(X, y)

    save_path = cfg.get("train", {}).get("save_path")
    if save and save_path:
        save_joblib(model.model, save_path)

    return model


def main():
    cfg = load_config("xgboost")
    X_train, y_train = load_train_data(cfg.get("data"))
    train_xgboost(X_train, y_train, cfg, save=True)
    print("XGBoost training completed successfully.")


if __name__ == "__main__":
    main()

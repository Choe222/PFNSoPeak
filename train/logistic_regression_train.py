import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.logistic_regression import LogisticRegressionModel
from utils import load_config, load_train_data, save_joblib


def train_logistic_regression(X, y, config: dict = None, save: bool = True) -> LogisticRegressionModel:
    cfg = config or load_config("logistic_regression")
    model = LogisticRegressionModel(**cfg.get("model", {}))
    model.model.fit(X, y)

    save_path = cfg.get("train", {}).get("save_path")
    if save and save_path:
        save_joblib(model.model, save_path)

    return model


def main():
    cfg = load_config("logistic_regression")
    X_train, y_train = load_train_data(cfg.get("data"))
    train_logistic_regression(X_train, y_train, cfg, save=True)
    print("Logistic Regression training completed successfully.")


if __name__ == "__main__":
    main()

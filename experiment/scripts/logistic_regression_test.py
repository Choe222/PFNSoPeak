import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.logistic_regression import LogisticRegressionModel
from train.logistic_regression_train import train_logistic_regression
from utils import (
    evaluate_overall_metrics,
    load_config,
    load_joblib,
    load_test_data,
    load_train_data,
    resolve_path,
)


def get_logistic_regression_model(cfg: dict = None) -> LogisticRegressionModel:
    cfg = cfg or load_config("logistic_regression")
    model_path = resolve_path(cfg["train"]["save_path"])
    if model_path.exists():
        model = LogisticRegressionModel(**cfg.get("model", {}))
        model.model = load_joblib(model_path)
        return model

    X_train, y_train = load_train_data(cfg.get("data"))
    return train_logistic_regression(X_train, y_train, cfg, save=True)


def main():
    cfg = load_config("logistic_regression")
    X_test, y_test = load_test_data(cfg.get("data"))

    model = get_logistic_regression_model(cfg)
    y_pred = model(X_test)

    exp_cfg = cfg["experiments"]["overall"]
    evaluate_overall_metrics(
        y_true=y_test,
        y_pred=y_pred,
        title=exp_cfg["title"],
        save_path=exp_cfg["output_path"],
    )


if __name__ == "__main__":
    main()

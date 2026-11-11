import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.xgboost_model import XGBoostModel
from train.xgboost_train import train_xgboost
from utils import (
    evaluate_subpop_metrics,
    load_config,
    load_joblib,
    load_label_mapping,
    load_test_data,
    load_train_data,
    resolve_path,
)


def get_xgboost_model(cfg: dict = None) -> XGBoostModel:
    cfg = cfg or load_config("xgboost")
    model_path = resolve_path(cfg["train"]["save_path"])
    if model_path.exists():
        model = XGBoostModel(**cfg.get("model", {}))
        model.model = load_joblib(model_path)
        return model

    X_train, y_train = load_train_data(cfg.get("data"))
    return train_xgboost(X_train, y_train, cfg, save=True)


def main():
    cfg = load_config("xgboost")
    X_test, y_test = load_test_data(cfg.get("data"))
    labels, pop_names = load_label_mapping(cfg.get("data"))

    model = get_xgboost_model(cfg)
    y_pred = model(X_test)

    subpop_cfg = cfg["experiments"]["subpop"]
    evaluate_subpop_metrics(
        y_true=y_test,
        y_pred=y_pred,
        labels=labels,
        pop_names=pop_names,
        model_display_name=subpop_cfg["display_name"],
        hist_path=subpop_cfg["hist_path"],
        cm_path=subpop_cfg["cm_path"],
        hist_title=subpop_cfg.get("hist_title"),
        cm_title=subpop_cfg.get("cm_title"),
    )


if __name__ == "__main__":
    main()

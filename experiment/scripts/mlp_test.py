import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.mlp_model import MLPModel
from train.mlp_train import train_mlp
from utils import (
    evaluate_subpop_metrics,
    load_config,
    load_joblib,
    load_label_mapping,
    load_test_data,
    load_train_data,
    resolve_path,
    save_joblib,
)


def get_mlp_model(cfg: dict = None) -> MLPModel:
    cfg = cfg or load_config("mlp")
    model_path = resolve_path(cfg["train"]["save_path"])
    fallback_path = resolve_path(cfg["train"].get("fallback_stacking_path", ""))

    if model_path.exists():
        model = MLPModel(**cfg.get("model", {}))
        model.model = load_joblib(model_path)
        return model

    if fallback_path.exists():
        stacking_data = load_joblib(fallback_path)
        model = MLPModel(**cfg.get("model", {}))
        model.model = stacking_data["mlp"]
        save_joblib(model.model, model_path)
        return model

    X_train, y_train = load_train_data(cfg.get("data"))
    return train_mlp(X_train, y_train, cfg, save=True)


def main():
    cfg = load_config("mlp")
    X_test, y_test = load_test_data(cfg.get("data"))
    labels, pop_names = load_label_mapping(cfg.get("data"))

    model = get_mlp_model(cfg)
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

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.rf_mlp_stacking import RFMLPStackingModel
from train.rf_mlp_stacking_train import train_rf_mlp_stacking
from utils import (
    evaluate_subpop_metrics,
    load_config,
    load_joblib,
    load_label_mapping,
    load_test_data,
    load_train_data,
    resolve_path,
)


def get_rf_mlp_stacking_model(cfg: dict = None) -> RFMLPStackingModel:
    cfg = cfg or load_config("rf_mlp_stacking")
    model_path = resolve_path(cfg["train"]["save_path"])
    if model_path.exists():
        model_cfg = cfg.get("model", {})
        model = RFMLPStackingModel(
            n_splits=model_cfg.get("n_splits", 5),
            random_state=model_cfg.get("random_state", 42),
            n_jobs=model_cfg.get("n_jobs", -1),
            rf_params=model_cfg.get("rf"),
            mlp_params=model_cfg.get("mlp"),
            meta_lr_params=model_cfg.get("meta_lr"),
        )
        data = load_joblib(model_path)
        model.rf = data["rf"]
        model.mlp = data["mlp"]
        model.meta_lr = data["meta_lr"]
        return model

    X_train, y_train = load_train_data(cfg.get("data"))
    return train_rf_mlp_stacking(X_train, y_train, cfg, save=True)


def main():
    cfg = load_config("rf_mlp_stacking")
    X_test, y_test = load_test_data(cfg.get("data"))
    labels, pop_names = load_label_mapping(cfg.get("data"))

    model = get_rf_mlp_stacking_model(cfg)
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

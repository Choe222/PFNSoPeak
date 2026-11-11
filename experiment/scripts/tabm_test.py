import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.tabm import TabM
from train.tabm_train import train_tabm
from utils import (
    evaluate_subpop_metrics,
    get_device,
    load_config,
    load_label_mapping,
    load_test_data,
    load_torch_checkpoint,
    load_train_data,
    resolve_path,
)


def load_pretrained_tabm(model_path: Path, device: torch.device) -> TabM:
    ckpt = load_torch_checkpoint(model_path, device=device)
    model = TabM(
        in_features=ckpt["in_features"],
        out_features=ckpt["out_features"],
        n_layers=ckpt["n_layers"],
        dim=ckpt["dim"],
        k=ckpt["k"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_tabm(model: TabM, X, device: torch.device) -> np.ndarray:
    model.eval()
    X_arr = np.asarray(X, dtype=np.float32)
    X_t = torch.from_numpy(X_arr).to(device)
    logits_mean = model(X_t).mean(dim=1)
    preds = torch.argmax(logits_mean, dim=-1)
    return preds.cpu().numpy()


def main():
    cfg = load_config("tabm")
    device = get_device(cfg["train"].get("device", "auto"))

    X_test, y_test = load_test_data(cfg.get("data"))
    labels, pop_names = load_label_mapping(cfg.get("data"))

    model_path = resolve_path(cfg["train"]["save_path"])
    if not model_path.exists():
        X_train, y_train = load_train_data(cfg.get("data"))
        model = train_tabm(X_train, y_train, cfg)
    else:
        model = load_pretrained_tabm(model_path, device)

    y_pred = predict_tabm(model, X_test, device)

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

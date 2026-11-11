import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.tabm import TabM
from utils import (
    get_device,
    load_config,
    load_train_data,
    save_torch_checkpoint,
    set_seed,
)


def train_tabm(X, y, config: dict = None) -> TabM:
    cfg = config or load_config("tabm")
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    random_state = train_cfg.get("random_state", 42)
    set_seed(random_state)
    device = get_device(train_cfg.get("device", "auto"))

    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64).ravel()

    in_features = X_arr.shape[1]
    out_features = int(np.max(y_arr)) + 1

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_arr,
        y_arr,
        test_size=train_cfg.get("val_ratio", 0.15),
        stratify=y_arr,
        random_state=random_state,
    )

    train_dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.get("batch_size", 128),
        shuffle=True,
    )

    X_val_t = torch.from_numpy(X_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    model = TabM(
        in_features=in_features,
        out_features=out_features,
        n_layers=model_cfg.get("n_layers", 2),
        dim=model_cfg.get("dim", 128),
        k=model_cfg.get("k", 32),
        dropout=model_cfg.get("dropout", 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    max_epochs = train_cfg.get("max_epochs", 300)
    patience = train_cfg.get("patience", 16)
    grad_clip = train_cfg.get("grad_clip", 1.0)

    for _ in range(max_epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            logits_k = model(bx)
            B, k, C = logits_k.shape
            y_expanded = by.unsqueeze(1).expand(B, k)
            loss = criterion(logits_k.reshape(B * k, C), y_expanded.reshape(B * k))

            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits_mean = model(X_val_t).mean(dim=1)
            val_loss = criterion(val_logits_mean, y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    save_path = train_cfg.get("save_path")
    if save_path:
        checkpoint = {
            "in_features": model.in_features,
            "out_features": model.out_features,
            "n_layers": model.n_layers,
            "dim": model.dim,
            "k": model.k,
            "dropout": model.dropout,
            "state_dict": model.state_dict(),
        }
        save_torch_checkpoint(checkpoint, save_path)

    return model


def main():
    cfg = load_config("tabm")
    X_train, y_train = load_train_data(cfg.get("data"))
    train_tabm(X_train, y_train, cfg)
    print("TabM training completed successfully.")


if __name__ == "__main__":
    main()

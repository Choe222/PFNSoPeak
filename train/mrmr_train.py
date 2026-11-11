import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from models.mrmr_selector import MRMRFeatureSelector
from utils import load_config, load_train_data, save_joblib


def train_mrmr(X, y, config: dict = None, save: bool = True) -> MRMRFeatureSelector:
    cfg = config or load_config("mrmr")
    model_cfg = cfg.get("model", {})

    selector = MRMRFeatureSelector(
        k=model_cfg.get("k", 192),
        criterion=model_cfg.get("criterion", "MID"),
        eps=float(model_cfg.get("eps", 1e-12)),
    )

    if isinstance(X, pd.DataFrame):
        feature_names = list(X.columns)
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
        feature_names = [f"f_{i}" for i in range(X_arr.shape[1])]

    y_arr = np.asarray(y).ravel()
    n_features = X_arr.shape[1]
    k = min(selector.k, n_features)

    selector.relevance_ = np.array(
        [mutual_info_score(X_arr[:, i], y_arr) for i in range(n_features)]
    )
    selector.redundancy_matrix_ = np.zeros((n_features, n_features), dtype=np.float64)

    selected = []
    remaining = list(range(n_features))
    history = []

    first_idx = int(np.argmax(selector.relevance_))
    selected.append(first_idx)
    remaining.remove(first_idx)

    history.append(
        {
            "step": 1,
            "feature_idx": first_idx,
            "feature_name": feature_names[first_idx],
            "relevance": float(selector.relevance_[first_idx]),
            "redundancy": 0.0,
            "mrmr_score": float(selector.relevance_[first_idx]),
        }
    )

    cum_redundancy = np.zeros(n_features, dtype=np.float64)

    for step in range(2, k + 1):
        last_selected = selected[-1]
        for idx in remaining:
            mi_val = mutual_info_score(X_arr[:, idx], X_arr[:, last_selected])
            selector.redundancy_matrix_[idx, last_selected] = mi_val
            selector.redundancy_matrix_[last_selected, idx] = mi_val
            cum_redundancy[idx] += mi_val

        avg_redundancy = cum_redundancy[remaining] / len(selected)
        rel = selector.relevance_[remaining]

        if selector.criterion == "MIQ":
            scores = rel / (avg_redundancy + selector.eps)
        else:
            scores = rel - avg_redundancy

        best_pos = int(np.argmax(scores))
        best_idx = remaining[best_pos]
        best_score = float(scores[best_pos])
        best_red = float(avg_redundancy[best_pos])

        selected.append(best_idx)
        remaining.remove(best_idx)

        history.append(
            {
                "step": step,
                "feature_idx": best_idx,
                "feature_name": feature_names[best_idx],
                "relevance": float(selector.relevance_[best_idx]),
                "redundancy": best_red,
                "mrmr_score": best_score,
            }
        )

    selector.selected_indices_ = selected
    selector.selected_features_ = [feature_names[i] for i in selected]
    selector.history_ = history

    save_path = cfg.get("train", {}).get("save_path")
    if save and save_path:
        save_joblib(selector, save_path)

    return selector


def main():
    cfg = load_config("mrmr")
    X_train, y_train = load_train_data(cfg.get("data"))
    selector = train_mrmr(X_train, y_train, cfg, save=True)
    print(f"mRMR feature selection (k={selector.k}) completed successfully.")


if __name__ == "__main__":
    main()

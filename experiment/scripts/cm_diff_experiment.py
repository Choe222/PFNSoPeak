import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiment.scripts.mlp_test import get_mlp_model
from experiment.scripts.random_forest_test import get_random_forest_model
from experiment.scripts.rf_mlp_stacking_test import get_rf_mlp_stacking_model
from utils import (
    load_config,
    load_label_mapping,
    load_test_data,
    resolve_path,
)


def get_cm_plot(y_true, y_pred, labels):
    cm_norm_true = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return cm_norm_true.T


def plot_diff_matrix(diff_matrix, pop_names, title, save_path):
    out_path = resolve_path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(pop_names))
    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    im = ax.imshow(diff_matrix, interpolation="nearest", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Confusion Matrix Difference (-1.0: Dark Blue -> 0.0: Neutral -> +1.0: Dark Red)",
        fontsize=10,
    )

    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels(pop_names, rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(pop_names, fontweight="bold")
    ax.set_xlabel("Actual Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    for i in range(len(pop_names)):
        for j in range(len(pop_names)):
            val = diff_matrix[i, j]
            if abs(val) > 0.005 or i == j:
                color = "white" if abs(val) > 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:+.2f}" if abs(val) >= 0.005 else "0.00",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                    fontweight="bold" if i == j else "normal",
                )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main():
    stacking_cfg = load_config("rf_mlp_stacking")
    X_test, y_test = load_test_data(stacking_cfg.get("data"))
    labels, pop_names = load_label_mapping(stacking_cfg.get("data"))

    stacking_model = get_rf_mlp_stacking_model(stacking_cfg)
    mlp_model = get_mlp_model(load_config("mlp"))
    rf_model = get_random_forest_model(load_config("random_forest"))

    y_pred_mlp = mlp_model(X_test)
    y_pred_rf = rf_model(X_test)
    y_pred_stacking = stacking_model(X_test)

    cm_mlp = get_cm_plot(y_test, y_pred_mlp, labels)
    cm_rf = get_cm_plot(y_test, y_pred_rf, labels)
    cm_stacking = get_cm_plot(y_test, y_pred_stacking, labels)

    diff_mlp_stacking = cm_mlp - cm_stacking
    diff_rf_stacking = cm_rf - cm_stacking

    diff_cfg = stacking_cfg["experiments"]["cm_diff"]
    mlp_diff_png = plot_diff_matrix(
        diff_mlp_stacking,
        pop_names,
        "Confusion Matrix Difference: MLP - RF_MLP_Stacking (26 Populations)",
        diff_cfg["mlp_diff_png"],
    )
    rf_diff_png = plot_diff_matrix(
        diff_rf_stacking,
        pop_names,
        "Confusion Matrix Difference: RF - RF_MLP_Stacking (26 Populations)",
        diff_cfg["rf_diff_png"],
    )

    diag_df = pd.DataFrame(
        {
            "Population": pop_names,
            "MLP_Acc": np.diag(cm_mlp),
            "RF_Acc": np.diag(cm_rf),
            "Stacking_Acc": np.diag(cm_stacking),
            "MLP_minus_Stacking": np.diag(diff_mlp_stacking),
            "RF_minus_Stacking": np.diag(diff_rf_stacking),
        }
    )

    print("=== Diagonal (Exact Match) Difference Summary ===")
    print(diag_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved Matrix 1 (MLP - RF_MLP_Stacking): {mlp_diff_png}")
    print(f"Saved Matrix 2 (RF - RF_MLP_Stacking):  {rf_diff_png}")


if __name__ == "__main__":
    main()

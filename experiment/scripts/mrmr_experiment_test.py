import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from train.catboost_train import train_catboost
from train.logistic_regression_train import train_logistic_regression
from train.mrmr_train import train_mrmr
from train.random_forest_train import train_random_forest
from train.rf_mlp_stacking_train import train_rf_mlp_stacking
from train.xgboost_train import train_xgboost
from utils import (
    load_config,
    load_joblib,
    load_test_data,
    load_train_data,
    resolve_path,
)

PREV_RESULTS = [
    (5, "Logistic Regression", 0.2695, 0.1833, 0.1877),
    (5, "Random Forest", 0.2615, 0.1779, 0.1818),
    (5, "XGBoost", 0.2615, 0.1789, 0.1827),
    (5, "CatBoost", 0.2615, 0.1795, 0.1832),
    (5, "RF + MLP Stacking", 0.2655, 0.1743, 0.1779),
    (10, "Logistic Regression", 0.2814, 0.2464, 0.2513),
    (10, "Random Forest", 0.2695, 0.2569, 0.2591),
    (10, "XGBoost", 0.2735, 0.2588, 0.2602),
    (10, "CatBoost", 0.2735, 0.2591, 0.2605),
    (10, "RF + MLP Stacking", 0.2715, 0.2251, 0.2225),
    (15, "Logistic Regression", 0.3134, 0.2853, 0.2913),
    (15, "Random Forest", 0.3174, 0.3124, 0.3111),
    (15, "XGBoost", 0.3134, 0.3076, 0.3054),
    (15, "CatBoost", 0.3034, 0.2966, 0.2979),
    (15, "RF + MLP Stacking", 0.3194, 0.2794, 0.2859),
    (20, "Logistic Regression", 0.3353, 0.3142, 0.3201),
    (20, "Random Forest", 0.3234, 0.3111, 0.3137),
    (20, "XGBoost", 0.3333, 0.3246, 0.3245),
    (20, "CatBoost", 0.3313, 0.3241, 0.3236),
    (20, "RF + MLP Stacking", 0.3333, 0.2820, 0.2905),
    (30, "Logistic Regression", 0.3633, 0.3437, 0.3537),
    (30, "Random Forest", 0.3733, 0.3590, 0.3637),
    (30, "XGBoost", 0.3733, 0.3662, 0.3669),
    (30, "CatBoost", 0.3713, 0.3627, 0.3630),
    (30, "RF + MLP Stacking", 0.3373, 0.3150, 0.3172),
    (50, "Logistic Regression", 0.4152, 0.4025, 0.4099),
    (50, "Random Forest", 0.4251, 0.4133, 0.4168),
    (50, "XGBoost", 0.4251, 0.4227, 0.4232),
    (50, "CatBoost", 0.4232, 0.4122, 0.4176),
    (50, "RF + MLP Stacking", 0.4072, 0.3654, 0.3713),
    (55, "Logistic Regression", 0.4032, 0.3978, 0.4020),
    (55, "Random Forest", 0.4391, 0.4310, 0.4327),
    (55, "XGBoost", 0.4351, 0.4333, 0.4353),
    (55, "CatBoost", 0.4132, 0.4096, 0.4106),
    (55, "RF + MLP Stacking", 0.3733, 0.3596, 0.3609),
    (100, "Logistic Regression", 0.4531, 0.4502, 0.4539),
    (100, "Random Forest", 0.4790, 0.4601, 0.4648),
    (100, "XGBoost", 0.4571, 0.4460, 0.4501),
    (100, "CatBoost", 0.4471, 0.4379, 0.4413),
    (100, "RF + MLP Stacking", 0.4451, 0.4049, 0.4077),
    (192, "Logistic Regression", 0.4232, 0.4205, 0.4235),
    (192, "Random Forest", 0.4950, 0.4729, 0.4817),
    (192, "XGBoost", 0.4890, 0.4796, 0.4840),
    (192, "CatBoost", 0.5010, 0.4861, 0.4932),
    (192, "RF + MLP Stacking", 0.5170, 0.5056, 0.5064),
]

TRAIN_FNS = {
    "Logistic Regression": lambda X, y: train_logistic_regression(X, y, save=False),
    "Random Forest": lambda X, y: train_random_forest(X, y, save=False),
    "XGBoost": lambda X, y: train_xgboost(X, y, save=False),
    "CatBoost": lambda X, y: train_catboost(X, y, save=False),
    "RF + MLP Stacking": lambda X, y: train_rf_mlp_stacking(X, y, save=False),
}


def format_pivot_table(df, metric, all_k, model_names):
    pivot = df.pivot(index="Model", columns="k", values=metric).loc[model_names, all_k]
    header = " | ".join([f"**k={k}**" for k in all_k])
    lines = [
        f"| Model | {header} |",
        "| :--- | " + " | ".join([":---:" for _ in all_k]) + " |",
    ]
    for model_name, row in pivot.iterrows():
        max_val = row.max()
        vals = [f"**{v:.4f}**" if v == max_val else f"{v:.4f}" for v in row]
        lines.append(f"| **{model_name}** | " + " | ".join(vals) + " |")
    return lines


def main():
    cfg = load_config("mrmr")
    X_train, y_train = load_train_data(cfg.get("data"))
    X_test, y_test = load_test_data(cfg.get("data"))

    selector_path = resolve_path(cfg["train"]["save_path"])
    if selector_path.exists():
        selector = load_joblib(selector_path)
    else:
        selector = train_mrmr(X_train, y_train, cfg, save=True)

    exp_cfg = cfg["experiments"]["mrmr_sweep"]
    all_k_list = exp_cfg["all_k_list"]
    new_k_list = exp_cfg["new_k_list"]
    full_k = exp_cfg["full_k"]
    model_names = exp_cfg["model_names"]
    plot_path = resolve_path(exp_cfg["plot_path"])
    md_path = resolve_path(exp_cfg["markdown_path"])

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    records = [
        {"k": k, "Model": m, "Accuracy": acc, "F1_Macro": f1m, "F1_Weighted": f1w}
        for k, m, acc, f1m, f1w in PREV_RESULTS
    ]

    for k in new_k_list:
        print(f"Running experiments for k = {k} features...")
        X_tr_k = selector(X_train, k=k)
        X_te_k = selector(X_test, k=k)

        for model_name in model_names:
            model = TRAIN_FNS[model_name](X_tr_k, y_train)
            y_pred = model(X_te_k)

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")

            records.append(
                {
                    "k": k,
                    "Model": model_name,
                    "Accuracy": acc,
                    "F1_Macro": f1_macro,
                    "F1_Weighted": f1_weighted,
                }
            )
            print(
                f"  {model_name}: Acc={acc:.4f}, F1_Macro={f1_macro:.4f}, F1_Weighted={f1_weighted:.4f}"
            )

    results_df = pd.DataFrame(records).sort_values(["Model", "k"])

    all_k = all_k_list + [full_k]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    for model_name in model_names:
        sub = results_df[results_df["Model"] == model_name].sort_values("k")
        axes[0].plot(sub["k"], sub["Accuracy"], marker="o", linewidth=2, label=model_name)
        axes[1].plot(sub["k"], sub["F1_Macro"], marker="s", linewidth=2, label=model_name)

    for ax, title, ylabel in [
        (axes[0], "Accuracy vs Number of mRMR Features (k)", "Accuracy"),
        (axes[1], "F1 Score (Macro) vs Number of mRMR Features (k)", "F1 Score (Macro)"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Number of Selected Features (k)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_k)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    if md_path.exists():
        old_md = md_path.read_text(encoding="utf-8")
        sec3_idx = old_md.find("## 3.")
        prefix_md = old_md[:sec3_idx].rstrip() if sec3_idx != -1 else old_md.rstrip()
    else:
        prefix_md = "# mRMR Feature Selection Experiment Results (mRMR-MID)"

    md_lines = [
        prefix_md,
        "",
        "## 3. Accuracy Comparison across Number of Selected Features ($k$)",
        "",
        *format_pivot_table(results_df, "Accuracy", all_k, model_names),
        "",
        "## 4. F1 Score (Macro) Comparison across Number of Selected Features ($k$)",
        "",
        *format_pivot_table(results_df, "F1_Macro", all_k, model_names),
        "",
        "## 5. F1 Score (Weighted) Comparison across Number of Selected Features ($k$)",
        "",
        *format_pivot_table(results_df, "F1_Weighted", all_k, model_names),
    ]

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("Completed updating charts and markdown tables.")


if __name__ == "__main__":
    main()

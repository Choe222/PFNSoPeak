import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT_DIR = Path(__file__).resolve().parent


def load_env(env_path: Union[str, Path] = None) -> Dict[str, str]:
    """Load key-value pairs from a .env file into os.environ if not already set."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path or (ROOT_DIR / ".env"), override=False)
    except ImportError:
        path = Path(env_path) if env_path else (ROOT_DIR / ".env")
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = val
    return dict(os.environ)


load_env()


def get_env_path(env_key: str, default_rel_path: Union[str, Path]) -> Path:
    """Resolve a path from an environment variable if set, otherwise use default_rel_path."""
    env_val = os.environ.get(env_key)
    if env_val:
        return resolve_path(env_val)
    return resolve_path(default_rel_path)


def load_config(config_name_or_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML configuration file by name (from configs/) or path."""
    path = Path(config_name_or_path)
    if not path.is_absolute():
        if not path.exists():
            config_dir = get_env_path("CONFIG_DIR", "configs")
            candidate = config_dir / path
            if not candidate.suffix:
                candidate = candidate.with_suffix(".yaml")
            path = candidate
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_or_abs_path: Union[str, Path]) -> Path:
    """Resolve path relative to ROOT_DIR, applying environment directory overrides when applicable."""
    path = Path(rel_or_abs_path)
    if path.is_absolute():
        return path
    rel_posix = path.as_posix()
    if os.environ.get("DATA_DIR") and rel_posix.startswith("data/"):
        base = Path(os.environ["DATA_DIR"])
        base = base if base.is_absolute() else (ROOT_DIR / base)
        return base / rel_posix[len("data/") :]
    if os.environ.get("MODEL_SAVE_DIR") and rel_posix.startswith("experiment/models/saved/"):
        base = Path(os.environ["MODEL_SAVE_DIR"])
        base = base if base.is_absolute() else (ROOT_DIR / base)
        return base / rel_posix[len("experiment/models/saved/") :]
    if os.environ.get("RESULTS_DIR") and rel_posix.startswith("experiment/results/"):
        base = Path(os.environ["RESULTS_DIR"])
        base = base if base.is_absolute() else (ROOT_DIR / base)
        return base / rel_posix[len("experiment/results/") :]
    return ROOT_DIR / path


def set_seed(seed: int = None) -> None:
    """Set random seeds across random, numpy, and torch for reproducibility."""
    if seed is None:
        seed = int(os.environ.get("RANDOM_SEED", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_cfg: str = None) -> torch.device:
    """Resolve PyTorch computation device."""
    device_cfg = device_cfg or os.environ.get("DEVICE", "auto")
    if device_cfg and device_cfg != "auto":
        return torch.device(device_cfg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_data_file(data_cfg: Dict[str, Any], cfg_key: str, default_filename: str) -> Path:
    """Resolve data file path considering DATA_DIR environment override."""
    data_cfg = data_cfg or {}
    if os.environ.get("DATA_DIR") and cfg_key not in data_cfg:
        return get_env_path("DATA_DIR", "data") / default_filename
    rel_path = data_cfg.get(cfg_key, f"data/{default_filename}")
    if os.environ.get("DATA_DIR") and str(rel_path).startswith("data/"):
        return get_env_path("DATA_DIR", "data") / Path(rel_path).name
    return resolve_path(rel_path)


def load_train_data(data_cfg: Dict[str, Any] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load X_train and y_train from configured paths."""
    x_path = _resolve_data_file(data_cfg, "x_train", "X_train.csv")
    y_path = _resolve_data_file(data_cfg, "y_train", "y_train.csv")
    X_train = pd.read_csv(x_path)
    y_train = pd.read_csv(y_path).values.ravel()
    return X_train, y_train


def load_test_data(data_cfg: Dict[str, Any] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """Load X_test and y_test from configured paths."""
    x_path = _resolve_data_file(data_cfg, "x_test", "X_test.csv")
    y_path = _resolve_data_file(data_cfg, "y_test", "y_test.csv")
    X_test = pd.read_csv(x_path)
    y_test = pd.read_csv(y_path).values.ravel()
    return X_test, y_test


def load_label_mapping(data_cfg: Dict[str, Any] = None) -> Tuple[List[int], List[str]]:
    """Load sorted label IDs and corresponding population names."""
    mapping_path = _resolve_data_file(data_cfg, "label_mapping", "label_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        label_mapping = json.load(f)
    inv_map = {int(v): k for k, v in label_mapping.items()}
    labels = sorted(inv_map.keys())
    pop_names = [inv_map[i] for i in labels]
    return labels, pop_names


def save_joblib(obj: Any, path: Union[str, Path]) -> Path:
    """Save object to disk using joblib."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, out_path)
    return out_path


def load_joblib(path: Union[str, Path]) -> Any:
    """Load object from disk using joblib."""
    return joblib.load(resolve_path(path))


def save_torch_checkpoint(checkpoint: Dict[str, Any], path: Union[str, Path]) -> Path:
    """Save PyTorch checkpoint dictionary to disk."""
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    return out_path


def load_torch_checkpoint(path: Union[str, Path], device: torch.device = None) -> Dict[str, Any]:
    """Load PyTorch checkpoint dictionary from disk."""
    device = device or get_device()
    return torch.load(resolve_path(path), map_location=device, weights_only=True)


def evaluate_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Union[str, Path],
) -> Dict[str, float]:
    """Compute overall Accuracy, Macro F1, Weighted F1 and save summary bar chart."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1 Score (Macro)": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1 Score (Weighted)": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    out_path = resolve_path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(
        list(metrics.keys()),
        list(metrics.values()),
        color=["#2563eb", "#16a34a", "#0d9488"],
        width=0.5,
    )
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.02,
            f"{h:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print(f"Results saved to: {out_path}")
    return metrics


def evaluate_subpop_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    pop_names: List[str],
    model_display_name: str,
    hist_path: Union[str, Path],
    cm_path: Union[str, Path],
    hist_title: str = None,
    cm_title: str = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Compute overall & per-population metrics and plot histogram + confusion matrix."""
    overall_acc = float(accuracy_score(y_true, y_pred))
    overall_f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    overall_f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm_norm_true = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    cm_plot = cm_norm_true.T

    per_pop_acc = np.diag(cm_norm_true)
    per_pop_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    hist_out = resolve_path(hist_path)
    cm_out = resolve_path(cm_path)
    hist_out.parent.mkdir(parents=True, exist_ok=True)
    cm_out.parent.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(
        {
            "Population": pop_names,
            "Label_ID": labels,
            "Accuracy": per_pop_acc,
            "F1_Score": per_pop_f1,
        }
    )

    x = np.arange(len(pop_names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(16, 6), dpi=150)
    bars_acc = ax.bar(x - width / 2, per_pop_acc, width, label="Accuracy", color="#2563eb")
    bars_f1 = ax.bar(x + width / 2, per_pop_f1, width, label="F1 Score", color="#16a34a")

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_xlabel("Population")
    ax.set_title(
        hist_title
        or f"{model_display_name} - Accuracy & F1 Score across 26 Populations (Acc: {overall_acc:.4f}, Macro F1: {overall_f1_macro:.4f})"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(pop_names, rotation=45, ha="right", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")

    for bars in (bars_acc, bars_f1):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.015,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )

    plt.tight_layout()
    plt.savefig(hist_out)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 12), dpi=150)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="RdBu_r", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Prediction Ratio (0.0: Dark Blue / Error -> 1.0: Dark Red / Exact)", fontsize=10)

    ax.set_xticks(x)
    ax.set_yticks(x)
    ax.set_xticklabels(pop_names, rotation=45, ha="right", fontweight="bold")
    ax.set_yticklabels(pop_names, fontweight="bold")
    ax.set_xlabel("Actual Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Predicted Label", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        cm_title or f"{model_display_name} - Normalized Confusion Matrix (26 Populations)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )

    for i in range(len(pop_names)):
        for j in range(len(pop_names)):
            val = cm_plot[i, j]
            if val > 0.005 or i == j:
                color = "white" if (val < 0.25 or val > 0.75) else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                    fontweight="bold" if i == j else "normal",
                )

    plt.tight_layout()
    plt.savefig(cm_out)
    plt.close()

    print(f"=== {model_display_name} Overall Metrics ===")
    print(f"Overall Accuracy   : {overall_acc:.4f}")
    print(f"F1 Score (Macro)   : {overall_f1_macro:.4f}")
    print(f"F1 Score (Weighted): {overall_f1_weighted:.4f}\n")
    print(f"=== {model_display_name} Per-Population Evaluation (26 Populations) ===")
    print(metrics_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved Histogram: {hist_out}\nSaved Confusion Matrix: {cm_out}")

    overall = {
        "accuracy": overall_acc,
        "f1_macro": overall_f1_macro,
        "f1_weighted": overall_f1_weighted,
    }
    return overall, metrics_df

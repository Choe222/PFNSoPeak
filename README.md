# Human Population Ancestry Classification on Tabular SNP Data

## 1. Project Overview

This repository provides a machine learning and deep learning benchmarking framework for human subpopulation ancestry classification across 26 populations (grouped into 5 super-populations: AFR, AMR, EAS, EUR, SAS) using Single Nucleotide Polymorphism (SNP) tabular features.

The project implements and compares classical statistical models, gradient boosted decision trees, stacking ensembles, mutual information feature selection (mRMR), and parameter-efficient deep tabular ensembles (`TabM` and `TabM_mini` based on BatchEnsemble adapters), alongside Prior-Data Fitted Networks (`TabPFN`).

## 2. Repository Structure

| Directory / File | Purpose |
| :--- | :--- |
| `configs/` | Stores YAML configuration files specifying model hyperparameters, training settings, data paths, and output locations. |
| `data/` | Holds label mapping metadata (`label_mapping.json`, `super_sup.json`) and local training/testing CSV files (`X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`). Raw datasets are ignored by Git. |
| `models/` | Contains modular model definitions including `TabM`, `TabM_mini`, `RFMLPStackingModel`, `MRMRFeatureSelector`, and wrappers for `CatBoost`, `XGBoost`, `RandomForest`, and `LogisticRegression`. |
| `train/` | Contains training entry-point scripts that fit models on the training dataset and serialize checkpoints to the configured model directory. |
| `experiment/scripts/` | Contains evaluation scripts that load trained checkpoints (or train from scratch if absent), compute overall and per-population metrics, and generate visual plots. |
| `experiment/results/` | Stores generated evaluation figures, including per-population accuracy/F1 histograms, normalized confusion matrices, confusion matrix difference heatmaps, and mRMR feature sweep curves. |
| `utils.py` | Provides centralized path resolution, `.env` configuration loading, random seed management, dataset I/O, and standardized plotting routines. |
| `PFNtoTabM.pdf` | Technical presentation detailing the theoretical transition from Maximum Likelihood Estimation and Prior-Data Fitted Networks (PFNs) to Deep Ensembles and TabM. |

## 3. Installation

### Prerequisites

- **Python:** 3.10 to 3.12 recommended (compatible with 3.10+).
- **Hardware:** CPU is sufficient for classical models, mRMR, and `TabM` / `TabM_mini` training. A CUDA-capable GPU is optional for accelerating PyTorch models (`TabM`, `TabM_mini`) and required (minimum 12 GB VRAM) only if running `TabPFN` locally via `models/tabpfn.ipynb`.

### Environment Setup

1. Clone the repository and navigate to the project root:

```bash
git clone <repository-url>
cd PFNSoPeak
```

2. Create and activate a virtual environment:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install the required dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Configuration

### Environment Variables (`.env`)

Local directory paths, execution device preferences, and API tokens can be customized via a `.env` file in the project root without modifying source code or YAML files.

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Supported variables in `.env`:

- `DATA_DIR`: Directory containing `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`, and `label_mapping.json` (default: `data`).
- `CONFIG_DIR`: Directory containing model YAML configurations (default: `configs`).
- `MODEL_SAVE_DIR`: Directory where trained `.joblib` and `.pt` checkpoints are saved (default: `experiment/models/saved`).
- `RESULTS_DIR`: Directory where evaluation figures and tables are written (default: `experiment/results`).
- `LOG_DIR`: Directory for optional training logs such as CatBoost diagnostic files (default: `logs`).
- `RANDOM_SEED`: Global random seed for NumPy, Python, and PyTorch (default: `42`).
- `DEVICE`: Computation device for PyTorch models: `auto`, `cpu`, or `cuda` (default: `auto`).
- `TABPFN_TOKEN`: Access token for Prior Labs API when using the `TabPFN` client backend in `models/tabpfn.ipynb`.

### Dataset Preparation

Datasets are not tracked in Git. Before running training or evaluation scripts, place the following preprocessed CSV files inside `data/` (or the directory specified by `DATA_DIR`):
y
- `data/X_train.csv`: Training feature matrix (e.g., 192 SNP features encoded as allele counts `0, 1, 2`).
- `data/y_train.csv`: Training target labels (integer class IDs `0` to `25` with header `target`).
- `data/X_test.csv`: Testing feature matrix matching the columns of `X_train.csv`.
- `data/y_test.csv`: Testing target labels (integer class IDs `0` to `25` with header `target`).

The metadata files `data/label_mapping.json` and `data/super_sup.json` are included in the repository.

## 5. Usage

All commands should be executed from the root directory of the repository.

### Training Models

Run any script in `train/` to fit the corresponding model and save its checkpoint to `experiment/models/saved/`:

```bash
python train/logistic_regression_train.py
python train/random_forest_train.py
python train/xgboost_train.py
python train/catboost_train.py
python train/mlp_train.py
python train/rf_mlp_stacking_train.py
python train/tabm_mini_train.py
python train/tabm_train.py
python train/mrmr_train.py
```

### Reproducing Experiments and Evaluations

Run any script in `experiment/scripts/` to evaluate a model on the test set and generate plots in `experiment/results/`. If a saved checkpoint does not exist in `experiment/models/saved/`, the evaluation script will automatically train the model first:

```bash
# Overall accuracy and F1 bar charts
python experiment/scripts/logistic_regression_test.py
python experiment/scripts/catboost_test.py

# Per-population histograms and normalized confusion matrices (26 populations)
python experiment/scripts/random_forest_test.py
python experiment/scripts/xgboost_test.py
python experiment/scripts/mlp_test.py
python experiment/scripts/rf_mlp_stacking_test.py
python experiment/scripts/tabm_mini_test.py
python experiment/scripts/tabm_test.py

# Confusion matrix difference analysis (MLP vs Stacking, RF vs Stacking)
python experiment/scripts/cm_diff_experiment.py

# mRMR feature selection sweep across k features
python experiment/scripts/mrmr_experiment_test.py
```

## 6. Results

Experimental outputs are organized under `experiment/results/`:

- **Overall Model Evaluations:**
  - `experiment/results/catboost_results.png`
  - `experiment/results/logistic_regression_results.png`
  - `experiment/results/model_comparison_result.png`
- **mRMR Feature Selection Sweep:**
  - `experiment/results/mrmr/mrmr_results.png`: Accuracy and Macro F1 progression curves across feature subset sizes $k \in \{5, 10, 15, 20, 30, 50, 55, 100, 120, 140, 150, 160, 175, 192\}$.
- **Subpopulation Histograms (`experiment/results/subpop/histogram/`):**
  - Per-population Accuracy and F1 Score bar charts across all 26 populations for `MLP`, `Random Forest`, `XGBoost`, `RF + MLP Stacking`, `TabM_mini`, and `TabM` (`dim` 128, 256, 512).
- **Subpopulation Confusion Matrices (`experiment/results/subpop/confusion_matrix/`):**
  - Normalized 26x26 confusion matrices for each model, including `TabPFN` (`tabpfn_confusion_matrix.png`) and pairwise differential confusion matrices (`mlp_minus_rf_mlp_stacking_confusion_matrix.png`, `rf_minus_rf_mlp_stacking_confusion_matrix.png`).

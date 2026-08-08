# AMR Prediction from EHR Data (Stanford-ARMD)

<p align="center">
  <b>Predicting antimicrobial resistance from longitudinal EHR data</b>
</p>

Code accompanying: **"Predicting antimicrobial resistance using electronic health
record data: A machine learning analysis of the Stanford-ARMD dataset."**

This repository contains the full pipeline used to train and evaluate
logistic regression, optimized XGBoost, and TabTransformer deep-learning
models for predicting antimicrobial resistance (AMR) from longitudinal EHR
data. Each modeling track is kept in its own folder, all tracks share a
single preprocessing step, and all settings live in one configuration file.

## What is in this repository

```
AMR_EHR_StanfordARMD_ML/
├── README.md
├── LICENSE
├── requirements.txt          # one shared Python environment
├── .gitignore
├── config.yaml               # one main config file (paths, seed, model settings)
├── config_loader.py          # helper used by all scripts to read config.yaml
├── dataset/
│   └── README.md             # data must be copied here by the user
├── preprocessing/
│   └── build_dl_features.py  # builds the shared amr_analysis_bundle.joblib
├── logistic_regression/
│   └── lr_dl_matched.py      # logistic regression benchmark
├── xgboost/
│   ├── xgb_config.py         # shared XGBoost configuration
│   ├── xgb_common.py         # shared helpers
│   ├── run_all.py            # run the canonical XGBoost scripts
│   ├── check_sr_ratios.py
│   ├── 01_train_xgb_variations.py
│   ├── 02_analyze_best_xgb.py
│   ├── 03_compare_xgb_models.py
│   └── imbalance_handling/   # seven strategies from manuscript Table 1
│       ├── 00_preprocess.py
│       ├── 01_baseline_xgb.py
│       ├── 02_class_weights.py
│       ├── 03_smote.py
│       ├── 04_undersampling.py
│       ├── 05_threshold_optimization.py
│       ├── 06_kfold_cv.py
│       ├── 07_bayesian_optuna.py
│       ├── common.py
│       ├── utils.py
│       ├── run_all.py
│       └── visualization/
├── deep_learning/
│   ├── train_tabtransformer.py
│   ├── analyze_tabtransformer.py
│   └── ablation/
│       ├── tabtransformer_ablation.py
│       ├── bootstrap_ci.py
│       └── final_loss_plot.py
└── models/
    └── tabtransformer/
        └── .gitkeep          # trained model is written here
```

## Setup

Create one virtual environment and install the single requirements file:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

GPU (CUDA) is required only for the TabTransformer training step. The XGBoost
and logistic-regression steps run on CPU.

## Prerequisites

- Python 3.10 or later
- Enough free disk space for the raw CSV files and the generated bundle
- A CUDA GPU is required only for the TabTransformer training and analysis
  scripts; XGBoost and logistic regression will run on CPU
- Docker is **not** used in this pipeline

## Data

The ARMD-Stanford dataset is **not** included in this repository. The user who
runs the pipeline must copy the raw CSV files into the `dataset/` folder. See
`dataset/README.md` for the exact file list. Raw data and generated artifacts
are Git-ignored.

After running `preprocessing/build_dl_features.py`, the same `dataset/` folder
will also contain `amr_analysis_bundle.joblib`. This bundle is the shared
preprocessed input for all three modeling tracks (logistic regression,
XGBoost, and TabTransformer), so every model trains and evaluates on identical
train / validation / test splits.

## Running the pipeline

All commands below assume your virtual environment is activated and your
working directory is the repository root.

### 1. Preprocessing (run once)

```bash
python preprocessing/build_dl_features.py
```

This produces `dataset/amr_analysis_bundle.joblib`.

### 2. Modeling tracks (run in any order after preprocessing)

Logistic regression:

```bash
python logistic_regression/lr_dl_matched.py
```

Canonical XGBoost experiments:

```bash
python xgboost/01_train_xgb_variations.py
python xgboost/02_analyze_best_xgb.py
python xgboost/03_compare_xgb_models.py
# or run all three at once
python xgboost/run_all.py
```

XGBoost imbalance-handling experiments (manuscript Table 1):

```bash
python xgboost/imbalance_handling/run_all.py
```

TabTransformer deep learning:

```bash
python deep_learning/train_tabtransformer.py
python deep_learning/analyze_tabtransformer.py
```

TabTransformer ablation analysis:

```bash
python deep_learning/ablation/tabtransformer_ablation.py
python deep_learning/ablation/bootstrap_ci.py
python deep_learning/ablation/final_loss_plot.py
```

## Configuration

All adjustable parameters live in `config.yaml`: the random seed,
train/validation/test split sizes, XGBoost experiment settings, and
TabTransformer training settings. Each script reads from this single config
file via `config_loader.py`.

Key sections in `config.yaml`:

- `seed` - global random seed
- `paths` - shared input/output directories
- `split` - train/validation/test split sizes
- `xgboost_imbalance` - seven-strategy imbalance-handling experiments
- `xgboost` - DL-feature-matched XGBoost workflow
- `tabtransformer` - TabTransformer training and analysis settings

## Reproducibility

- A single fixed random seed is used throughout (`config.yaml: seed`).
- The same `dataset/amr_analysis_bundle.joblib` is loaded by every modeling
track, ensuring identical splits and preprocessing.
- Feature engineering restricts predictors to information available before the
relevant culture to prevent information leakage.
- Path strings are read from `config.yaml` rather than hardcoded, so moving
the repository or renaming the data folder only requires one edit.

## Citation

If you use this code, please cite the manuscript (citation to be added on
publication or preprint release).

## License

See `LICENSE`.

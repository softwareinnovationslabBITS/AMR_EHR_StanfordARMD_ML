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
│   └── logistic_regression_dl_matched.py  # logistic regression benchmark
├── xgboost/
│   ├── xgb_config.py                      # shared XGBoost configuration
│   ├── xgb_common.py                      # shared helpers
│   ├── run_xgb_pipeline.py                # runs the DL-matched XGBoost scripts
│   ├── check_sr_ratios.py
│   ├── train_xgb_variations.py
│   ├── analyze_best_xgb.py
│   └── compare_xgb_models.py
├── deep_learning/
│   ├── train_tabtransformer.py
│   ├── analyze_tabtransformer.py
│   └── ablation/
│       ├── tabtransformer_ablation.py
│       ├── bootstrap_ci.py
│       └── tabtransformer_loss_evaluation.py
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

## Data

The ARMD-Stanford dataset is **not** included. Copy the raw CSV files into the
`dataset/` folder; see `dataset/README.md` for the required files. Raw data and
generated artifacts are Git-ignored.

Running `preprocessing/build_dl_features.py` produces
`dataset/amr_analysis_bundle.joblib`, the shared preprocessed input for all
modeling tracks.

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
python logistic_regression/logistic_regression_dl_matched.py
```

XGBoost DL-matched experiments:

```bash
python xgboost/run_xgb_pipeline.py
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
python deep_learning/ablation/tabtransformer_loss_evaluation.py
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
- `xgboost` - DL-feature-matched XGBoost workflow
- `tabtransformer` - TabTransformer training and analysis settings

## Feature Representation

The preprocessing step compiles raw electronic health record (EHR) data into three distinct feature sets for model ingestion:

- **Categorical Features (8):** Includes patient age and gender (encoded), organism, target antibiotic, culture type, ordering mode, and the year and month of the order.
- **Continuous Features:** Includes raw laboratory test results, vital signs, area deprivation index (ADI) scores/state ranks, nursing home visit counts, history of prior resistant infections, days since the first documented resistance, and comorbidity counts.
- **Binary Features:** Includes hospital ward indicators (IP, OP, ER, ICU), specific comorbidity flags, prior exposures to specific antibiotic classes or subtypes, prior infecting organism types, and prior surgical/medical procedures.

## Model Architectures

### 1. TabTransformer Deep Learning
Designed for heterogeneous tabular data, this neural network processes categorical and numerical inputs in separate streams before fusing them:
- **Categorical Stream (Transformer):** 8 categorical inputs are converted into low-dimensional embeddings, projected to a uniform size (64), and processed by a 4-layer multi-head self-attention encoder (8 heads, 256 feedforward units, GELU activation, dropout).
- **Numerical/Binary Stream:** Continuous features are batch-normalized and passed to a projection MLP (128 units, GELU). Binary features map through a linear layer (64 units, GELU). A wide connection bypasses these layers to project concatenated inputs directly (64 units).
- **Fusion Classifier:** Categorical tokens are flattened, concatenated with numerical and wide embeddings, and mapped to logits via a final feed-forward network (hidden layers: 512, 256, and 128 units, with batch normalization, GELU, and dropout).

### 2. XGBoost Baseline & Variations
Gradient-boosted decision trees trained using the `hist` tree method. Evaluated variations include class-weight adjustment, SMOTE-NC synthetic sampling, random undersampling, classification threshold optimization (using MCC), 5-fold cross-validation, and Optuna hyperparameter optimization.

### 3. Logistic Regression
A standard L2-regularized logistic regression benchmark trained on the same standardized continuous features, one-hot encoded categorical variables, and binary flags.

## Preprocessing Bundle Structure

The `dataset/amr_analysis_bundle.joblib` file generated by `build_dl_features.py` contains a dictionary containing:
- **`X_train`, `X_val`, `X_test`**: Numpy float32 feature matrices.
- **`y_train`, `y_val`, `y_test`**: Numpy float32 label arrays (1 for resistant, 0 for susceptible).
- **`keys_train`, `keys_val`, `keys_test`**: Order tracking keys (`order_proc_id_coded`) aligning row-by-row with feature matrices.
- **`CAT_FEATURES`, `CONT_FEATURES`, `BINARY_FEATURES`, `ALL_FEATURES`**: List of column/feature names.
- **`cat_idx`, `cont_idx`, `bin_idx`**: Index lists mapping columns in the matrices to their corresponding data types.
- **`cat_cardinalities`, `cat_embed_dims`**: Feature cardinalities and computed target embedding dimensions for TabTransformer layers.
- **`scaler`, `org_le`, `ab_le`**: Scalers and label encoders fit on the training data.

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

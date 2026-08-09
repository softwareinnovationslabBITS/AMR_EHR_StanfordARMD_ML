#!/usr/bin/env bash
# Source: created in repo, no external source
# run_full_pipeline.sh
# --------------------
# Runs the entire AMR EHR Stanford-ARMD pipeline end-to-end.
# Designed for a remote Linux server: uses nohup so the workflow keeps
# running if the SSH / VS Code remote session disconnects.
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   nohup ./run_full_pipeline.sh > pipeline.log 2>&1 &
#
# Then you can safely close your terminal. To check progress later:
#   tail -f pipeline.log
#
# To stop the pipeline:
#   kill $(cat run_full_pipeline.pid)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "============================================================"
echo "AMR EHR Stanford-ARMD full pipeline"
echo "Started at: $(date)"
echo "Working directory: $REPO_ROOT"
echo "============================================================"

# Activate virtual environment
if [[ -f .venv/bin/activate ]]; then
    echo "[SETUP] Activating virtual environment..."
    source .venv/bin/activate
else
    echo "[ERROR] Virtual environment not found. Run: python3 -m venv .venv"
    exit 1
fi

# Save PID so the process can be killed later
echo $$ > run_full_pipeline.pid

# Helper to run each step and log clearly
run_step() {
    local step_name="$1"
    local script_path="$2"

    echo ""
    echo "============================================================"
    echo "STEP: $step_name"
    echo "Running: python $script_path"
    echo "Started at: $(date)"
    echo "============================================================"

    if python "$script_path"; then
        echo "[OK] $step_name completed at $(date)"
    else
        echo "[ERROR] $step_name failed at $(date)"
        echo "Check the log above for details."
        exit 1
    fi
}

# 1. Preprocessing (run once)
run_step "Preprocessing" "preprocessing/build_dl_features.py"

# 2. Logistic regression benchmark
run_step "Logistic Regression" "logistic_regression/logistic_regression_dl_matched.py"

# 3. Canonical XGBoost experiments
run_step "XGBoost DL-matched" "xgboost/run_xgb_pipeline.py"

# 5. TabTransformer training
run_step "TabTransformer Training" "deep_learning/train_tabtransformer.py"

# 6. TabTransformer analysis
run_step "TabTransformer Analysis" "deep_learning/analyze_tabtransformer.py"

# 7. TabTransformer ablation
run_step "TabTransformer Ablation" "deep_learning/ablation/tabtransformer_ablation.py"

# 8. Bootstrap CI for baseline TabTransformer
run_step "TabTransformer Bootstrap CI" "deep_learning/ablation/bootstrap_ci.py"

# 9. Final loss evaluation plot
run_step "TabTransformer Loss Evaluation" "deep_learning/ablation/tabtransformer_loss_evaluation.py"

echo ""
echo "============================================================"
echo "ALL PIPELINE STEPS COMPLETED SUCCESSFULLY"
echo "Finished at: $(date)"
echo "============================================================"

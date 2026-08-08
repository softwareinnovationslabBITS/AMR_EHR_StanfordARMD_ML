# Source: /AMR_Stanford/XGB_feature_engg/run_all.py
"""
run_all.py
-----------
Runs every script in order (00 through 07) and prints the final comparison
table at the end. Each script is run as a subprocess so a crash in one
technique doesn't take down the rest — you still get partial results
logged for everything that succeeded.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time

SCRIPTS = [
    "00_common.py",
    "01_baseline_xgb.py",
    "02_class_weights.py",
    "03_smote.py",
    "04_undersampling.py",
    "05_threshold_optimization.py",
    "06_kfold_cv.py",
    "07_bayesian_optuna.py",
]


def run_script(path):
    print("\n" + "#" * 70)
    print(f"#  RUNNING: {path}")
    print("#" * 70)
    t0 = time.time()
    result = subprocess.run([sys.executable, path])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[WARN] {path} exited with code {result.returncode} after {elapsed:.1f}s — continuing.")
    else:
        print(f"[OK] {path} finished in {elapsed:.1f}s")


def main():
    overall_start = time.time()
    for script in SCRIPTS:
        run_script(script)

    total_elapsed = time.time() - overall_start
    print("\n" + "#" * 70)
    print(f"#  ALL SCRIPTS FINISHED — total time: {total_elapsed/60:.1f} min")
    print("#" * 70)

    from utils import build_comparison_table
    df = build_comparison_table()
    if not df.empty:
        print("\nFINAL COMPARISON TABLE (sorted by MCC, descending):\n")
        print(df.to_string(index=False))
        df.to_csv("./results/comparison_table.csv", index=False)
        print("\n[LOG] Saved -> ./results/comparison_table.csv")
    else:
        print("[WARN] No results were logged — check individual script output above.")


if __name__ == "__main__":
    main()

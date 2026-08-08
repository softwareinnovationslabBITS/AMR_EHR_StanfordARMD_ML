# Source: /AMR_Stanford/py_codes/xg_classw/run_all_visualizations.py
"""
run_all_visualizations.py
----------------------------
Runs gen_viz_01.py through gen_viz_07.py in sequence, generating the full
visualization suite for every trained technique. Each script is a separate
subprocess so a failure on one technique (e.g. a model that wasn't trained
yet) doesn't block the rest.

Run this AFTER you've trained the models you care about (01-07 in the
amr_imbalance suite) — each gen_viz_XX.py loads its model from
./saved_models/<method>/ rather than retraining anything.

Usage:
    python run_all_visualizations.py
"""

import subprocess
import sys
import time

SCRIPTS = [
    "gen_viz_01.py",
    "gen_viz_02.py",
    "gen_viz_03.py",
    "gen_viz_04.py",
    "gen_viz_05.py",
    "gen_viz_06.py",
    "gen_viz_07.py",
]


def run_script(path):
    print("\n" + "#" * 70)
    print(f"#  RUNNING: {path}")
    print("#" * 70)
    t0 = time.time()
    result = subprocess.run([sys.executable, path])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[WARN] {path} exited with code {result.returncode} after {elapsed:.1f}s "
              f"— likely no saved model found for this technique yet. Continuing.")
    else:
        print(f"[OK] {path} finished in {elapsed:.1f}s")


def main():
    overall_start = time.time()
    for script in SCRIPTS:
        run_script(script)
    total_elapsed = time.time() - overall_start
    print("\n" + "#" * 70)
    print(f"#  ALL VISUALIZATION SCRIPTS FINISHED — total time: {total_elapsed/60:.1f} min")
    print(f"#  Check ./visualizations/<method_name>/ for each technique's plots.")
    print("#" * 70)


if __name__ == "__main__":
    main()

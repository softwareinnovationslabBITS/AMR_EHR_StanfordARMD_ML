# Source: /AMR_Stanford/DL_codes/amr_project/xgb_dl_feature_matched_project/run_all_xgb_dl_matched.py
"""Run training, comparison and best-model analysis in sequence."""
import subprocess,sys,time
from pathlib import Path
# #migrate: resolve scripts relative to this file so the runner works from any cwd
_HERE = Path(__file__).resolve().parent
SCRIPTS=[
    _HERE/'01_train_xgb_variations.py',
    _HERE/'03_compare_xgb_models.py',
    _HERE/'02_analyze_best_xgb.py',
]
def main():
    start=time.time()
    for script in SCRIPTS:
        print('\n'+'#'*80+f'\nRUNNING {script.name}\n'+'#'*80)
        result=subprocess.run([sys.executable,'-u',str(script)],cwd=str(_HERE))
        if result.returncode!=0: raise SystemExit(result.returncode)
    print(f'Complete workflow finished in {(time.time()-start)/3600:.2f} hours')
if __name__=='__main__': main()

"""Run training, comparison and best-model analysis in sequence."""
import subprocess,sys,time
SCRIPTS=['01_train_xgb_variations.py','03_compare_xgb_models.py','02_analyze_best_xgb.py']
def main():
    start=time.time()
    for script in SCRIPTS:
        print('\n'+'#'*80+f'\nRUNNING {script}\n'+'#'*80)
        result=subprocess.run([sys.executable,'-u',script])
        if result.returncode!=0: raise SystemExit(result.returncode)
    print(f'Complete workflow finished in {(time.time()-start)/3600:.2f} hours')
if __name__=='__main__': main()

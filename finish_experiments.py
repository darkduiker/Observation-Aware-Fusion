"""Evaluate only completed experiments, rejecting partially written exports."""
from pathlib import Path
import subprocess,sys,json
import pandas as pd
r=Path(__file__).resolve().parents[1]
assert len(list((r/'runs').glob('seed_*/fold_*/complete.json')))==15,'Primary training still running'
assert len(list((r/'baseline_runs').glob('seed_*_fold_*/selection.json')))==15,'Baseline training still running'
# Regenerate aggregate files from complete fold outputs.
sys.path.insert(0,str(r/'code'))
from run_nested import aggregate,MODELS
from run_baselines import aggregate as aggregate_baselines
aggregate([pd.read_csv(f) for f in sorted((r/'runs').glob('seed_*/fold_*/predictions.csv'))],r/'runs')
aggregate_baselines([pd.read_csv(f) for f in sorted((r/'baseline_runs').glob('seed_*_fold_*/predictions.csv'))],r/'baseline_runs')
s=pd.concat([pd.read_csv(f) for f in sorted((r/'runs').glob('seed_*/fold_*/stress.csv'))],ignore_index=True)
external=s.loc[s.cohort.eq('fengxian')].groupby(['seed','dropout','mask','patient_id','stage_id','target','cohort'],as_index=False)[MODELS].mean()
pd.concat([s.loc[s.cohort.eq('xuhui_oof')],external],ignore_index=True).to_csv(r/'runs/stress_predictions.csv',index=False)
for script in ['evaluate.py','six_model_results.py','extra_analysis.py']:
 subprocess.run([sys.executable,str(r/'code'/script)],cwd=r,check=True)
(r/'results/RESULTS_COMPLETE.json').write_text(json.dumps({'seeds':[2026,2027,2028],'primary_folds':15,'baseline_folds':15,'queries':1596,'bootstrap_primary':10000,'bootstrap_sensitivity':1000},indent=2))

from pathlib import Path
import sys,numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from evaluate import scores
from fusion import patient_weights
r=ROOT/'results';pfile=r/'six_model_predictions.csv'
if not pfile.exists():pfile=ROOT/'private_predictions/six_model_predictions.csv'
d=pd.read_csv(pfile);mods=['LR_current','RF_current','LightGBM_current','GRUD','Raindrop_portable','OAF_shrink'];rows=[]
for site,g in d.groupby('cohort'):
 w=patient_weights(g.patient_id);y=g.target.to_numpy();p=g[mods].to_numpy();u,iv=np.unique(g.patient_id,return_inverse=True);rng=np.random.default_rng(20261001+(0 if site=='fengxian' else 1000));boot=[];point=np.array([scores(y,p[:,j],w) for j in range(6)])
 for _ in range(5000):
  ww=w*rng.multinomial(len(u),np.full(len(u),1/len(u)))[iv]
  if len(np.unique(y[ww>0]))==2:boot.append([scores(y,p[:,j],ww) for j in range(6)])
 boot=np.array(boot)
 for j,m in enumerate(mods):
  for k,metric in enumerate(['AUROC','AUPRC','Brier']):
   lo,hi=np.quantile(boot[:,j,k],[.025,.975]);rows.append(dict(cohort=site,model=m,metric=metric,estimate=point[j,k],ci_low=lo,ci_high=hi,bootstrap=5000))
 print('DONE',site,flush=True)
pd.DataFrame(rows).to_csv(r/'benchmark_5000.csv',index=False)

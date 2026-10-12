"""Matched fresh six-model benchmark; never import legacy table numbers."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from evaluate import scores
from fusion import patient_weights
MODELS=['LR_current','RF_current','LightGBM_current','GRUD','Raindrop_portable','OAF_shrink']
LABELS=['LR current','RF current','LightGBM current','GRU-D','Raindrop portable','OAF with shrinkage']
def main(n):
    root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
    keys=['patient_id','stage_id','target','cohort']
    a=pd.read_csv(root/'runs/ensemble_predictions.csv');b=pd.read_csv(root/'baseline_runs/ensemble_predictions.csv')
    d=a.merge(b,on=keys,validate='one_to_one');assert len(d)==len(a)==len(b)==1596
    d[keys+MODELS].to_csv(out/'six_model_predictions.csv',index=False)
    rows=[];paired=[];fig,ax=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for si,(site,g) in enumerate(d.groupby('cohort',sort=False)):
        ids=g.patient_id.to_numpy();unique,inv=np.unique(ids,return_inverse=True);w=patient_weights(ids);y=g.target.to_numpy()
        p=g[MODELS].to_numpy();point=np.array([scores(y,p[:,j],w) for j in range(6)])
        rng=np.random.default_rng(20261001);boot=[]
        for _ in range(n):
            bw=w*rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))[inv]
            if len(np.unique(y[bw>0]))==2:boot.append([scores(y,p[:,j],bw) for j in range(6)])
        boot=np.asarray(boot)
        for j,model in enumerate(MODELS):
            for k,metric in enumerate(['AUROC','AUPRC','Brier']):
                lo,hi=np.quantile(boot[:,j,k],[.025,.975]);rows.append(dict(cohort=site,model=model,metric=metric,estimate=point[j,k],ci_low=lo,ci_high=hi))
            fpr,tpr,_=roc_curve(y,p[:,j],sample_weight=w);pr,re,_=precision_recall_curve(y,p[:,j],sample_weight=w)
            ax[si,0].plot(fpr,tpr,label=f'{LABELS[j]} ({point[j,0]:.3f})');ax[si,1].plot(re,pr,label=f'{LABELS[j]} ({point[j,1]:.3f})')
        for j in range(5):
            for k,metric in enumerate(['AUROC','AUPRC','Brier']):
                delta=boot[:,5,k]-boot[:,j,k];lo,hi=np.quantile(delta,[.025,.975]);paired.append(dict(cohort=site,reference=MODELS[j],metric=metric,estimate=point[5,k]-point[j,k],ci_low=lo,ci_high=hi))
        title='Xuhui OOF' if site=='xuhui_oof' else 'Fengxian exploratory'
        for k in range(2):ax[si,k].set_title(title);ax[si,k].legend(fontsize=7);ax[si,k].grid(alpha=.2)
        ax[si,0].set(xlabel='False positive rate',ylabel='True positive rate');ax[si,1].set(xlabel='Recall',ylabel='Precision')
    pd.DataFrame(rows).to_csv(out/'six_model_metrics.csv',index=False);pd.DataFrame(paired).to_csv(out/'OAF_vs_five_models.csv',index=False)
    fig.savefig(out/'Figure4_six_model_ROC_PR.png',dpi=300);fig.savefig(out/'Figure4_six_model_ROC_PR.pdf');plt.close(fig)
    print(pd.DataFrame(rows).pivot(index=['cohort','model'],columns='metric',values='estimate').to_string(),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=int,default=10000);main(ap.parse_args().bootstrap)

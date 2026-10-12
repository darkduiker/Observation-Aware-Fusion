from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve
from fusion import patient_weights
from six_model_results import MODELS,LABELS
from evaluate import scores
r=Path(__file__).resolve().parents[1];d=pd.read_csv(r/'results/six_model_predictions.csv')
plt.rcParams.update({'font.size':10})
fig,axes=plt.subplots(2,2,figsize=(7.6,8.4),layout='constrained')
for si,site in enumerate(['xuhui_oof','fengxian']):
 g=d.loc[d.cohort.eq(site)];w=patient_weights(g.patient_id);y=g.target
 for m,label in zip(MODELS,LABELS):
  score=scores(y,g[m],w);f,t,_=roc_curve(y,g[m],sample_weight=w);p,re,_=precision_recall_curve(y,g[m],sample_weight=w)
  axes[si,0].plot(f,t,label=f'{label} ({score[0]:.3f})');axes[si,1].plot(re,p,label=f'{label} ({score[1]:.3f})')
 for k,ax in enumerate(axes[si]):
  ax.set_title('Xuhui OOF' if si==0 else 'Fengxian exploratory');ax.legend(fontsize=7.6);ax.grid(alpha=.2);ax.set_xlim(0,1);ax.set_ylim(0,1)
 axes[si,0].set(xlabel='False positive rate',ylabel='True positive rate');axes[si,1].set(xlabel='Recall',ylabel='Precision')
for ext in ['png','pdf','svg']:fig.savefig(r/'results'/f'Figure4_six_model_ROC_PR.{ext}',dpi=300)

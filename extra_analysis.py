from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,brier_score_loss,balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from fusion import patient_weights
from evaluate import scores
from run_nested import MODELS

def standardized_ap(y,p,w,pi):
 f,t,_=roc_curve(y,p,sample_weight=w,drop_intermediate=False)
 den=pi*t+(1-pi)*f;precision=np.divide(pi*t,den,out=np.ones_like(t),where=den>0)
 return float(np.sum(np.diff(t)*precision[1:]))
def main():
 root=Path(__file__).resolve().parents[1];out=root/'results';d=pd.read_csv(root/'runs/ensemble_predictions.csv')
 assert len(d)==1596
 six=pd.read_csv(out/'six_model_predictions.csv');ba=[]
 for site,g in six.groupby('cohort'):
  for m in ['LR_current','RF_current','LightGBM_current','GRUD','Raindrop_portable','OAF_shrink']:
   ba.append(dict(cohort=site,model=m,BA=balanced_accuracy_score(g.target,g[m].ge(.5),sample_weight=patient_weights(g.patient_id))))
 pd.DataFrame(ba).to_csv(out/'balanced_accuracy.csv',index=False)
 a=d.loc[d.cohort.eq('xuhui_oof')];pi=np.average(a.target,weights=patient_weights(a.patient_id));rows=[];phases=[];cal=[]
 fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
 for ax,(site,g) in zip(axes,d.groupby('cohort',sort=False)):
  y=g.target.to_numpy();ids=g.patient_id.to_numpy();w=patient_weights(ids);u,inv=np.unique(ids,return_inverse=True)
  candidates=['RF_current','GRUD','Fixed_half','Global_weight','OAF_original','OAF_shrink','OAF_reliable_shrink'];point=np.array([standardized_ap(y,g[m],w,pi) for m in candidates]);rng=np.random.default_rng(771);draw=[]
  for _ in range(1000):
   ww=w*rng.multinomial(len(u),np.full(len(u),1/len(u)))[inv]
   if len(np.unique(y[ww>0]))==2:draw.append([standardized_ap(y,g[m],ww,pi) for m in candidates])
  draw=np.array(draw)
  for j,m in enumerate(candidates):
   lo,hi=np.quantile(draw[:,j],[.025,.975]);dl,dh=np.quantile(draw[:,j]-draw[:,0],[.025,.975]);rows.append(dict(cohort=site,model=m,reference_prevalence=pi,standardized_AP=point[j],ci_low=lo,ci_high=hi,difference_vs_RF=point[j]-point[0],difference_ci_low=dl,difference_ci_high=dh))
  for m in ['RF_current','GRUD','Fixed_half','OAF_original','OAF_shrink']:
   prob=g[m].to_numpy();logit=np.log(np.clip(prob,1e-6,1-1e-6)/np.clip(1-prob,1e-6,1-1e-6));lr=LogisticRegression(C=1e6,max_iter=1000).fit(logit[:,None],y,sample_weight=w)
   cal.append(dict(cohort=site,model=m,calibration_intercept=lr.intercept_[0],calibration_slope=lr.coef_[0,0],Brier=brier_score_loss(y,prob,sample_weight=w)))
   bins=pd.qcut(prob,5,labels=False,duplicates='drop');xp=[];yp=[]
   for b in np.unique(bins):
    sel=bins==b;xp.append(np.average(prob[sel],weights=w[sel]));yp.append(np.average(y[sel],weights=w[sel]))
   ax.plot(xp,yp,'o-',label=m.replace('_',' '))
  ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlabel='Mean predicted intervention probability',ylabel='Observed intervention proportion',title=site);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.legend(fontsize=8)
  for stage,sub in g.groupby('stage_id'):
   yy=sub.target.to_numpy();iid=sub.patient_id.to_numpy();ww=patient_weights(iid);uni,iv=np.unique(iid,return_inverse=True);points=np.array([scores(yy,sub[m],ww) for m in candidates]);r=np.random.default_rng(771+int(stage));boot=[]
   for _ in range(1000):
    bw=ww*r.multinomial(len(uni),np.full(len(uni),1/len(uni)))[iv]
    if len(np.unique(yy[bw>0]))==2:boot.append([scores(yy,sub[m],bw) for m in candidates])
   boot=np.array(boot)
   for j,m in enumerate(candidates):
    lo,hi=np.quantile(boot[:,j,1]-boot[:,0,1],[.025,.975]);phases.append(dict(cohort=site,stage_id=int(stage),model=m,patients=len(uni),events=int(yy.sum()),AUROC=points[j,0],AUPRC=points[j,1],Brier=points[j,2],AUPRC_difference_vs_RF=points[j,1]-points[0,1],difference_ci_low=lo,difference_ci_high=hi))
 pd.DataFrame(rows).to_csv(out/'standardized_AP.csv',index=False);pd.DataFrame(phases).to_csv(out/'phase_performance.csv',index=False);pd.DataFrame(cal).to_csv(out/'calibration_metrics.csv',index=False)
 fig.savefig(out/'Figure5_calibration.png',dpi=300);fig.savefig(out/'Figure5_calibration.pdf');plt.close(fig)
 print(pd.DataFrame(rows).loc[lambda x:x.model.eq('OAF_shrink')].to_string(index=False),flush=True)
if __name__=='__main__':main()

"""Paired common-roster contrasts, descriptive cross-campus gaps and dropout."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from evaluate import scores
from fusion import patient_weights
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'common_experiment_verified'
NAMES=['A','B','C','D','E','Fixed_half','Global_weight','OAF_original','OAF_same6_no_shrink','OAF_shrink']
CONTRASTS=[('B','A'),('C','A'),('D','C'),('D','A'),('E','C'),('OAF_shrink','A'),('Fixed_half','A'),('Global_weight','A'),('OAF_same6_no_shrink','A'),('OAF_shrink','D'),('OAF_shrink','Fixed_half'),('OAF_shrink','Global_weight'),('OAF_shrink','OAF_same6_no_shrink')]

def standard_ap(y,p,w,pi=209/736):
 f,t,_=roc_curve(y,p,sample_weight=w,drop_intermediate=False);den=pi*t+(1-pi)*f;pr=np.divide(pi*t,den,out=np.ones_like(t),where=den>0)
 return np.sum(np.diff(t)*pr[1:])

def metrics(y,p,w):return np.r_[scores(y,p,w)[:3],standard_ap(y,p,w)]

def main(n=5000):
 keys=['patient_id','stage_id','target','cohort'];new=pd.read_csv(OUT/'predictions_ensemble.csv');old=pd.read_csv(ROOT/'runs/ensemble_predictions.csv');d=new.merge(old[keys+['RF_current','Fixed_half','Global_weight','OAF_original','OAF_shrink']],on=keys,validate='one_to_one');assert len(d)==1596 and np.allclose(d.A,d.RF_current,atol=1e-12)
 d.to_csv(OUT/'common_predictions.csv',index=False);absolute=[];paired=[];G=[];siteboots={};sitepoints={};metricnames=['AUROC','AUPRC','Brier','Standardized_AP']
 for site,g in d.groupby('cohort'):
  y=g.target.to_numpy();p=g[NAMES].to_numpy();w=patient_weights(g.patient_id);u,iv=np.unique(g.patient_id,return_inverse=True);point=np.array([metrics(y,p[:,j],w) for j in range(len(NAMES))]);rng=np.random.default_rng(20261001+(0 if site=='fengxian' else 1000));boot=[]
  for i in range(n):
   ww=w*rng.multinomial(len(u),np.full(len(u),1/len(u)))[iv]
   if len(np.unique(y[ww>0]))==2:boot.append([metrics(y,p[:,j],ww) for j in range(len(NAMES))])
  boot=np.asarray(boot);siteboots[site]=boot;sitepoints[site]=point
  for j,name in enumerate(NAMES):
   for k,m in enumerate(metricnames):
    lo,hi=np.quantile(boot[:,j,k],[.025,.975]);absolute.append(dict(cohort=site,model=name,metric=m,estimate=point[j,k],ci_low=lo,ci_high=hi))
  for plus,minus in CONTRASTS:
   a=NAMES.index(plus);b=NAMES.index(minus)
   for k,m in enumerate(metricnames):
    delta=boot[:,a,k]-boot[:,b,k];lo,hi=np.quantile(delta,[.025,.975]);paired.append(dict(cohort=site,contrast=f'{plus}-{minus}',metric=m,estimate=point[a,k]-point[b,k],ci_low=lo,ci_high=hi))
  print('BOOT DONE',site,flush=True)
 for plus,minus in CONTRASTS:
  a=NAMES.index(plus);b=NAMES.index(minus)
  for k,m in enumerate(metricnames):
   A=sitepoints['xuhui_oof'];B=sitepoints['fengxian'];ba=siteboots['xuhui_oof'];bb=siteboots['fengxian'];n=min(len(ba),len(bb));delta=bb[:n,a,k]-bb[:n,b,k]-ba[:n,a,k]+ba[:n,b,k];lo,hi=np.quantile(delta,[.025,.975]);G.append(dict(contrast=f'{plus}-{minus}',metric=m,siteA_increment=A[a,k]-A[b,k],siteB_increment=B[a,k]-B[b,k],gap=(B[a,k]-B[b,k])-(A[a,k]-A[b,k]),ci_low=lo,ci_high=hi))
 pd.DataFrame(absolute).to_csv(OUT/'absolute.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired.csv',index=False);pd.DataFrame(G).to_csv(OUT/'cross_campus_gaps.csv',index=False)
 # Seed-by-seed contrasts, without selecting a seed.
 sd=pd.read_csv(OUT/'predictions_by_seed.csv');od=pd.read_csv(ROOT/'runs/predictions.csv');sd=sd.merge(od[keys+['seed','Fixed_half','Global_weight','OAF_original','OAF_shrink']],on=keys+['seed'],validate='one_to_one');seedrows=[]
 for (seed,site),g in sd.groupby(['seed','cohort']):
  w=patient_weights(g.patient_id)
  for name in NAMES:seedrows.append(dict(seed=seed,cohort=site,model=name,**dict(zip(metricnames,metrics(g.target.to_numpy(),g[name].to_numpy(),w)))))
 pd.DataFrame(seedrows).to_csv(OUT/'seed_metrics.csv',index=False)
 # Mask-level scores: mask ranges are not patient confidence intervals.
 sk=keys+['dropout','mask'];new=pd.read_csv(OUT/'stress_ensemble.csv');old=pd.read_csv(ROOT/'runs/stress_predictions.csv').groupby(sk,as_index=False)[['RF_current','Fixed_half','Global_weight','OAF_original','OAF_shrink']].mean();st=new.merge(old,on=sk,validate='one_to_one');assert np.allclose(st.A,st.RF_current)
 stress=[]
 for (site,rate,mask),g in st.groupby(['cohort','dropout','mask']):
  w=patient_weights(g.patient_id)
  for name in NAMES:stress.append(dict(cohort=site,dropout=rate,mask=mask,model=name,**dict(zip(metricnames,metrics(g.target.to_numpy(),g[name].to_numpy(),w)))))
 stress=pd.DataFrame(stress);stress.to_csv(OUT/'stress_metrics.csv',index=False);st.to_csv(OUT/'stress_predictions.csv',index=False)
 robustness(st,d,n)
 a=pd.DataFrame(absolute);summary=[]
 for (site,rate,name),g in stress.groupby(['cohort','dropout','model']):
  base=a.loc[a.cohort.eq(site)&a.model.eq(name)&a.metric.eq('AUPRC'),'estimate'].iloc[0];diff=g.AUPRC-base;summary.append(dict(cohort=site,dropout=rate,model=name,AUPRC=g.AUPRC.mean(),change_from_unmasked=diff.mean(),mask_change_low=diff.quantile(.025),mask_change_high=diff.quantile(.975),worst_mask_change=diff.min(),negative_masks=int((diff<0).sum())))
 pd.DataFrame(summary).to_csv(OUT/'stress_summary.csv',index=False)
 draw(a,pd.DataFrame(paired),stress)
 (OUT/'EVALUATION_COMPLETE.json').write_text(json.dumps({'bootstrap':n,'conditional_on_fitted_predictions':True,'multiplicity_adjustment':False,'same_current_anchor_asserted':True,'queries':1596},indent=2))
 print(a.pivot(index=['cohort','model'],columns='metric',values='estimate').to_string(),flush=True)

def robustness(st,base,n):
 rows=[];keys=['patient_id','stage_id','target','cohort'];cols=['D','E','Fixed_half','Global_weight','OAF_same6_no_shrink','OAF_shrink']
 merged=st.merge(base[keys+cols],on=keys,suffixes=('_masked','_base'),validate='many_to_one')
 for (site,rate),g in merged.groupby(['cohort','dropout']):
  response={};risk={}
  for m in cols:
   x=g[keys].copy();x['response']=abs(g[m+'_masked']-g[m+'_base']);x['risk_change']=(g.target-g[m+'_masked'])**2-(g.target-g[m+'_base'])**2
   z=x.groupby(['patient_id','stage_id'])[['response','risk_change']].mean().groupby('patient_id').mean();response[m]=z.response.to_numpy();risk[m]=z.risk_change.to_numpy()
  for m in cols[:-1]:
   rng=np.random.default_rng(902);v=response[m]-response['OAF_shrink'];r=risk[m]-risk['OAF_shrink'];indices=rng.integers(0,len(v),(n,len(v)));lo,hi=np.quantile(v[indices].mean(axis=1),[.025,.975]);rl,rh=np.quantile(r[indices].mean(axis=1),[.025,.975])
   rows.append(dict(cohort=site,dropout=rate,control=m,control_probability_response=response[m].mean(),OAF_probability_response=response['OAF_shrink'].mean(),response_attenuation=v.mean(),ci_low=lo,ci_high=hi,control_brier_change=risk[m].mean(),OAF_brier_change=risk['OAF_shrink'].mean(),brier_change_attenuation=r.mean(),brier_ci_low=rl,brier_ci_high=rh))
 pd.DataFrame(rows).to_csv(OUT/'paired_probability_response.csv',index=False)


def draw(a,paired,stress):
 plt.rcParams.update({'font.size':10});fig,axes=plt.subplots(1,2,figsize=(8,4.5),layout='constrained')
 chosen=['B-A','C-A','D-A','E-C','Fixed_half-A','Global_weight-A','OAF_same6_no_shrink-A','OAF_shrink-A']
 for ax,site in zip(axes,['xuhui_oof','fengxian']):
  g=paired.loc[paired.cohort.eq(site)&paired.metric.eq('AUPRC')].set_index('contrast').loc[chosen];v=g.estimate.to_numpy();yy=np.arange(len(g));ax.errorbar(v,yy,xerr=np.vstack([v-g.ci_low,g.ci_high-v]),fmt='o',capsize=3);ax.set_yticks(yy,[{'B-A':'R minus RF','C-A':'Change minus RF','D-A':'Change + R minus RF','E-C':'OPR minus change','Fixed_half-A':'Fixed blend minus RF','Global_weight-A':'Global blend minus RF','OAF_same6_no_shrink-A':'Unshrunk OAF minus RF','OAF_shrink-A':'OAF minus RF'}[c] for c in chosen]);ax.invert_yaxis();ax.axvline(0,color='gray');ax.set(title=site,xlabel='Paired AUPRC difference');ax.grid(axis='x',alpha=.2)
  if site=='fengxian':ax.set_yticklabels([])
 for ext in ['png','pdf']:fig.savefig(OUT/f'Figure_common_paired.{ext}',dpi=300)
 plt.close(fig);fig,axes=plt.subplots(1,2,figsize=(8,4.5),layout='constrained')
 for ax,site in zip(axes,['xuhui_oof','fengxian']):
  for name in ['A','C','D','E','Fixed_half','Global_weight','OAF_same6_no_shrink','OAF_shrink']:
   base=a.loc[a.cohort.eq(site)&a.model.eq(name)&a.metric.eq('AUPRC'),'estimate'].iloc[0];v=[base];rates=[0,.2,.4]
   for rate in rates[1:]:v.append(stress.loc[stress.cohort.eq(site)&stress.model.eq(name)&stress.dropout.eq(rate),'AUPRC'].mean())
   ax.plot(rates,v,'o-',label={'A':'RF current','C':'Change RF','D':'Change + R RF','E':'OPR RF','Fixed_half':'Fixed 50:50','Global_weight':'Global weight','OAF_same6_no_shrink':'OAF no shrinkage','OAF_shrink':'OAF shrinkage'}[name])
  ax.set(title=site,xlabel='Older history removal probability',ylabel='Patient-equal AUPRC')
 axes[1].legend(fontsize=7,loc='center left',bbox_to_anchor=(1.02,.5))
 for ext in ['png','pdf']:fig.savefig(OUT/f'Figure_common_dropout.{ext}',dpi=300)
 plt.close(fig)
if __name__=='__main__':main()

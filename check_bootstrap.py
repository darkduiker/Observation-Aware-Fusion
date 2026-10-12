from pathlib import Path
import pandas as pd,numpy as np,json
P=Path(__file__).resolve().parents[1];d=pd.read_csv(P/'private_predictions/common_common_predictions.csv');names=['A','B','C','D','E','Fixed_half','Global_weight','OAF_original','OAF_same6_no_shrink','OAF_shrink'];draws={};points={};checks=[]
def ap(y,p,W):
 ix=np.argsort(-p,kind='mergesort');yy=y[ix];sw=W[:,ix];ends=np.r_[np.flatnonzero(np.diff(p[ix])!=0),len(y)-1];tp=np.cumsum(sw*yy,axis=1)[:,ends];tot=np.cumsum(sw,axis=1)[:,ends];prec=np.divide(tp,tot,out=np.zeros_like(tp),where=tot!=0);mass=np.diff(np.c_[np.zeros(len(W)),tp],axis=1);return np.sum(prec*mass,axis=1)/np.sum(sw*yy,axis=1)
for site,g in d.groupby('cohort'):
 ids=g.patient_id.to_numpy();u,iv,cnt=np.unique(ids,return_inverse=True,return_counts=True);w=1/cnt[iv];rng=np.random.default_rng(20261001+(0 if site=='fengxian' else 1000));mult=rng.multinomial(len(u),np.full(len(u),1/len(u)),size=5000);W=w*mult[:,iv];y=g.target.to_numpy();bs=np.column_stack([ap(y,g[n].to_numpy(),W) for n in names]);draws[site]=bs;points[site]=np.array([ap(y,g[n].to_numpy(),w[None,:])[0] for n in names]);reported=pd.read_csv(P/'results/common_paired.csv');reported=reported.loc[reported.cohort.eq('Site B' if site=='fengxian' else 'Site A')&reported.metric.eq('AUPRC')]
 for z in reported.itertuples():
  plus,minus=z.contrast.split('-');a=names.index(plus);b=names.index(minus);lo,hi=np.quantile(bs[:,a]-bs[:,b],[.025,.975]);assert max(abs(lo-z.ci_low),abs(hi-z.ci_high))<1e-12;checks.append(dict(cohort=site,contrast=z.contrast,recomputed_ci_low=lo,recomputed_ci_high=hi))
 print('independent 5000 AP bootstrap',site,'passed',flush=True)
G=pd.read_csv(P/'results/common_cross_campus_gaps.csv')
for z in G.loc[G.metric.eq('AUPRC')].itertuples():
 plus,minus=z.contrast.split('-');a=names.index(plus);b=names.index(minus);x=draws['fengxian'][:,a]-draws['fengxian'][:,b]-draws['xuhui_oof'][:,a]+draws['xuhui_oof'][:,b];lo,hi=np.quantile(x,[.025,.975]);assert max(abs(lo-z.ci_low),abs(hi-z.ci_high))<1e-12
pd.DataFrame(checks).to_csv(P/'verification/independent_bootstrap_check.csv',index=False)
(P/'verification/bootstrap_verification.json').write_text(json.dumps({'draws':5000,'paired_AP_intervals_checked':len(checks),'cross_campus_AP_gap_intervals_checked':len(G.loc[G.metric.eq('AUPRC')]),'algorithm':'independent vectorized weighted average-precision calculation on paired patient multiplicities','matches_saved_intervals_to':1e-12},indent=2))

"""Matched phase-first RF A-E decomposition and exact six-feature no-shrink control."""
from pathlib import Path
import json,pickle,copy
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from run_nested import load_data,build_samples,observations,thin_samples
from fusion import patient_weights,Fusion
from opal_fusion.baselines.tabular import tabularize_samples
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'common_experiment_verified'
DELTA=['aop','cervical_dilation','cervical_length','chin_chest_angle','head_direction_angle','head_station','hpd_recorded','midline_angle','spine_occiput_angle']

def change(samples,prep):
 # Last two distinct observation times, retaining the final record at each time.
 x=np.full((len(samples),len(DELTA)),np.nan)
 for i,s in enumerate(samples):
  for j,n in enumerate(DELTA):
   if n not in prep.variable_to_id:raise KeyError(n)
   rows=np.flatnonzero(s['variable_id']==prep.variable_to_id[n]); times={}
   for pos in rows:times[float(s['recency_hours'][pos])]=float(s['value'][pos])
   if len(times)>=2:
    t=sorted(times);x[i,j]=times[t[0]]-times[t[1]]
 return x,np.isfinite(x).astype(float)

def arms(S,delta,R,median):
 d=np.where(np.isfinite(delta),delta,median)
 return {'A':S,'B':np.c_[S,R],'C':np.c_[S,d],'D':np.c_[S,d,R],'E':np.c_[S,d]}

def main():
 OUT.mkdir(exist_ok=True);p,e,q,folds=load_data(ROOT/'processed_private');ext=p.loc[p.eligible&p.site.eq('fengxian'),'patient_id'].to_numpy(int)
 # Declared before inspecting new matched-arm predictions.
 (OUT/'PROTOCOL.json').write_text(json.dumps({'queries':1596,'seeds':[2026,2027,2028],'delta_variables':DELTA,'delta':'last two distinct times; normalized by saved outer-training statistics','R':'nine per-variable repeat-availability flags','E':'five copies at rates 0,.1,.2,.3,.4; fixed outer-training change medians','RF':'same saved outer fitting pool, seed,500trees,leaf3,balanced_subsample,patient weights','OAF_same6_no_shrink':'same six features, standardization and regularization as primary OAF but lambda=0','inference':'no target labels used in fitting','purpose':'exploratory problem-driven bridge, not independent confirmation'},indent=2))
 for seed in [2026,2027,2028]:
  for fold in range(5):
   dest=OUT/f'seed_{seed}_fold_{fold}';dest.mkdir(exist_ok=True)
   if (dest/'complete.json').exists():continue
   print('MATCHED FIT',seed,fold,flush=True)
   record=ROOT/'runs'/f'seed_{seed}'/f'fold_{fold}';saved=pickle.load(open(record/'models.pkl','rb'));prep=saved['prep']
   train=folds.loc[folds.fold.ne(fold),'patient_id'].to_numpy(int);valid=folds.loc[folds.fold.eq(fold),'patient_id'].to_numpy(int)
   ss=build_samples(p,e,q,train,prep);tab=tabularize_samples(ss,prep.n_variables,'current');delta,R=change(ss,prep)
   med=np.array([np.median(v[np.isfinite(v)]) if np.isfinite(v).any() else 0. for v in delta.T]);Xs=arms(tab.features,delta,R,med);w=patient_weights(tab.metadata.patient_id.to_numpy());models={'A':saved['rf']}
   for name in ['B','C','D','E']:
    X=Xs[name];y=tab.target;ww=w
    if name=='E':
     rng=np.random.default_rng(seed+fold+67000);xx=[]
     for rate in [0,.1,.2,.3,.4]:
      dd=delta.copy();mask=(rng.random(dd.shape)<rate)&np.isfinite(dd);dd[mask]=np.nan;xx.append(arms(tab.features,dd,R,med)['C'])
     X=np.vstack(xx);y=np.tile(y,5);ww=np.tile(w/5,5)
    model=clone(saved['rf']);model.set_params(n_jobs=2)
    model.fit(X,y,sample_weight=ww);models[name]=model
   # Inner records are in the same query ordering as the saved score table; key merge makes alignment explicit.
   inner=pd.read_csv(record/'inner_expert_predictions.csv');obs,rel=observations(ss);meta=tab.metadata[['patient_id','stage_id']].copy()
   for k in range(6):meta[f'o{k}']=obs[:,k]
   meta['repeat_support']=rel;inner=inner.merge(meta,on=['patient_id','stage_id'],validate='one_to_one')
   gate=Fusion('shrink',0.).fit(inner.current.to_numpy(),inner.history.to_numpy(),inner[[f'o{k}' for k in range(6)]].to_numpy(),inner.repeat_support.to_numpy(),inner.target.to_numpy(),inner.patient_id.to_numpy())
   frames=[];stressframes=[];old=pd.read_csv(record/'predictions.csv');oldstress=pd.read_csv(record/'stress.csv')
   for site,ids in [('xuhui_oof',valid),('fengxian',ext)]:
    test=build_samples(p,e,q,ids,prep);tt=tabularize_samples(test,prep.n_variables,'current');dd,rr=change(test,prep);xx=arms(tt.features,dd,rr,med)
    df=tt.metadata.copy();df['cohort']=site;df['seed']=seed;df['fold']=fold
    for n in models:df[n]=models[n].predict_proba(xx[n])[:,1]
    ref=old.loc[old.cohort.eq(site)].merge(df[['patient_id','stage_id']],on=['patient_id','stage_id'],validate='one_to_one').set_index(['patient_id','stage_id']).loc[pd.MultiIndex.from_frame(df[['patient_id','stage_id']])]
    assert np.allclose(df.A,ref.RF_current,atol=1e-12)
    oo,rrr=observations(test);df['OAF_same6_no_shrink']=gate.predict(ref.RF_current.to_numpy(),ref.GRUD.to_numpy(),oo,rrr);frames.append(df)
    for rate in [.2,.4]:
     for mask in range(10):
      th=thin_samples(test,rate,202600+mask+round(rate*1000));t=tabularize_samples(th,prep.n_variables,'current');assert np.array_equal(t.features,tt.features)
      dd,rr=change(th,prep);xx=arms(t.features,dd,rr,med);st=df[['patient_id','stage_id','target','cohort','seed','fold']].copy()
      for n in models:st[n]=models[n].predict_proba(xx[n])[:,1]
      hs=oldstress.loc[oldstress.cohort.eq(site)&oldstress.dropout.eq(rate)&oldstress['mask'].eq(mask)].set_index(['patient_id','stage_id']).loc[pd.MultiIndex.from_frame(st[['patient_id','stage_id']])]
      oo,rrr=observations(th);st['OAF_same6_no_shrink']=gate.predict(hs.RF_current.to_numpy(),hs.GRUD.to_numpy(),oo,rrr);st['dropout']=rate;st['mask']=mask;stressframes.append(st)
   pd.concat(frames).to_csv(dest/'predictions.csv',index=False);pd.concat(stressframes).to_csv(dest/'stress.csv',index=False)
   (dest/'complete.json').write_text(json.dumps({'rf_random_state':saved['rf'].random_state,'train_ids':train.tolist(),'test_ids':valid.tolist(),'delta_medians':med.tolist(),'gate_beta':gate.beta.tolist(),'gate_optimizer_success':gate.optimizer_success},indent=2))
   print('MATCHED DONE',seed,fold,flush=True)
 aggregate()

def aggregate():
 names=['A','B','C','D','E','OAF_same6_no_shrink'];keys=['seed','patient_id','stage_id','target','cohort']
 for name in ['predictions','stress']:
  d=pd.concat([pd.read_csv(z) for z in sorted(OUT.glob(f'seed_*/{name}.csv'))],ignore_index=True)
  kk=keys+(['dropout','mask'] if name=='stress' else []);d=d.groupby(kk,as_index=False)[names].mean();d.to_csv(OUT/f'{name}_by_seed.csv',index=False)
  d.groupby([k for k in kk if k!='seed'],as_index=False)[names].mean().to_csv(OUT/f'{name}_ensemble.csv',index=False)
if __name__=='__main__':main()

"""Campus separability using query-time observation features only."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from fusion import patient_weights

def main():
 root=Path(__file__).resolve().parents[1];out=root/'results';out.mkdir(exist_ok=True)
 p=pd.read_csv(root/'processed_private/patients.csv');e=pd.read_csv(root/'processed_private/events.csv');q=pd.read_csv(root/'processed_private/landmarks.csv')
 p=p.loc[p.eligible.astype(str).str.lower().eq('true')&p.site.isin(['xuhui','fengxian'])]
 q=q.loc[q.patient_id.isin(p.patient_id)].copy();e=e.loc[e.enabled.astype(str).str.lower().eq('true')]
 e.event_time=pd.to_datetime(e.event_time,format='mixed',errors='coerce');q.query_time=pd.to_datetime(q.query_time,format='mixed')
 from opal_fusion.data.labor_schema import EVENT_SPECS
 names=[s.name for s in EVENT_SPECS if s.enabled];groups={i:g for i,g in e.groupby('patient_id')};rows=[];feat=[]
 for r in q.itertuples():
  g=groups[r.patient_id];g=g.loc[g.event_time.le(r.query_time)&g.event_time.notna()];M=[];R=[];C=[];T=[]
  for n in names:
   times=g.loc[g.variable.eq(n),'event_time'];count=times.nunique();M.append(count>=1);R.append(count>=2);C.append(np.log1p(count));T.append(np.log1p((r.query_time-times.max()).total_seconds()/3600) if count else 0.)
  rows.append({'patient_id':r.patient_id,'stage':r.stage,'target':r.outcome_binary,'campus':int(r.site=='fengxian')});feat.append(np.r_[M,R,C,T])
 d=pd.DataFrame(rows);X=np.array(feat,float);unique=d.drop_duplicates('patient_id');ids=d.patient_id.to_numpy();w=patient_weights(ids);y=d.campus.to_numpy();splits=list(StratifiedKFold(5,shuffle=True,random_state=20261001).split(unique.patient_id,unique.campus))
 matrices={'repeat_only':X[:,14:28],'full_query_observation':X};metrics=[];preds={}
 for name,x in matrices.items():
  for model in ['LR','RF']:
   pred=np.zeros(len(d))
   for a,b in splits:
    tr=np.isin(ids,unique.patient_id.to_numpy()[a]);va=np.isin(ids,unique.patient_id.to_numpy()[b]);wt=patient_weights(ids[tr])
    clf=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000)) if model=='LR' else RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight='balanced_subsample',random_state=20261001,n_jobs=2)
    clf.fit(x[tr],y[tr],**({'logisticregression__sample_weight':wt} if model=='LR' else {'sample_weight':wt}));pred[va]=clf.predict_proba(x[va])[:,1]
   key=name+'_'+model;preds[key]=pred
   u,inv=np.unique(ids,return_inverse=True);rng=np.random.default_rng(123);boots=[]
   for _ in range(1000):
    ww=w*rng.multinomial(len(u),np.full(len(u),1/len(u)))[inv]
    if len(np.unique(y[ww>0]))==2:boots.append(roc_auc_score(y,pred,sample_weight=ww))
   lo,hi=np.quantile(boots,[.025,.975]);metrics.append(dict(feature_set=name,learner=model,stratum='all',AUROC=roc_auc_score(y,pred,sample_weight=w),ci_low=lo,ci_high=hi))
   for field in ['stage','target']:
    for value in d[field].unique():
     sel=d[field].eq(value).to_numpy();metrics.append(dict(feature_set=name,learner=model,stratum=f'{field}={value}',AUROC=roc_auc_score(y[sel],pred[sel],sample_weight=patient_weights(ids[sel]))))
 pd.DataFrame(metrics).to_csv(out/'query_time_site_diagnostic.csv',index=False)
 d.assign(**preds).to_csv(out/'query_time_site_predictions.csv',index=False)
 # Patient labels are permuted as whole clusters; predictions regenerated.
 rng=np.random.default_rng(1234);null=[];x=matrices['repeat_only'];uid=unique.patient_id.to_numpy()
 for perm in range(100):
  mapping=dict(zip(uid,rng.permutation(unique.campus.to_numpy())));yp=np.array([mapping[i] for i in ids]);pred=np.zeros(len(d))
  for a,b in StratifiedKFold(5,shuffle=True,random_state=20261001).split(uid,np.array([mapping[i] for i in uid])):
   tr=np.isin(ids,uid[a]);va=np.isin(ids,uid[b]);clf=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=2000));clf.fit(x[tr],yp[tr],logisticregression__sample_weight=patient_weights(ids[tr]));pred[va]=clf.predict_proba(x[va])[:,1]
  null.append(roc_auc_score(yp,pred,sample_weight=w))
 observed=roc_auc_score(y,preds['repeat_only_LR'],sample_weight=w);pd.DataFrame({'AUROC':null}).to_csv(out/'site_permutation_null.csv',index=False)
 (out/'site_permutation.json').write_text(json.dumps({'observed':observed,'permutations':100,'p_plus_one':(1+sum(v>=observed for v in null))/101},indent=2))
 print(pd.DataFrame(metrics).to_string(index=False),flush=True)
if __name__=='__main__':main()

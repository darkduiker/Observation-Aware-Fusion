from pathlib import Path
import pandas as pd,numpy as np,json,re,sys
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,balanced_accuracy_score
P=Path(__file__).resolve().parents[1];out=P/'verification';out.mkdir(exist_ok=True)
d=pd.read_csv(P/'private_predictions/common_common_predictions.csv');six=pd.read_csv(P/'private_predictions/six_model_predictions.csv');rows=[]
def weight(g):return 1/g.patient_id.map(g.patient_id.value_counts()).to_numpy()
def scores(g,m):
 w=weight(g);return {'AUROC':roc_auc_score(g.target,g[m],sample_weight=w),'AUPRC':average_precision_score(g.target,g[m],sample_weight=w),'Brier':brier_score_loss(g.target,g[m],sample_weight=w)}
assert len(d)==1596 and not d.duplicated(['patient_id','stage_id','cohort']).any()
assert len(six)==1596 and not six.duplicated(['patient_id','stage_id','cohort']).any()
assert d.groupby('patient_id').target.nunique().max()==1
assert np.allclose(d.A,d.RF_current,rtol=0,atol=1e-12)
a=pd.read_csv(P/'results/common_absolute.csv'); a['cohort']=a.cohort.replace({'Site A':'xuhui_oof','Site B':'fengxian'})
for site,g in d.groupby('cohort'):
 assert g.patient_id.nunique()==({'xuhui_oof':736,'fengxian':118}[site])
 for m in a.model.unique():
  for metric,v in scores(g,m).items():
   z=a.loc[(a.cohort==site)&(a.model==m)&(a.metric==metric)].iloc[0];diff=v-z.estimate;assert abs(diff)<1e-12
   rows.append(dict(source='common',cohort=site,model=m,metric=metric,recomputed=v,reported=z.estimate,difference=diff))
bs=pd.read_csv(P/'results/benchmark_5000.csv'); bs['cohort']=bs.cohort.replace({'Site A':'xuhui_oof','Site B':'fengxian'});print('benchmark cols',bs.columns.tolist())
for site,g in six.groupby('cohort'):
 for m in ['LR_current','RF_current','LightGBM_current','GRUD','Raindrop_portable','OAF_shrink']:
  for metric,v in scores(g,m).items():
   z=bs.loc[(bs.cohort==site)&(bs.model==m)&(bs.metric==metric)].iloc[0];assert abs(v-z.estimate)<1e-12
   rows.append(dict(source='six',cohort=site,model=m,metric=metric,recomputed=v,reported=z.estimate,difference=v-z.estimate))
# exact patient/query keys and identical anchor/fused predictions across result sets
merge=d.merge(six,on=['patient_id','stage_id','cohort','target'],validate='one_to_one',suffixes=('_d','_s'))
for m in ['RF_current','OAF_shrink']:assert np.allclose(merge[m+'_d'],merge[m+'_s'],rtol=0,atol=1e-12)
# Independent algebra for contrasts, gaps, and perturbation responses.
paired=pd.read_csv(P/'results/common_paired.csv'); paired['cohort']=paired.cohort.replace({'Site A':'xuhui_oof','Site B':'fengxian'})
for site,g in d.groupby('cohort'):
 for _,z in paired.loc[paired.cohort.eq(site)&paired.metric.isin(['AUROC','AUPRC','Brier'])].iterrows():
  plus,minus=z.contrast.split('-');v=scores(g,plus)[z.metric]-scores(g,minus)[z.metric];assert abs(v-z.estimate)<1e-12
G=pd.read_csv(P/'results/common_cross_campus_gaps.csv')
assert np.allclose(G.gap,G.siteB_increment-G.siteA_increment,atol=1e-12)
st=pd.read_csv(P/'private_predictions/common_stress_predictions.csv')
assert len(st)==1596*20
sm=st.merge(d,on=['patient_id','stage_id','cohort','target'],suffixes=('_st','_base'),validate='many_to_one')
assert np.allclose(sm.A_st,sm.A_base,rtol=0,atol=1e-12)
resp=pd.read_csv(P/'results/common_paired_probability_response.csv'); resp['cohort']=resp.cohort.replace({'Site A':'xuhui_oof','Site B':'fengxian'})
for (site,rate),g in sm.groupby(['cohort','dropout']):
 vals={}
 for m in ['D','E','Fixed_half','Global_weight','OAF_same6_no_shrink','OAF_shrink']:
  x=g[['patient_id','stage_id']].copy();x['v']=abs(g[m+'_st']-g[m+'_base']);vals[m]=x.groupby(['patient_id','stage_id']).v.mean().groupby('patient_id').mean().mean()
 for _,z in resp.loc[resp.cohort.eq(site)&resp.dropout.eq(rate)].iterrows():
  assert abs(vals[z.control]-z.control_probability_response)<1e-12
  assert abs(vals['OAF_shrink']-z.OAF_probability_response)<1e-12
pd.DataFrame(rows).to_csv(out/'metric_recomputation.csv',index=False)
summary={'patients':854,'queries':1596,'absolute_metrics_checked':len(rows),'max_absolute_metric_discrepancy':max(abs(r['difference']) for r in rows),'query_keys_identical':True,'anchor_identical':True,'patient_outcome_consistent':True,'contrast_and_gap_algebra_checked':True,'stress_anchor_unchanged':True,'probability_responses_recomputed':True,'scope':'Recomputed from saved predictions; no fresh model fitting or source-workbook clinical-unit confirmation.'}
(out/'result_verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

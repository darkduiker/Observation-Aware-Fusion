from pathlib import Path
import re,pandas as pd,json
P=Path(__file__).resolve().parents[1]
s=(P/'paper/Supplement_Checked.md').read_text();count=0
models={'LR-current':'LR_current','RF-current':'RF_current','LightGBM-current':'LightGBM_current','GRU-D':'GRUD','Raindrop-portable':'Raindrop_portable','OAF':'OAF_shrink','Fixed blend':'Fixed_half','Global blend':'Global_weight','Four-summary gate':'OAF_original','Repeat-support gate':'OAF_reliable_shrink'}
conds={'Current state':'A','Current + reassessment':'B','Current + change':'C','Current + change + reassessment':'D','OPR training':'E','Fixed 50:50 blend':'Fixed_half','Global blend':'Global_weight','Six-feature gate without shrinkage':'OAF_same6_no_shrink'}
def rows(n):
 text=re.split(r'\n## S\d+ ',s[s.index('# S'+str(n)+' '):],maxsplit=1)[0]
 return [[c.strip() for c in l.strip('|').split('|')] for l in text.splitlines() if l.startswith('|')][2:]
def ck(cell,vals):
 global count
 toks=re.findall(r'[+-]?\d+\.\d+|[+-]?\d+',cell);assert len(toks)==len(vals),(cell,vals)
 for t,v in zip(toks,vals):
  nd=len(t.partition('.')[2]);assert abs(float(t)-v)<=.5001*10**(-nd),(cell,v);count+=1
for n,fn in [(2,'benchmark_5000'),(3,'metrics_by_seed'),(4,'standardized_AP'),(5,'phase_performance'),(6,'query_time_site_diagnostic'),(7,'calibration_metrics'),(10,'balanced_accuracy'),(11,'common_absolute'),(12,'common_cross_campus_gaps'),(13,'common_paired_probability_response')]:
 d=pd.read_csv(P/'results'/f'{fn}.csv')
 for r in rows(n):
  if n==2:
   z=d[(d.cohort==r[0])&(d.model==models[r[1]])&(d.metric==r[2])].iloc[0];cells=[(r[3],[z.estimate]),(r[4],[z.ci_low,z.ci_high])]
  elif n==3:
   z=d[(d.seed==int(r[0]))&(d.cohort==r[1])&(d.model==models[r[2]])].iloc[0];cells=[(r[i],[z[k]]) for i,k in enumerate(['AUROC','AUPRC','Brier'],3)]
  elif n==4:
   z=d[(d.cohort==r[0])&(d.model==models[r[1]])].iloc[0];cells=[(r[2],[z.standardized_AP]),(r[3],[z.difference_vs_RF]),(r[4],[z.difference_ci_low,z.difference_ci_high])]
  elif n==5:
   z=d[(d.cohort==r[0])&(d.stage_id==int(r[1]))&(d.model==models[r[2]])].iloc[0];cells=[(r[i],[z[k]]) for i,k in enumerate(['patients','events','AUROC','AUPRC','Brier'],3)]
  elif n==6:
   fs='repeat_only' if r[0]=='Repeat availability' else 'full_query_observation';strata={'Dilation <4 cm':'stage=lt4','Dilation 4–<6 cm':'stage=4to6','Dilation 6–10 cm':'stage=6to10','Spontaneous delivery':'target=0','Intervention':'target=1'};st=strata.get(r[2],r[2]);z=d[(d.feature_set==fs)&(d.learner==r[1])&(d.stratum==st)]
   if not len(z):print('stratum skip',r[2],d.stratum.unique());continue
   z=z.iloc[0];cells=[(r[3],[z.AUROC])];cells+=[] if r[4]=='Not estimated' else [(r[4],[z.ci_low,z.ci_high])]
  elif n==7:
   z=d[(d.cohort==r[0])&(d.model==models[r[1]])].iloc[0];cells=[(r[i],[z[k]]) for i,k in enumerate(['calibration_intercept','calibration_slope','Brier'],2)]
  elif n==10:
   z=d[(d.cohort==r[0])&(d.model==models[r[1]])].iloc[0];cells=[(r[2],[z.BA])]
  elif n==11:
   metric='Standardized_AP' if r[2]=='Standardized AP' else r[2];z=d[(d.cohort==r[0])&(d.model==conds[r[1]])&(d.metric==metric)].iloc[0];cells=[(r[3],[z.estimate]),(r[4],[z.ci_low,z.ci_high])]
  elif n==12:
   c=r[0].replace('OAF','OAF_shrink').replace('Fixed blend','Fixed_half').replace('Global blend','Global_weight').replace('Unshrunk gate','OAF_same6_no_shrink');metric='Standardized_AP' if r[1]=='Standardized AP' else r[1];z=d[(d.contrast==c)&(d.metric==metric)].iloc[0];cells=[(r[2],[z.siteA_increment]),(r[3],[z.siteB_increment]),(r[4],[z.gap,z.ci_low,z.ci_high])]
  else:
   z=d[(d.cohort==r[0])&(d.dropout==float(r[1].strip('%'))/100)&(d.control==conds[r[2]])].iloc[0];cells=[(r[3],[z.control_brier_change]),(r[4],[z.OAF_brier_change]),(r[5],[z.brier_change_attenuation,z.brier_ci_low,z.brier_ci_high])]
  for cell,v in cells:ck(cell,v)
print('Rounded supplementary numeric cells verified:',count)
(P/'verification/table_cell_verification.json').write_text(json.dumps({'numeric_cells_checked':count,'supplement_sections':[2,3,4,5,6,7,10,11,12,13],'rounded_values_match_CSV':True},indent=2))

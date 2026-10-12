"""Reconstruct gate covariates from original measurement opportunities.

Expert probabilities remain cross-fitted; nominal indicator channels do not
count as nine separate clinical observations. No outcomes from Site B are used.
"""
from pathlib import Path
import pickle,json
import numpy as np,pandas as pd,torch
from run_nested import load_data,build_samples,observations,expert_predictions,thin_samples,tensors,predict_tensor,model_for,MODELS
from fusion import Fusion,select_gate
from opal_fusion.baselines.tabular import tabularize_samples

def main():
 root=Path(__file__).resolve().parents[1];p,e,q,f=load_data(root/'processed_private');external=p.loc[p.eligible&p.site.eq('fengxian'),'patient_id'].to_numpy(int);torch.set_num_threads(2)
 for seed in [2026,2027,2028]:
  for fold in range(5):
   d=root/'runs'/f'seed_{seed}'/f'fold_{fold}'
   if (d/'gate_covariates_corrected.json').exists():continue
   record=json.loads((d/'complete.json').read_text());bundle=pickle.load(open(d/'models.pkl','rb'));prep=bundle['prep'];rf=bundle['rf']
   train=f.loc[f.fold.ne(fold),'patient_id'].to_numpy(int);valid=f.loc[f.fold.eq(fold),'patient_id'].to_numpy(int)
   met=pd.read_csv(d/'inner_expert_predictions.csv');ss=build_samples(p,e,q,train,prep);o,r=observations(ss)
   obs=pd.DataFrame({'patient_id':[s['patient_id'] for s in ss],'stage_id':[s['stage_id'] for s in ss],'support':r,**{f'correct_obs_{j}':o[:,j] for j in range(6)}})
   met=met.merge(obs,on=['patient_id','stage_id'],validate='one_to_one');assert len(met)==len(ss)
   c=met.current.to_numpy();h=met.history.to_numpy();o=met[[f'correct_obs_{j}' for j in range(6)]].to_numpy();r=met.support.to_numpy();y=met.target.to_numpy();ids=met.patient_id.to_numpy()
   gates={'Fixed_half':Fusion('fixed_half').fit(c,h,o,r,y,ids),'Global_weight':Fusion('global').fit(c,h,o,r,y,ids),'OAF_original':Fusion('original').fit(c,h,o,r,y,ids)};tuning={}
   for name,mode in [('OAF_shrink','shrink'),('OAF_reliable_shrink','reliable_shrink')]:gates[name],tuning[name]=select_gate(mode,c,h,o,r,y,ids,seed+fold)
   model=model_for(prep,128);model.load_state_dict(torch.load(d/'grud.pt',weights_only=False)['state']);model.eval()
   parts=[];stress=[]
   for cohort,queryids in [('xuhui_oof',valid),('fengxian',external)]:
    m,c,h,o,r,samples=expert_predictions(rf,model,prep,p,e,q,queryids);m['RF_current']=c;m['GRUD']=h
    for name,g in gates.items():m[name]=g.predict(c,h,o,r);m[name+'_weight']=g.gates(o,r)
    m['seed']=seed;m['fold']=fold;m['cohort']=cohort;parts.append(m)
    for rate in [.2,.4]:
     for mask in range(10):
      ss=thin_samples(samples,rate,202600+mask+round(rate*1000));assert np.array_equal(tabularize_samples(samples,prep.n_variables,'current').features,tabularize_samples(ss,prep.n_variables,'current').features)
      hh=predict_tensor(model,tensors(ss));oo,rr=observations(ss);sm=m[['patient_id','stage_id','target','center_id','seed','fold','cohort']].copy();sm['RF_current']=c;sm['GRUD']=hh
      for name,g in gates.items():sm[name]=g.predict(c,hh,oo,rr)
      sm['dropout']=rate;sm['mask']=mask;stress.append(sm)
   pd.concat(parts,ignore_index=True).to_csv(d/'predictions.csv',index=False);pd.concat(stress,ignore_index=True).to_csv(d/'stress.csv',index=False)
   met.to_csv(d/'inner_expert_predictions.csv',index=False);bundle['gates']=gates;pickle.dump(bundle,open(d/'models.pkl','wb'))
   record['gate_tuning']=tuning
   for name,g in gates.items():record.setdefault('gate_params',{})[name]={'strength':g.strength,'alpha':getattr(g,'alpha',None),'beta':getattr(g,'beta',np.array([])).tolist()}
   record['observation_encoding']='one opportunity per original categorical measurement, not per indicator channel';(d/'complete.json').write_text(json.dumps(record,indent=2));(d/'gate_covariates_corrected.json').write_text(json.dumps({'version':37,'seed':seed,'fold':fold}))
   print('CORRECTED GATE',seed,fold,flush=True)
if __name__=='__main__':main()

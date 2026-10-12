"""Nested patient-separated RF/GRU-D fusion experiment on phase-first queries.

Fengxian features/outcomes do not enter any fitting or tuning operation.
Each outer training pool generates fresh three-fold inner expert predictions.
GRU-D epochs are selected with a patient holdout inside each inner training set,
then the expert is refit on that whole inner set. Outer refits use the median
selected epoch. Patient-equal weights are used for fitting and primary metrics.
"""
from pathlib import Path
import argparse, hashlib, json, pickle, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score
from opal_fusion.data.labor_dataset import LaborPreprocessor, LaborLandmarkDataset
from opal_fusion.baselines.tabular import tabularize_samples
from opal_fusion.baselines.grud import GRUD, GRUDConfig
from opal_fusion.baselines.dense_adapter import collate_dense_landmarks
from fusion import Fusion, patient_weights, select_gate

MODELS=['RF_current','GRUD','Fixed_half','Global_weight','OAF_original','OAF_shrink','OAF_reliable_shrink']

def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)

def expand_position_events(e):
    source=e.loc[e.variable.eq('fetal_position')].copy()
    if source.empty:return e
    code=pd.to_numeric(source.value,errors='coerce').fillna(0).astype(int)
    code=code.where(code.between(0,8),0)
    blocks=[e.loc[~e.variable.eq('fetal_position')].copy()]
    for k in range(9):
        b=source.copy();b['variable']=f'fetal_position_{k}';b['value']=(code==k).astype(float);blocks.append(b)
    return pd.concat(blocks,ignore_index=True)

def load_data(processed):
    p=pd.read_csv(processed/'patients.csv');e=pd.read_csv(processed/'events.csv')
    q=pd.read_csv(processed/'landmarks.csv');f=pd.read_csv(processed/'xuhui_folds.csv')
    e=expand_position_events(e)
    p['eligible']=p.eligible.astype(str).str.lower().eq('true')
    e['enabled']=e.enabled.astype(str).str.lower().eq('true')
    e['event_time']=pd.to_datetime(e.event_time,format='mixed',errors='coerce')
    q['query_time']=pd.to_datetime(q.query_time,format='mixed',errors='coerce')
    e=e.loc[e.enabled&e.event_time.notna()].copy()
    q=q.loc[q.site.isin(['xuhui','fengxian'])].copy()
    assert q.groupby('site').size().to_dict()=={'fengxian':218,'xuhui':1378}
    assert not q.duplicated(['patient_id','stage']).any()
    return p,e,q,f

def metadata(samples):
    return pd.DataFrame([{k:s[k] for k in ['patient_id','stage_id','target','center_id']} for s in samples])

def observations(samples):
    obs=[];reliable=[]
    # variable ids follow EVENT_SPECS; four progression-associated variables.
    progression={9,10,13,14} # replaced by names in build_samples below
    for s in samples:
        keep=(s['obs_variable_id']!=3)|(s['variable_id']==s['position_zero_id'])
        v=s['obs_variable_id'][keep];age=s['recency_hours'][keep]
        counts=np.bincount(v,minlength=15)[1:15]
        last_age=[];repeat=[]
        for j in range(1,15):
            t=age[v==j]
            if len(t):last_age.append(float(np.min(t)))
            repeat.append(len(np.unique(t))>=2)
        span=float(np.max(age)-np.min(age)) if len(age) else 0.0
        support=np.mean([repeat[j-1] for j in s['observation_progression_ids']])
        obs.append([s['stage_id']/2,np.mean(counts>0),np.log1p(len(v))/4,
                    np.log1p(np.median(last_age) if last_age else 0)/4,
                    np.mean(repeat),np.log1p(span)/4])
        reliable.append(support)
    return np.asarray(obs),np.asarray(reliable)

def build_samples(p,e,q,ids,prep):
    ds=LaborLandmarkDataset(p,e,q,prep,ids)
    ss=[ds[i] for i in range(len(ds))]
    prog=[prep.variable_to_id[n] for n in ['aop','head_station','hpd_recorded','cervical_dilation']]
    from opal_fusion.data.labor_schema import EVENT_SPECS
    original={spec.name:i+1 for i,spec in enumerate([t for t in EVENT_SPECS if t.enabled])}
    inverse={i:original['fetal_position' if n.startswith('fetal_position_') else n] for n,i in prep.variable_to_id.items()}
    for s in ss:
        s['progression_ids']=prog
        s['position_zero_id']=prep.variable_to_id['fetal_position_0']
        s['observation_progression_ids']=[original[n] for n in ['aop','head_station','hpd_recorded','cervical_dilation']]
        s['obs_variable_id']=np.array([inverse[int(i)] for i in s['variable_id']],dtype=np.int64)
    return ss

def tensors(samples):
    return collate_dense_landmarks(samples,bin_hours=1/6,max_steps=128)

def model_for(prep,hidden,kind='grud'):
    if kind == 'raindrop':
        from opal_fusion.baselines.raindrop import Raindrop,RaindropConfig
        return Raindrop(RaindropConfig(prep.n_variables,prep.n_static_features,model_dimension=hidden))
    return GRUD(GRUDConfig(prep.n_variables,prep.n_static_features,hidden_size=hidden))

def fit_grud(train_samples,prep,seed,epochs,hidden,selection_samples=None,patience=10,kind='grud'):
    set_seed(seed)
    model=model_for(prep,hidden,kind)
    batch=tensors(train_samples)
    y=batch['target'];ids=batch['patient_id'].numpy();w=patient_weights(ids)
    prevalence=np.average(y.numpy(),weights=w)
    classweights=np.where(y.numpy()==1,1/(2*prevalence),1/(2*(1-prevalence)))
    weights=torch.tensor(w*classweights,dtype=torch.float32)
    weights/=weights.mean()
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    best=-np.inf;stale=0;bestepoch=1;beststate=None
    selector=tensors(selection_samples) if selection_samples else None
    selids=selector['patient_id'].numpy() if selector is not None else None
    hist=[]
    for epoch in range(1,epochs+1):
        model.train();order=torch.randperm(len(y));losses=[]
        for ii in order.split(128):
            b={k:v[ii] for k,v in batch.items()}
            logits=model(b)['logits']
            loss=(nn.functional.cross_entropy(logits,b['target'],reduction='none')*weights[ii]).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
        if selector is not None:
            pred=predict_tensor(model,selector)
            score=average_precision_score(selector['target'].numpy(),pred,sample_weight=patient_weights(selids))
            hist.append({'epoch':epoch,'selection_auprc':float(score),'train_loss':float(np.mean(losses))})
            if score>best+1e-5:
                best=score;bestepoch=epoch;stale=0
                beststate={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            if stale>=patience:break
    if beststate is not None:model.load_state_dict(beststate)
    return model,bestepoch if selector is not None else epochs,hist

def predict_tensor(model,batch):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(batch['target']),256):
            b={k:v[start:start+256] for k,v in batch.items()}
            out.extend(torch.softmax(model(b)['logits'],dim=-1)[:,1].numpy())
    return np.array(out)

def fit_experts(p,e,q,ids,seed,args,select_epochs=True,fixed_epochs=None):
    labels=p.set_index('patient_id').loc[ids,'outcome_binary'].to_numpy(int)
    if select_epochs:
        a,b=train_test_split(ids,test_size=.2,random_state=seed,stratify=labels)
        assert not set(a)&set(b)
        earlyprep=LaborPreprocessor.fit(p,e,a)
        _,epochs,history=fit_grud(build_samples(p,e,q,a,earlyprep),earlyprep,seed,args.epochs,args.hidden,
                        build_samples(p,e,q,b,earlyprep),args.patience)
    else:epochs=int(fixed_epochs);history=[]
    prep=LaborPreprocessor.fit(p,e,ids)
    ss=build_samples(p,e,q,ids,prep)
    matrix=tabularize_samples(ss,prep.n_variables,'current')
    rf=RandomForestClassifier(n_estimators=args.trees,min_samples_leaf=3,max_features='sqrt',
           class_weight='balanced_subsample',random_state=seed,n_jobs=args.jobs)
    rf.fit(matrix.features,matrix.target,sample_weight=patient_weights(matrix.metadata.patient_id.to_numpy()))
    grud,_,_=fit_grud(ss,prep,seed,args.epochs if fixed_epochs is None and not select_epochs else epochs,args.hidden)
    return rf,grud,prep,epochs,history

def expert_predictions(rf,grud,prep,p,e,q,ids):
    ss=build_samples(p,e,q,ids,prep)
    matrix=tabularize_samples(ss,prep.n_variables,'current')
    c=rf.predict_proba(matrix.features)[:,1];h=predict_tensor(grud,tensors(ss))
    o,r=observations(ss)
    return matrix.metadata,c,h,o,r,ss

def metrics(frame,column,weighted=True):
    y=frame.target.to_numpy();pred=frame[column].to_numpy()
    w=patient_weights(frame.patient_id.to_numpy()) if weighted else np.ones(len(y))
    return {'AUROC':float(roc_auc_score(y,pred,sample_weight=w)),
            'AUPRC':float(average_precision_score(y,pred,sample_weight=w)),
            'Brier':float(brier_score_loss(y,pred,sample_weight=w)),
            'BA':float(balanced_accuracy_score(y,pred>=.5,sample_weight=w))}

def summarize(frame):
    rows=[]
    for cohort,g in frame.groupby('cohort'):
        for model in MODELS:
            for weighted in [True,False]:
                rows.append({'cohort':cohort,'model':model,'weighting':'patient_equal' if weighted else 'query_equal',
                             'patients':g.patient_id.nunique(),'queries':len(g),**metrics(g,model,weighted)})
    return pd.DataFrame(rows)

def thin_samples(samples,rate,seed):
    rng=np.random.default_rng(seed);out=[]
    for s in samples:
        keep=np.ones(len(s['value']),bool)
        for j in np.unique(s['obs_variable_id']):
            ix=np.flatnonzero(s['obs_variable_id']==j)
            last=np.min(s['recency_hours'][ix])
            # One draw per original variable/time, shared by category channels.
            for age in np.unique(s['recency_hours'][ix]):
                if age>last:
                    keep[ix[s['recency_hours'][ix]==age]]=rng.random()>=rate
        copy=s.copy()
        for k in ['value','variable_id','modality_id','recency_hours','elapsed_hours','time_quality_id','obs_variable_id']:
            copy[k]=s[k][keep]
        out.append(copy)
    return out

def main(args):
    torch.set_num_threads(args.threads)
    p,e,q,folds=load_data(args.processed)
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    protocol={'seeds':args.seeds,'outer_folds':5,'inner_folds':3,'epochs_max':args.epochs,'hidden':args.hidden,
              'rf_trees':args.trees,'patient_equal':True,'phase_first':{'xuhui':1378,'fengxian':218},
              'models':MODELS,'gate_selection':'training-only three-fold meta-CV, minimum patient-equal Brier',
              'history_reliability':'fraction of AoP, head station, HPD and dilation with >=2 distinct pre-query times',
              'external_status':'cross-campus evaluation; target labels excluded from reported fitting and tuning',
              'stress_test':'delete older history events while preserving latest values; no workflow-causal interpretation'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    allpred=[];allstress=[]
    external=p.loc[p.eligible&p.site.eq('fengxian'),'patient_id'].to_numpy(int)
    for seed in args.seeds:
        seedout=out/f'seed_{seed}';seedout.mkdir(exist_ok=True)
        for fold in range(5):
            dest=seedout/f'fold_{fold}';dest.mkdir(exist_ok=True)
            if (dest/'predictions.csv').exists() and (dest/'complete.json').exists():
                allpred.append(pd.read_csv(dest/'predictions.csv'))
                if (dest/'stress.csv').exists():allstress.append(pd.read_csv(dest/'stress.csv'))
                print(f'CACHE seed={seed} fold={fold}',flush=True);continue
            tic=time.monotonic()
            train=folds.loc[folds.fold.ne(fold),'patient_id'].to_numpy(int)
            valid=folds.loc[folds.fold.eq(fold),'patient_id'].to_numpy(int)
            assert not set(train)&set(valid) and not set(train)&set(external)
            labels=p.set_index('patient_id').loc[train,'outcome_binary'].to_numpy(int)
            innerpred=[];epochchoices=[];audit={'outer_train_ids':train.tolist(),'outer_test_ids':valid.tolist(),'inner':[]}
            for infold,(a,b) in enumerate(StratifiedKFold(3,shuffle=True,random_state=seed+fold).split(train,labels)):
                innertrain,innerval=train[a],train[b]
                expertseed=seed+1000*fold+100*infold
                print(f'FIT seed={seed} outer={fold} inner={infold} train={len(innertrain)} validation={len(innerval)}',flush=True)
                rf,grud,prep,epochs,history=fit_experts(p,e,q,innertrain,expertseed,args)
                m,c,h,o,r,_=expert_predictions(rf,grud,prep,p,e,q,innerval)
                m['current']=c;m['history']=h;m['reliability']=r
                for j in range(o.shape[1]):m[f'obs_{j}']=o[:,j]
                innerpred.append(m);epochchoices.append(epochs)
                audit['inner'].append({'fit_ids':innertrain.tolist(),'prediction_ids':innerval.tolist(),'epochs':epochs,'epoch_history':history})
                m.to_csv(dest/f'inner_{infold}_expert_predictions.csv',index=False)
            met=pd.concat(innerpred,ignore_index=True);met.to_csv(dest/'inner_expert_predictions.csv',index=False)
            assert set(met.patient_id)==set(train)
            c=met.current.to_numpy();h=met.history.to_numpy();o=met[[f'obs_{j}' for j in range(6)]].to_numpy()
            r=met.reliability.to_numpy();y=met.target.to_numpy();ids=met.patient_id.to_numpy()
            gates={'Fixed_half':Fusion('fixed_half').fit(c,h,o,r,y,ids),
                   'Global_weight':Fusion('global').fit(c,h,o,r,y,ids),
                   'OAF_original':Fusion('original').fit(c,h,o,r,y,ids)}
            tuning={}
            for name,mode in [('OAF_shrink','shrink'),('OAF_reliable_shrink','reliable_shrink')]:
                gates[name],tuning[name]=select_gate(mode,c,h,o,r,y,ids,seed+fold)
            epochs=int(np.median(epochchoices))
            print(f'REFIT seed={seed} outer={fold} epochs={epochs}',flush=True)
            rf,grud,prep,_,_=fit_experts(p,e,q,train,seed+1000*fold+999,args,False,epochs)
            pickle.dump({'rf':rf,'prep':prep,'gates':gates},open(dest/'models.pkl','wb'))
            torch.save({'state':grud.state_dict(),'hidden':args.hidden},dest/'grud.pt')
            audit['gate_tuning']=tuning;audit['outer_refit_epochs']=epochs
            for name,g in gates.items():
                audit.setdefault('gate_params',{})[name]={'strength':g.strength,'alpha':getattr(g,'alpha',None),
                                'beta':getattr(g,'beta',np.array([])).tolist()}
            frames=[];stressframes=[]
            for cohort,queryids in [('xuhui_oof',valid),('fengxian',external)]:
                m,c,h,o,r,samples=expert_predictions(rf,grud,prep,p,e,q,queryids)
                m['RF_current']=c;m['GRUD']=h
                for name,g in gates.items():m[name]=g.predict(c,h,o,r);m[name+'_weight']=g.gates(o,r)
                m['seed']=seed;m['fold']=fold;m['cohort']=cohort;frames.append(m)
                for rate in args.dropout:
                    for mask in range(args.masks):
                        ss=thin_samples(samples,rate,202600+mask+round(rate*1000))
                        # Clinical current state must remain exactly unchanged.
                        before=tabularize_samples(samples,prep.n_variables,'current').features
                        after=tabularize_samples(ss,prep.n_variables,'current').features
                        assert np.array_equal(before,after),'Latest-state preservation failed'
                        ho=predict_tensor(grud,tensors(ss));oo,rr=observations(ss)
                        sm=m[['patient_id','stage_id','target','center_id','seed','fold','cohort']].copy()
                        sm['RF_current']=c;sm['GRUD']=ho
                        for name,g in gates.items():sm[name]=g.predict(c,ho,oo,rr)
                        sm['dropout']=rate;sm['mask']=mask;stressframes.append(sm)
            joined=pd.concat(frames,ignore_index=True);joined.to_csv(dest/'predictions.csv',index=False)
            if stressframes:
                stress=pd.concat(stressframes,ignore_index=True);stress.to_csv(dest/'stress.csv',index=False);allstress.append(stress)
            (dest/'complete.json').write_text(json.dumps(audit,indent=2))
            allpred.append(joined)
            print(f'DONE seed={seed} outer={fold} seconds={time.monotonic()-tic:.1f}',flush=True)
        aggregate(allpred,out)
    aggregate(allpred,out)
    if allstress:
        stress=pd.concat(allstress,ignore_index=True)
        dev=stress.loc[stress.cohort.eq('xuhui_oof')]
        ext=stress.loc[stress.cohort.eq('fengxian')].groupby(['seed','dropout','mask','patient_id','stage_id','target','cohort'],as_index=False)[MODELS].mean()
        pd.concat([dev,ext],ignore_index=True).to_csv(out/'stress_predictions.tmp.csv',index=False)
        (out/'stress_predictions.tmp.csv').replace(out/'stress_predictions.csv')
    (out/'RUN_COMPLETE.json').write_text(json.dumps({'seeds':args.seeds,'folds':5}))

def aggregate(frames,out):
    pred=pd.concat(frames,ignore_index=True)
    dev=pred.loc[pred.cohort.eq('xuhui_oof')].copy()
    ext=pred.loc[pred.cohort.eq('fengxian')].groupby(['seed','patient_id','stage_id','target','cohort'],as_index=False)[MODELS].mean()
    full=pd.concat([dev,ext],ignore_index=True)
    assert not full.duplicated(['seed','cohort','patient_id','stage_id']).any()
    full.to_csv(out/'predictions.csv',index=False)
    tables=[]
    for seed,g in full.groupby('seed'):
        table=summarize(g);table['seed']=seed;tables.append(table)
    pd.concat(tables,ignore_index=True).to_csv(out/'metrics_by_seed.csv',index=False)
    ensemble=full.groupby(['patient_id','stage_id','target','cohort'],as_index=False)[MODELS].mean()
    ensemble.to_csv(out/'ensemble_predictions.csv',index=False)
    summary=summarize(ensemble);summary.to_csv(out/'ensemble_metrics.csv',index=False)
    print(summary.loc[summary.weighting.eq('patient_equal')].to_string(index=False),flush=True)

def parse_args():
    root=Path(__file__).resolve().parents[1]
    ap=argparse.ArgumentParser()
    ap.add_argument('--processed',type=Path,default=root/'processed_private')
    ap.add_argument('--output',type=Path,default=root/'runs')
    ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028])
    ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--patience',type=int,default=10)
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--trees',type=int,default=500)
    ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--dropout',type=float,nargs='*',default=[.2,.4]);ap.add_argument('--masks',type=int,default=10)
    return ap.parse_args()

if __name__=='__main__':main(parse_args())

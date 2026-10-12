"""Additional LR/LightGBM/portable-Raindrop baseline refits on the same outer roster."""
from pathlib import Path
import json,pickle,time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from run_nested import load_data,build_samples,fit_grud,tensors,predict_tensor,metrics
from opal_fusion.data.labor_dataset import LaborPreprocessor
from opal_fusion.baselines.tabular import tabularize_samples
from fusion import patient_weights

def main(args):
    torch.set_num_threads(args.threads)
    p,e,q,folds=load_data(args.processed);external=p.loc[p.eligible&p.site.eq('fengxian'),'patient_id'].to_numpy(int)
    out=args.output;out.mkdir(parents=True,exist_ok=True);frames=[]
    for seed in args.seeds:
        for fold in range(5):
            d=out/f'seed_{seed}_fold_{fold}';d.mkdir(exist_ok=True)
            if (d/'predictions.csv').exists():frames.append(pd.read_csv(d/'predictions.csv'));continue
            tic=time.monotonic();print('BASELINE FIT',seed,fold,flush=True)
            train=folds.loc[folds.fold.ne(fold),'patient_id'].to_numpy(int);valid=folds.loc[folds.fold.eq(fold),'patient_id'].to_numpy(int)
            labels=p.set_index('patient_id').loc[train,'outcome_binary'].to_numpy(int)
            a,b=train_test_split(train,test_size=.2,random_state=seed+fold,stratify=labels)
            earlyprep=LaborPreprocessor.fit(p,e,a)
            _,ep,hist=fit_grud(build_samples(p,e,q,a,earlyprep),earlyprep,seed+fold,args.epochs,args.hidden,
                                build_samples(p,e,q,b,earlyprep),args.patience,kind='raindrop')
            prep=LaborPreprocessor.fit(p,e,train);ss=build_samples(p,e,q,train,prep)
            m=tabularize_samples(ss,prep.n_variables,'current');w=patient_weights(m.metadata.patient_id.to_numpy())
            lr=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=5000))
            lr.fit(m.features,m.target,logisticregression__sample_weight=w)
            lgb=LGBMClassifier(n_estimators=500,learning_rate=.03,num_leaves=15,min_child_samples=15,
                   subsample=.9,colsample_bytree=.9,reg_lambda=1,class_weight='balanced',random_state=seed+fold,n_jobs=args.jobs,verbosity=-1)
            lgb.fit(m.features,m.target,sample_weight=w)
            rd,_,_=fit_grud(ss,prep,seed+fold,ep,args.hidden,kind='raindrop')
            pickle.dump({'lr':lr,'lgb':lgb,'prep':prep},open(d/'models.pkl','wb'))
            torch.save({'state':rd.state_dict(),'hidden':args.hidden},d/'raindrop.pt')
            part=[]
            for site,ids in [('xuhui_oof',valid),('fengxian',external)]:
                samples=build_samples(p,e,q,ids,prep);tab=tabularize_samples(samples,prep.n_variables,'current')
                df=tab.metadata.copy();df['LR_current']=lr.predict_proba(tab.features)[:,1]
                df['LightGBM_current']=lgb.predict_proba(tab.features)[:,1];df['Raindrop_portable']=predict_tensor(rd,tensors(samples))
                df['seed']=seed;df['fold']=fold;df['cohort']=site;part.append(df)
            joined=pd.concat(part,ignore_index=True);joined.to_csv(d/'predictions.csv',index=False);frames.append(joined)
            (d/'selection.json').write_text(json.dumps({'fit_ids':a.tolist(),'selection_ids':b.tolist(),'outer_test_ids':valid.tolist(),'best_epoch':ep,'history':hist},indent=2))
            print('BASELINE DONE',seed,fold,'seconds',round(time.monotonic()-tic,1),flush=True)
            aggregate(frames,out)
    aggregate(frames,out)

def aggregate(frames,out):
    models=['LR_current','LightGBM_current','Raindrop_portable'];d=pd.concat(frames,ignore_index=True)
    ext=d.loc[d.cohort.eq('fengxian')].groupby(['seed','patient_id','stage_id','target','cohort'],as_index=False)[models].mean()
    dev=d.loc[d.cohort.eq('xuhui_oof')];d=pd.concat([dev,ext],ignore_index=True)
    d.to_csv(out/'predictions.csv',index=False)
    rows=[]
    for (seed,site),g in d.groupby(['seed','cohort']):
        for model in models:rows.append({'seed':seed,'cohort':site,'model':model,**metrics(g,model)})
    pd.DataFrame(rows).to_csv(out/'metrics_by_seed.csv',index=False)
    d.groupby(['patient_id','stage_id','target','cohort'],as_index=False)[models].mean().to_csv(out/'ensemble_predictions.csv',index=False)

if __name__=='__main__':
    import argparse
    root=Path(__file__).resolve().parents[1];ap=argparse.ArgumentParser()
    ap.add_argument('--processed',type=Path,default=root/'processed_private');ap.add_argument('--output',type=Path,default=root/'baseline_runs')
    ap.add_argument('--seeds',type=int,nargs='+',default=[2026,2027,2028]);ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--patience',type=int,default=10);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--threads',type=int,default=1);ap.add_argument('--jobs',type=int,default=2)
    main(ap.parse_args())

"""Paired patient-cluster inference for frozen nested predictions."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from fusion import patient_weights
from run_nested import MODELS,metrics

LABELS={'RF_current':'RF current','GRUD':'GRU-D','Fixed_half':'Fixed 50:50',
        'Global_weight':'Learned global weight','OAF_original':'Original OAF',
        'OAF_shrink':'OAF with shrinkage','OAF_reliable_shrink':'OAF with repeat support'}
SITES={'xuhui_oof':'Xuhui OOF','fengxian':'Fengxian exploratory'}

def scores(y,p,w):
    return [roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w),
            brier_score_loss(y,p,sample_weight=w)]

def bootstrap(pred,n,seed):
    absolute=[];paired=[]
    for cohort,g in pred.groupby('cohort'):
        ids=g.patient_id.to_numpy();unique,inv=np.unique(ids,return_inverse=True)
        y=g.target.to_numpy();basew=patient_weights(ids)
        prob=g[MODELS].to_numpy();point=np.array([scores(y,prob[:,j],basew) for j in range(len(MODELS))])
        rng=np.random.default_rng(seed)
        draws=[]
        for boot in range(n):
            sampled=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)))
            w=basew*sampled[inv]
            if len(np.unique(y[w>0]))<2:continue
            draws.append([scores(y,prob[:,j],w) for j in range(len(MODELS))])
        draws=np.asarray(draws)
        for j,model in enumerate(MODELS):
            for k,metric in enumerate(['AUROC','AUPRC','Brier']):
                lo,hi=np.quantile(draws[:,j,k],[.025,.975])
                absolute.append({'cohort':cohort,'model':model,'metric':metric,'estimate':point[j,k],
                                 'ci_low':lo,'ci_high':hi,'bootstrap_valid':len(draws)})
                if j:
                    factor=-1 if metric=='Brier' else 1
                    d=factor*(draws[:,j,k]-draws[:,0,k]);lo,hi=np.quantile(d,[.025,.975])
                    paired.append({'cohort':cohort,'model':model,'reference':'RF_current',
                                   'metric':metric if metric!='Brier' else 'Brier_improvement',
                                   'estimate':factor*(point[j,k]-point[0,k]),'ci_low':lo,'ci_high':hi,
                                   'bootstrap_valid':len(draws)})
        print('BOOTSTRAP',cohort,len(draws),flush=True)
    return pd.DataFrame(absolute),pd.DataFrame(paired)

def figures(absolute,paired,stress,out):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
    fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    for row,site in enumerate(['xuhui_oof','fengxian']):
        for col,metric in enumerate(['AUROC','AUPRC','Brier']):
            d=absolute.loc[absolute.cohort.eq(site)&absolute.metric.eq(metric)].set_index('model').loc[MODELS]
            y=np.arange(len(d));v=d.estimate.to_numpy()
            axes[row,col].errorbar(v,y,xerr=np.vstack([v-d.ci_low.to_numpy(),d.ci_high.to_numpy()-v]),fmt='o',capsize=3)
            axes[row,col].set_yticks(y,[LABELS[n] for n in d.index]);axes[row,col].invert_yaxis()
            axes[row,col].set_title(f'{SITES[site]} — {metric}');axes[row,col].grid(axis='x',alpha=.2)
    fig.savefig(out/'Figure1_nested_performance.png',dpi=300)
    fig.savefig(out/'Figure1_nested_performance.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8,4.5),layout='constrained')
    for ax,site in zip(axes,['xuhui_oof','fengxian']):
        d=paired.loc[paired.cohort.eq(site)&paired.metric.eq('AUPRC')].set_index('model').loc[MODELS[1:]]
        v=d.estimate.to_numpy();y=np.arange(len(d))
        ax.axvline(0,color='gray',lw=1)
        ax.errorbar(v,y,xerr=np.vstack([v-d.ci_low.to_numpy(),d.ci_high.to_numpy()-v]),fmt='o',capsize=3)
        ax.set_yticks(y,[LABELS[n] for n in d.index]);ax.invert_yaxis();ax.set_title(SITES[site]);ax.set_xlabel('AUPRC difference from RF current');ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    fig.savefig(out/'Figure2_paired_AUPRC.png',dpi=300);fig.savefig(out/'Figure2_paired_AUPRC.pdf');plt.close(fig)
    if stress is not None:
        fig,axes=plt.subplots(1,2,figsize=(8,4.5),layout='constrained')
        chosen=['RF_current','GRUD','Global_weight','OAF_original','OAF_shrink','OAF_reliable_shrink']
        for ax,site in zip(axes,['xuhui_oof','fengxian']):
            for model in chosen:
                a=absolute.loc[absolute.cohort.eq(site)&absolute.model.eq(model)&absolute.metric.eq('AUPRC')].estimate.iloc[0]
                d=stress.loc[stress.cohort.eq(site)&stress.model.eq(model)]
                rates=[0.0]+sorted(d.dropout.unique().tolist());means=[a];lows=[a];highs=[a]
                for rate in rates[1:]:
                    vals=d.loc[d.dropout.eq(rate),'AUPRC'];means.append(vals.mean());lows.append(vals.quantile(.025));highs.append(vals.quantile(.975))
                ax.plot(rates,means,'o-',label=LABELS[model]);ax.fill_between(rates,lows,highs,alpha=.1)
            ax.set_title(SITES[site]);ax.set_xlabel('Older-history dropout probability');ax.set_ylabel('Patient-equal AUPRC')
        axes[1].legend(fontsize=9)
        fig.savefig(out/'Figure3_history_dropout.png',dpi=300);fig.savefig(out/'Figure3_history_dropout.pdf');plt.close(fig)

def report(absolute,paired,seedmetrics,out):
    lines=['# OAF 嵌套患者实验结果','',
           '本报告记录实际新拟合结果。旧 Table 5 数值不与本轮数值混用；本轮重新拟合专家并使用患者等权训练与主评价。',
           '', '## 设计','',
           '- 徐汇：736 人、1,378 个 phase-first 查询；奉贤：118 人、218 个查询。',
           '- 五个固定外层患者折。每个外层训练池重新做三折专家交叉拟合；GRU-D 轮数由更内层患者留出选择。',
           '- RF 和 GRU-D 最终在整个外层训练池重拟合；门控使用该池内交叉拟合的专家预测。',
           '- 三个随机种子。奉贤标签不进入训练、早停、门控、超参数选择。',
           '- 收缩强度只由训练池内元模型三折患者分组 Brier 选择。复测支持是四个进展相关变量中有两个不同观测时刻的比例，不能等同于测量质量。',
           '- 下列区间是固定预测下的 10,000 次配对患者簇 bootstrap；种子敏感性单独报告。',
           '', '## 三种子平均预测的患者等权结果','',
           '| Model | Xuhui AUROC | Xuhui AUPRC | Fengxian AUROC | Fengxian AUPRC |',
           '|---|---:|---:|---:|---:|']
    def val(site,model,metric):return absolute.loc[absolute.cohort.eq(site)&absolute.model.eq(model)&absolute.metric.eq(metric)].estimate.iloc[0]
    for model in MODELS:
        vals=[val(site,model,metric) for site in ['xuhui_oof','fengxian'] for metric in ['AUROC','AUPRC']]
        lines.append('| '+LABELS[model]+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    lines+=['','## 相对 RF 的配对 AUPRC 差','', '| Model | Xuhui difference [95% interval] | Fengxian difference [95% interval] |','|---|---:|---:|']
    for model in MODELS[1:]:
        vals=[]
        for site in ['xuhui_oof','fengxian']:
            d=paired.loc[paired.cohort.eq(site)&paired.model.eq(model)&paired.metric.eq('AUPRC')].iloc[0]
            vals.append(f"{d.estimate:+.4f} [{d.ci_low:+.4f}, {d.ci_high:+.4f}]")
        lines.append('| '+LABELS[model]+' | '+' | '.join(vals)+' |')
    lines+=['','## 解释限制','',
            '所有方案在已查看过的同一医院系统第二院区上作探索性评价。重复同一外层分折的三个随机种子不是三个独立验证队列。',
            '历史 dropout 保留每个变量的最新记录，属于有控制的历史删减压力测试，不模拟全部临床复查机制。',
            '当前数据没有干预决策时间戳，静态变量缺少逐字段采集时间；结果不能证明决策前预警能力。',
            'LR、LightGBM、portable Raindrop 另行在相同外层患者及查询集重新拟合，六模型结果见 six_model_metrics.csv；其早停规则与 GRU-D 不完全相同。',
            '拟合和早停引入了与旧基线不同的患者权重与训练量，本轮结论必须来自本轮配对 RF 对照。']
    (out/'EXPERIMENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main(args):
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    pred=pd.read_csv(args.runs/'ensemble_predictions.csv')
    absolute,paired=bootstrap(pred,args.bootstrap,20261001)
    absolute.to_csv(out/'absolute_patient_bootstrap.csv',index=False);paired.to_csv(out/'paired_vs_RF.csv',index=False)
    seeds=pd.read_csv(args.runs/'metrics_by_seed.csv')
    seeds.to_csv(out/'metrics_by_seed.csv',index=False)
    deltas=[]
    for (seed,cohort),g in seeds.loc[seeds.weighting.eq('patient_equal')].groupby(['seed','cohort']):
        rf=g.loc[g.model.eq('RF_current')].iloc[0]
        for row in g.itertuples():
            deltas.append({'seed':seed,'cohort':cohort,'model':row.model,'AUROC_difference':row.AUROC-rf.AUROC,
                           'AUPRC_difference':row.AUPRC-rf.AUPRC,'Brier_improvement':rf.Brier-row.Brier})
    pd.DataFrame(deltas).to_csv(out/'paired_point_differences_by_seed.csv',index=False)
    stresspath=args.runs/'stress_predictions.csv';stressmetrics=None
    if stresspath.exists():
        s=pd.read_csv(stresspath)
        baseline=pd.read_csv(args.runs/'predictions.csv')
        checked=s.merge(baseline[['seed','cohort','patient_id','stage_id','RF_current']],on=['seed','cohort','patient_id','stage_id'],suffixes=('','_baseline'),validate='many_to_one')
        assert np.allclose(checked.RF_current,checked.RF_current_baseline), 'Current RF changed under thinning'
        assert set(s.seed)==set(baseline.seed), 'Incomplete stress seed export'
        assert len(s)==len(baseline)*len(s.dropout.unique())*len(s['mask'].unique()), 'Incomplete stress rows'
        # Average predictions across seeds before summarizing each paired mask.
        s=s.groupby(['cohort','dropout','mask','patient_id','stage_id','target'],as_index=False)[MODELS].mean()
        rows=[]
        for (cohort,dropout,mask),g in s.groupby(['cohort','dropout','mask']):
            for model in MODELS:rows.append({'cohort':cohort,'dropout':dropout,'mask':mask,'model':model,**metrics(g,model)})
        stressmetrics=pd.DataFrame(rows);stressmetrics.to_csv(out/'history_dropout_metrics.csv',index=False)
    figures(absolute,paired,stressmetrics,out);report(absolute,paired,seeds,out)
    print(paired.loc[paired.metric.eq('AUPRC')].to_string(index=False),flush=True)

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser()
    p.add_argument('--runs',type=Path,default=root/'runs');p.add_argument('--output',type=Path,default=root/'results')
    p.add_argument('--bootstrap',type=int,default=10000);main(p.parse_args())

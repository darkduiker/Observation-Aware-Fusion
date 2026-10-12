from pathlib import Path
import sys
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
R=Path(__file__).resolve().parents[1]; F=R/'figures';D=R/'results'
def read(name):
 p=D/name
 if not p.exists():p=R/'private_predictions'/name
 if not p.exists() and name.startswith('common_'):p=R/'common_experiment_verified'/name[7:]
 d=pd.read_csv(p)
 if "cohort" in d:d["cohort"]=d["cohort"].replace({"Site A":"xuhui_oof","Site B":"fengxian"})
 return d
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
 for ext in ['png','pdf','svg']:fig.savefig(F/f'{name}.{ext}',dpi=300,bbox_inches='tight')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5.5));ax.set(xlim=(0,10),ylim=(-.2,6));ax.axis('off')
def box(x,y,w,h,t,c):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',facecolor=c,edgecolor='#34495e'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
def arrow(x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#34495e','lw':1.3})
box(.25,4.7,9.3,.75,'Sparse pre-query record  |  Same patients, phase-first queries and folds','#e8edf2')
box(.3,2.85,4.15,1.2,'Information-source decomposition\nS: current state\nΔ: last-two recorded change\nR: reassessment availability','#e4eef7')
box(5.15,3.45,1.8,.8,'Current RF\nanchor pS','#e4eef7');box(7.55,3.45,1.8,.8,'GRU-D\nhistory pH','#e7f0e5')
box(5.15,1.95,4.2,1,'Observation summaries → gate g\nShrinkage of historical weight\npOAF = pS + g(pH − pS)','#f8edda')
box(.3,.95,4.15,1.1,'Matched RF conditions\nA=S   B=S+R   C=S+Δ   D=S+Δ+R\nE: randomized-change training','#e4eef7')
box(.3,-.0,9.05,.65,'Shared evaluation\nPaired increments · simple blends · cross-campus shifts · history deletion','#e8edf2')
arrow(2.4,4.65,2.4,4.12);arrow(7.4,4.65,7.4,4.3);arrow(2.4,2.8,2.4,2.15);arrow(6,3.4,6.2,3.0);arrow(8.5,3.4,8.2,3.0);arrow(2.4,.9,2.4,.7);arrow(7.25,1.9,7.25,.7)
save(fig,'Figure_integrated_framework')
p=read('common_paired.csv');cs=['B-A','C-A','D-A','E-C','OAF_shrink-A','OAF_shrink-D','OAF_shrink-Global_weight','OAF_shrink-OAF_same6_no_shrink'];labs=['Reassessment RF − current RF','Change RF − current RF','Combined RF − current RF','OPR RF − change RF','OAF − current RF','OAF − combined RF','OAF − global blend','OAF − matched unshrunk gate']
fig,axes=plt.subplots(1,2,figsize=(10,5),sharey=True,layout='constrained')
for ax,site,title in zip(axes,['xuhui_oof','fengxian'],['Site A · OOF','Site B · exploratory']):
 for i,c in enumerate(cs):
  z=p.loc[(p.cohort==site)&(p.contrast==c)&(p.metric=='AUPRC')].iloc[0];ax.errorbar(z.estimate,i,xerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt='o',color='#245a81',capsize=3)
 ax.axvline(0,color='gray',lw=1);ax.set(title=title,xlabel='Paired AUPRC difference');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(cs)),labs);axes[0].invert_yaxis();save(fig,'Figure_common_paired')
d=read('six_model_predictions.csv');models=['LR_current','RF_current','LightGBM_current','GRUD','Raindrop_portable','OAF_shrink'];labels=['LR-current','RF-current','LightGBM-current','GRU-D','Raindrop-portable','OAF']
fig,axes=plt.subplots(2,2,figsize=(8,8),layout='constrained')
for si,site in enumerate(['xuhui_oof','fengxian']):
 g=d[d.cohort==site];w=1/g.groupby('patient_id').patient_id.transform('size');y=g.target
 for m,label in zip(models,labels):
  f,t,_=roc_curve(y,g[m],sample_weight=w);pr,re,_=precision_recall_curve(y,g[m],sample_weight=w)
  axes[si,0].plot(f,t,label=f'{label} ({roc_auc_score(y,g[m],sample_weight=w):.3f})');axes[si,1].plot(re,pr,label=f'{label} ({average_precision_score(y,g[m],sample_weight=w):.3f})')
 for ax in axes[si]:ax.set(title='Site A · OOF' if si==0 else 'Site B · exploratory',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=7.5);ax.grid(alpha=.15)
 axes[si,0].set(xlabel='False positive rate',ylabel='True positive rate');axes[si,1].set(xlabel='Recall',ylabel='Precision')
save(fig,'Figure4_six_model_ROC_PR')
s=read('common_stress_summary.csv');a=read('common_absolute.csv')
fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
for ax,site,title in zip(axes,['xuhui_oof','fengxian'],['Site A · OOF','Site B · exploratory']):
 for m,l in zip(['A','D','Global_weight','OAF_same6_no_shrink','OAF_shrink'],['Current RF','Combined RF','Global blend','Unshrunk gate','OAF']):
  b=a.loc[(a.cohort==site)&(a.model==m)&(a.metric=='AUPRC'),'estimate'].iloc[0];z=s[(s.cohort==site)&(s.model==m)].sort_values('dropout');ax.plot([0]+list(z.dropout),[b]+list(z.AUPRC),marker='o',label=l)
 ax.set(title=title,xlabel='Older-record deletion probability',ylabel='AUPRC');ax.grid(alpha=.2)
axes[1].legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8);save(fig,'Figure_common_dropout')
fig,axes=plt.subplots(1,2,figsize=(8,3.7),layout='constrained')
for ax,site,title in zip(axes,['xuhui_oof','fengxian'],['Site A · OOF','Site B · exploratory']):
 g=d[d.cohort==site].copy();w=1/g.groupby('patient_id').patient_id.transform('size')
 for m,l in zip(['RF_current','GRUD','OAF_shrink'],['Current RF','GRU-D','OAF']):
  bins=pd.qcut(g[m],5,labels=False,duplicates='drop');xp=[];yp=[]
  for b in sorted(bins.unique()):idx=bins==b;xp.append(np.average(g.loc[idx,m],weights=w[idx]));yp.append(np.average(g.loc[idx,'target'],weights=w[idx]))
  ax.plot(xp,yp,'o-',label=l)
 ax.plot([0,1],[0,1],'--',color='gray');ax.set(title=title,xlabel='Mean predicted probability',ylabel='Observed intervention fraction',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
save(fig,'Figure5_calibration')

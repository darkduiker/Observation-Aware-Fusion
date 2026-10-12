from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parents[1];F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
blue='#23668C';teal='#24877B';orange='#B46C2D';gray='#556575'
def save(fig,n):
 for ext in ['png','svg','pdf']:fig.savefig(F/f'{n}.{ext}',dpi=350,bbox_inches='tight',facecolor='white')
 plt.close(fig)
fig,ax=plt.subplots(figsize=(12,7.3));ax.set(xlim=(0,12),ylim=(0,7.3));ax.axis('off')
def box(x,y,w,h,text,fc='#F2F6F9',ec='#AFBDCA',fs=10,weight='normal'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.025,rounding_size=0.08',fc=fc,ec=ec,lw=1));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color='#182D3C',weight=weight)
def arr(x,y,u,v):ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops={'arrowstyle':'-|>','lw':1.35,'color':gray})
for x,w,title in [(0.12,3.46,'A  Sparse assessment history'),(3.78,3.9,'B  Decompose and regulate'),(7.9,3.94,'C  Matched evaluation')]:
 box(x,.95,w,5.95,'',fc='white');box(x,6.42,w,.48,title,fc='#DAEAF4',ec='#DAEAF4',fs=11,weight='bold')
ax.text(1.85,6.06,'Illustrative pre-query record',ha='center',fontsize=10,color=gray)
ax.plot([.45,3.25],[5.48,5.48],color=gray,lw=1.4)
for x,t,vals in [(.72,'Earlier','AoP: 112°\nDilation:\n3 cm\nStation: −2'),(1.65,'Previous','AoP: 124°\nDilation:\n5 cm\nStation: −1'),(3.03,'Query t','AoP: 136°\nDilation:\n8 cm\nStation: 0')]:
 ax.scatter(x,5.48,s=33,color=blue,zorder=3);ax.text(x,5.8,t,ha='center',fontsize=8,weight='bold');box(x-.4,4.12,.8,1.07,vals,fs=7.6);arr(x,5.4,x,5.2)
ax.text(1.85,3.68,'Unequal intervals\nPartial measurements',ha='center',fontsize=9,color=gray)
box(.4,2.69,2.9,.83,'First query within each\ndilation phase\nHistory includes only\nrecords at or before t',fs=9.5)
box(.4,1.32,2.9,1.06,'854 patients\n1,596 shared queries\nSite A: 736 / 1,378\nSite B: 118 / 218',fs=10)
box(4.02,5.52,1.04,.56,'S\nCurrent\nstate',fs=9)
box(5.24,5.52,1.04,.56,'Δ\nLast-two\nchange',fs=9)
box(6.46,5.52,.99,.56,'R\nRepeat\navailability',fs=9)
box(4.02,4.53,3.42,.66,'RF conditions: A=S, B=[S,R]\nC=[S,Δ̃], D=[S,Δ̃,R]; E=OPR training',fs=9.5)
for x in [4.54,5.76,6.96]:arr(x,5.48,x,5.23)
box(4.02,3.43,1.55,.64,'Current RF\nAnchor c',fc='#E8F1F8')
box(5.9,3.43,1.55,.64,'GRU-D\nHistory h',fc='#E5F2EC')
box(4.02,2.16,3.43,.9,'Observation summaries → gate g\nHistorical-weight shrinkage λ\np = c + g(h − c)',fc='#FBF0E2',fs=10)
arr(4.8,3.39,5.1,3.1);arr(6.66,3.39,6.37,3.1)
ax.text(5.75,1.49,'Experts and gate fitted within training folds',ha='center',fontsize=8.8,color=gray)
for y,h,t in [(5.38,.66,'Same queries, encoding\nand patient folds\nPatient-equal scoring; three seeds'),(4.18,.83,'1  Diagnose feature contribution\nA–E paired increments\nObservation process → campus'),(2.9,.89,'2  Evaluate OAF and its controls\nCurrent RF • combined-history RF\nFixed / global blend • unshrunk gate'),(1.42,.98,'3  Test perturbation response\nDelete older records\nPreserve current state\nPatient bootstrap + mask variability')]:box(8.15,y,3.42,h,t,fs=9.5)
for y,z in [(5.35,5.05),(4.15,3.84),(2.87,2.43)]:arr(9.86,y,9.86,z)
box(.12,.12,11.72,.53,'Site A: patient-grouped development     |     Site B: exploratory cross-campus evaluation',fc='#EAF0F4',fs=10.5)
save(fig,'Figure_integrated_framework')
a=pd.read_csv(R/'results/common_absolute.csv');p=pd.read_csv(R/'results/common_paired.csv');gap=pd.read_csv(R/'results/common_cross_campus_gaps.csv');s=pd.read_csv(R/'results/query_time_site_diagnostic.csv')
fig,axs=plt.subplots(2,2,figsize=(12,9.2));fig.subplots_adjust(left=.19,right=.98,bottom=.10,top=.94,wspace=.42,hspace=.58)
# A performance A-E, identical axis and no lines implying time
ax=axs[0,0];xs=np.arange(5)
for site,c,off,label in [('Site A',blue,-.09,'Site A · OOF'),('Site B',orange,.09,'Site B · exploratory')]:
 z=a[(a.cohort==site)&(a.metric=='AUPRC')].set_index('model').loc[list('ABCDE')];ax.plot(xs+off,z.estimate,'o-',color=c,label=label,lw=1.5)
 for x,v in zip(xs+off,z.estimate):ax.text(x,v+(.013 if site=='Site A' else -.022),f'{v:.3f}',ha='center',fontsize=8,color=c)
ax.set_xticks(xs,['A\nCurrent','B\n+R','C\n+Δ̃','D\n+Δ̃+R','E\nOPR']);ax.set(ylim=(.625,.79),ylabel='Patient-equal AUPRC',title='A  Matched RF information conditions');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9,loc='upper left');ax.text(.5,-.25,'Points: three-seed averaged predictions; conditions are not time steps.',transform=ax.transAxes,ha='center',fontsize=8,color=gray)
# B site classifier
ax=axs[0,1]
for learner,off,c in [('LR',-.15,blue),('RF',.15,teal)]:
 z=s[(s.learner==learner)&(s.stratum=='all')].set_index('feature_set').loc[['repeat_only','full_query_observation']]
 ax.errorbar(np.arange(2)+off,z.AUROC,yerr=[z.AUROC-z.ci_low,z.ci_high-z.AUROC],fmt='o',color=c,label=learner,capsize=4,ms=6)
 for x,v in zip(np.arange(2)+off,z.AUROC):ax.text(x,v+.053,f'{v:.3f}',ha='center',fontsize=9,color=c)
ax.axhline(.5,color=gray,ls='--',lw=1);ax.set(xticks=[0,1],xticklabels=['R only','Full observation\nprocess'],xlim=(-.4,1.4),ylim=(.48,.87),ylabel='Campus-classification AUROC',title='B  Observation-process separability');ax.legend(fontsize=9,loc='upper left');ax.grid(axis='y',alpha=.2)
# C gaps forest includes relevant 5contrasts
ax=axs[1,0];cs=['B-A','C-A','D-C','D-A','E-C'];labs=['B − A  Reassessment','C − A  Recorded change','D − C  R given change','D − A  Combined history','E − C  OPR training']
for i,c in enumerate(cs):
 z=gap[(gap.contrast==c)&(gap.metric=='AUPRC')].iloc[0];ax.errorbar(z.gap,i,xerr=[[z.gap-z.ci_low],[z.ci_high-z.gap]],fmt='o',color=blue,capsize=3)
ax.axvline(0,color=gray,lw=1);ax.set_yticks(range(5),labs);ax.invert_yaxis();ax.set(xlim=(-.14,.08),xlabel='G = Site B increment − Site A increment',title='C  Cross-campus increment change');ax.grid(axis='x',alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(5))
# D fusion paired
ax=axs[1,1];cs=['OAF_shrink-A','OAF_shrink-D','OAF_shrink-Global_weight','OAF_shrink-OAF_same6_no_shrink'];labels=['OAF − current RF','OAF − combined RF','OAF − global blend','OAF − unshrunk gate']
for site,off,c,label in [('Site A',-.12,blue,'Site A'),('Site B',.12,orange,'Site B')]:
 for i,k in enumerate(cs):
  z=p[(p.cohort==site)&(p.contrast==k)&(p.metric=='AUPRC')].iloc[0];ax.errorbar(z.estimate,i+off,xerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt='o',color=c,capsize=3,label=label if i==0 else None)
ax.axvline(0,color=gray,lw=1);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set(xlim=(-.035,.16),xlabel='Paired AUPRC difference',title='D  Regulated fusion versus controls');ax.legend(fontsize=8,loc='lower right');ax.grid(axis='x',alpha=.2);ax.xaxis.set_major_locator(MaxNLocator(5))
fig.text(.5,.025,'Panels B–D: 95% patient-cluster bootstrap intervals on fixed predictions. Site B is exploratory.',ha='center',fontsize=9,color=gray)
save(fig,'Figure_common_paired')
print('Saved 2 figures in PNG/PDF/SVG; all points and intervals read directly from result CSVs.')

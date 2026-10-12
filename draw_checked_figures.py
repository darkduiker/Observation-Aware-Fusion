from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,Circle
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import pandas as pd,numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8.2,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
INK='#203342';MUTED='#647786';BLUE='#23749B';TEAL='#228976';GOLD='#C4823E';LINE='#CAD5DD';PALE='#F3F6F8'
def save(f,n):
 for ext in ['png','svg','pdf']:f.savefig(OUT/f'{n}.{ext}',dpi=600,bbox_inches='tight',pad_inches=.10)
 plt.close(f)
f,ax=plt.subplots(figsize=(7.4,4.65));ax.set(xlim=(0,100),ylim=(0,62));ax.axis('off')
def text(x,y,t,size=8.5,c=INK,ha='left',va='center',weight='normal'):
 return ax.text(x,y,t,fontsize=size,color=c,ha=ha,va=va,weight=weight)
def card(x,y,w,h,t='',fc='white',ec=LINE,size=8.2):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.16,rounding_size=0.9',fc=fc,ec=ec,lw=.65));text(x+w/2,y+h/2,t,size,ha='center')
def arrow(x,y,u,v,c=MUTED,lw=.9,ls='-'):
 ax.annotate('',xy=(u,v),xytext=(x,y),arrowprops=dict(arrowstyle='-|>',mutation_scale=8,lw=lw,color=c,linestyle=ls,shrinkA=1,shrinkB=1))
def panel(x,w,k,title):
 ax.add_patch(Rectangle((x,57.8),w,3.6,fc=PALE,ec='none'));text(x+.8,59.6,k,10,BLUE,weight='bold');text(x+4,59.6,title,8.7,weight='bold')
panel(0,29,'a','Sparse records');panel(32,37,'b','Decompose and regulate');panel(72,28,'c','Shared evaluation')
# clinical acquisition matrix
text(1,54.4,'History available at query t',8.2,c=MUTED)
times=[10.3,15.2,24.6];yrows=[48.8,44.6,40.4,36.2];labels=['AoP','HPD','Dilation','Position']
for y,label,pres in zip(yrows,labels,[[1,1,1],[1,0,1],[1,1,1],[0,1,1]]):
 text(.8,y,label,7.6)
 ax.plot([9,26.5],[y,y],color=LINE,lw=.6)
 for x,yes in zip(times,pres):
  if yes:ax.scatter(x,y,s=24,fc=BLUE,ec='white',lw=.6,zorder=3)
  else:ax.scatter(x,y,s=20,fc='white',ec=LINE,lw=.6,zorder=3)
ax.plot([26.5,26.5],[33.7,51.4],ls=(0,(3,2)),color=GOLD,lw=1)
for x,l in zip(times,[r'$t_1$',r'$t_2$',r'$t_3$']):text(x,52.2,l,6.8,c=MUTED,ha='center')
arrow(9,33.7,28.1,33.7);text(27.9,32,'t',8,c=GOLD,ha='center')
text(14,30.4,'Unequal intervals\nPartial records',7.1,c=MUTED,ha='center')
card(.6,18.5,27.5,8.1,'First eligible query\nwithin each dilation phase',fc=PALE,size=8)
text(1,14.5,'Common query roster',8.2,weight='bold')
text(1,10.9,'854 patients  ·  1,596 queries',8.2)
# B representation
for x,s,lab,fc in [(33,'S','Current\nstate','#E8F1F7'),(45.3,'Δ','Last-two\nchange','#E6F3EE'),(57.6,'R','Repeat\navailability','#FCF0E1')]:
 card(x,46.6,10.5,8.7,fc=fc);text(x+5.25,52.8,s,11,ha='center',weight='bold');text(x+5.25,49.1,lab,7.5,ha='center');arrow(x+5.25,46.2,x+5.25,43.9)
text(33,42.7,'Matched RF conditions',7.9,weight='bold')
text(33,39.6,r'$A=S\quad B=[S,R]$',8.2)
text(33,36.5,r'$C=[S,\widetilde{\Delta}]\quad D=[S,\widetilde{\Delta},R]$',8.2)
text(33,33.4,'E: randomized-change training',7.2,c=MUTED)
ax.plot([33,68],[30.6,30.6],color=LINE,lw=.6)
text(33,28.3,'Observation-aware fusion',8.3,weight='bold')
card(33,18.6,14.1,6.4,'Current RF\nAnchor c',fc='#E8F1F7',size=8)
card(53.9,18.6,14.1,6.4,'GRU-D\nHistory h',fc='#E6F3EE',size=8)
arrow(40,18.2,46,12.4);arrow(60.8,18.2,55,12.4)
card(35.2,4.9,30.7,7.2,fc='#FCF0E1')
text(50.5,9.6,r'$p=c+g(h-c)$',10,ha='center')
text(50.5,6.7,'Observation gate g + shrinkage λ',7.0,ha='center')
# C evaluation minimalist numbered blocks, no enclosed panel
for y,num,title,sub in [(51.5,'1','Diagnose','A–E paired increments\nCampus predictability'),(36,'2','Compare','RF · temporal models · OAF\nBlends · unshrunk gate'),(20.4,'3','Perturb','Delete older records\nPreserve current state')]:
 ax.add_patch(Circle((74.2,y),1.6,fc='#E8F1F7',ec='none'));text(74.2,y,num,8,BLUE,ha='center',weight='bold');text(77,y,title,8.5,weight='bold');text(77,y-4.8,sub,7.4,c=MUTED)
text(72.8,5.8,'Patient-equal scores\nPaired patient bootstrap',7.3,c=MUTED)
# restrained campus strip
ax.plot([0,100],[1.3,1.3],color=LINE,lw=.65)
text(0,-1.7,'Site A  736 patients / 1,378 queries',7.6,c=BLUE)
text(100,-1.7,'Site B  118 patients / 218 queries',7.6,c=GOLD,ha='right')
text(0,-4.7,'Patient-grouped development',7,c=MUTED);text(100,-4.7,'Exploratory cross-campus evaluation',7,c=MUTED,ha='right')
ax.set_ylim(-6.2,62)
save(f,'Figure_integrated_framework')
# result figure compact, aligned, differentiated shapes
A=pd.read_csv(ROOT/'figure_data/common_absolute.csv');P=pd.read_csv(ROOT/'figure_data/common_paired.csv');G=pd.read_csv(ROOT/'figure_data/common_cross_campus_gaps.csv');S=pd.read_csv(ROOT/'figure_data/query_time_site_diagnostic.csv')
f=plt.figure(figsize=(7.4,6.25));gs=f.add_gridspec(2,2,left=.19,right=.98,bottom=.10,top=.90,wspace=.55,hspace=.48);axes=[f.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
def title(a,letter,t):
 a.text(-.02,1.095,letter,transform=a.transAxes,fontsize=11,weight='bold',color=BLUE,va='bottom');a.text(.075,1.095,t,transform=a.transAxes,fontsize=8.8,weight='bold',color=INK,va='bottom')
def grid(a,axis='x'):
 a.grid(axis=axis,color='#E9EEF1',lw=.6);a.set_axisbelow(True);a.tick_params(length=3,color=MUTED);a.spines['left'].set_color(LINE);a.spines['bottom'].set_color(LINE)
# A dumbbell, no artificial condition trajectory
ax=axes[0];labs=['A  Current','B  +Reassessment','C  +Change','D  +Both','E  OPR training']
for i,m in enumerate('ABCDE'):
 vals=[A[(A.cohort==site)&(A.model==m)&(A.metric=='AUPRC')].estimate.iloc[0] for site in ['Site A','Site B']]
 ax.plot(vals,[i,i],color=LINE,lw=1.6,zorder=1)
 for v,c,mk in zip(vals,[BLUE,GOLD],['o','s']):ax.scatter(v,i,s=23,c=c,marker=mk,zorder=3,edgecolors='white',linewidths=.5)
ax.set(yticks=np.arange(5),yticklabels=labs,xlim=(.64,.78),ylim=(4.55,-.55),xlabel='Patient-equal AUPRC');ax.xaxis.set_major_locator(MultipleLocator(.04));grid(ax);title(ax,'a','Matched RF performance')
for c,mk,l in [(BLUE,'o','Site A · OOF'),(GOLD,'s','Site B · exploratory')]:ax.plot([],[],mk,color=c,label=l,ms=4)
f.legend(*ax.get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.59,1.0),frameon=False,fontsize=8,ncol=2,handletextpad=.5,columnspacing=2.5)
# B separated point interval
ax=axes[1]
for learner,off,c,mk in [('LR',-.13,BLUE,'o'),('RF',.13,TEAL,'D')]:
 for i,typ in enumerate(['repeat_only','full_query_observation']):
  z=S[(S.learner==learner)&(S.stratum=='all')&(S.feature_set==typ)].iloc[0]
  ax.errorbar(i+off,z.AUROC,yerr=[[z.AUROC-z.ci_low],[z.ci_high-z.AUROC]],fmt=mk,color=c,ms=4,lw=1.2,capsize=2.5,label=learner if i==0 else None)
  ax.text(i+off,z.ci_high+.023,f'{z.AUROC:.3f}',ha='center',fontsize=7.5,color=c)
ax.axhline(.5,color=MUTED,lw=.75,ls=(0,(3,3)));ax.set(xlim=(-.45,1.45),ylim=(.48,.86),xticks=[0,1],xticklabels=['R only','Full process'],ylabel='Campus AUROC');ax.yaxis.set_major_locator(MultipleLocator(.1));grid(ax,'y');title(ax,'b','Campus separability');ax.legend(loc='upper left',frameon=False,fontsize=7,ncol=2,handletextpad=.3,columnspacing=.8)
# C effect intervals
ax=axes[2];cs=['B-A','C-A','D-C','D-A','E-C'];labs=['B − A  Reassessment','C − A  Change','D − C  R | change','D − A  Both','E − C  OPR']
for i,k in enumerate(cs):
 z=G[(G.contrast==k)&(G.metric=='AUPRC')].iloc[0]
 ax.errorbar(z.gap,i,xerr=[[z.gap-z.ci_low],[z.ci_high-z.gap]],fmt='o',c=BLUE,ms=4,lw=1.2,capsize=2.5)
ax.axvline(0,c=MUTED,lw=.8);ax.set(yticks=np.arange(5),yticklabels=labs,ylim=(4.55,-.55),xlim=(-.13,.035),xlabel='Cross-campus increment gap G');ax.xaxis.set_major_locator(MultipleLocator(.05));ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));grid(ax);title(ax,'c','Increment stability')
# D comparisons title carries common OAF minus
ax=axes[3];cs=['OAF_shrink-A','OAF_shrink-D','OAF_shrink-Global_weight','OAF_shrink-OAF_same6_no_shrink'];labs=['Current RF','Combined RF','Global blend','Unshrunk gate']
for site,off,col,mk in [('Site A',-.14,BLUE,'o'),('Site B',.14,GOLD,'s')]:
 for i,k in enumerate(cs):
  z=P[(P.cohort==site)&(P.contrast==k)&(P.metric=='AUPRC')].iloc[0]
  ax.errorbar(z.estimate,i+off,xerr=[[z.estimate-z.ci_low],[z.ci_high-z.estimate]],fmt=mk,c=col,ms=4,lw=1.2,capsize=2.3)
ax.axvline(0,c=MUTED,lw=.8);ax.set(yticks=np.arange(4),yticklabels=labs,ylim=(3.6,-.6),xlim=(-.03,.16),xlabel='OAF − comparator AUPRC');ax.xaxis.set_major_locator(MultipleLocator(.05));ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));grid(ax);title(ax,'d','Fusion controls')
f.text(.19,.026,'Intervals: 95% patient-cluster bootstrap, conditional on fitted predictions.',fontsize=7,c=MUTED)
save(f,'Figure_common_paired')
print('Generated Figure 1 and Figure 2: 600-dpi PNG, vector PDF and editable SVG.')

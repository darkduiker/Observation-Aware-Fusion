from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
r=Path(__file__).resolve().parents[1]/'results';r.mkdir(exist_ok=True)
f,a=plt.subplots(figsize=(10,6.5));a.set(xlim=(0,10),ylim=(0,6.5));a.axis('off')
def box(x,y,w,h,s,c='#edf3f8'):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',facecolor=c,edgecolor='#476579',linewidth=1.1));a.text(x+w/2,y+h/2,s,ha='center',va='center',fontsize=10)
def arrow(x,y,u,v):a.add_patch(FancyArrowPatch((x,y),(u,v),arrowstyle='-|>',mutation_scale=12,color='#476579'))
a.text(.1,6.2,'A  Query-time score fusion',fontsize=13,weight='bold')
box(.15,4.55,2.25,.9,'Latest state\nNominal position channels');box(.15,3.2,2.25,.9,'Pre-query history\nValues, masks, time gaps');box(3,4.6,1.7,.8,'RF expert\nCurrent score c');box(3,3.25,1.7,.8,'GRU-D expert\nHistory score h');box(5.3,4,2,.8,'Convex score blend\np = c + g(h − c)','#e9f2eb');box(7.9,4,1.8,.8,'Intervention\nprobability');arrow(2.46,5,2.95,5);arrow(2.46,3.65,2.95,3.65);arrow(4.75,5,5.28,4.5);arrow(4.75,3.65,5.28,4.3);arrow(7.37,4.4,7.85,4.4)
box(4.8,2.55,2.9,.8,'Observation summaries\nSix features, 14 original variables');arrow(6.25,3.4,6.25,3.94);a.text(7.9,2.95,'Shrinkage penalizes g²\nTraining-only selection',fontsize=10,ha='left',va='center')
a.text(.1,2.1,'B  Fitting within each outer patient fold',fontsize=13,weight='bold')
box(.15,.7,2.4,.9,'Outer training patients\nThree inner expert folds');box(3.15,.7,2.9,.9,'Cross-fitted expert scores\nPatient-grouped gate selection');box(6.65,.7,3.05,.9,'Refit experts and apply gate\nOuter test / exploratory campus');arrow(2.61,1.15,3.09,1.15);arrow(6.11,1.15,6.59,1.15)
a.text(.2,.25,'No outer-test or Fengxian outcome enters fitting or gate selection.',fontsize=10,color='#344e60')
f.savefig(r/'Figure0_method.png',dpi=300,bbox_inches='tight');f.savefig(r/'Figure0_method.pdf',bbox_inches='tight');f.savefig(r/'Figure0_method.svg',bbox_inches='tight');plt.close(f)

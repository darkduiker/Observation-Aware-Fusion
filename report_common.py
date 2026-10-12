from pathlib import Path
import pandas as pd,numpy as np
R=Path(__file__).resolve().parents[1];O=R/'common_experiment_verified'
a=pd.read_csv(O/'absolute.csv');p=pd.read_csv(O/'paired.csv');g=pd.read_csv(O/'cross_campus_gaps.csv');s=pd.read_csv(O/'stress_summary.csv');r=pd.read_csv(O/'paired_probability_response.csv')
def row(t,site,contrast,metric='AUPRC'):return t.loc[t.cohort.eq(site)&t.contrast.eq(contrast)&t.metric.eq(metric)].iloc[0]
def f(v):return f'{v:+.4f}'
def ci(z):return f'{f(z.estimate)} [{f(z.ci_low)}, {f(z.ci_high)}]'
def tab(cols,rows):return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(map(str,z))+' |' for z in rows])
point=a.pivot(index=['cohort','model'],columns='metric',values='estimate');rows=[]
for m in ['A','B','C','D','E','Fixed_half','Global_weight','OAF_same6_no_shrink','OAF_shrink']:
 aa=point.loc[('xuhui_oof',m)];bb=point.loc[('fengxian',m)];rows.append([m,f'{aa.AUROC:.4f}',f'{aa.AUPRC:.4f}',f'{bb.AUROC:.4f}',f'{bb.AUPRC:.4f}'])
T1=tab(['模型/条件','徐汇 AUROC','徐汇 AUPRC','奉贤 AUROC','奉贤 AUPRC'],rows)
contrasts=['B-A','C-A','D-A','E-C','OAF_shrink-A','OAF_shrink-D','OAF_shrink-Fixed_half','OAF_shrink-Global_weight','OAF_shrink-OAF_same6_no_shrink']
T2=tab(['AUPRC 配对比较','徐汇差值 [95%区间]','奉贤差值 [95%区间]'],[[c,ci(row(p,'xuhui_oof',c)),ci(row(p,'fengxian',c))] for c in contrasts])
G=tab(['比较','徐汇增量','奉贤增量','跨院区差值 [95%区间]'],[[z.contrast,f(z.siteA_increment),f(z.siteB_increment),f'{f(z.gap)} [{f(z.ci_low)}, {f(z.ci_high)}]'] for z in g.loc[g.metric.eq('AUPRC')&g.contrast.isin(['B-A','C-A','D-A','OAF_shrink-A','Global_weight-A'])].itertuples()])
Rob=tab(['院区','控制模型','控制概率响应','OAF概率响应','响应减少 [95%区间]'],[[z.cohort,z.control,f'{z.control_probability_response:.5f}',f'{z.OAF_probability_response:.5f}',f'{z.response_attenuation:.5f} [{z.ci_low:.5f}, {z.ci_high:.5f}]'] for z in r.loc[r.dropout.eq(.4)].itertuples()])
text='''# 同口径共同实验结果

本次是 v36 问题与 v37 方法的直接桥接，不是将旧全地标结果和新阶段首次结果相减。患者854人；同一1596个阶段首次查询；同一五外层折、三个种子、名义胎方位编码、91维当前态、患者等权评分。A的逐条概率与v37 RF完全相同。所有新RF条件克隆该折RF的全部参数和随机种子。

## 实验定义

A=S；B=S+R；C=S+中位数填补变化；D=S+变化+R；E与C推理输入相同，但训练加入0/10/20/30/40%随机变化遮蔽的五份等权副本。变化是九个数值变量在查询前最后两次不同时间的记录差值；不对胎方位做减法。缺失变化的中位数仅由该折训练患者计算。S已经包含当前可得性和recency；分解并不实现生理与观察机制的因果独立。

九变量：AoP、宫口扩张、宫颈长度、颏胸角、胎头方向角、胎头先露、HPD、中线角、脊柱枕骨角。保留宽变量集是为桥接原分解，不能把这些差值全部解释成真实产程进展。

普通融合包含固定平均和训练内拟合的全局权重。另加入“相同六个输入、不收缩”的门控对照，它与主OAF使用同一内层交叉拟合概率、门控标准化、L2惩罚，唯一取消lambda*g²收缩项。旧四输入门控仍保留在原始CSV中，但不用于单独归因收缩效果。

## 完整点估计

'''+T1+'\n\n## 配对不确定性\n\n'+T2+'\n\n固定预测下5000次患者簇bootstrap，区间不包含模型重训练不确定性，也未作多重比较调整。Brier与共同患病率标准化AP见CSV。\n\n## 描述性跨院区增量差\n\n'+G+'\n\n上述gap是两院区预测增量之差，受拟合模型组合、病例构成和AUPRC患病率影响；不是因果效应，也不能把gap接近零直接当成问题已解决。\n\n## 旧历史减少时的预测响应\n\n'+Rob+'\n\n这里“概率响应”指保留当前值、随机删除40%旧记录后，最终概率变化绝对值的患者等权平均。先对十个固定配对mask求平均，再患者簇bootstrap。正响应减少支持对这类删除更不敏感，不能证明对真实院区观察策略不变或更准确。A的输入和预测在所有mask下精确不变。各模型AUPRC变化、mask范围、Brier变化分别见stress_summary.csv和paired_probability_response.csv。\n\n## 结论判定规则\n\n两院区OAF相对A点估计均为正，仅表示收益方向保持；若区间含零，不写稳定显著提高。OAF相对简单融合的区间不支持优势，就不能把融合收益归因新增模块。如果C/D在当前口径没有复现原来的跨院区负增量，更不能把OAF的更高分数当成修复旧问题。概率响应减小与预测风险改善是不同结论。\n\n本实验不使用奉贤标签选择新模型、门控或mask；奉贤此前已被查看过，因此仍为探索性评价。源患者级字段无逐字段采集时间，这个局限没有被本实验消除。\n'
interpret=[]
for site,title in [('xuhui_oof','徐汇'),('fengxian','奉贤')]:
 z=row(p,site,'OAF_shrink-A');interpret.append(title+' OAF相对RF的AUPRC增量：'+ci(z)+'；'+('区间包含零。' if z.ci_low<=0<=z.ci_high else '条件区间不包含零，但属于探索性比较。'))
 z=row(p,site,'OAF_shrink-OAF_same6_no_shrink');interpret.append(title+' 收缩相对同六输入不收缩门控：'+ci(z)+'。')
 c=row(p,site,'C-A');dd=row(p,site,'D-A');interpret.append(title+' 变化单独增量'+ci(c)+'，变化与再次评估合并增量'+ci(dd)+'。')
rr=r.loc[r.dropout.eq(.4)&r.control.eq('OAF_same6_no_shrink')]
if (rr.ci_low>0).all():interpret.append('相同六输入下，收缩门控在两院区都减小了40%旧记录删除的概率响应；这是对指定扰动的敏感性缓解证据，不等于临床观察策略不变或域泛化已解决。')
else:interpret.append('对指定旧记录删除，尚未得到两院区一致的概率响应减小区间证据。')
text=text.replace('## 完整点估计','## 直接判定\n\n'+'\n\n'.join(interpret)+'\n\n## 完整点估计')
(O/'共同实验结果.md').write_text(text)

print(T1);print(T2);print(Rob)

insert="""## Matched bridge experiment

We evaluated the original information-source contrasts and fusion models on the corrected phase-first roster. All conditions used the same 854 patients, 1,596 queries, outer patient folds, nominal fetal-position encoding, current-state representation, and patient-equal scoring. The newly fitted representation conditions cloned every parameter, including the random seed, of the saved current-state RF in each fold. The current-state probabilities were verified to equal the saved anchor exactly. Last-two changes were defined over distinct pre-query times for nine numeric variables; missing changes used training-pool medians. Reassessment availability marked whether such a change could be observed. OPR used five training copies at removal rates 0%, 10%, 20%, 30%, and 40%, each with one fifth of the patient-query weight.

To isolate the shrinkage contribution, we additionally fitted an unpenalized historical-weight gate using the identical six observation features, expert cross-fitted scores, standardization, and coefficient regularization as OAF. Only the historical-weight penalty was removed. We did not use the earlier four-feature gate as an isolated ablation of shrinkage. Paired patient bootstrap intervals used 5,000 draws conditional on the saved predictions. Cross-campus increment differences remain descriptive rather than causal transportability effects.

"""
for site,title in [('xuhui_oof','Xuhui'),('fengxian','Fengxian')]:
 c=row(p,site,'C-A');d=row(p,site,'D-A');o=row(p,site,'OAF_shrink-A');h=row(p,site,'OAF_shrink-D');no=row(p,site,'OAF_shrink-OAF_same6_no_shrink')
 insert+=f'In {title}, recorded-change and combined-history AUPRC increments were {ci(c)} and {ci(d)}, respectively. The OAF increment over the current-state RF was {ci(o)}; the difference versus combined-history RF was {ci(h)}. Shrinkage versus the matched six-feature gate changed AUPRC by {ci(no)}.\n\n'
for z in rr.itertuples():
 insert+=f'Under 40% removal of older observations, the patient-equal mean absolute probability response was {z.OAF_probability_response:.5f} for OAF and {z.control_probability_response:.5f} for its matched unshrunk gate in {z.cohort}; response attenuation was {z.response_attenuation:.5f} ({z.ci_low:.5f} to {z.ci_high:.5f}). These response intervals condition on ten fixed masks and the fitted models.\n\n'
insert+='The matched analysis supported lower probability sensitivity to removal of older records for shrinkage fusion than for the otherwise matched gate. Discriminative benefits require separate paired evidence. A global blend had still smaller probability responses in both campuses, so sensitivity attenuation is not a unique advantage of OAF. Positive OAF point increments at both campuses do not establish superiority to simple blending, observation-policy invariance, or elimination of the cross-campus problem. The numeric-change models and score-fusion models encode history differently; the resulting comparison evaluates complete prediction procedures, not a causal intervention on one isolated information source.\n'
(O/'Common_experiment_manuscript_insert.md').write_text(insert)

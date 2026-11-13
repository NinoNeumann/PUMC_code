"""Conditional patient inference and reports for the finite inference-only study."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
def holm(p):
    p=np.asarray(p);ix=np.argsort(p);out=np.empty(len(p));out[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))));return out
def pvalue(q):
    q=q.copy();q['p_two_sided_conditional_normal']=q.apply(lambda r:float(2*norm.sf(abs(r.estimate/r.SE))) if r.SE>0 else (1. if r.estimate==0 else 0.),axis=1);return q
def independent_C(q):
    per=[];raw=[]
    for j in range(1,21):
        numerator=0.;denominator=0
        for f in range(1,6):
            a=q[q.fold.eq(f)&q.imputation.eq(j)];assert len(a)==228;t=a.Survival_Days.to_numpy();e=a.Death.to_numpy(bool);v=a.score.to_numpy();i,k=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
            n=float(((v[i]>v[k])+.5*(v[i]==v[k])).sum());numerator+=n;denominator+=len(i);raw.append(n/len(i))
        per.append(numerator/denominator)
    return float(np.mean(per)),float(np.mean(raw))
def reference_frames(s):
    b=s.b
    def collect(root,stem):return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in root.glob(f'outer/fold*/shared/mi*/{stem}.csv')],ignore_index=True)
    refs={name:pd.read_csv(b.io.source.r.old.REF/f'predictions_{name}.csv',float_precision='round_trip') for name in ['ordinary_cox','RCS_ridge','original_CACox']}
    refs['C_J_SP_I20']=collect(b.source17.OUT,'C_J_SP_I20')
    for name in b.NAM:refs['Native20_'+name+'_I20']=collect(b.source20.OUT,name+'_I20')
    refs['Previous20_S_I20']=collect(b.source20.OUT,'S_I20');refs['Previous19_S_I20']=collect(s.ROOT/'results/cacox_numeric_embedding/20261031_v1','S_I20')
    refs['C_J_S1_a1_I20']=collect(s.ROOT/'results/cacox_prediction_anchor/20261031_v1','C_J_S1_a1_I20')
    refs['P_C_I20']=collect(s.ROOT/'results/cacox_shared_rank_objective/20261031_v1','P_C_I20')
    refs['I20_previous']=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in b.io.source.r.prior.OUT.glob('outer/fold*/mi*/I20_selection.csv')],ignore_index=True)
    return refs
def report_all(s):
    out,b,m,ev=s.OUT,s.b,s.m,s.ev
    frames={r:pd.concat([pd.read_csv(p,float_precision='round_trip') for p in out.glob(f'outer/fold*/shared/mi*/{r}.csv')],ignore_index=True) for r in s.ROLES};refs=reference_frames(s);frames.update(refs)
    summary=dict(at=m.stamp(),models={},comparisons={},interval_scope=ev.SCOPE);tables={};details={};metrics=[];ones=[];folds=[];influences={}
    for role,q in frames.items():
        b.io.source.r.old.check(q);m.assert_paired(frames['RCS_ridge'],q);table,IF=ev.evaluate(q);one=m.individual_cindices(q);assert len(one)==100
        C,mean=independent_C(q);a=table[table.metric.eq('C_index')].iloc[0];assert abs(C-a.estimate)<1e-12 and abs(mean-one.C_index.mean())<1e-12
        summary['models'][role]=dict(C_index=C,CI95=[float(a.lower),float(a.upper)],mean_100=mean,range_100=[float(one.C_index.min()),float(one.C_index.max())]);tables[role]=table;details[role]=IF
        metrics.append(table.assign(model_id=role));ones.append(one.assign(model_id=role));folds.append(one.groupby('fold').C_index.agg(['mean','min','max']).reset_index().assign(model_id=role))
        for metric,v in IF.items():
            for key,value in v.items():influences[role+'__'+metric+'__'+key]=value
        s.rg.record(out/'monitor',False);s.status(status='running',phase='metrics_and_report',models_evaluated=len(tables),progress=s.progress())
    metrics=pd.concat(metrics);one=pd.concat(ones);metrics.to_csv(out/'metrics_pooled.csv',index=False);one.to_csv(out/'model_cindex.csv',index=False);pd.concat(folds).to_csv(out/'fold_cindex.csv',index=False);np.savez_compressed(out/'conditional_influences.npz',**influences)
    pair=[]
    references=['ordinary_cox','RCS_ridge','C_J_SP_I20','Previous20_S_I20','Previous19_S_I20','C_J_S1_a1_I20','P_C_I20','I20_previous',*['Native20_'+name+'_I20' for name in b.NAM]]
    for role in s.ROLES:
        summary['comparisons'][role]={}
        for ref in references:
            if role==b.CONTROL and ref=='C_J_SP_I20':continue
            q=pvalue(ev.paired(tables[ref],details[ref],tables[role],details[role]));pair.append(q.assign(model_id=role,reference_id=ref));summary['comparisons'][role][ref]=q[q.metric.eq('C_index')].iloc[0].to_dict()
    pair=pd.concat(pair,ignore_index=True);primary=pair.reference_id.isin(['ordinary_cox','RCS_ridge'])&pair.metric.eq('C_index');assert primary.sum()==36
    pair['p_Holm_primary_C_family36']=np.nan;pair.loc[primary,'p_Holm_primary_C_family36']=holm(pair.loc[primary,'p_two_sided_conditional_normal']);pair.to_csv(out/'paired_differences.csv',index=False)
    matched=[]
    for c in b.CONDITIONS:
        if c['weight17'] not in [.25,.5,.75]:continue
        for ref in [b.CONTROL,c['source']+'_w000']:
            q=pvalue(ev.paired(tables[ref],details[ref],tables[c['candidate']],details[c['candidate']]));q=q[q.metric.eq('C_index')].copy();q['alternative']=c['candidate'];q['reference']=ref;matched.append(q)
    contrast=pd.concat(matched,ignore_index=True);assert len(contrast)==24;contrast['p_Holm_matched_mixtures24']=holm(contrast.p_two_sided_conditional_normal);contrast.to_csv(out/'procedure_contrasts.csv',index=False)
    general=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in out.glob('outer/fold*/shared/mi*/generalization.csv')],ignore_index=True);assert len(general)==1800;general['gap']=general.train_C-general.test_C;general.to_csv(out/'generalization_each_model.csv',index=False);general.groupby(['role','fold'])[['train_C','test_C','gap']].mean().reset_index().to_csv(out/'generalization_by_fold.csv',index=False);summary['generalization']=general.groupby('role')[['train_C','test_C','gap']].mean().to_dict('index')
    comp=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in out.glob('inner/fold*/inner*/complementarity.csv')],ignore_index=True);assert len(comp)==2000;comp.to_csv(out/'complementarity_internal.csv',index=False);summary['complementarity']=comp.groupby('source')[['validation_spearman','validation_pair_disagreement']].mean().to_dict('index')
    frozen=s.read(out/'FROZEN_CHOICES.json');assert s.freeze(False)==frozen['choices'];summary['choices']=frozen['choices'];pd.DataFrame([dict(fold=f,**v['condition'],inner_C=v['inner_C']) for f,v in frozen['choices'].items()]).to_csv(out/'selection_by_fold.csv',index=False)
    resources=pd.DataFrame([json.loads(line) for line in (out/'monitor/resources.jsonl').read_text().splitlines()]);summary['resources']=dict(max_physical_used_GiB=float(resources.physical_used_GiB.max()),max_owned_RSS_GiB=float(resources.owned_RSS_GiB.max()),minimum_available_GiB=float(resources.available_GiB.min()),median_sampled_system_GPU=float(resources.GPU_utilization_percent.median()),peak_system_GPU=float(resources.GPU_utilization_percent.max()),peak_system_GPU_MiB=float(resources.GPU_used_MiB.max()))
    assert summary['resources']['max_physical_used_GiB']<=s.rg.LIMIT/s.rg.GiB and summary['resources']['max_owned_RSS_GiB']<=8 and summary['resources']['minimum_available_GiB']>=4
    assert s.progress()==dict(inner_groups=25,outer_groups=5,outer_MI_units=100)
    summary['previous_protected_attempt']=s.read(out/'previous_attempt/PRESERVATION.json')
    summary['execution']=dict(version='stream_v2',workers=1,blocks=dict(candidates=1,MI_views=4,pairs=4096),benchmark=s.read(out/'benchmark/SUMMARY.json'),previous_RAM_guard_stop_preserved=True)
    summary['inventory']=dict(**s.progress(),original_MI_views=600,source_weight_states_replayed=150,inner_C_rows=8500,outer_outputs=1800,new_model_fits=0,folds=5,MI=20,skipped=0,preflight_groups=2,preflight_source_weight_replays=10,preflight_C_rows=680,benchmark_group_evaluations=s.read(out/'benchmark/SUMMARY.json')['group_evaluations'],benchmark_warmup_group_evaluations=s.read(out/'benchmark/SUMMARY.json')['warmup_group_evaluations'])
    summary['minimum_target_met']={r:summary['models'][r]['C_index']>=.76 for r in s.ROLES};summary['strict_RCS_plus002_met']={r:summary['models'][r]['C_index']-summary['models']['RCS_ridge']['C_index']>=.02 for r in s.ROLES};m.write_json(out/'summary.json',summary);plot(s,summary,one,resources)
    lines=['# 迭代21：冻结TabM组合与分组NAM组合的有限混合','',f'报告时间：{m.stamp()}。','',
        '20的加性网络内部选择C=.7535286208未超过17 .7541125624，100内部无封顶。本轮检验两种归纳偏置的预测是否互补，复用已有来源，不增加NN/树/核拟合。并非把模型平均必然有效当作前提。',
        '固定17 C_J_SP_I20为控制，四来源为固定20 A8_NLL/A8_C/A16_NLL/A16_C的完整I20组合，全部来源预先纳入，不按外层成绩挑最好NAM。每内部单元只用该单元原训练/验证患者的源权重和预测，不混用其他内折模型。',
        '以每MI训练患者的完整风险均值/总体SD对齐：tNAM=(sNAM-muNAM)/sdNAM*sd17+mu17；新风险=w17*s17+(1-w17)*tNAM。SD最小1e-8。w17仅0/.25/.5/.75/1，共同w1控制只计一次，共17固定候选+E_selected。w1显式保留源17风险和概率逐位一致；w0仅保证NAM排序及C相同，重新Breslow概率不保证等于原NAM校准。不是对原临床X做新标准化。',
        '每来源组合含同一原I20，但风险单位不同，新混合的最终树有效系数可变；原树权重/预测和各来源alpha不变，不宣称最终系数仍等于源alpha。无RCS或弃用v2风险主支路。',
        '全部1140人149事件/29候选/Boruta+LASSO15,9,8,9,8/原5外折5内折/R MICE20maxit60/600原Encoder与数组、列顺序和患者不变。每折选择先mean20MI每内折实际C再mean5内折，1e-12平局带与固定顺序，全部五折选择冻结后才新外层。没有按外层回改权重、模型来源或指标。','',
        '| 方法 | 汇总C（统一患者条件95%CI） | 100输出均值 | 最小–最大 |','|---|---|---|---|']
    for role,a in summary['models'].items():lines.append(f"| {role} | {a['C_index']:.7f}（{a['CI95'][0]:.7f}～{a['CI95'][1]:.7f}） | {a['mean_100']:.7f} | {a['range_100'][0]:.7f}～{a['range_100'][1]:.7f} |")
    lines+=['','## 各时点AUC和C','',metrics[['model_id','metric','estimate','lower','upper']].to_markdown(index=False),'','## 与基线和近期版本配对差','',pair[pair.metric.eq('C_index')][['model_id','reference_id','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_primary_C_family36']].to_markdown(index=False),'','## 24项中间权重对端点探索','',contrast[['alternative','reference','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_matched_mixtures24']].to_markdown(index=False),
        '主要18流程对两基线共36项C，另12中间混合对17及自身w0共24项探索，各自Holm仅覆盖本轮家族；不校正历史外层多轮查看。控制w1对源17不重复做自身差值检验。患者条件IF先跨MI平均再平方，条件于固定模型/划分/已实现MI，不含重新训练/重选型/重插补或开发乐观；不把100相关MI输出当独立t检验样本。',
        '','## 内部互补性、选择、泛化与资源','',json.dumps({k:summary[k] for k in ['complementarity','choices','generalization','inventory','resources','execution','previous_protected_attempt','minimum_target_met','strict_RCS_plus002_met']},ensure_ascii=False,indent=2),
        '','## 预算、完整性与限制','',
        '0新模型拟合。150原NN保存状态、600原输入视图、8500内部C、五选择、100MI外层单元×18=1800输出均核验。源模型已训练/验证，本轮只作CUDA组合与评估；不能把1800输出说成1800次独立训练。保存包装记录源权重与SHA、原Encoder、训练风险统计、固定权重和训练Breslow。',
        '合成标量/NumPy/CUDA风险、w1身份、w0正仿射排序、常数SD与输入不变门控通过；两固定内部组10源状态CUDA重放、40MI视图、680候选C/训练C独立重算及端点对原17/20内部C一致。原双进程执行在0完整单元时触发27.772GiB物理内存保护，实际样本28.496GiB、可用3.276GiB，已停止自有子进程并保留原记录。执行v2使用单进程、1候选×4MI×4096对子分块累计相同int64胜对数；680预检C逐位一致，完整两组源读取/推理/评分与原单路缓存逐位一致，实际计时见execution.benchmark。额外2组计时/1组warmup与正式25+5组分开。此修复不改变模型、网格或统计定义，也不声称单路比旧双路吞吐更高。',
        '物理27.772GiB/可用4GiB/研究RSS8GiB保护不提高，不关闭其他应用。系统GPU含其他应用；该流程无新训练且缓存I/O/哈希/指标可由CPU处理，GPU占用间歇不表示未使用CUDA。图为内部权重响应，不是虚构的新神经训练loss曲线。',
        '内部相关性/对子分歧仅是互补性诊断，不保证外层改善或因果解释。训练测试gap缩小不能单独称成功。纯排序风险经训练Breslow校准，不保证正确PH分布。目标仍完整五折20MI至少.76，精确RCS+.02另列，未增加80/100或p门槛。',
        '固定名单/I20内部依赖、内部验证被早停和权重选择复用、in-sample树风险、外层反复查看和MI相关的已有偏倚保留；不宣称完全嵌套独立无偏确认。E_selected为内部流程，外层最高固定候选不是内部选择的替身。一轮结束不等于总体目标完成。',
        '[原始模型库集成研究](https://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf)提供验证预测组合动机。本轮是固定有限五点凸组合/生存C适配，不是论文无限贪心或直接疗效复现。',
        '','![分布](cindex_distribution.png)','','![内部权重响应](weight_selection.png)','','![资源](resources.png)']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');assert s.signature()==s.read(out/'manifest.json') and m.verify_completed(out/'verification')
    for phase in ['inner','outer']:
        for args in s.jobs(phase):
            d=s.directory(phase,args[0],args[1] if phase=='inner' else None);assert m.verify_completed(d)
            if phase=='outer':
                for j in range(1,21):assert m.verify_completed(d/f'mi{j:02d}')
    m.write_json(out/'FINAL_INTEGRITY.json',dict(status='passed',inventory=summary['inventory'],source_weights_and_original_inputs_unchanged=True,all_fivefold_choices_before_outer=True,all150_source_weights_and_mixtures_replayed=True,control17_risks_probabilities_bitwise_identical=True,all8500_internalC_and1800_outputs_verified=True,successful_execution_resource_guards_passed=True,previous_attempt_RAM_guard_stop_preserved=True,new_model_fits=0))
    s.log('完整计算结束','有限异构组合对固定基线及上轮的比较',summary,'独立与视觉复核另计；原目标不因单轮结束而完成。')
    m.mark_complete(out,['summary.json','FINAL_REPORT.md','FINAL_INTEGRITY.json','metrics_pooled.csv','model_cindex.csv','fold_cindex.csv','paired_differences.csv','procedure_contrasts.csv','conditional_influences.npz','generalization_each_model.csv','generalization_by_fold.csv','complementarity_internal.csv','selection_by_fold.csv','FROZEN_CHOICES.json','cindex_distribution.png','weight_selection.png','resources.png','verification/COMPLETE.json']);s.status(status='complete',phase='complete',progress=s.progress());s.event('complete')
def plot(s,summary,one,resources):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    out=s.OUT;order=['ordinary_cox','RCS_ridge','Previous20_S_I20',*s.ROLES];C=np.array([summary['models'][r]['C_index'] for r in order]);ci=np.array([summary['models'][r]['CI95'] for r in order]);fig,axes=plt.subplots(1,2,figsize=(17,15))
    axes[0].errorbar(C,range(len(order)),xerr=[C-ci[:,0],ci[:,1]-C],fmt='o',capsize=3);axes[0].set(yticks=range(len(order)),yticklabels=order,xlabel='Pooled C / fixed-model/fold/MI conditional95CI');axes[0].invert_yaxis()
    axes[1].boxplot([one[one.model_id.eq(r)].C_index for r in order],vert=False);axes[1].set(yticks=range(1,len(order)+1),yticklabels=order,xlabel='100 correlated outer MI-output C-indices');axes[1].invert_yaxis()
    for ax in axes:ax.axvline(.76,color='gray',ls='--');ax.tick_params(axis='y',labelsize=9)
    fig.suptitle('Fixed17/NAM-I20 finite mixtures:full5fold20MI');fig.tight_layout();fig.savefig(out/'cindex_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,12));weights=[0.,.25,.5,.75,1.]
    for f,ax in zip(range(1,6),axes.flat):
        q=pd.concat([pd.read_csv(s.directory('inner',f,k)/'grid.csv',float_precision='round_trip') for k in range(1,6)]);means=q.groupby(['candidate','inner_fold']).C_index.mean().groupby('candidate').mean()
        for name in s.b.NAM:
            values=[means[s.b.CONTROL] if w==1. else means[f'{name}_w{int(w*100):03d}'] for w in weights];ax.plot(weights,values,'o-',label=name)
        c=summary['choices'][str(f)];ax.set(title=f"Outer fold{f}:inner choice {c['candidate']}",xlabel='Weight of frozen17 score',ylabel='Mean fiveinnerfold C after mean20MI');ax.axvline(c['condition']['weight17'],color='gray',ls='--');ax.legend(fontsize=8)
    axes.flat[-1].axis('off');fig.suptitle('Internal weight responses only;no new neural training curves');fig.tight_layout();fig.savefig(out/'weight_selection.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);t=resources.unix-resources.unix.min();axes[0].plot(t,resources.physical_used_GiB,label='Physical used');axes[0].plot(t,resources.owned_RSS_GiB,label='Study RSS');axes[0].axhline(s.rg.LIMIT/s.rg.GiB,color='red',ls='--',label='Physical guard');axes[0].set_ylabel('GiB');axes[0].legend();q=resources.dropna(subset=['GPU_utilization_percent']);axes[1].plot(q.unix-resources.unix.min(),q.GPU_utilization_percent);axes[1].set(xlabel='Monitored seconds',ylabel='System GPU %',ylim=(0,105));fig.tight_layout();fig.savefig(out/'resources.png',dpi=180);plt.close(fig)

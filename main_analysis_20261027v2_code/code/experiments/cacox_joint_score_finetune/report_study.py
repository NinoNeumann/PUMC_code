"""Uniform fixed-model conditional inference and full iteration17 report."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
def independent_C(q):
    values=[];raw=[]
    for j in range(1,21):
        num=0.;den=0
        for f in range(1,6):
            a=q[q.fold.eq(f)&q.imputation.eq(j)];assert len(a)==228;t=a.Survival_Days.to_numpy();e=a.Death.to_numpy(bool);v=a.score.to_numpy()
            i,k=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
            win=float(((v[i]>v[k])+.5*(v[i]==v[k])).sum());num+=win;den+=len(i);raw.append(win/len(i))
        values.append(num/den)
    return float(np.mean(values)),float(np.mean(raw))
def holm(p):
    p=np.asarray(p);order=np.argsort(p);z=np.empty(len(p));z[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return z
def factorial(s,summary,details):
    rows=[]
    for warm in ['NLL','C']:
        for kind in ['NN','I20']:
            for name,alt,ref in [('joint_softplus_minus_NN_softplus','J_SP','N_SP'),('joint_sigmoid1_minus_joint_softplus','J_S1','J_SP'),('joint_sigmoid01_minus_joint_softplus','J_S01','J_SP')]:
                a=warm+'_'+alt+'_'+kind;r=warm+'_'+ref+'_'+kind;x=details[a]['C_index'];y=details[r]['C_index'];np.testing.assert_array_equal(x['patient_ids'],y['patient_ids'])
                z=s.ev.interval(summary['models'][a]['C_index']-summary['models'][r]['C_index'],x['mean_IF']-y['mean_IF'],bounded=False)
                p=float(2*norm.sf(abs(z['estimate']/z['SE']))) if z['SE'] else (1. if z['estimate']==0 else 0.)
                rows.append(dict(warmstart=warm,kind=kind,contrast=name,alternative=a,reference=r,p_two_sided_conditional_normal=p,**z))
    q=pd.DataFrame(rows);assert len(q)==12;q['p_Holm_exploratory_procedure12']=holm(q.p_two_sided_conditional_normal);q.to_csv(s.OUT/'procedure_contrasts.csv',index=False);return q

def report_all(s):
    out,m,b,ev=s.OUT,s.m,s.b,s.ev
    def collect(root,pattern):return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in root.glob(pattern)],ignore_index=True)
    frames={r:collect(out,f'outer/fold*/shared/mi*/{r}.csv') for r in s.ROLES}
    for role in ['S0_shared_Cox','J0_I20_shared','JC_I20_selected']:frames[role]=collect(b.io.OUT,f'outer/fold*/shared/mi*/{role}.csv')
    for role in b.legacy.previous.ROLES:frames[role]=collect(b.legacy.previous.OUT,f'outer/fold*/shared/mi*/{role}.csv')
    previous_roles=[cid+'_'+kind for cid in ['H_NLL','H_C','P_NLL','P_C'] for kind in ['NN','I20']]+['S_NN','S_I20']
    previous_aliases={r:('Previous15_'+r if r in s.ROLES else r) for r in previous_roles}
    for role,alias in previous_aliases.items():
        assert alias not in frames;frames[alias]=collect(b.legacy.OUT,f'outer/fold*/shared/mi*/{role}.csv')
    source16=s.ROOT/'results/cacox_loss_aggregation/20261031_v1';roles16=[cid+'_'+kind for cid in ['CM_NLL','CM_C','RM_NLL','RM_C'] for kind in ['NN','I20']]+['S_NN','S_I20'];aliases16={r:('Previous16_'+r if r in s.ROLES else r) for r in roles16}
    for role,alias in aliases16.items():
        assert alias not in frames;frames[alias]=collect(source16,f'outer/fold*/shared/mi*/{role}.csv')
    for role,root,stem in [('I20_previous',b.io.source.r.prior.OUT,'I20_selection'),('B2_sharedNN_ranktree_MS',s.ROOT/'results/cacox_shared_nn_ranktree/20261031_v1','B2_sharedNN_ranktree_MS')]:frames[role]=collect(root,f'outer/fold*/mi*/{stem}.csv')
    for role in ['ordinary_cox','RCS_ridge','original_CACox']:frames[role]=pd.read_csv(b.io.source.r.old.REF/f'predictions_{role}.csv',float_precision='round_trip')
    summary=dict(at=m.stamp(),models={},comparisons={},interval_scope=ev.SCOPE);tables={};details={};metrics=[];ones=[];folds=[];influences={}
    for role,data in frames.items():
        b.io.source.r.old.check(data);m.assert_paired(frames['RCS_ridge'],data);table,IF=ev.evaluate(data);tables[role]=table;details[role]=IF
        one=m.individual_cindices(data);assert len(one)==100;C,mean=independent_C(data);a=table[table.metric.eq('C_index')].iloc[0];assert abs(C-float(a.estimate))<1e-12 and abs(mean-float(one.C_index.mean()))<1e-12
        summary['models'][role]=dict(C_index=C,CI95=[float(a.lower),float(a.upper)],mean_100=mean,range_100=[float(one.C_index.min()),float(one.C_index.max())])
        metrics.append(table.assign(model_id=role));ones.append(one.assign(model_id=role));folds.append(one.groupby('fold').C_index.agg(['mean','min','max']).reset_index().assign(model_id=role))
        for metric,v in IF.items():
            for key,value in v.items():influences[role+'__'+metric+'__'+key]=value
        s.rg.record(out/'monitor',False);s.status(status='running',phase='metrics_and_report',models_evaluated=len(tables),progress=s.progress())
    metrics=pd.concat(metrics);one=pd.concat(ones);metrics.to_csv(out/'metrics_pooled.csv',index=False);one.to_csv(out/'model_cindex.csv',index=False);pd.concat(folds).to_csv(out/'fold_cindex.csv',index=False);np.savez_compressed(out/'conditional_influences.npz',**influences)
    pairs=[]
    references=['ordinary_cox','RCS_ridge','I20_previous','S0_shared_Cox','J0_I20_shared','JC_I20_selected','C1_Cstop_NN','C1_Cstop_I20','E3_NLL_I20','E3_Cstop_I20','B2_sharedNN_ranktree_MS',*previous_aliases.values(),*aliases16.values(),*s.ROLES]
    for role in s.ROLES:
        summary['comparisons'][role]={}
        for ref in references:
            if ref==role:continue
            q=ev.paired(tables[ref],details[ref],tables[role],details[role]);q['p_two_sided_conditional_normal']=q.apply(lambda v:float(2*norm.sf(abs(v.estimate/v.SE))) if v.SE>0 else (1. if v.estimate==0 else 0.),axis=1)
            pairs.append(q.assign(model_id=role,reference_id=ref));summary['comparisons'][role][ref]=q[q.metric.eq('C_index')].iloc[0].to_dict()
    pair=pd.concat(pairs,ignore_index=True);pair['p_Holm_primary_C_family18']=np.nan;primary=pair.model_id.str.endswith('_I20')&pair.reference_id.isin(['ordinary_cox','RCS_ridge'])&pair.metric.eq('C_index');assert primary.sum()==18
    pair.loc[primary,'p_Holm_primary_C_family18']=holm(pair.loc[primary,'p_two_sided_conditional_normal']);pair.to_csv(out/'paired_differences.csv',index=False)
    general=collect(out,'outer/fold*/shared/mi*/generalization.csv');assert len(general)==1800;general['gap']=general.train_C-general.test_C;general.to_csv(out/'generalization_each_model.csv',index=False)
    general.groupby(['role','fold'])[['train_C','test_C','gap']].mean().reset_index().to_csv(out/'generalization_by_fold.csv',index=False);summary['generalization']=general.groupby('role')[['train_C','test_C','gap']].mean().to_dict('index')
    frozen=s.read(out/'FROZEN_CHOICES.json');assert s.freeze(False)=={k:frozen[k] for k in ['epochs','choices']};summary['epochs']=frozen['epochs'];summary['choices']=frozen['choices'];convergence=[]
    for phase in ['inner','outer']:
        for p in (out/phase).glob('fold*/*/fits/*/audit.json'):
            a=s.read(p);assert a['cuda_train'];convergence.append(dict(phase=phase,fold=a['fold'],inner_fold=a['inner_fold'],condition=a['condition']['condition_id'],monitor=a['condition']['monitor'],warmstart=a['condition']['warmstart'],target=a['condition']['target'],temperature=a['condition']['temperature'],
                seed=a['seed'],best_epoch=a['best_epoch'],stopped_epoch=a['stopped_epoch'],cap_hit=a['cap_hit'],fit_seconds=a['fit_seconds'],parameters=a['parameters'],initial_state_SHA=a['initial_state_SHA']))
    conv=pd.DataFrame(convergence);assert conv.phase.eq('inner').sum()==200 and conv.phase.eq('outer').sum()==40;conv.to_csv(out/'convergence.csv',index=False)
    prefixes=pd.concat([pd.read_csv(p,float_precision='round_trip').assign(group=str(p.parent.relative_to(out))) for p in out.glob('*er/fold*/*/warmstart_checks.csv')],ignore_index=True);assert len(prefixes)==60 and prefixes.shared_source_initial_state.all();prefixes.to_csv(out/'all_warmstart_checks.csv',index=False)
    summary['convergence']=dict(inner_cap_count=int(conv[conv.phase.eq('inner')].cap_hit.sum()),by_condition=conv[conv.phase.eq('inner')].groupby('condition')[['best_epoch','stopped_epoch','fit_seconds']].mean().to_dict('index'),parameters_by_fold=conv.groupby('fold').parameters.first().to_dict(),matching_warmstarts=60)
    resources=pd.DataFrame([json.loads(line) for line in (out/'monitor/resources.jsonl').read_text().splitlines()]);summary['resources']=dict(max_physical_used_GiB=float(resources.physical_used_GiB.max()),max_owned_RSS_GiB=float(resources.owned_RSS_GiB.max()),minimum_available_GiB=float(resources.available_GiB.min()),median_sampled_system_GPU=float(resources.GPU_utilization_percent.median()),peak_system_GPU=float(resources.GPU_utilization_percent.max()),peak_system_GPU_MiB=float(resources.GPU_used_MiB.max()))
    assert summary['resources']['max_physical_used_GiB']<=s.rg.LIMIT/s.rg.GiB and summary['resources']['max_owned_RSS_GiB']<=8
    pr=s.progress();assert pr==dict(inner_groups=25,inner_new_NN=200,outer_groups=5,outer_new_NN=40,outer_MI_units=100)
    summary['inventory']=dict(**pr,reused_inner_NN=50,reused_outer_NN=10,new_MI_outputs=1800,inner_MI_views=500,inner_candidate_grid_rows=10000,folds=5,MI=20,skipped=0,new_tree_kernel_fits=0,
        extra_prefits=s.read(out/'preflight/PREFLIGHT.json')['full20MI_additional_fit_calls'],extra_zero_step_checks=4,extra_benchmark_fullfits=s.read(out/'benchmark/SUMMARY.json')['full_shared_fits_in_passed_cases'],extra_threeepoch_warmups=s.read(out/'benchmark/SUMMARY.json')['warmup3epoch_fits'],zero_budget_outer_states=int(((conv.phase=='outer')&(conv.best_epoch==0)).sum()),benchmark_workers=s.read(out/'RUNTIME_PLAN.json')['workers'])
    summary['minimum_target_met']={r:summary['models'][r]['C_index']>=.76 for r in s.ROLES};summary['strict_RCS_plus002_met']={r:summary['models'][r]['C_index']-summary['models']['RCS_ridge']['C_index']>=.02 for r in s.ROLES}
    factor=factorial(s,summary,details);summary['procedure_contrasts']=factor.to_dict('records');m.write_json(out/'summary.json',summary);plot(s,summary,one,resources)
    lines=['# 迭代17：固定树与神经网络的最终组合目标微调','',f'报告时间：{m.stamp()}。','',
        '16没有超过15最好P_C_I20=.75375144，尚未达到.76。本轮固定I20树、原数据和神经结构，从15 P_NLL及P_C两个固定起点微调，检验最终组合目标；不按外层择起点。',
        '每起点四条件：N_SP继续纯NN softplus（匹配额外训练/优化器对照），J_SP优化组合softplus，J_S1/J_S01优化组合sigmoid温度1/.1。全用最终组合内部C早停，起点epoch0可回退；追加上限500/耐心150，AdamW lr.0001/WD.0003/dropout.15/clip5，不搜索。骨干仍TabM8成员2×64。',
        '最终组合=(1-alpha)*Zsource(I20)+alpha*Zsource(NN)，alpha固定.5/.75/.5/.5/.5。全部输出统计来自该训练分区原起点NN及树，微调和推理全程冻结，不能在最后按新NN重新标准化。输入字段/患者/插补/Encoder未变；这是模型输出组合。',
        '1140人149事件、29候选、Boruta/LASSO名单15/9/8/9/8、原R MICE20/maxit60、全部内外划分及每MI编码数组不变。200新内部/40新外层状态（0预算对应0优化步另列）、50/10只读起点；18流程各100MI输出，不是1800独立NN训练。40新预算/10来源预算/10选择全部内部冻结才新外层。',
        '','| 方法 | 汇总C（统一条件95%CI） | 100MI输出均值 | 最小–最大 |','|---|---|---|---|']
    for r,a in summary['models'].items():lines.append(f"| {r} | {a['C_index']:.7f}（{a['CI95'][0]:.7f}～{a['CI95'][1]:.7f}） | {a['mean_100']:.7f} | {a['range_100'][0]:.7f}～{a['range_100'][1]:.7f} |")
    lines+=['','## C及AUC','',metrics[['model_id','metric','estimate','lower','upper']].to_markdown(index=False),'','## 配对差值','',pair[pair.metric.eq('C_index')][['model_id','reference_id','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_primary_C_family18']].to_markdown(index=False),
        '','区间条件于固定模型/划分/已实现20MI，患者影响函数先跨MI平均再平方，不覆盖重新训练/选型/插补不确定性；与历史Rubin口径不同。九组合流程对两基线十八个C检验Holm只覆盖该家族，不校正此前多轮开发。其他探索比较不当独立确认。',
        '','## 匹配流程对比（探索性）','',factor[['warmstart','kind','contrast','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_exploratory_procedure12']].to_markdown(index=False),'','## 完整预算、收敛、资源','',json.dumps({k:summary[k] for k in ['epochs','choices','inventory','convergence','generalization','resources','minimum_target_met','strict_RCS_plus002_met']},ensure_ascii=False,indent=2),
        '','## 工程证据及限制','',
        '240新微调状态+60来源状态全部回放，600原MI输入和Encoder、10000内部C、所有预算/选择、1800包装风险与训练Breslow概率重放。每患者组两个起点、每起点四方法初态一致；目标不同后训练轨迹不应强求一致。额外预检17完整微调/4零步数检查，1/2路基准24完整微调/3次三步warmup单列。',
        '0预算并非任务跳过，保存原起点完整状态且仍评测该折20MI。学习曲线epoch0为微调前eval、后续训练目标含dropout，不能把0到1变化全当训练改进。新源控制NN概率使用原rawNN风险；组合用冻结源输出统计，各自开发Breslow只作附加校准，不保证排序目标满足PH。',
        '可比较对和C汇总定义不变；C使用score，AUC使用各时点保存风险，100输出算术均值另列。值/解析梯度CPU/CUDA误差检查、起点/零步数复现和保存/重排/逐MI推理/重复均通过，不以预检C冒充外层。',
        '三类预定流程对比按两个起点和NN/I20共12探索比较，Holm只覆盖该家族；九个I20流程对两基线共18主要C检验另外Holm，不校正历史多轮查看。不是按外层选择最好的温度/起点或新增组合。',
        '有界sigmoid为平滑Harrell错序率动机，有限温度可出现loss下降而离散C下降，且对极差对子梯度会饱和。因此实际监控组合C，不能以loss相似就保证提高。J_SP与N_SP对比区分组合目标和继续训练，新低学习率是否全局最好并未探索。',
        '训练树风险为in-sample，神经分支可能过度补偿树的训练特点；不能偷偷引入看过该内部验证结局的其他折树来构造训练OOF。固定名单/继承I20内部依赖、外层反复查看、同队列继续开发和MI相关仍在，不称独立完全嵌套无偏验证。患者条件IF区间不覆盖重新训练/选型/插补或开发乐观偏倚。',
        'GPU训练/推理，physical27.772GiB/available4GiB/ownedRSS8GiB保护不提高；系统GPU包括其他程序，末段缺失采样不填0。训练gap下降或内部C提升均不能单独称成功；至少.76目标及约RCS+.02不变，无新p/80模型门槛。',
        '','![分布](cindex_distribution.png)','','![学习曲线](learning_curves.png)','','![资源](resources.png)']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');assert s.signature()==s.read(out/'manifest.json') and m.verify_completed(out/'verification')
    for phase in ['inner','outer']:
        for args in s.jobs(phase):
            d=s.directory(phase,args[0],args[1] if phase=='inner' else None);assert m.verify_completed(d)
            for p in (d/'fits').iterdir():assert m.verify_completed(p)
            if phase=='outer':
                for j in range(1,21):assert m.verify_completed(d/f'mi{j:02d}')
    m.write_json(out/'FINAL_INTEGRITY.json',dict(status='passed',inventory=summary['inventory'],source_and_data_unchanged=True,all_fivefold_choices_before_outer=True,all_saved_weights_risks_probabilities_replayed=True,all60_matching_warmstart_states_verified=True,all1800_outputs_and_C_verified=True,uniform_conditional_inference=True,resource_budget_passed=True))
    s.log('完整实验结束','五折20MI固定树最终组合目标微调',summary,'对照baseline及15/16；目标仍须独立和视觉审核。')
    m.mark_complete(out,['summary.json','FINAL_REPORT.md','FINAL_INTEGRITY.json','metrics_pooled.csv','model_cindex.csv','fold_cindex.csv','paired_differences.csv','conditional_influences.npz','generalization_each_model.csv','generalization_by_fold.csv',
        'procedure_contrasts.csv','convergence.csv','all_warmstart_checks.csv','FROZEN_CHOICES.json','cindex_distribution.png','learning_curves.png','resources.png','verification/COMPLETE.json']);s.status(status='complete',phase='complete',progress=s.progress());s.event('complete')

def plot(s,summary,one,resources):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    out=s.OUT;order=['ordinary_cox','RCS_ridge','JC_I20_selected','P_NLL_I20','P_C_I20','RM_C_I20',*s.ROLES];C=np.array([summary['models'][r]['C_index'] for r in order]);CI=np.array([summary['models'][r]['CI95'] for r in order]);fig,axes=plt.subplots(1,2,figsize=(17,16))
    axes[0].errorbar(C,range(len(order)),xerr=[C-CI[:,0],CI[:,1]-C],fmt='o',capsize=3);axes[0].set(yticks=range(len(order)),yticklabels=order,xlabel='Pooled C / fixed-model/fold/realizedMI conditional95CI');axes[0].invert_yaxis();axes[1].boxplot([one[one.model_id.eq(r)].C_index for r in order],vert=False);axes[1].set(yticks=range(1,len(order)+1),yticklabels=order,xlabel='100 correlated outer MI-output C-indices');axes[1].invert_yaxis()
    for ax in axes:ax.axvline(.76,c='gray',ls='--')
    fig.suptitle('Frozen-I20 final-score neural fine-tuning:full5fold20MI');fig.tight_layout();fig.savefig(out/'cindex_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for cid in s.NEW:
        q=pd.read_csv(out/f'inner/fold01/inner01/fits/{cid}/training_trace.csv');axes[0,0 if cid.endswith('_SP') else 1].plot(q.epoch,q.training_requested_loss,label=cid)
        axes[1,0].plot(q.epoch,q.validation_combined_C,label=cid);axes[1,1].plot(q.epoch,q.validation_NN_C,label=cid)
    for ax,title in zip(axes.flat,['NN/combined softplus training targets','Combined sigmoid training targets','Validation finalcombined C','Validation NN-only C']):ax.set(title=title,xlabel='Additional epoch;0=initial eval');ax.legend(fontsize=8)
    fig.suptitle('Fixed fold1/inner1;actual inner curves,not outertest selection');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);x=resources.unix-resources.unix.min();axes[0].plot(x,resources.physical_used_GiB,label='Physical used');axes[0].plot(x,resources.owned_RSS_GiB,label='Study RSS');axes[0].axhline(s.rg.LIMIT/s.rg.GiB,c='r',ls='--',label='Physical guard');axes[0].set_ylabel('GiB');axes[0].legend();q=resources.dropna(subset=['GPU_utilization_percent']);axes[1].plot(q.unix-resources.unix.min(),q.GPU_utilization_percent);axes[1].set(xlabel='Monitored seconds',ylabel='System GPU %',ylim=(0,105));fig.tight_layout();fig.savefig(out/'resources.png',dpi=180);plt.close(fig)

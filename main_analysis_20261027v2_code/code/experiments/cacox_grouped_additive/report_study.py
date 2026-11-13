"""Iteration20: uniform conditional inference, grouped additive architecture contrasts."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm

def independent_C(q):
    values=[];raw=[]
    for j in range(1,21):
        numerator=0.;denominator=0
        for f in range(1,6):
            a=q[q.fold.eq(f)&q.imputation.eq(j)];assert len(a)==228
            t=a.Survival_Days.to_numpy();e=a.Death.to_numpy(bool);v=a.score.to_numpy()
            i,k=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
            wins=float(((v[i]>v[k])+.5*(v[i]==v[k])).sum());numerator+=wins;denominator+=len(i);raw.append(wins/len(i))
        values.append(numerator/denominator)
    return float(np.mean(values)),float(np.mean(raw))

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return out

def pvalue(q):
    q=q.copy();q['p_two_sided_conditional_normal']=q.apply(lambda v:float(2*norm.sf(abs(v.estimate/v.SE))) if v.SE>0 else (1. if v.estimate==0 else 0.),axis=1);return q

def report_all(s):
    out,m,b,ev=s.OUT,s.m,s.b,s.ev
    def collect(root,pattern):
        paths=list(root.glob(pattern));assert paths;return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in paths],ignore_index=True)
    frames={r:collect(out,f'outer/fold*/shared/mi*/{r}.csv') for r in s.ROLES}
    for role in ['P_NLL_NN','P_NLL_I20','P_C_NN','P_C_I20']:frames[role]=collect(b.legacy.OUT,f'outer/fold*/shared/mi*/{role}.csv')
    for role in ['C_J_SP_NN','C_J_SP_I20']:frames[role]=collect(s.ROOT/'results/cacox_joint_score_finetune/20261031_v1',f'outer/fold*/shared/mi*/{role}.csv')
    for role in ['C_J_S1_a1_NN','C_J_S1_a1_I20']:frames[role]=collect(s.ROOT/'results/cacox_prediction_anchor/20261031_v1',f'outer/fold*/shared/mi*/{role}.csv')
    for role in ['S_NN','S_I20']:frames['Previous19_'+role]=collect(s.ROOT/'results/cacox_numeric_embedding/20261031_v1',f'outer/fold*/shared/mi*/{role}.csv')
    for role in ['JC_I20_selected']:frames[role]=collect(b.io.OUT,f'outer/fold*/shared/mi*/{role}.csv')
    frames['I20_previous']=collect(b.io.source.r.prior.OUT,'outer/fold*/mi*/I20_selection.csv')
    for role in ['ordinary_cox','RCS_ridge','original_CACox']:frames[role]=pd.read_csv(b.io.source.r.old.REF/f'predictions_{role}.csv',float_precision='round_trip')
    tables={};details={};metrics=[];ones=[];folds=[];influences={};summary=dict(at=m.stamp(),models={},comparisons={},interval_scope=ev.SCOPE)
    for role,data in frames.items():
        b.io.source.r.old.check(data);m.assert_paired(frames['RCS_ridge'],data);table,IF=ev.evaluate(data);tables[role]=table;details[role]=IF
        one=m.individual_cindices(data);assert len(one)==100;C,mean=independent_C(data);a=table[table.metric.eq('C_index')].iloc[0]
        assert abs(C-float(a.estimate))<1e-12 and abs(mean-one.C_index.mean())<1e-12
        summary['models'][role]=dict(C_index=C,CI95=[float(a.lower),float(a.upper)],mean_100=mean,range_100=[float(one.C_index.min()),float(one.C_index.max())])
        metrics.append(table.assign(model_id=role));ones.append(one.assign(model_id=role));folds.append(one.groupby('fold').C_index.agg(['mean','min','max']).reset_index().assign(model_id=role))
        for metric,v in IF.items():
            for key,value in v.items():influences[role+'__'+metric+'__'+key]=value
        s.rg.record(out/'monitor',False);s.status(status='running',phase='metrics_and_report',models_evaluated=len(tables),progress=s.progress())
    metrics=pd.concat(metrics);one=pd.concat(ones);metrics.to_csv(out/'metrics_pooled.csv',index=False);one.to_csv(out/'model_cindex.csv',index=False);pd.concat(folds).to_csv(out/'fold_cindex.csv',index=False);np.savez_compressed(out/'conditional_influences.npz',**influences)
    refs=['ordinary_cox','RCS_ridge','I20_previous','JC_I20_selected','P_NLL_NN','P_C_NN','P_NLL_I20','P_C_I20','C_J_SP_I20','C_J_S1_a1_I20','Previous19_S_I20']
    pairs=[]
    for role in s.ROLES:
        summary['comparisons'][role]={}
        for ref in refs:
            q=pvalue(ev.paired(tables[ref],details[ref],tables[role],details[role]));pairs.append(q.assign(model_id=role,reference_id=ref));summary['comparisons'][role][ref]=q[q.metric.eq('C_index')].iloc[0].to_dict()
    pair=pd.concat(pairs,ignore_index=True);pair['p_Holm_primary_C_family10']=np.nan;primary=pair.model_id.str.endswith('_I20')&pair.reference_id.isin(['ordinary_cox','RCS_ridge'])&pair.metric.eq('C_index');assert primary.sum()==10
    pair.loc[primary,'p_Holm_primary_C_family10']=holm(pair.loc[primary,'p_two_sided_conditional_normal']);pair.to_csv(out/'paired_differences.csv',index=False)
    matched=[]
    for monitor in ['NLL','C']:
        for kind in ['NN','I20']:
            for alternative,reference in [(f'A8_{monitor}_{kind}',f'P_{monitor}_{kind}'),(f'A16_{monitor}_{kind}',f'P_{monitor}_{kind}'),(f'A16_{monitor}_{kind}',f'A8_{monitor}_{kind}')]:
                row=pvalue(ev.paired(tables[reference],details[reference],tables[alternative],details[alternative]));row=row[row.metric.eq('C_index')].copy();row['alternative']=alternative;row['reference']=reference;matched.append(row)
    contrasts=pd.concat(matched,ignore_index=True);assert len(contrasts)==12;contrasts['p_Holm_additive12']=holm(contrasts.p_two_sided_conditional_normal);contrasts.to_csv(out/'procedure_contrasts.csv',index=False)
    general=collect(out,'outer/fold*/shared/mi*/generalization.csv');assert len(general)==1000;general['gap']=general.train_C-general.test_C;general.to_csv(out/'generalization_each_model.csv',index=False);general.groupby(['role','fold'])[['train_C','test_C','gap']].mean().reset_index().to_csv(out/'generalization_by_fold.csv',index=False);summary['generalization']=general.groupby('role')[['train_C','test_C','gap']].mean().to_dict('index')
    frozen=s.read(out/'FROZEN_CHOICES.json');assert s.freeze(False)=={k:frozen[k] for k in ['epochs','choices']};summary['epochs']=frozen['epochs'];summary['choices']=frozen['choices'];convergence=[];architecture=[]
    for phase in ['inner','outer']:
        for p in (out/phase).glob('fold*/*/fits/*/audit.json'):
            a=s.read(p);assert a['cuda_train'];c=a['condition'];convergence.append(dict(phase=phase,fold=a['fold'],inner_fold=a['inner_fold'],condition=c['condition_id'],monitor=c['monitor'],NAM_width=c['width'],seed=a['seed'],best_epoch=a['best_epoch'],stopped_epoch=a['stopped_epoch'],cap_hit=a['cap_hit'],fit_seconds=a['fit_seconds'],parameters=a['parameters'],active_parameters=a['active_parameters'],initial_state_SHA=a['initial_state_SHA']))
            architecture.append(dict(phase=phase,fold=a['fold'],inner_fold=a['inner_fold'],condition=c['condition_id'],spec=a['architecture_spec']))
    conv=pd.DataFrame(convergence);assert conv.phase.eq('inner').sum()==100 and conv.phase.eq('outer').sum()==20;conv.to_csv(out/'convergence.csv',index=False);m.write_json(out/'all_architecture_specs.json',architecture)
    prefixes=collect(out,'*er/fold*/*/monitor_prefix_checks.csv');assert len(prefixes)==60 and prefixes.training_monitor_prefix_error.eq(0).all();prefixes.to_csv(out/'all_monitor_prefix_checks.csv',index=False)
    summary['convergence']=dict(inner_cap_count=int(conv[conv.phase.eq('inner')].cap_hit.sum()),by_condition=conv[conv.phase.eq('inner')].groupby('condition')[['best_epoch','stopped_epoch','fit_seconds','parameters','active_parameters']].mean().to_dict('index'),matched_monitor_prefixes=60)
    resources=pd.DataFrame([json.loads(line) for line in (out/'monitor/resources.jsonl').read_text().splitlines()]);summary['resources']=dict(max_physical_used_GiB=float(resources.physical_used_GiB.max()),max_owned_RSS_GiB=float(resources.owned_RSS_GiB.max()),minimum_available_GiB=float(resources.available_GiB.min()),median_sampled_system_GPU=float(resources.GPU_utilization_percent.median()),peak_system_GPU=float(resources.GPU_utilization_percent.max()),peak_system_GPU_MiB=float(resources.GPU_used_MiB.max()))
    assert summary['resources']['max_physical_used_GiB']<=s.rg.LIMIT/s.rg.GiB and summary['resources']['max_owned_RSS_GiB']<=8 and summary['resources']['minimum_available_GiB']>=4
    pr=s.progress();assert pr==dict(inner_groups=25,inner_new_NN=100,outer_groups=5,outer_new_NN=20,outer_MI_units=100)
    summary['inventory']=dict(**pr,reused_inner_NN=50,reused_outer_NN=10,new_MI_outputs=1000,inner_MI_views=500,inner_candidate_grid_rows=6000,folds=5,MI=20,skipped=0,new_tree_kernel_fits=0,extra_prefits=s.read(out/'preflight/PREFLIGHT.json')['new_preflight_NN'],extra_benchmark_fullfits=s.read(out/'benchmark/SUMMARY.json')['full_shared_fits_in_passed_cases'],extra_threeepoch_warmups=s.read(out/'benchmark/SUMMARY.json')['warmup3epoch_fits'])
    summary['minimum_target_met']={r:summary['models'][r]['C_index']>=.76 for r in s.ROLES};summary['strict_RCS_plus002_met']={r:summary['models'][r]['C_index']-summary['models']['RCS_ridge']['C_index']>=.02 for r in s.ROLES};m.write_json(out/'summary.json',summary);plot(s,summary,one,resources)
    lines=report_lines(s,summary,metrics,pair,contrasts)
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');assert s.signature()==s.read(out/'manifest.json') and m.verify_completed(out/'verification')
    for phase in ['inner','outer']:
        for args in s.jobs(phase):
            d=s.directory(phase,args[0],args[1] if phase=='inner' else None);assert m.verify_completed(d)
            for fit in (d/'fits').iterdir():assert m.verify_completed(fit)
            if phase=='outer':
                for j in range(1,21):assert m.verify_completed(d/f'mi{j:02d}')
    m.write_json(out/'FINAL_INTEGRITY.json',dict(status='passed',inventory=summary['inventory'],all_original_inputs_encoders_unchanged=True,all_original_variable_groups_padding_verified=True,all_choices_before_new_outer=True,all_saved_weights_risks_probabilities_replayed=True,all60_monitor_prefixes_identical=True,all1000_outputs_and_C_verified=True,uniform_conditional_inference=True,resource_budget_passed=True))
    s.log('完整计算结束','分组NAM宽8/16匹配比较，独立与视觉复核另计',summary,'不将单轮完成当总目标完成。')
    m.mark_complete(out,['summary.json','FINAL_REPORT.md','FINAL_INTEGRITY.json','metrics_pooled.csv','model_cindex.csv','fold_cindex.csv','paired_differences.csv','procedure_contrasts.csv','conditional_influences.npz','generalization_each_model.csv','generalization_by_fold.csv','convergence.csv','all_architecture_specs.json','all_monitor_prefix_checks.csv','FROZEN_CHOICES.json','cindex_distribution.png','learning_curves.png','resources.png','verification/COMPLETE.json']);s.status(status='complete',phase='complete',progress=s.progress());s.event('complete')

def plot(s,summary,one,resources):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    out=s.OUT;order=['ordinary_cox','RCS_ridge','P_C_I20','C_J_SP_I20','C_J_S1_a1_I20','Previous19_S_I20',*[r for r in s.ROLES if r.endswith('_I20')]]
    C=np.array([summary['models'][r]['C_index'] for r in order]);ci=np.array([summary['models'][r]['CI95'] for r in order]);fig,axes=plt.subplots(1,2,figsize=(16,9))
    axes[0].errorbar(C,range(len(order)),xerr=[C-ci[:,0],ci[:,1]-C],fmt='o',capsize=3);axes[0].set(yticks=range(len(order)),yticklabels=order,xlabel='Pooled C / fixed-model/fold/realizedMI conditional95CI');axes[0].invert_yaxis()
    axes[1].boxplot([one[one.model_id.eq(r)].C_index for r in order],vert=False);axes[1].set(yticks=range(1,len(order)+1),yticklabels=order,xlabel='100 correlated outer MI-output C-indices');axes[1].invert_yaxis()
    for ax in axes:ax.axvline(.76,c='gray',ls='--')
    fig.suptitle('Grouped additive ensemble:fixed full5fold20MI');fig.tight_layout();fig.savefig(out/'cindex_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for cid in s.NEW:
        q=pd.read_csv(out/f'inner/fold01/inner01/fits/{cid}/training_trace.csv')
        axes[0,0].plot(q.epoch,q.training_mean_view_ensemble_pair_loss,label=cid);axes[0,1].plot(q.epoch,q.gradient_norm_before_clip,label=cid);axes[1,0].plot(q.epoch,q.validation_mean_view_ensemble_Efron,label=cid);axes[1,1].plot(q.epoch,q.validation_C_index,label=cid)
    for ax,title in zip(axes.flat,['Training mean-risk pair loss','Gradient norm before clip','Validation ensemble Efron','Validation C']):ax.set(title=title,xlabel='Epoch');ax.legend(fontsize=8)
    fig.suptitle('Fixed fold1/inner1;actual inner records,not outertest selection');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);x=resources.unix-resources.unix.min();axes[0].plot(x,resources.physical_used_GiB,label='Physical used');axes[0].plot(x,resources.owned_RSS_GiB,label='Study RSS');axes[0].axhline(s.rg.LIMIT/s.rg.GiB,c='r',ls='--',label='Physical guard');axes[0].set_ylabel('GiB');axes[0].legend();q=resources.dropna(subset=['GPU_utilization_percent']);axes[1].plot(q.unix-resources.unix.min(),q.GPU_utilization_percent);axes[1].set(xlabel='Monitored seconds',ylabel='System GPU %',ylim=(0,105));fig.tight_layout();fig.savefig(out/'resources.png',dpi=180);plt.close(fig)

def report_lines(s,summary,metrics,pair,contrasts):
    lines=['# 迭代20：共享MI分组加性风险网络','',f'报告时间：{s.m.stamp()}。','',
        '19连续嵌入内部选择C=.75091817，未超过17最佳.75411256，且100内部0封顶。增加表示灵活性未改善外层；本轮测试限制任意跨变量交互的加性归纳偏置，不把它提前认定为唯一根因或必然有效。',
        '按原Encoder.variables分组，一个临床变量的全部原dummy共同输入一个小子网；每个子网两层ReLU/Dropout.15，宽8或16。每个成员风险直接对变量子网输出求和；八个成员分别参数化，mean8风险参与纯排序训练。无RCS/v2样条主风险分支、额外交互或学到的混合权重。内部padding有mask，所有原列恰出现一次，不删除变量/不改编码。总参数与排除padding后的有效参数分列。',
        '数据侧1140人149事件、29候选、名单15/9/8/9/8、5外折/5内折、R MICE20maxit60、每MI Encoder/数组列顺序与患者完全固定。20MI各自用原患者对子，不能看作20n独立样本；风险在模型内不跨MI聚合为单一患者预测来更换C口径。',
        '四条件A8_NLL/A8_C/A16_NLL/A16_C，共同AdamW.001/WD.0003/clip5/max3000/150严格早停，目标沿15 mean20MI可比较对softplus(mean8风险)。同宽度两监控初始化/训练前缀相同，不同宽度不要求同初态或容量。新模型配置明确为NAM宽8/16，不把只读TabM控制的width64当作新模型宽度。15 P_NLL/P_C原标量TabM只读对照，权重、I20原树/核及NNalpha.5/.75/.5/.5/.5不改。',
        '每新NN只用自己的训练risk均值SD进行原I20组合。全部30预算/10内部选择在新外层之前冻结；每条件预算ceilmedian5内最佳轮数，选型先mean20MI每内折再mean5内折、固定平局顺序，可回退15原控制。报告所有固定流程与内部S，不用外层最高单流程替代内部选择。','',
        '| 方法 | 汇总C（统一患者条件95%CI） | 100输出均值 | 最小–最大 |','|---|---|---|---|']
    for role,a in summary['models'].items():lines.append(f"| {role} | {a['C_index']:.7f}（{a['CI95'][0]:.7f}～{a['CI95'][1]:.7f}） | {a['mean_100']:.7f} | {a['range_100'][0]:.7f}～{a['range_100'][1]:.7f} |")
    lines += ['','## C及各时点AUC','',metrics[['model_id','metric','estimate','lower','upper']].to_markdown(index=False),'','## 与基线/15/17/18/19的配对差','',pair[pair.metric.eq('C_index')][['model_id','reference_id','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_primary_C_family10']].to_markdown(index=False),'','## 十二项匹配探索','',contrasts[['alternative','reference','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_additive12']].to_markdown(index=False),
        '八项新NAM对同监控raw控制（NN/I20），四项宽16对宽8（NLL/C×NN/I20）组成预注册12项探索。主要五个I20对两基线十项C的Holm独立列。两种校正只覆盖本轮家族，不校正历史反复查看外层。患者条件IF先跨MI平均再平方，条件于固定模型/划分/实现MI，不含重训练/重选型/重插补及此前开发乐观；相关100模型不当独立t检验样本。',
        '','## 选择、轮数、泛化、资源','',json.dumps({key:summary[key] for key in ['choices','epochs','inventory','convergence','generalization','resources','minimum_target_met','strict_RCS_plus002_met']},ensure_ascii=False,indent=2),
        '','## 完整性与解释边界','',
        '100新内部/20新外层共享NN状态、50/10来源，600原MI视图，6000内部候选C及1000外层MI输出；不是1000个独立NN拟合。全部保存状态CUDA回放、分组/原列完整性/padding零权重、原输入/源码签名、预算/内部选择、风险与训练Breslow概率复算；60组同宽度监控训练前缀误差0。',
        '14完整20MI预检含4原控制重放、8新候选和2重复；另12次完整并发计时与3次三步warmup，正式拟合单列。1路4拟合54.511秒/13.628秒每次，2路8拟合87.960秒/10.995秒每次；全部逐轮/风险/组合C差0，按吞吐选2，而不是比较不同工作量裸墙钟。物理27.772GiB/可用4GiB/研究RSS8GiB保护不提高，GPU系统采样包括用户其他应用。',
        '原型只用7个合成人，逐变量逐成员独立标量oracle的值/输入与参数梯度、mask梯度、变量独立性和CUDA前后向/恢复通过；它不算临床试跑或达标结果。旧M2单隐层8/lr7e-6/WD.01、498/500封顶，本轮两隐层、八成员、共享MI、不同优化率及排序目标，不能把差异归于单一参数，也不冒充论文逐字复现。',
        '训练loss下降、训练测试gap缩小或内部C提高均不自动表示外层改善；若有3000封顶需按轨迹说明，不能直接判结构无效。纯排序后的Breslow只是训练概率校准，不保证PH分布正确。最低目标仍.76，精确RCS+.02另列，没有新增80/100或p门槛。',
        '固定名单及继承I20内部依赖、in-sample树预测、同队列持续开发/外层反复查看、相关MI局限保留；不称完全嵌套独立无偏验证，不将单轮完成当总目标完成。保存包装包含原Encoder、训练统计与来源权重路径，不是独立单文件部署。',
        '[NAM原文](https://arxiv.org/abs/2004.13912)、[CoxNAM出版方摘要](https://www.sciencedirect.com/science/article/pii/S0957417423007200)、[生存NAM扩展](https://arxiv.org/abs/2211.07814)提供方法动机，不保证本队列效果。未复制上游TF预处理/划分/指标。当前为本地PyTorch纯排序适配。',
        '','![C-index](cindex_distribution.png)','','![固定内部单元学习曲线](learning_curves.png)','','![资源](resources.png)']
    return lines

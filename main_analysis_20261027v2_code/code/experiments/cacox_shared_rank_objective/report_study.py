"""Uniform fixed-model conditional inference and full iteration15 report."""
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
def report_all(s):
    out,m,b,ev=s.OUT,s.m,s.b,s.ev
    def collect(root,pattern):return pd.concat([pd.read_csv(p,float_precision='round_trip') for p in root.glob(pattern)],ignore_index=True)
    frames={r:collect(out,f'outer/fold*/shared/mi*/{r}.csv') for r in s.ROLES}
    for role in ['S0_shared_Cox','J0_I20_shared','JC_I20_selected']:frames[role]=collect(b.io.OUT,f'outer/fold*/shared/mi*/{role}.csv')
    for role in b.previous.ROLES:frames[role]=collect(b.previous.OUT,f'outer/fold*/shared/mi*/{role}.csv')
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
    references=['ordinary_cox','RCS_ridge','I20_previous','S0_shared_Cox','J0_I20_shared','JC_I20_selected','C1_Cstop_NN','C1_Cstop_I20','E3_NLL_I20','E3_Cstop_I20','B2_sharedNN_ranktree_MS',*s.ROLES]
    for role in s.ROLES:
        summary['comparisons'][role]={}
        for ref in references:
            if ref==role:continue
            q=ev.paired(tables[ref],details[ref],tables[role],details[role]);q['p_two_sided_conditional_normal']=q.apply(lambda v:float(2*norm.sf(abs(v.estimate/v.SE))) if v.SE>0 else (1. if v.estimate==0 else 0.),axis=1)
            pairs.append(q.assign(model_id=role,reference_id=ref));summary['comparisons'][role][ref]=q[q.metric.eq('C_index')].iloc[0].to_dict()
    pair=pd.concat(pairs,ignore_index=True);pair['p_Holm_primary_C_family10']=np.nan;primary=pair.model_id.str.endswith('_I20')&pair.reference_id.isin(['ordinary_cox','RCS_ridge'])&pair.metric.eq('C_index');assert primary.sum()==10
    pair.loc[primary,'p_Holm_primary_C_family10']=holm(pair.loc[primary,'p_two_sided_conditional_normal']);pair.to_csv(out/'paired_differences.csv',index=False)
    general=collect(out,'outer/fold*/shared/mi*/generalization.csv');assert len(general)==1000;general['gap']=general.train_C-general.test_C;general.to_csv(out/'generalization_each_model.csv',index=False)
    general.groupby(['role','fold'])[['train_C','test_C','gap']].mean().reset_index().to_csv(out/'generalization_by_fold.csv',index=False);summary['generalization']=general.groupby('role')[['train_C','test_C','gap']].mean().to_dict('index')
    frozen=s.read(out/'FROZEN_CHOICES.json');assert s.freeze(False)=={k:frozen[k] for k in ['epochs','choices']};summary['epochs']=frozen['epochs'];summary['choices']=frozen['choices'];convergence=[]
    for phase in ['inner','outer']:
        for p in (out/phase).glob('fold*/*/fits/*/audit.json'):
            a=s.read(p);assert a['cuda_train'];convergence.append(dict(phase=phase,fold=a['fold'],inner_fold=a['inner_fold'],condition=a['condition']['condition_id'],monitor=a['condition']['monitor'],rank_weight=a['condition']['rank_weight'],
                seed=a['seed'],best_epoch=a['best_epoch'],stopped_epoch=a['stopped_epoch'],cap_hit=a['cap_hit'],fit_seconds=a['fit_seconds'],parameters=a['parameters'],initial_state_SHA=a['initial_state_SHA']))
    conv=pd.DataFrame(convergence);assert conv.phase.eq('inner').sum()==100 and conv.phase.eq('outer').sum()==20;conv.to_csv(out/'convergence.csv',index=False)
    prefixes=pd.concat([pd.read_csv(p,float_precision='round_trip').assign(group=str(p.parent.relative_to(out))) for p in out.glob('*er/fold*/*/monitor_prefix_checks.csv')],ignore_index=True);assert len(prefixes)==60 and prefixes.training_monitor_prefix_error.eq(0).all();prefixes.to_csv(out/'all_monitor_prefix_checks.csv',index=False)
    summary['convergence']=dict(inner_cap_count=int(conv[conv.phase.eq('inner')].cap_hit.sum()),by_condition=conv[conv.phase.eq('inner')].groupby('condition')[['best_epoch','stopped_epoch','fit_seconds']].mean().to_dict('index'),parameters_by_fold=conv.groupby('fold').parameters.first().to_dict(),matched_monitor_prefixes=60)
    resources=pd.DataFrame([json.loads(line) for line in (out/'monitor/resources.jsonl').read_text().splitlines()]);summary['resources']=dict(max_physical_used_GiB=float(resources.physical_used_GiB.max()),max_owned_RSS_GiB=float(resources.owned_RSS_GiB.max()),minimum_available_GiB=float(resources.available_GiB.min()),median_sampled_system_GPU=float(resources.GPU_utilization_percent.median()),peak_system_GPU=float(resources.GPU_utilization_percent.max()),peak_system_GPU_MiB=float(resources.GPU_used_MiB.max()))
    assert summary['resources']['max_physical_used_GiB']<=s.rg.LIMIT/s.rg.GiB and summary['resources']['max_owned_RSS_GiB']<=8
    pr=s.progress();assert pr==dict(inner_groups=25,inner_new_NN=100,outer_groups=5,outer_new_NN=20,outer_MI_units=100)
    summary['inventory']=dict(**pr,reused_inner_NN=50,reused_outer_NN=10,new_MI_outputs=1000,inner_MI_views=500,inner_candidate_grid_rows=6000,folds=5,MI=20,skipped=0,new_tree_kernel_fits=0,
        extra_prefits=s.read(out/'preflight/PREFLIGHT.json')['new_preflight_NN'],extra_benchmark_fullfits=s.read(out/'benchmark/SUMMARY.json')['full_shared_fits_in_passed_cases'],extra_threeepoch_warmups=s.read(out/'benchmark/SUMMARY.json')['warmup3epoch_fits'])
    summary['minimum_target_met']={r:summary['models'][r]['C_index']>=.76 for r in s.ROLES};summary['strict_RCS_plus002_met']={r:summary['models'][r]['C_index']-summary['models']['RCS_ridge']['C_index']>=.02 for r in s.ROLES}
    m.write_json(out/'summary.json',summary);plot(s,summary,one,resources)
    lines=['# 迭代15：共享平均风险的排序训练目标','',f'报告时间：{m.stamp()}。','',
        '14同类独立seed排序高度一致，集成未超过12最佳.7526393；只改C监控也未改善。本轮固定数据、共享TabM架构/优化器和I20，检验最终平均风险的训练目标。不是原始Transformer，也不是弃用v2。',
        'H=.5平均20MI×8成员精确Efron+.5平均20MI可比较对softplus(mean8风险差)；P=纯pair项。H/P各NLL和C监控，固定seed0。四流程各纯NN/I20，另纯NN/组合内部有限选择可回退两个只读原控制。所有五折预算/选择冻结后才新外层，全部条件均输出，不按外层选seed/参数。',
        '固定1140人149死亡/29候选/名单15,9,8,9,8/R MICE20/maxit60/所有内外患者划分/每MI原Encoder和数组。官方紧凑TabM8成员2×64/dropout.15、AdamW.001/WD.0003、clip5/max3000/patience150不变。I20原权重/排序树只读，组合NN权重.5/.75/.5/.5/.5，训练均值/SD，不新训树/核或加RCS预测分支。',
        '','| 方法 | 汇总C（统一条件95%CI） | 100MI输出均值 | 最小–最大 |','|---|---|---|---|']
    for r,a in summary['models'].items():lines.append(f"| {r} | {a['C_index']:.7f}（{a['CI95'][0]:.7f}～{a['CI95'][1]:.7f}） | {a['mean_100']:.7f} | {a['range_100'][0]:.7f}～{a['range_100'][1]:.7f} |")
    lines+=['','## C及AUC','',metrics[['model_id','metric','estimate','lower','upper']].to_markdown(index=False),'','## 配对差值','',pair[pair.metric.eq('C_index')][['model_id','reference_id','estimate','lower','upper','p_two_sided_conditional_normal','p_Holm_primary_C_family10']].to_markdown(index=False),
        '','区间条件于固定模型/划分/已实现20MI，患者影响函数先跨MI平均再平方，不覆盖重新训练/选型/插补不确定性；与历史Rubin口径不同。五组合流程对两基线十个C检验Holm只覆盖该家族，不校正此前多轮开发。其他探索比较不当独立确认。',
        '','## 完整预算、收敛、资源','',json.dumps({k:summary[k] for k in ['epochs','choices','inventory','convergence','generalization','resources','minimum_target_met','strict_RCS_plus002_met']},ensure_ascii=False,indent=2),
        '','## 工程证据及限制','',
        '100新内部/20新外层共享NN，各含20MI；复用50内部/10外层控制，1000MI输出不是1000独立训练。60个同目标不同监控内外共同训练前缀完全一致，目标不同才改变梯度；同初态并不代表最终参数相同。所有保存状态/600输入/6000内部C/1000风险与训练Breslow概率回放，签名/预算重算。',
        '纯排序及混合目标的Breslow是开发集附加校准，不保证PH分布正确；C用原风险，AUC用保存生存风险。排序项基于原Harrell对子约定，不以20n作为独立患者，不跨MI造对子；没有更改mean20MI C的原汇总定义。',
        '合成CPU/CUDA独立oracle、两个固定内部患者组的14次完整预检（含4原控制/8候选/2重复）通过，控制逐轮/保存风险误差0；四类型均衡24任务1/2/3路完整基准逐轮/风险/C一致后选最快。额外验证与正式拟合数分列，不能用预检C冒充外层效果。',
        '[二人部分似然研究](https://academic.oup.com/biostatistics/article/25/1/134/6775140)和[DeepHit](https://ojs.aaai.org/index.php/AAAI/article/view/11842)提供平滑排序动机；本机softplus标量风险不是两论文完整复现，也不是离散C本身，loss下降不保证测试C提高。原Transformer和树G1已有排序损失，不说首次尝试。',
        '同队列持续开发、固定名单及继承I20内部依赖、外层反复查看、相关MI局限仍在，不宣称完全嵌套独立无偏验证。训练gap缩小或内部C提高不能单独称效果改善。若封顶，不能直接归类为方法无效。',
        '权重含原20Encoder元数据；每MI wrappers记录实际新/只读状态来源和SHA、预算、训练标准化与Breslow。不是独立单文件部署。图为固定f1inner1实际记录，不代表所有单元或外层调参曲线；没有虚构源控制未保存的C轨迹。',
        '','![分布](cindex_distribution.png)','','![学习曲线](learning_curves.png)','','![资源](resources.png)']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8');assert s.signature()==s.read(out/'manifest.json') and m.verify_completed(out/'verification')
    for phase in ['inner','outer']:
        for args in s.jobs(phase):
            d=s.directory(phase,args[0],args[1] if phase=='inner' else None);assert m.verify_completed(d)
            for p in (d/'fits').iterdir():assert m.verify_completed(p)
            if phase=='outer':
                for j in range(1,21):assert m.verify_completed(d/f'mi{j:02d}')
    m.write_json(out/'FINAL_INTEGRITY.json',dict(status='passed',inventory=summary['inventory'],source_and_data_unchanged=True,all_fivefold_choices_before_outer=True,all_saved_weights_risks_probabilities_replayed=True,all60_matching_monitor_prefixes_identical=True,all1000_outputs_and_C_verified=True,uniform_conditional_inference=True,resource_budget_passed=True))
    s.log('完整实验结束','全部五折20MI比较四训练/监控设置及内部回退选择',summary,'报告baseline/12最佳/14上轮；目标是否达标仍须最终审核。')
    m.mark_complete(out,['summary.json','FINAL_REPORT.md','FINAL_INTEGRITY.json','metrics_pooled.csv','model_cindex.csv','fold_cindex.csv','paired_differences.csv','conditional_influences.npz','generalization_each_model.csv','generalization_by_fold.csv',
        'convergence.csv','all_monitor_prefix_checks.csv','FROZEN_CHOICES.json','cindex_distribution.png','learning_curves.png','resources.png','verification/COMPLETE.json']);s.status(status='complete',phase='complete',progress=s.progress());s.event('complete')

def plot(s,summary,one,resources):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    out=s.OUT;order=['ordinary_cox','RCS_ridge','I20_previous','S0_shared_Cox','J0_I20_shared','JC_I20_selected','C1_Cstop_I20','E3_NLL_I20',*s.ROLES]
    C=np.array([summary['models'][r]['C_index'] for r in order]);CI=np.array([summary['models'][r]['CI95'] for r in order]);fig,axes=plt.subplots(1,2,figsize=(16,13))
    axes[0].errorbar(C,range(len(order)),xerr=[C-CI[:,0],CI[:,1]-C],fmt='o',capsize=3);axes[0].set(yticks=range(len(order)),yticklabels=order,xlabel='Pooled C / fixed-model/fold/realizedMI conditional95CI');axes[0].invert_yaxis()
    axes[1].boxplot([one[one.model_id.eq(r)].C_index for r in order],vert=False);axes[1].set(yticks=range(1,len(order)+1),yticklabels=order,xlabel='100 correlated outer MI-output C-indices');axes[1].invert_yaxis()
    for ax in axes:ax.axvline(.76,c='gray',ls='--')
    fig.suptitle('Shared mean-risk pair objective:fixed full5fold20MI');fig.tight_layout();fig.savefig(out/'cindex_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8));source=pd.read_csv(s.b.source_dir(1,1,'NLL')/'training_trace.csv')
    axes[0,0].plot(source.epoch,source.training_mean_view_member_Efron,label='source Cox NLLstop');axes[1,0].plot(source.epoch,source.validation_mean_view_ensemble_Efron,label='source Cox NLLstop')
    for cid in s.NEW:
        q=pd.read_csv(out/f'inner/fold01/inner01/fits/{cid}/training_trace.csv')
        if cid.endswith('NLL'):axes[0,0].plot(q.epoch,q.training_mean_view_member_Efron,label=cid);axes[0,1].plot(q.epoch,q.training_mean_view_ensemble_pair_loss,label=cid)
        axes[1,0].plot(q.epoch,q.validation_mean_view_ensemble_Efron,label=cid);axes[1,1].plot(q.epoch,q.validation_C_index,label=cid)
    for ax,title in zip(axes.flat,['Training mean-member Efron','Training mean-risk pair loss','Validation Efron','Validation C:recorded new arms']):ax.set(title=title,xlabel='Epoch');ax.legend(fontsize=8)
    fig.suptitle('Fixed fold1/inner1;actual inner curves,not outertest selection');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);x=resources.unix-resources.unix.min();axes[0].plot(x,resources.physical_used_GiB,label='Physical used');axes[0].plot(x,resources.owned_RSS_GiB,label='Study RSS');axes[0].axhline(s.rg.LIMIT/s.rg.GiB,c='r',ls='--',label='Physical guard');axes[0].set_ylabel('GiB');axes[0].legend()
    q=resources.dropna(subset=['GPU_utilization_percent']);axes[1].plot(q.unix-resources.unix.min(),q.GPU_utilization_percent);axes[1].set(xlabel='Monitored seconds',ylabel='System GPU %',ylim=(0,105));fig.tight_layout();fig.savefig(out/'resources.png',dpi=180);plt.close(fig)

"""Paired full-5x20 outcomes and independently recounted patient concordance."""
import json,itertools
import numpy as np
import pandas as pd


def independent_C(q):
    pooled=[];individual=[]
    for j in range(1,21):
        wins=0.;pairs=0
        for f in range(1,6):
            z=q[q.fold.eq(f)&q.imputation.eq(j)];assert len(z)==228
            t=z.Survival_Days.to_numpy(float);e=z.Death.to_numpy(bool);risk=z.score.to_numpy(float)
            i,k=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
            n=float(((risk[i]>risk[k])+.5*(risk[i]==risk[k])).sum());wins+=n;pairs+=len(i);individual.append(n/len(i))
        pooled.append(wins/pairs)
    return float(np.mean(pooled)),float(np.mean(individual))


def report_all(r):
    out,m,s=r.OUT,r.m,r.s
    frames={name:pd.concat([pd.read_csv(p,float_precision='round_trip') for p in out.glob(f'outer/fold*/mi*/{name}.csv')],ignore_index=True) for name in r.ROLES}
    source_dirs={'I20_previous':(s.r.prior.OUT,'I20_selection'),'MS1_I20_multiscale':(s.r.OUT,'MS1_I20_multiscale'),'E1_pooled_epoch_MS':(s.last.OUT,'E1_pooled_epoch_MS'),
                 'U0_patient_pair':(s.OUT,'U0_patient_pair'),'U1_patient_pair_MS':(s.OUT,'U1_patient_pair_MS')}
    for name,(path,stem) in source_dirs.items():frames[name]=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in path.glob(f'outer/fold*/mi*/{stem}.csv')],ignore_index=True)
    frames['RCS_ridge']=pd.read_csv(s.r.old.REF/'predictions_RCS_ridge.csv',float_precision='round_trip')
    frames['original_CACox']=pd.read_csv(s.r.old.REF/'predictions_original_CACox.csv',float_precision='round_trip')
    history=pd.read_csv(s.OUT/'metrics_pooled.csv',float_precision='round_trip');summary=dict(at=m.stamp(),models={},comparisons={});metrics=[];ones=[];folds=[];recounts=[]
    for name,q in frames.items():
        s.r.old.check(q);m.assert_paired(frames['RCS_ridge'],q)
        metric=s.r.old.pooled(q) if name in [*r.ROLES,'original_CACox'] else history[history.model_id.eq(name)].drop(columns='model_id').copy()
        assert len(metric)==4;metric.insert(0,'model_id',name);metrics.append(metric)
        one=m.individual_cindices(q);assert len(one)==100;one.insert(0,'model_id',name);ones.append(one)
        fold=one.groupby('fold').C_index.agg(['mean','min','max']).reset_index();fold.insert(0,'model_id',name);folds.append(fold)
        v=metric.iloc[0];summary['models'][name]=dict(C_index=float(v.estimate),CI95=[float(v.lower),float(v.upper)],mean_100=float(one.C_index.mean()),range_100=[float(one.C_index.min()),float(one.C_index.max())])
        if name in r.ROLES:
            C,mean=independent_C(q);assert abs(C-v.estimate)<1e-12 and abs(mean-one.C_index.mean())<1e-12
            recounts.append(dict(model_id=name,independent_pooled_C=C,independent_mean100=mean,error=abs(C-v.estimate)))
    metric=pd.concat(metrics);one=pd.concat(ones);metric.to_csv(out/'metrics_pooled.csv',index=False);one.to_csv(out/'model_cindex.csv',index=False);pd.concat(folds).to_csv(out/'fold_cindex.csv',index=False)
    pd.DataFrame(recounts).to_csv(out/'independent_C.csv',index=False);differences=[]
    for role in r.ROLES:
        summary['comparisons'][role]={}
        for ref in ['RCS_ridge','original_CACox','I20_previous','MS1_I20_multiscale','E1_pooled_epoch_MS','U1_patient_pair_MS']:
            v=s.r.old.paired(frames[ref],frames[role]);summary['comparisons'][role][ref]=v;differences.append(dict(model_id=role,reference_id=ref,**v))
    for alt,ref in [('N8_shared_Cox','N1_single_Cox'),('J8_I20_shared','J1_I20_single')]:
        v=s.r.old.paired(frames[ref],frames[alt]);summary['comparisons'][alt][ref]=v;differences.append(dict(model_id=alt,reference_id=ref,**v))
    pd.DataFrame(differences).to_csv(out/'paired_differences.csv',index=False)
    general=pd.concat([pd.read_csv(p,float_precision='round_trip') for p in out.glob('outer/fold*/mi*/generalization.csv')],ignore_index=True);general['gap']=general.train_C-general.test_C
    general.to_csv(out/'generalization_each_model.csv',index=False);general.groupby(['role','fold'])[['train_C','test_C','gap','control']].mean().reset_index().to_csv(out/'generalization_by_fold.csv',index=False)
    summary['generalization']=general.groupby('role')[['train_C','test_C','gap','control']].mean().to_dict('index');summary['choices']=r.read(out/'FROZEN_CHOICES.json')['choices']
    convergence=[]
    for p in out.glob('inner/fold*/mi*/fits/*/audit.json'):
        a=r.read(p);trace=pd.read_csv(p.parent/'training_trace.csv',float_precision='round_trip');assert len(trace)==a['stopped_epoch']
        assert np.isfinite(trace.select_dtypes('number')).all().all()
        assert int(trace.loc[trace.validation_ensemble_Efron.idxmin(),'epoch'])==a['best_epoch']
        assert a['cuda_train'] and a['config']==r.CM[p.parent.name]
        convergence.append(dict(fold=a['fold'],imputation=a['imputation'],inner_fold=a['inner_fold'],config_id=p.parent.name,
            best_epoch=a['best_epoch'],stopped_epoch=a['stopped_epoch'],cap_hit=a['cap_hit'],fit_seconds=a['fit_seconds'],parameters=a['parameters'],
            gradient_clipped_fraction=float((trace.gradient_norm_before_clip>r.b.COMMON['gradient_clip']).mean())))
    conv=pd.DataFrame(convergence);assert len(conv)==2000;conv.to_csv(out/'convergence.csv',index=False)
    summary['convergence']=conv.groupby('config_id')[['best_epoch','stopped_epoch','cap_hit','fit_seconds','gradient_clipped_fraction']].mean().to_dict('index')
    resources=[json.loads(row) for row in (out/'monitor/resources.jsonl').read_text().splitlines()];resource=pd.DataFrame(resources)
    summary['resources']=dict(max_physical_used_GiB=float(resource.physical_used_GiB.max()),max_owned_RSS_GiB=float(resource.owned_RSS_GiB.max()),minimum_available_GiB=float(resource.available_GiB.min()),
        peak_GPU_utilization=float(resource.GPU_utilization_percent.max()),median_sampled_GPU=float(resource.GPU_utilization_percent.median()),peak_GPU_MiB=float(resource.GPU_used_MiB.max()))
    summary['inventory']=dict(inner_units=500,new_inner_NN=2000,outer_units=100,new_outer_NN=sum(r.read(p)['new_neural_models'] for p in out.glob('outer/fold*/mi*/audit.json')),outer_predictions=400,
        benchmark_additional_NN=72,benchmark_three_epoch_warmups=6,preflight_additional_NN=4,no_new_I20_tree_or_old_NN_fits=True)
    assert sum(1 for _ in out.glob('outer/fold*/mi*/fits/*/COMPLETE.json'))==summary['inventory']['new_outer_NN']
    summary['minimum_target_met']={role:summary['models'][role]['C_index']>=.76 for role in r.ROLES}
    summary['strict_plus002_met']={role:summary['comparisons'][role]['RCS_ridge']['estimate']>=.02 for role in r.ROLES}
    assert summary['resources']['max_physical_used_GiB']<=r.rg.LIMIT/r.rg.GiB and summary['resources']['max_owned_RSS_GiB']<=8
    m.write_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    order=['RCS_ridge','original_CACox','I20_previous','MS1_I20_multiscale','E1_pooled_epoch_MS','U1_patient_pair_MS',*r.ROLES]
    cs=np.array([summary['models'][name]['C_index'] for name in order]);ci=np.array([summary['models'][name]['CI95'] for name in order]);fig,axes=plt.subplots(1,2,figsize=(16,9))
    axes[0].errorbar(cs,range(len(order)),xerr=[cs-ci[:,0],ci[:,1]-cs],fmt='o',capsize=3);axes[0].set(yticks=range(len(order)),yticklabels=order,xlabel='Pooled C-index / conditional 95%CI');axes[0].invert_yaxis()
    axes[1].boxplot([one[one.model_id.eq(name)].C_index for name in order],vert=False);axes[1].set(yticks=range(1,len(order)+1),yticklabels=order,xlabel='100 correlated outer-model C-indices');axes[1].invert_yaxis()
    for ax in axes:ax.axvline(.76,ls='--',c='gray')
    fig.suptitle('Compact official TabM-Cox: matched single/shared members and I20 fusion');fig.tight_layout();fig.savefig(out/'cindex_distribution.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5));first=out/'inner/fold01/mi01_inner01'
    for c in r.b.CONFIGS:
        q=pd.read_csv(first/'fits'/c['config_id']/'training_trace.csv');axes[0].plot(q.epoch,q.training_member_mean_Efron,label=c['config_id']);axes[1].plot(q.epoch,q.validation_ensemble_Efron,label=c['config_id'])
    axes[0].set(title='Training: dropout mean member Efron',xlabel='Epoch',ylabel='Mean event NLL');axes[1].set(title='Internal validation: mean-risk Efron',xlabel='Epoch',ylabel='Mean event NLL')
    for ax in axes:ax.legend(fontsize=8)
    fig.suptitle('Predetermined fold1 / MI1 / inner1; different losses, not an outer-test curve');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True);x=resource.unix-resource.unix.min()
    axes[0].plot(x,resource.physical_used_GiB,label='System physical used');axes[0].plot(x,resource.owned_RSS_GiB,label='Study RSS');axes[0].axhline(r.rg.LIMIT/r.rg.GiB,ls='--',c='r',label='Physical guard');axes[0].set_ylabel('GiB');axes[0].legend()
    v=resource.dropna(subset=['GPU_utilization_percent']);axes[1].plot(v.unix-resource.unix.min(),v.GPU_utilization_percent);axes[1].set(xlabel='Monitored seconds',ylabel='GPU %',ylim=(0,105));fig.tight_layout();fig.savefig(out/'resources.png',dpi=180);plt.close(fig)
    lines=['# 紧凑TabM-Cox与原I20组合：完整5折×20MI','',f'完成：{m.stamp()}。','',
        '问题：09患者级正则缓解上一轮对子抽样的退步，但不及原I20/MS1/E1。此轮检验紧凑共享参数神经集成能否提供额外泛化信号，不改数据侧。',
        '旧M4已是8成员TabM＋Weibull/SurvHL/WAS，非Cox；固定极小LR训练437/500封顶。本轮不是第一次试TabM，也不把多因素变化声称为纯换头或纯架构因果对照。',
        '官方TabM0.0.3，同2层×64，Dropout.15，无额外数值嵌入/分箱。匹配k1与k8；每成员精确Efron损失平均训练、预测平均log风险；验证按平均风险Efron损失早停。AdamW LR.0003/.001，WD.0003、clip5、max3000、patience150。',
        '完全同一1140人、29候选、名单15/9/8/9/8、R MICE20与内外成员/Encoder。每折20MI均值后5内折均值选LR和组合NN权重0/.25/.5/.75/1。端点保持原生风险，内部权重按各当前训练均值/SD标准化。所有选择及100内部最佳轮数中位数冻结后才外层；原I20模型不变。',
        '共2000内部新NN拟合，外层新模型按实际共享/回退统计，4组各100结果。训练/验证曲线只来自预先固定内部单元，不逐轮查看外层。',
        '','| 方法 | 汇总C（条件95%CI） | 100模型均值 | 最小–最大 |','|---|---|---|---|']
    for name in order:
        a=summary['models'][name];lines.append(f"| {name} | {a['C_index']:.6f}（{a['CI95'][0]:.6f}～{a['CI95'][1]:.6f}） | {a['mean_100']:.6f} | {a['range_100'][0]:.6f}～{a['range_100'][1]:.6f} |")
    for title,value in [('配对差与条件95CI',summary['comparisons']),('内部冻结配置和轮数',summary['choices']),('开发/测试差距',summary['generalization']),('封顶/优化诊断',summary['convergence']),('模型数量',summary['inventory']),('资源',summary['resources']),('最低.76目标',summary['minimum_target_met']),('额外精确RCS+.02检查',summary['strict_plus002_met'])]:lines+=['','## '+title,'',json.dumps(value,ensure_ascii=False,indent=2)]
    lines+=['','## C与各时点AUC','',metric[['model_id','metric','estimate','lower','upper']].to_markdown(index=False),'',
        '独立数学Efron值/梯度/tie顺序/平移及极端风险检查、实际内部GPU重复和保存回放、1/2/3路并发逐轮及风险一致性通过。CUDA训练和推理；模型保存成员权重及Encoder/源模型/患者ID，最终全部模型重放风险和训练Breslow校准/生存概率。独立患者可比较对重数核对汇总C与100均值，二者分开报告。',
        '同队列继续开发、固定名单/继承I20参数的内部依赖、外层反复查看及插补模型相关局限仍在。条件95CI不是独立外部无偏验证，不因训练gap缩小称成功，不用外层重新选参。有限网格/预算不能宣称全局最优；封顶须单独解释。无RCS预测分支、弃用v2或敏感性。',
        '[TabM原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c1ba41c694834aeef91ae161711d4939-Abstract-Conference.html)；[固定官方代码](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4)。其分类/回归证据不保证此临床生存任务提升。',
        '时间/目的/修改/结果位置见EXPERIMENTS.md，参数及源码manifest冻结。所有inner/outer/neural.pt、训练曲线/成员预测、wrappers、C/AUC/配对/模型/折/差距/资源和图均在本目录。',
        '', '![C-index](cindex_distribution.png)','','![内部曲线](learning_curves.png)','','![资源](resources.png)']
    (out/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    assert r.signature()==r.read(out/'manifest.json') and r.progress()==dict(inner=500,outer=100) and m.verify_completed(out/'verification')
    for p in itertools.chain(out.glob('inner/fold*/mi*/COMPLETE.json'),out.glob('outer/fold*/mi*/COMPLETE.json')):assert m.verify_completed(p.parent)
    assert sum(1 for _ in out.glob('inner/fold*/mi*/fits/*/COMPLETE.json'))==2000
    m.write_json(out/'FINAL_INTEGRITY.json',dict(status='passed',folds=5,imputations=20,skipped=0,inventory=summary['inventory'],
        saved_member_risks_and_survival_replayed=True,independent_C=True,source_data_and_I20_unchanged=True,all_choices_before_outer=True,resource_budget_passed=True))
    r.log('正式实验完整结束','比较新神经与I20组合、匹配RCS和上轮',summary,'真实完整5×20；是否达标按汇总C，不将内部预检当外层。')
    m.mark_complete(out,['summary.json','FINAL_REPORT.md','FINAL_INTEGRITY.json','metrics_pooled.csv','model_cindex.csv','fold_cindex.csv','paired_differences.csv','independent_C.csv',
        'convergence.csv','generalization_each_model.csv','generalization_by_fold.csv','FROZEN_CHOICES.json','cindex_distribution.png','learning_curves.png','resources.png','verification/COMPLETE.json'])
    r.status(status='complete',phase='complete',progress=r.progress());r.event('complete')

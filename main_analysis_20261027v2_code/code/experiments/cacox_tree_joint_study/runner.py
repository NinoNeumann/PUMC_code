"""Prespecified joint tree/blend selection from inner predictions only."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import argparse,json,sys,time,subprocess,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import torch
import joblib
import psutil
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/cacox_original_head_study/src'))
from cacox_original_head_study.runtime import configure,stamp,sha
from cacox_original_head_study.storage import write_json,mark_complete,verify_completed,run_lock
from cacox_original_head_study.data import Bundle
from cacox_original_head_study.training import fit,FittedModel
from cacox_original_head_study.models import make_model
from cacox_original_head_study._numeric_encoder import Encoder
from cacox_original_head_study._cox_prediction import breslow,survival_from_baseline
from cacox_original_head_study.evaluation import normalize,evaluate,paired_difference,individual_cindices,assert_complete,assert_paired

OUT=PROJECT/'results/cacox_tree_joint_study/20261030_v1'
ORIGINAL=PROJECT/'results/cacox_tree_selected29_integrated_20261028_v1'
COMMON=json.loads((PROJECT/'experiments/cacox_original_head_study/configs/study_v1.json').read_text())['common']
SPEC=dict(model_id='O0_original_cox',kind='cox',bins=1)
DEPTHS=[1,2,3];COUNTS=[75,150,300,600];ALPHAS=[.1,.25,.5,.75,.9]
VARIANTS=['J1_probability730','J2_probability1095','J3_standardized_logrisk']
TREE_SETTINGS=dict(loss='coxph',learning_rate=.03,subsample=.8,min_samples_leaf=25,max_features=None)
PLAN=dict(study='cacox_tree_joint_study_20261030_v1',common=COMMON,depths=DEPTHS,trees=COUNTS,alpha_CACox=ALPHAS,
    variants=VARIANTS,tree_settings=TREE_SETTINGS,workers=5,
    selection='one joint depth/trees/alpha per outer fold and variant, max mean C over 5 inner folds x20 MI; tie fewer trees, smaller depth, alpha nearest .5',
    target='pooled C >= matched ordinary Cox + .02; no 80/100 requirement',target_C=.7590991164041492,
    data='1140 patients,29 candidates,original R MICE and Boruta+LASSO lists15/9/8/9/8; no new preprocessing',
    no_outer_tuning=True,all_inner_before_outer=True,no_v2=True)


def bundle(fold):return Bundle(ORIGINAL/f'bundles/repeat01_fold{fold:02d}/main')
def status(path,**kw):
    try:write_json(path,dict(at=stamp(),pid=os.getpid(),**kw))
    except OSError as e:print('TELEMETRY_RETRY',repr(e),flush=True)
def record(title,purpose,params,location,result,start=None):
    with (OUT/'EXPERIMENTS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## {title}\n\n- 时间：{start or stamp()} → {stamp()}\n- 目的：{purpose}\n- 参数：`{json.dumps(params,ensure_ascii=False)}`\n- 结果位置：{location}\n- 结果说明：{result}\n')
def standardize(train,val):
    sd=max(float(np.std(train)),1e-8);return (train-np.mean(train))/sd,(val-np.mean(train))/sd
def scores(fitted,train,val):
    fitted.net.eval()
    with torch.no_grad():
        a=fitted.net(torch.as_tensor(fitted.encoder.transform(train),device='cuda')).cpu().numpy().reshape(-1).astype(float)
        b=fitted.net(torch.as_tensor(fitted.encoder.transform(val),device='cuda')).cpu().numpy().reshape(-1).astype(float)
    return a,b,survival_from_baseline(a,fitted.baseline,COMMON['times']),survival_from_baseline(b,fitted.baseline,COMMON['times'])
def tree_arrays(train,val,variables,fold,j,k,depth,counts):
    enc=Encoder(variables,'robust');x=enc.fit_transform(train).astype(float);vx=enc.transform(val).astype(float)
    seed=100000+fold*1000+j*10+k
    model=GradientBoostingSurvivalAnalysis(**TREE_SETTINGS,max_depth=depth,n_estimators=max(counts),random_state=seed).fit(
        x,Surv.from_arrays(train.Death.to_numpy(bool),train.Survival_Days.to_numpy(float)))
    found={}
    for count,(a,b) in enumerate(zip(model.staged_predict(x),model.staged_predict(vx)),1):
        if count in counts:
            base=breslow(train.Survival_Days,train.Death,a)
            found[count]=(a,b,survival_from_baseline(a,base,COMMON['times']),survival_from_baseline(b,base,COMMON['times']))
    return enc,model,found
def c_evaluator(frame):
    t=frame.Survival_Days.to_numpy();e=frame.Death.to_numpy()
    i,j=np.nonzero((e[:,None]==1)&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&(e[None,:]==0))))
    if not len(i):raise ValueError('No comparable pairs')
    return lambda s:float(np.mean((s[i]>s[j])+.5*(s[i]==s[j])))
def blend(c,t,alpha,variant):
    if variant=='J1_probability730':return 1-(alpha*c[1][:,1]+(1-alpha)*t[1][:,1])
    if variant=='J2_probability1095':return 1-(alpha*c[1][:,2]+(1-alpha)*t[1][:,2])
    return alpha*c[0]+(1-alpha)*t[0]


def inner_job(fold,j):
    b=bundle(fold);done=[]
    for k in range(1,6):
        d=OUT/f'inner/fold{fold:02d}/mi{j:02d}_inner{k:02d}'
        if verify_completed(d):continue
        start=time.monotonic();train,val=b.pair(j,k)
        if (set(train.patient_id)|set(val.patient_id))&set(b.raw_assessment.patient_id):raise ValueError('Outer leakage')
        state=dict(stage='inner',fold=fold,imputation=j,inner_fold=k)
        fitted,trace,audit=fit(train,b.variables,SPEC,COMMON,COMMON['seed_base']+(fold-1)*1000+j*10000+k*100,
            validation=val,progress=lambda p:status(d/'RUN_STATUS.json',**state,epoch=p))
        ca,cv,cas,cvs=scores(fitted,train,val);cz,cvz=standardize(ca,cv)
        arrays=dict(c_train_score=ca,c_val_score=cv,c_train_survival=cas,c_val_survival=cvs,
            train_ids=train.patient_id.to_numpy(),val_ids=val.patient_id.to_numpy())
        del fitted;C=c_evaluator(val);rows=[]
        for depth in DEPTHS:
            status(d/'RUN_STATUS.json',**state,tree_depth=depth)
            _,_,found=tree_arrays(train,val,b.variables,fold,j,k,depth,COUNTS)
            for count,(ta,tv,tas,tvs) in found.items():
                tz,tvz=standardize(ta,tv)
                arrays.update({f'train_score_d{depth}_n{count}':ta,f'val_score_d{depth}_n{count}':tv,
                    f'train_survival_d{depth}_n{count}':tas,f'val_survival_d{depth}_n{count}':tvs})
                for variant in VARIANTS:
                    for alpha in ALPHAS:
                        rows.append(dict(fold=fold,imputation=j,inner_fold=k,variant=variant,depth=depth,trees=count,alpha=alpha,
                            C_index=C(blend((cvz,cvs),(tvz,tvs),alpha,variant))))
        d.mkdir(parents=True,exist_ok=True);np.savez_compressed(d/'predictions.npz',**arrays)
        trace.to_csv(d/'training_trace.csv',index=False);pd.DataFrame(rows).to_csv(d/'grid.csv',index=False)
        write_json(d/'audit.json',dict(**audit,elapsed_seconds=time.monotonic()-start,variables=b.variables,outer_evaluated=False))
        mark_complete(d,['predictions.npz','training_trace.csv','grid.csv','audit.json'])
    return dict(fold=fold,imputation=j,phase='inner')


def load_original(fold,j):
    key=f'repeat01_fold{fold:02d}'
    d=ORIGINAL/f'gpu_runs/{key}_attempt01/{key}/main/imputation_{j:02d}'
    proof=json.loads((d/'COMPLETE.json').read_text());assert sha(d/'cacox.pt')==proof['outputs']['cacox.pt']
    obj=torch.load(d/'cacox.pt',map_location='cpu',weights_only=False)
    enc=Encoder(obj['encoder']['variables'],COMMON['scale']);enc.__dict__.update(obj['encoder'])
    net=make_model(len(enc.columns),SPEC,COMMON,obj['seed']);net.load_state_dict(obj['state_dict'])
    return FittedModel(net,enc,SPEC,COMMON,None,obj['baseline'])


def select():
    choices={}
    for fold in range(1,6):
        paths=list((OUT/f'inner/fold{fold:02d}').glob('mi*/grid.csv'));assert len(paths)==100
        data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
        means=data.groupby(['variant','depth','trees','alpha']).C_index.agg(['mean','count']).reset_index()
        assert means['count'].eq(100).all();means.to_csv(OUT/f'selection_grid_fold{fold:02d}.csv',index=False)
        choices[str(fold)]={}
        for variant in VARIANTS:
            q=means[means.variant.eq(variant)].copy();q['distance']=abs(q.alpha-.5)
            row=q.sort_values(['mean','trees','depth','distance','alpha'],ascending=[False,True,True,True,True]).iloc[0]
            choices[str(fold)][variant]=dict(depth=int(row.depth),trees=int(row.trees),alpha=float(row.alpha),inner_C=float(row['mean']))
    write_json(OUT/'FROZEN_CHOICES.json',dict(at=stamp(),inner_fits=500,choices=choices,no_outer_evaluated=True))
    record('内部联合选择完成','在100个内部验证单元上选择每折共同配置',choices,OUT/'FROZEN_CHOICES.json','三个方案全部五折冻结，随后才开始外层。')


def outer_job(fold,j):
    d=OUT/f'outer/fold{fold:02d}/mi{j:02d}'
    if verify_completed(d):return dict(fold=fold,imputation=j,phase='outer',reused=True)
    frozen=json.loads((OUT/'FROZEN_CHOICES.json').read_text());assert frozen['inner_fits']==500
    choices=frozen['choices'][str(fold)];b=bundle(fold);train,val=b.pair(j)
    fitted=load_original(fold,j);ca,cv,cas,cvs=scores(fitted,train,val);cz,cvz=standardize(ca,cv)
    del fitted;rows=[];outputs=[]
    for variant in VARIANTS:
        choice=choices[variant];depth=choice['depth'];count=choice['trees'];alpha=choice['alpha']
        enc,model,found=tree_arrays(train,val,b.variables,fold,j,9,depth,[count])
        ta,tv,tas,tvs=found[count];tz,tvz=standardize(ta,tv)
        risk=blend((cvz,cvs),(tvz,tvs),alpha,variant)
        if variant=='J3_standardized_logrisk':
            train_risk=alpha*cz+(1-alpha)*tz
            base=breslow(train.Survival_Days,train.Death,train_risk)
            survival=survival_from_baseline(risk,base,COMMON['times'])
        else:base=None;survival=alpha*cvs+(1-alpha)*tvs
        q=val[['patient_id','Death','Survival_Days']].copy();q['repeat_id']=1;q['fold']=fold;q['imputation']=j;q['score']=risk
        for i,t in enumerate(COMMON['times']):q[f'survival_{t}d']=survival[:,i]
        q=normalize(q,variant,PLAN['study'],'blended survival risk' if base is None else 'blended standardized Cox log-risk')
        d.mkdir(parents=True,exist_ok=True);q.to_csv(d/f'{variant}.csv',index=False)
        joblib.dump(dict(tree=model,encoder=enc.__dict__,choice=choice,baseline=base,
            c_mean=float(np.mean(ca)),c_sd=float(np.std(ca)),t_mean=float(np.mean(ta)),t_sd=float(np.std(ta)),
            original_CACox_reference=str(ORIGINAL),fold=fold,imputation=j),d/f'{variant}.joblib')
        outputs.extend([f'{variant}.csv',f'{variant}.joblib'])
    write_json(d/'audit.json',dict(fold=fold,imputation=j,choices=choices,variables=b.variables,at=stamp()))
    mark_complete(d,outputs+['audit.json']);return dict(fold=fold,imputation=j,phase='outer')


def fingerprints():
    source={str(p.relative_to(PROJECT)):sha(p) for p in HERE.glob('*') if p.is_file()}
    package=PROJECT/'experiments/cacox_original_head_study/src/cacox_original_head_study'
    source.update({str(p.relative_to(PROJECT)):sha(p) for p in package.rglob('*.py')})
    inputs={}
    for fold in range(1,6):
        b=bundle(fold);assert verify_completed(b.directory)
        assert len(b.variables)==[15,9,8,9,8][fold-1]
        for p in [b.directory/'COMPLETE.json',b.directory/'bundle.json']:inputs[str(p.relative_to(PROJECT))]=sha(p)
        key=f'repeat01_fold{fold:02d}'
        for j in range(1,21):
            d=ORIGINAL/f'gpu_runs/{key}_attempt01/{key}/main/imputation_{j:02d}'
            assert verify_completed(d)
            for p in [d/'cacox.pt',d/'training.json']:inputs[str(p.relative_to(PROJECT))]=sha(p)
    ref=PROJECT/'results/ordinary_cox_matched_comparison/20261030_v1'
    assert verify_completed(ref)
    for p in [ref/'comparisons/predictions_ordinary_cox.csv',ref/'comparisons/summary.json',
        ORIGINAL/'comparison/paired_predictions_all20.csv']:inputs[str(p.relative_to(PROJECT))]=sha(p)
    return dict(source=source,inputs=inputs,plan=PLAN)


def report():
    frames={v:pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted((OUT/'outer').glob(f'fold*/mi*/{v}.csv'))],ignore_index=True) for v in VARIANTS}
    plain=pd.read_csv(PROJECT/'results/ordinary_cox_matched_comparison/20261030_v1/comparisons/predictions_ordinary_cox.csv',float_precision='round_trip')
    wide=pd.read_csv(ORIGINAL/'comparison/paired_predictions_all20.csv',float_precision='round_trip')
    old=wide[['repeat_id','fold','imputation','patient_id','Death','Survival_Days']].copy();old['score']=wide.blend_score
    metrics=[];individual=[];diffs=[];summary=dict(at=stamp(),models={},comparisons={},target_met={})
    for role,q in dict(ordinary_cox=plain,previous_equal_blend=old,**frames).items():
        assert_complete(q);assert_paired(plain,q)
        _,m=evaluate(q,times=() if role=='previous_equal_blend' else (365,730,1095));m.insert(0,'model_id',role);metrics.append(m)
        one=individual_cindices(q);one.insert(0,'model_id',role);individual.append(one);c=m.iloc[0]
        summary['models'][role]=dict(C_index=float(c.estimate),CI95=[float(c.lower),float(c.upper)],mean_100=float(one.C_index.mean()),range_100=[float(one.C_index.min()),float(one.C_index.max())])
    for role,q in frames.items():
        summary['comparisons'][role]={}
        for ref,reference in [('ordinary_cox',plain),('previous_equal_blend',old)]:
            _,delta=paired_difference(reference,q,times=());delta.insert(0,'model_id',role);delta.insert(1,'reference_id',ref);diffs.append(delta)
            c=delta.iloc[0];summary['comparisons'][role][ref]=dict(delta=float(c.estimate),CI95=[float(c.lower),float(c.upper)],p_value=float(c.p_value_two_sided))
            if ref=='ordinary_cox':summary['target_met'][role]=bool(c.estimate>=.02)
    pd.concat(metrics).to_csv(OUT/'metrics_pooled.csv',index=False);models=pd.concat(individual);models.to_csv(OUT/'model_cindex.csv',index=False)
    pd.concat(diffs).to_csv(OUT/'paired_differences.csv',index=False);write_json(OUT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5));order=list(summary['models'])
    ax.boxplot([models[models.model_id.eq(v)].C_index for v in order]);ax.set(xticks=range(1,len(order)+1),xticklabels=order,ylabel='Outer test C-index')
    ax.tick_params(axis='x',labelrotation=15);fig.tight_layout();fig.savefig(OUT/'cindex_distribution.png',dpi=170);plt.close(fig)
    text=['# CACox＋tree联合优化结果','',f'完成时间：{stamp()}。','', '| 模型 | 汇总C-index（95%CI） | 100模型均值与范围 |','|---|---|---|']
    for role,q in summary['models'].items():text.append(f"| {role} | {q['C_index']:.5f}（{q['CI95'][0]:.5f}～{q['CI95'][1]:.5f}） | {q['mean_100']:.5f}（{q['range_100'][0]:.5f}～{q['range_100'][1]:.5f}） |")
    text+=['',json.dumps(summary['comparisons'],ensure_ascii=False,indent=2),'',f"达标：{summary['target_met']}",'',
        '仅内部联合选择；普通Cox与旧等权组合只读。当前为同队列继续开发，外层已多轮查看；区间未校正选择及多轮开发乐观偏差，100插补模型相关。',
        '', '![分布](cindex_distribution.png)']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(text),encoding='utf-8')
    manifest=json.loads((OUT/'manifest.json').read_text());assert fingerprints()==manifest
    write_json(OUT/'FINAL_INTEGRITY.json',dict(status='passed',inner=500,outer_per_variant=100,folds=5,skipped=0,source_and_inputs_unchanged=True))
    record('最终配对评价','评价相对普通Cox提升是否达到.02',summary['comparisons'],OUT,'三组完整100外层结果；目标判定：'+str(summary['target_met']))
    mark_complete(OUT,['summary.json','metrics_pooled.csv','model_cindex.csv','paired_differences.csv','FINAL_REPORT.md','FINAL_INTEGRITY.json','cindex_distribution.png'])
    status(OUT/'RUN_STATUS.json',status='complete',summary=summary)


def counts():return dict(inner=sum(1 for _ in (OUT/'inner').glob('fold*/mi*/COMPLETE.json')),outer=sum(1 for _ in (OUT/'outer').glob('fold*/mi*/COMPLETE.json')))
def schedule(phase):
    job=inner_job if phase=='inner' else outer_job
    pending=[(f,j) for f in range(1,6) for j in range(1,21)]
    with ProcessPoolExecutor(max_workers=5,initializer=configure) as pool:
        futures={pool.submit(job,f,j):(f,j) for f,j in pending}
        while futures:
            finished=[f for f in futures if f.done()]
            for f in finished:
                result=f.result();del futures[f]
                with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as log:log.write(json.dumps(dict(at=stamp(),**result))+'\n')
            status(OUT/'RUN_STATUS.json',status='running',phase=phase,progress=counts(),pending_jobs=len(futures),worker_pids=[p.pid for p in pool._processes.values()])
            if futures:time.sleep(3)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise ValueError('Frozen study already exists')
    status(OUT/'RUN_STATUS.json',status='preparing',phase='validation')
    write_json(OUT/'config.json',PLAN);manifest=fingerprints()
    train,val=bundle(1).pair(1,1)
    fitted,trace,audit=fit(train,bundle(1).variables,SPEC,COMMON,20271122,validation=val)
    old=pd.read_csv(ORIGINAL/'gpu_runs/repeat01_fold01_attempt01/repeat01_fold01/main/imputation_01/inner_training.csv',float_precision='round_trip').query('inner_fold==1')
    np.testing.assert_array_equal(trace.to_numpy(),old[['epoch','training_loss','inner_validation_loss']].to_numpy())
    del fitted
    # Confirm max600 uses the same first300 trees as max300 and the old code.
    enc,m,a=tree_arrays(train,val,bundle(1).variables,1,1,1,2,[75,150,300,600])
    enc2,m2,b=tree_arrays(train,val,bundle(1).variables,1,1,1,2,[300])
    np.testing.assert_array_equal(a[300][1],b[300][1])
    reference=np.array([fn(730.) for fn in m2.predict_survival_function(enc2.transform(val).astype(float))])
    np.testing.assert_allclose(reference,b[300][3][:,1],atol=1e-12,rtol=0)
    cv=pd.read_csv(ORIGINAL/'tree_models/repeat01_fold01/imputation_01/inner_cv.csv')
    expected=cv.query('inner_fold==1 and depth==2 and trees==300').C_index.iloc[0]
    assert abs(c_evaluator(val)(b[300][1])-expected)<1e-12
    write_json(OUT/'VALIDATION.json',dict(status='passed',at=stamp(),original_trace_error=0.,tree_prefix_error=0.,survival_parity=True,old_tree_C_parity=True,outer_evaluated=False))
    write_json(OUT/'manifest.json',manifest)
    record('方案冻结与试跑','原始CACox复现、树前缀及生存概率校验',PLAN,OUT/'VALIDATION.json','均通过，尚未开始正式内部及外层评价')


def run():
    assert fingerprints()==json.loads((OUT/'manifest.json').read_text())
    with run_lock(OUT):
        try:
            schedule('inner');assert counts()['inner']==500;select();schedule('outer');assert counts()['outer']==100
            status(OUT/'RUN_STATUS.json',status='running',phase='paired_summary',progress=counts());report()
        except BaseException as e:
            import traceback
            status(OUT/'RUN_STATUS.json',status='failed',error=repr(e),traceback=traceback.format_exc(),progress=counts());raise


def launch():
    assert fingerprints()==json.loads((OUT/'manifest.json').read_text())
    active=[]
    for p in psutil.process_iter(['pid','cmdline']):
        argv=p.info['cmdline'] or []
        if p.pid!=os.getpid() and any('cacox_tree_joint_study' in v for v in argv) and 'run' in argv:active.append(p.pid)
    if active:raise RuntimeError('Already running: '+str(active))
    if (OUT/'RUNNING.lock').exists():raise RuntimeError('Inspect stale lock before recovery')
    stdout=(OUT/'controller.out.log').open('a',encoding='utf-8');stderr=(OUT/'controller.err.log').open('a',encoding='utf-8')
    child=subprocess.Popen([sys.executable,str(__file__),'run'],cwd=PROJECT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    stdout.close();stderr.close();write_json(OUT/'LAUNCH.json',dict(at=stamp(),launcher_pid=child.pid,workers=5));print('LAUNCHED',child.pid)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','launch','run']);args=p.parse_args();configure()
    {'prepare':prepare,'launch':launch,'run':run}[args.command]()

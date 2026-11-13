"""Frozen five-fold/three-imputation CUDA pilot with paired CPU references."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import psutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import cuda_models as gpu
spec=importlib.util.spec_from_file_location('cpu_rank_forest_reference',ROOT/'experiments/cacox_rank_forest/runner.py')
cpu=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cpu)
m=cpu.m
from cacox_original_head_study import metrics_core as core
from cacox_original_head_study.evaluation import KEY

OUT=ROOT/'results/cacox_cuda_pilot/20261030_v1'
MIS=[1,2,3]
GPU_ROLES=['G1_cuda_rank','G2_cuda_survival_forest']
CPU_ROLES=['CPU_P1_rank','CPU_F1_forest']
ROLES=GPU_ROLES+CPU_ROLES
PLAN=dict(study='cacox_cuda_pilot_20261030_v1',imputations=MIS,outer_folds=[1,2,3,4,5],inner_folds=[1,2,3,4,5],
          inner_units=75,outer_units=15,selection='fixed MI indices, never selected by score; all inner choices frozen before any new outer evaluation',
          rank=dict(depths=[1,2],leaves=[25,50],trees=[0,75,150,300,600],eta=.03,subsample=.8,max_bin=32,offset_scale=.5,unit_working_hessian=True),
          forest=dict(depths=[3,5],leaves=[15,30],trees=300,max_features='sqrt',bins=32,bootstrap=True,min_leaf_unique=True),
          train_device='cuda:0',inference_device='cuda:0',no_neural_training=True,
          immutable='1140 patients,29 candidates,5 outer and5 inner partitions,R MICE20 existing files,selected15/9/8/9/8,originalCACox parameters weights predictions',
          cpu_reference='Same75 internal cells; old CPU grids reused read-only. Same15 external models refitted after choices frozen.',
          target='Exploratory subset estimate ordinary Cox +.02; not full20-MI confirmation',
          limitations='Histogram thresholds and RNG differ from old CPU tree libraries; fixed selected-list/cache dependencies; outer already viewed repeatedly; MI correlated.')


def status(**kw):
    m.status(OUT/'RUN_STATUS.json',controller_pid=os.getpid(),**kw)


def log(title,purpose,parameters,location,result,start=None):
    with (OUT/'EXPERIMENTS.md').open('a',encoding='utf-8') as f:
        f.write(f'\n## {title}\n\n- 时间：{start or m.stamp()} → {m.stamp()}\n- 目的：{purpose}\n- 参数：`{json.dumps(parameters,ensure_ascii=False)}`\n- 结果位置：{location}\n- 结果说明：{result}\n')


def progress():
    return dict(inner=sum(1 for _ in (OUT/'inner').glob('fold*/mi*/COMPLETE.json')),
                outer=sum(1 for _ in (OUT/'outer').glob('fold*/mi*/COMPLETE.json')))


def source_cache(f,j,k):
    p=m.OUT/f'inner/fold{f:02d}/mi{j:02d}_inner{k:02d}'
    if not m.verify_completed(p):
        raise ValueError('Missing or changed frozen neural prediction cache')
    return p


def signature():
    inputs={}
    for f in range(1,6):
        b=m.bundle(f)
        if not m.verify_completed(b.directory):
            raise ValueError('Unverified frozen bundle')
        assert len(b.variables)==[15,9,8,9,8][f-1]
        inputs[str((b.directory/'COMPLETE.json').relative_to(ROOT))]=m.sha(b.directory/'COMPLETE.json')
        for j in MIS:
            for k in range(1,6):
                p=source_cache(f,j,k)
                for name in ['COMPLETE.json','predictions.npz']:
                    inputs[str((p/name).relative_to(ROOT))]=m.sha(p/name)
                q=cpu.OUT/f'inner/fold{f:02d}/mi{j:02d}_inner{k:02d}'
                if not m.verify_completed(q):
                    raise ValueError('Missing CPU pilot grid; never silently change MI selection')
                inputs[str((q/'COMPLETE.json').relative_to(ROOT))]=m.sha(q/'COMPLETE.json')
                inputs[str((q/'grid.csv').relative_to(ROOT))]=m.sha(q/'grid.csv')
            key=f'repeat01_fold{f:02d}'
            p=m.ORIGINAL/f'gpu_runs/{key}_attempt01/{key}/main/imputation_{j:02d}'
            if not m.verify_completed(p):
                raise ValueError('Neural checkpoint signature failed')
            inputs[str((p/'cacox.pt').relative_to(ROOT))]=m.sha(p/'cacox.pt')
            for study in ['cacox_tree_joint_study','cacox_residual_tree']:
                q=ROOT/f'results/{study}/20261030_v1/outer/fold{f:02d}/mi{j:02d}'
                if not m.verify_completed(q):
                    raise ValueError('Unverified historical outer reference')
                inputs[str((q/'COMPLETE.json').relative_to(ROOT))]=m.sha(q/'COMPLETE.json')
    for p in [ROOT/f'results/ordinary_cox_matched_comparison/20261030_v1/comparisons/predictions_{role}.csv' for role in ['ordinary_cox','original_CACox','RCS_ridge']]:
        inputs[str(p.relative_to(ROOT))]=m.sha(p)
    files=list(HERE.glob('*.py'))+[HERE/'README.md',ROOT/'experiments/cacox_rank_forest/runner.py',ROOT/'experiments/cacox_rank_forest/learners.py',ROOT/'experiments/cacox_tree_joint_study/runner.py']
    files+=list((ROOT/'experiments/cacox_original_head_study/src').rglob('*.py'))
    return dict(plan=PLAN,source={str(p.relative_to(ROOT)):m.sha(p) for p in files},inputs=inputs,
                versions=dict(torch=torch.__version__,xgboost=gpu.xgb.__version__,cupy=gpu.cp.__version__))


def load_inner(f,j,k):
    b=m.bundle(f)
    train,val=b.pair(j,k)
    if (set(train.patient_id)|set(val.patient_id))&set(b.raw_assessment.patient_id):
        raise ValueError('Outer patients in inner partition')
    with np.load(source_cache(f,j,k)/'predictions.npz') as z:
        np.testing.assert_array_equal(z['train_ids'],train.patient_id)
        np.testing.assert_array_equal(z['val_ids'],val.patient_id)
        arrays={name:z[name].copy() for name in ['c_train_score','c_val_score','c_train_survival','c_val_survival']}
    enc=m.Encoder(b.variables,'robust')
    return b,train,val,enc,enc.fit_transform(train),enc.transform(val),arrays


def validate():
    d=OUT/'validation';d.mkdir(parents=True,exist_ok=True)
    status(status='preparing',phase='cuda_validation')
    assert torch.cuda.is_available()
    t=np.array([10.,10.,20.,30.,40.,50.]);e=np.array([1,0,1,0,1,1]);a=np.array([.2,-.4,.6,.3,-.7,.1])
    pair=gpu.comparable_pairs(t,e,'cuda')
    score=torch.tensor(a,device='cuda',dtype=torch.float64,requires_grad=True)
    i,j=pair
    loss=len(t)*torch.nn.functional.softplus(score[j]-score[i]).mean()
    np.testing.assert_allclose(gpu.rank_gradient(score.detach(),pair).cpu(),torch.autograd.grad(loss,score)[0].cpu(),atol=1e-12,rtol=0)
    assert (0,1) in set(zip(i.cpu().tolist(),j.cpu().tolist()))
    assert (1,0) not in set(zip(i.cpu().tolist(),j.cpu().tolist()))
    # Independent scalar log-rank oracle, including tied death/censor times.
    membership=np.array([1,0,1,0,0,1],bool);times=np.unique(t[e==1])
    y=np.array([(t>=u).sum() for u in times],float);de=np.array([((t==u)&(e==1)).sum() for u in times],float)
    yl=np.array([((t>=u)&membership).sum() for u in times],float);dl=np.array([((t==u)&(e==1)&membership).sum() for u in times],float)
    u=sum(dl-yl/y*de);v=sum(yl*(y-yl)*de*(y-de)/(y*y*np.maximum(y-1,1)))
    cuda_stat=gpu.HistogramSurvivalForest.logrank(*[torch.tensor(z,device='cuda') for z in [yl,dl,y,de]])
    np.testing.assert_allclose(float(cuda_stat),u*u/v,rtol=1e-12,atol=1e-12)
    b,train,val,enc,x,vx,z=load_inner(1,1,1)
    ca,cv=z['c_train_score'],z['c_val_score']
    timings=[];differences={}
    # Same histogram algorithm on CPU/CUDA; deliberately distinguish from sklearn exact splits.
    for device in ['cpu','cuda']:
        forest=gpu.HistogramSurvivalForest(3,30,201011,trees=300,device=device).fit(x,train.Survival_Days,train.Death)
        gpu.sync(device);start=time.perf_counter();pr=forest.predict_survival(vx);gpu.sync(device)
        timings.append(dict(role='histogram_RSF_same_algorithm',device=device,fit_seconds=forest.fit_seconds,inference_seconds=time.perf_counter()-start))
        if device=='cpu':
            reference=pr.cpu().numpy();reference_feature=forest.feature.cpu();reference_threshold=forest.threshold.cpu()
        else:
            differences['forest_cpu_cuda_max_abs']=float(np.max(np.abs(reference-pr.cpu().numpy())))
            np.testing.assert_allclose(reference,pr.cpu().numpy(),atol=1e-10,rtol=0)
            assert torch.equal(reference_feature,forest.feature.cpu()) and torch.equal(reference_threshold,forest.threshold.cpu())
            torch.save(forest.state(),d/'forest.pt')
            restored=gpu.HistogramSurvivalForest.restore(torch.load(d/'forest.pt',weights_only=False))
            np.testing.assert_array_equal(pr.cpu().numpy(),restored.predict_survival(vx).cpu().numpy())
            # Verify every leaf's KM probability using its actual bootstrap members.
            weights=np.stack([np.bincount(v,minlength=len(x)) for v in np.random.RandomState(forest.seed).randint(0,len(x),size=(forest.trees,len(x)))])
            xb=forest._bin(x).cpu().numpy();features=forest.feature.cpu().numpy();threshold=forest.threshold.cpu().numpy()
            route=np.zeros((forest.trees,len(x)),int)
            for level in range(forest.depth):
                ff=features[np.arange(forest.trees)[:,None],route];bb=threshold[np.arange(forest.trees)[:,None],route]
                route=np.where(ff>=0,2*route+1+(xb[np.arange(len(x))[None,:],np.maximum(ff,0)]>bb),route)
            for tree in [0,1,2]:
                for leaf in np.unique(route[tree]):
                    ww=weights[tree]*(route[tree]==leaf);tt=train.Survival_Days.to_numpy();ee=train.Death.to_numpy()
                    survival=[]
                    for horizon in forest.horizons:
                        sf=1.
                        for eventtime in np.unique(tt[(ee==1)&(ww>0)&(tt<=horizon)]):
                            sf*=1-ww[(tt==eventtime)&(ee==1)].sum()/ww[tt>=eventtime].sum()
                        survival.append(sf)
                    np.testing.assert_allclose(survival,forest.survival[tree,leaf].cpu(),atol=1e-12,rtol=0)
    # Full600-round CPU/GPU timing, not a few-epoch extrapolation.
    for device in ['cpu','cuda']:
        model,found,tele=gpu.fit_rank(x,train.Survival_Days,train.Death,.5*ca,vx,.5*cv,2,25,[75,300,600],101011,device=device)
        timings.append(dict(role='histogram_rank_same_objective',device=device,fit_seconds=tele['fit_seconds'],inference_seconds=tele['inference_seconds']))
        if device=='cpu':
            old=found[600][1]
        else:
            differences['rank_cpu_cuda_max_abs']=float(np.max(np.abs(old-found[600][1])))
            differences['rank_cpu_cuda_C_delta']=m.c_evaluator(val)(found[600][1])-m.c_evaluator(val)(old)
            assert tele['cuda_gradient_calls']==600
            model.save_model(d/'rank.ubj');restored=gpu.xgb.Booster(model_file=d/'rank.ubj');restored.set_param({'device':'cuda'})
            pp=restored.inplace_predict(gpu.cp.asarray(vx,dtype=gpu.cp.float32),predict_type='margin',base_margin=gpu.cp.asarray(.5*cv,dtype=gpu.cp.float32))
            np.testing.assert_array_equal(gpu.cp.asnumpy(pp).astype(float),found[600][1])
            pairs=gpu.comparable_pairs(train.Survival_Days,train.Death,'cuda')
            pi,pj=pairs
            def objective(values):
                q=torch.as_tensor(values,device='cuda',dtype=torch.float64)
                return float(torch.nn.functional.softplus(q[pj]-q[pi]).mean())
            assert objective(found[600][0])<objective(.5*ca)
            m.write_json(d/'rank_cuda_execution.json',tele)
    # End-to-end survival calibration also computed on CUDA.
    base=gpu.cuda_breslow(train.Survival_Days,train.Death,ca)
    expected=m.survival_from_baseline(cv,m.breslow(train.Survival_Days,train.Death,ca),m.COMMON['times'])
    np.testing.assert_allclose(gpu.cuda_survival(cv,base).cpu(),expected,atol=1e-12,rtol=1e-12)
    pd.DataFrame(timings).to_csv(d/'timings.csv',index=False)
    m.write_json(d/'VALIDATION.json',dict(at=m.stamp(),status='passed',gradient_autograd=True,censor_ties=True,
        logrank_scalar_oracle=True,leaf_KM_scalar_oracle=True,forest_cpu_cuda_structure_same=True,
        checkpoint_replay_exact=True,cuda_breslow_parity=True,differences=differences,
        ranking_cpu_cuda_not_required_identical='GPU/CPU histogram RNG and reductions may differ; not sklearn parity',
        gpu=torch.cuda.get_device_name(),versions=dict(torch=torch.__version__,xgboost=gpu.xgb.__version__,cupy=gpu.cp.__version__)))
    m.mark_complete(d,['VALIDATION.json','timings.csv','rank_cuda_execution.json','rank.ubj','forest.pt'])
    log('CUDA数学与性能验证','验证真正CUDA训练推理及删失逻辑',timings,d,'CPU/GPU差异与GPU实际执行记录见VALIDATION及rank_cuda_execution。')


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise ValueError('Protocol already frozen; do not overwrite')
    m.write_json(OUT/'config.json',PLAN)
    (OUT/'EXPERIMENTS.md').write_text('# CUDA小预算实验记录\n\n固定MI1/2/3，不按外层成绩选择；旧全量CPU任务用户终止，448内部完成、0外层，保留现场。\n',encoding='utf-8')
    before=signature()
    validate()
    assert signature()==before
    m.write_json(OUT/'manifest.json',before)
    log('协议冻结','固定预算及算法迁移差异',PLAN,OUT,'75内部、15外层；所有内层配置冻结后才外层。')
    status(status='ready',phase='validated',progress=progress())


def inner(f,j,k):
    d=OUT/f'inner/fold{f:02d}/mi{j:02d}_inner{k:02d}'
    if m.verify_completed(d):
        return
    d.mkdir(parents=True,exist_ok=True)
    b,train,val,enc,x,vx,z=load_inner(f,j,k)
    ca,cv=z['c_train_score'],z['c_val_score'];rows=[];timings=[]
    C=m.c_evaluator(val);TC=m.c_evaluator(train)
    for depth,leaf in itertools.product(PLAN['rank']['depths'],PLAN['rank']['leaves']):
        status(status='running',phase='inner',unit=dict(fold=f,imputation=j,inner_fold=k,role=GPU_ROLES[0],depth=depth,leaf=leaf),progress=progress())
        model,found,tele=gpu.fit_rank(x,train.Survival_Days,train.Death,.5*ca,vx,.5*cv,depth,leaf,PLAN['rank']['trees'],100000+f*1000+j*10+k)
        timings.append(dict(role=GPU_ROLES[0],depth=depth,leaf=leaf,**{a:tele[a] for a in ['fit_seconds','inference_seconds','gradient_calls','cuda_gradient_calls']}))
        for count,(tr,va) in found.items():
            rows.append(dict(role=GPU_ROLES[0],fold=f,imputation=j,inner_fold=k,depth=depth,leaf=leaf,trees=count,C_index=C(va),train_C=TC(tr)))
        del model
    for depth,leaf in itertools.product(PLAN['forest']['depths'],PLAN['forest']['leaves']):
        status(status='running',phase='inner',unit=dict(fold=f,imputation=j,inner_fold=k,role=GPU_ROLES[1],depth=depth,leaf=leaf),progress=progress())
        forest=gpu.HistogramSurvivalForest(depth,leaf,200000+f*1000+j*10+k).fit(x,train.Survival_Days,train.Death)
        gpu.sync();start=time.perf_counter();ts=forest.predict_survival(x);vs=forest.predict_survival(vx);gpu.sync()
        inference=time.perf_counter()-start
        risk=(1-.5*(torch.as_tensor(z['c_val_survival'],device='cuda')+vs))[:,1].cpu().numpy()
        trisk=(1-.5*(torch.as_tensor(z['c_train_survival'],device='cuda')+ts))[:,1].cpu().numpy()
        rows.append(dict(role=GPU_ROLES[1],fold=f,imputation=j,inner_fold=k,depth=depth,leaf=leaf,trees=300,C_index=C(risk),train_C=TC(trisk)))
        timings.append(dict(role=GPU_ROLES[1],depth=depth,leaf=leaf,fit_seconds=forest.fit_seconds,inference_seconds=inference))
        del forest,ts,vs
    old=cpu.OUT/f'inner/fold{f:02d}/mi{j:02d}_inner{k:02d}'
    cpu_grid=pd.read_csv(old/'grid.csv');cpu_grid['role']=cpu_grid.role.replace(dict(zip(cpu.ROLES,CPU_ROLES)))
    data=pd.concat([pd.DataFrame(rows),cpu_grid],ignore_index=True)
    data.to_csv(d/'grid.csv',index=False);pd.DataFrame(timings).to_csv(d/'timings.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),variables=b.variables,train_ids=train.patient_id.tolist(),validation_ids=val.patient_id.tolist(),
                                  no_outer_patients=True,no_neural_training=True,cpu_grid_reference=str(old),cuda_training=True,cuda_inference=True))
    m.mark_complete(d,['grid.csv','timings.csv','audit.json'])


def choose():
    choices={}
    for f in range(1,6):
        cells=list((OUT/f'inner/fold{f:02d}').glob('mi*/grid.csv'));assert len(cells)==15
        data=pd.concat([pd.read_csv(p) for p in cells],ignore_index=True)
        mean=data.groupby(['role','depth','leaf','trees']).C_index.agg(['mean','count']).reset_index()
        assert mean['count'].eq(15).all()
        mean.to_csv(OUT/f'selection_fold{f:02d}.csv',index=False)
        choices[str(f)]={}
        for role in ROLES:
            a=mean[mean.role.eq(role)].sort_values(['mean','trees','depth','leaf'],ascending=[False,True,True,False]).iloc[0]
            choices[str(f)][role]=dict(depth=int(a.depth),leaf=int(a.leaf),trees=int(a.trees),inner_C=float(a['mean']))
    m.write_json(OUT/'FROZEN_CHOICES.json',dict(at=m.stamp(),choices=choices,inner_units=75,imputations=MIS,outer_evaluated=False))
    log('内部选型冻结','GPU与CPU方法均按同75个内部单元选择',choices,OUT/'FROZEN_CHOICES.json','此后不允许使用外层成绩修改配置。')


def frame(val,f,j,role,score,survival):
    q=val[['patient_id','Death','Survival_Days']].copy();q['repeat_id']=1;q['fold']=f;q['imputation']=j;q['score']=score
    for c,t in enumerate(m.COMMON['times']):q[f'survival_{t}d']=survival[:,c]
    return m.normalize(q,role,PLAN['study'],'frozen CACox + survival-aware tree; CUDA histogram variant' if role in GPU_ROLES else 'CPU original exact-split variant')


def outer(f,j):
    d=OUT/f'outer/fold{f:02d}/mi{j:02d}'
    if m.verify_completed(d):return
    frozen=json.loads((OUT/'FROZEN_CHOICES.json').read_text());assert frozen['inner_units']==75
    b=m.bundle(f);train,val=b.pair(j);enc=m.Encoder(b.variables,'robust');x=enc.fit_transform(train);vx=enc.transform(val)
    fitted=m.load_original(f,j);ca,cv,cas,cvs=m.scores(fitted,train,val);del fitted
    d.mkdir(parents=True,exist_ok=True);outputs=[];audit=[]
    for role in ROLES:
        c=frozen['choices'][str(f)][role]
        depth,leaf,count=c['depth'],c['leaf'],c['trees']
        status(status='running',phase='outer',unit=dict(fold=f,imputation=j,role=role),progress=progress())
        start=time.perf_counter();inference=0.
        if role==GPU_ROLES[0]:
            if count:
                model,found,tele=gpu.fit_rank(x,train.Survival_Days,train.Death,.5*ca,vx,.5*cv,depth,leaf,[count],100000+f*1000+j*10+9)
                tr,va=found[count];model.save_model(d/f'{role}.ubj');outputs.append(f'{role}.ubj');inference=tele['inference_seconds']
            else:tr,va=.5*ca,.5*cv
            baseline=gpu.cuda_breslow(train.Survival_Days,train.Death,tr);survival=gpu.cuda_survival(va,baseline).cpu().numpy()
            state=dict(choice=c,encoder=enc.__dict__,baseline={a:v.cpu() for a,v in baseline.items()},offset_scale=.5,original_checkpoint_reference=str(m.ORIGINAL))
            torch.save(state,d/f'{role}.pt');outputs.append(f'{role}.pt')
        elif role==GPU_ROLES[1]:
            model=gpu.HistogramSurvivalForest(depth,leaf,200000+f*1000+j*10+9,trees=count).fit(x,train.Survival_Days,train.Death)
            gpu.sync();pt=time.perf_counter();ts=model.predict_survival(x);vs=model.predict_survival(vx)
            train_s=.5*(torch.as_tensor(cas,device='cuda')+ts);combined=.5*(torch.as_tensor(cvs,device='cuda')+vs)
            gpu.sync();inference=time.perf_counter()-pt;tr=(1-train_s[:,1]).cpu().numpy();va=(1-combined[:,1]).cpu().numpy();survival=combined.cpu().numpy()
            state=dict(forest=model.state(),encoder=enc.__dict__,choice=c,alpha=.5,original_checkpoint_reference=str(m.ORIGINAL))
            torch.save(state,d/f'{role}.pt');outputs.append(f'{role}.pt')
            restored=gpu.HistogramSurvivalForest.restore(state['forest'])
            np.testing.assert_array_equal(vs.cpu().numpy(),restored.predict_survival(vx).cpu().numpy())
            del model,restored
        elif role==CPU_ROLES[0]:
            if count:
                trees,found,_=cpu.learners.fit(x,train.Survival_Days,train.Death,.5*ca,vx,.5*cv,depth,leaf,[count],100000+f*1000+j*10+9)
                tr,va=found[count]
            else:trees=[];tr,va=.5*ca,.5*cv
            baseline=m.breslow(train.Survival_Days,train.Death,tr);survival=m.survival_from_baseline(va,baseline,m.COMMON['times'])
            import joblib
            joblib.dump(dict(trees=trees,encoder=enc.__dict__,choice=c,baseline=baseline),d/f'{role}.joblib');outputs.append(f'{role}.joblib')
        else:
            _,model,ts,vs=cpu.forest_fit(train,val,b.variables,f,j,9,depth,leaf)
            survival=.5*(cvs+vs);tr=1-.5*(cas[:,1]+ts[:,1]);va=1-survival[:,1]
            import joblib
            joblib.dump(dict(forest=model,encoder=enc.__dict__,choice=c,alpha=.5),d/f'{role}.joblib',compress=3);outputs.append(f'{role}.joblib')
        elapsed=time.perf_counter()-start
        frame(val,f,j,role,va,survival).to_csv(d/f'{role}.csv',index=False);outputs.append(f'{role}.csv')
        audit.append(dict(role=role,fold=f,imputation=j,train_C=m.c_evaluator(train)(tr),test_C=m.c_evaluator(val)(va),elapsed_seconds=elapsed,
                          inference_seconds=inference,device='cuda' if role in GPU_ROLES else 'cpu'))
    pd.DataFrame(audit).to_csv(d/'generalization.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),variables=b.variables,choices=frozen['choices'][str(f)],no_neural_training=True,imputation_preselected=True))
    m.mark_complete(d,outputs+['generalization.csv','audit.json'])


def check_subset(q):
    assert len(q)==1140*3 and set(q.imputation)==set(MIS) and set(q.fold)==set(range(1,6))
    assert not q[KEY].duplicated().any() and q.groupby(['fold','imputation']).size().eq(228).all()
    assert q.groupby('imputation').patient_id.nunique().eq(1140).all()


def pooled(q,times=(None,365,730,1095)):
    rows=[]
    for horizon in times:
        column='score' if horizon is None else f'risk_{horizon}d'
        a=[core.aggregate_metric(q[q.imputation.eq(j)],column,horizon) for j in MIS]
        rows.append(dict(metric='C_index' if horizon is None else f'AUC_{horizon}d',**core.rubin([v['estimate'] for v in a],[v['variance'] for v in a],True)))
    return pd.DataFrame(rows)


def paired(ref,alt):
    m.assert_paired(ref,alt);qs=[];us=[]
    for j in MIS:
        a=core.aggregate_metric(ref[ref.imputation.eq(j)]);b=core.aggregate_metric(alt[alt.imputation.eq(j)])
        ids=a['contributions'].keys()|b['contributions'].keys()
        qs.append(b['estimate']-a['estimate']);us.append(sum((b['contributions'].get(i,0)-a['contributions'].get(i,0))**2 for i in ids))
    return core.rubin(qs,us)


def report():
    frames={r:pd.concat([pd.read_csv(p,float_precision='round_trip') for p in (OUT/'outer').glob(f'fold*/mi*/{r}.csv')],ignore_index=True) for r in ROLES}
    def subset(q):return q[q.imputation.isin(MIS)].copy()
    ref=ROOT/'results/ordinary_cox_matched_comparison/20261030_v1/comparisons'
    frames['ordinary_cox']=subset(pd.read_csv(ref/'predictions_ordinary_cox.csv',float_precision='round_trip'))
    for role in ['RCS_ridge','original_CACox']:
        frames[role]=subset(pd.read_csv(ref/f'predictions_{role}.csv',float_precision='round_trip'))
    for name,study,role in [('J3_previous','cacox_tree_joint_study','J3_standardized_logrisk'),('C2_previous','cacox_residual_tree','C2_half_offset')]:
        frames[name]=subset(pd.concat([pd.read_csv(p,float_precision='round_trip') for p in (ROOT/f'results/{study}/20261030_v1/outer').glob(f'fold*/mi*/{role}.csv')],ignore_index=True))
    metrics=[];individual=[];diffs=[];summary=dict(at=m.stamp(),scope='5 outer folds x3 predetermined imputations, NOT full20 MI',models={},comparisons={},target_met_subset={})
    for name,q in frames.items():
        check_subset(q);m.assert_paired(frames['ordinary_cox'],q)
        metric=pooled(q);metric.insert(0,'model_id',name);metrics.append(metric)
        one=m.individual_cindices(q);one.insert(0,'model_id',name);individual.append(one)
        c=metric.iloc[0];summary['models'][name]=dict(C_index=float(c.estimate),CI95=[float(c.lower),float(c.upper)],mean_15=float(one.C_index.mean()),range_15=[float(one.C_index.min()),float(one.C_index.max())])
    for role,corresponding in zip(GPU_ROLES,CPU_ROLES):
        summary['comparisons'][role]={}
        for reference in ['ordinary_cox','C2_previous','J3_previous',corresponding]:
            a=paired(frames[reference],frames[role]);diffs.append(dict(model_id=role,reference_id=reference,**a));summary['comparisons'][role][reference]=a
        summary['target_met_subset'][role]=summary['comparisons'][role]['ordinary_cox']['estimate']>=.02
    pd.concat(metrics).to_csv(OUT/'metrics_pooled.csv',index=False)
    one=pd.concat(individual);one.to_csv(OUT/'model_cindex.csv',index=False);pd.DataFrame(diffs).to_csv(OUT/'paired_differences.csv',index=False)
    general=pd.concat([pd.read_csv(p) for p in (OUT/'outer').glob('fold*/mi*/generalization.csv')],ignore_index=True)
    general['gap']=general.train_C-general.test_C;general.to_csv(OUT/'generalization_each_model.csv',index=False)
    summary['generalization']=general.groupby('role')[['train_C','test_C','gap','elapsed_seconds']].mean().to_dict('index')
    timings=pd.concat([pd.read_csv(p).assign(cell=str(p.parent.relative_to(OUT))) for p in (OUT/'inner').glob('fold*/mi*/timings.csv')],ignore_index=True)
    timings.to_csv(OUT/'inner_timings.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5));order=list(frames)
    ax.boxplot([one[one.model_id.eq(v)].C_index for v in order]);ax.set(xticks=range(1,len(order)+1),xticklabels=order,ylabel='Outer-test C-index',title='Pilot: five folds x THREE predetermined imputations; paired correlated models')
    ax.tick_params(axis='x',labelrotation=25);fig.tight_layout();fig.savefig(OUT/'cindex_distribution.png',dpi=170);plt.close(fig)
    m.write_json(OUT/'summary.json',summary)
    lines=['# CUDA树与CACox小预算验证','',f'完成时间：{m.stamp()}。','',
           '仅预先固定的MI1/2/3，保留全部五外折/五内折；75内部、15外层单元。不是完整5×20，不据此声称完整达标。',
           '', 'GPU与旧CPU树算法存在直方图/随机数差异；训练分区分位折点不改变数据筛选缓存。',
           '', '## 外层C-index', '| 方法 | 汇总C（95%CI） | 15模型均值 | 最小–最大 |','|---|---|---|---|']
    for name,v in summary['models'].items():lines.append(f"| {name} | {v['C_index']:.5f} ({v['CI95'][0]:.5f}–{v['CI95'][1]:.5f}) | {v['mean_15']:.5f} | {v['range_15'][0]:.5f}–{v['range_15'][1]:.5f} |")
    lines+=['','## 配对差值',json.dumps(summary['comparisons'],ensure_ascii=False,indent=2),'',
            '训练和推理计时见validation/timings.csv、inner_timings.csv及generalization_each_model.csv。CUDA不保证小样本加速；迁移效果与性能分开解释。',
            '', '数据、划分、插补、筛选和CACox不变；没有神经重训。名单由外层开发集筛选及早停缓存的内层依赖仍保留；外层已多轮查看，属于同队列继续开发。插补模型相关；区间不包含反复探索的选择偏倚。',
            '', '各时点AUC见metrics_pooled.csv。分布图为15个相关模型，不作为15个独立样本推断。','', '![C-index](cindex_distribution.png)']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    assert signature()==json.loads((OUT/'manifest.json').read_text())
    for p in [*(OUT/'inner').glob('fold*/mi*/COMPLETE.json'),*(OUT/'outer').glob('fold*/mi*/COMPLETE.json')]:assert m.verify_completed(p.parent)
    m.write_json(OUT/'FINAL_INTEGRITY.json',dict(status='passed',inner_units=75,outer_units=15,models_per_new_role=15,imputations=MIS,folds=5,skipped_folds=0,source_inputs_unchanged=True,no_neural_training=True,cuda_validation_passed=m.verify_completed(OUT/'validation')))
    log('小预算配对比较完成','比较CUDA与CPU及历史基线',summary,OUT/'FINAL_REPORT.md','仅此固定15单元的探索性评价；不等于全量任务达标。')
    m.mark_complete(OUT,['summary.json','FINAL_REPORT.md','FINAL_INTEGRITY.json','metrics_pooled.csv','model_cindex.csv','paired_differences.csv','generalization_each_model.csv','inner_timings.csv','cindex_distribution.png','FROZEN_CHOICES.json'])
    status(status='complete',phase='complete',progress=progress())


def run():
    assert signature()==json.loads((OUT/'manifest.json').read_text())
    with m.run_lock(OUT):
        try:
            for f,j,k in itertools.product(range(1,6),MIS,range(1,6)):
                inner(f,j,k)
                with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(at=m.stamp(),phase='inner_complete',fold=f,imputation=j,inner_fold=k))+'\n')
            assert progress()['inner']==75
            choose()
            for f,j in itertools.product(range(1,6),MIS):
                outer(f,j)
                with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(at=m.stamp(),phase='outer_complete',fold=f,imputation=j))+'\n')
            assert progress()['outer']==15
            status(status='running',phase='report',progress=progress());report()
        except BaseException as e:
            status(status='failed',phase='exception',error=repr(e),traceback=traceback.format_exc(),progress=progress());raise


def launch():
    assert signature()==json.loads((OUT/'manifest.json').read_text())
    if (OUT/'RUNNING.lock').exists():raise RuntimeError('Run lock exists; do not duplicate')
    out=(OUT/'controller.out.log').open('a',encoding='utf-8');err=(OUT/'controller.err.log').open('a',encoding='utf-8')
    p=subprocess.Popen([sys.executable,str(__file__),'run'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    out.close();err.close();m.write_json(OUT/'LAUNCH.json',dict(at=m.stamp(),launcher_pid=p.pid,workers=1,device='cuda'))
    print('LAUNCHED',p.pid)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');m.configure()
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','launch','run']);args=parser.parse_args()
    {'prepare':prepare,'launch':launch,'run':run}[args.command]()

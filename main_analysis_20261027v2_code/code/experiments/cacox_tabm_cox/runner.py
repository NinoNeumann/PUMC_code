"""Frozen 5x20 TabM-Cox experiment: scalar risk, matched k1/k8, I20 fusion."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,importlib.util,inspect,itertools,json,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(HERE))
import tabm_backend as b
spec=importlib.util.spec_from_file_location('tabm_project_preflight',HERE/'preflight.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
s,m,rg,OUT=p.source,p.m,p.rg,p.OUT
CM={c['config_id']:c for c in b.CONFIGS};ROLES=['N1_single_Cox','N8_shared_Cox','J1_I20_single','J8_I20_shared']
ALPHAS=[.25,.5,.75,1.]
PLAN=dict(study='cacox_tabm_cox_20261031_v1',scope='fixed5outerx20MI,all500inner',configs=b.CONFIGS,common=b.COMMON,
    roles=ROLES,NN_weight_candidates=[0.,*ALPHAS],mixture='native endpoints;interior train mean/SD standardization,weight is NN',
    ensemble='mean per-member exact Efron train loss;mean log risk for inference;early stop ensemble Efron',
    input='unchanged selected fields and existing robust encoder,no new embedding/discretization/features',
    seed_base=910000,internal_selection='20MI mean per innerfold then5fold mean;ties I20 control,smaller NN weight,lower LR',
    outer_epochs='ceil median of100 best internal epochs for selected configuration;no outer early stopping',
    required_inner_units=500,required_inner_NN=2000,outer_units=100,outputs_per_role=100,maximum_outer_NN=400,
    I20_NN_tree_and_MS_unchanged=True,no_RCS_risk_branch=True,no_v2=True,no_sensitivity=True,
    data_immutable=s.PLAN['data_immutable'],benchmark_workers=[1,2,3],maximum_live_workers=3,
    physical_limit_GiB=rg.LIMIT/rg.GiB,reserve_GiB=4,owned_RSS_limit_GiB=8,
    historical_M4='already official8memberTabM withWeibull/WAS,notCox;lr7e-6/437caps;multiple other changes,notpure-head causal experiment',
    minimum_C=.76,strict_plus002_target=.7613465232424127,
    limitations='same cohort continued development;fixed-list/inheritedI20 internal dependence;repeated outer viewing;correlatedMI;conditionalCI notindependent')


def read(path):return s.read(path)
def status(**kw):m.status(OUT/'RUN_STATUS.json',controller_pid=os.getpid(),**kw)
def event(phase,**kw):
    with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(dict(at=m.stamp(),phase=phase,**kw))+'\n')
def log(title,purpose,params,result):
    with (OUT/'EXPERIMENTS.md').open('a',encoding='utf-8') as stream:
        stream.write(f'\n## {title}\n\n时间：{m.stamp()}\n\n目的：{purpose}\n\n参数：`{json.dumps(params,ensure_ascii=False)}`\n\n结果位置：{OUT}\n\n结果说明：{result}\n')
def jobs(phase):return list(itertools.product(range(1,6),range(1,21),range(1,6))) if phase=='inner' else list(itertools.product(range(1,6),range(1,21)))
def directory(phase,f,j,k=None):return OUT/f'{phase}/fold{f:02d}'/(f'mi{j:02d}_inner{k:02d}' if k is not None else f'mi{j:02d}')
def progress():return dict(inner=sum(1 for _ in OUT.glob('inner/fold*/mi*/COMPLETE.json')),outer=sum(1 for _ in OUT.glob('outer/fold*/mi*/COMPLETE.json')))
def unit(f,j,k=None):return s.unit(f,j,k)
def seed(f,j,k=None):return p.seed(f,j,k)
def blend(u,a,v,alpha):return s.r.mix(u['I20_train'],u['I20_val'],a,v,1-alpha)
def family(role):return 1 if role in ['N1_single_Cox','J1_I20_single'] else 8


def signature():
    assert m.verify_completed(s.OUT) and m.verify_completed(s.OUT/'interpretation')
    assert s.signature()==read(s.OUT/'manifest.json')
    assert m.verify_completed(OUT/'preflight') and m.verify_completed(OUT/'benchmark')
    for name,sha in read(OUT/'preflight/PREFLIGHT.json')['source_files'].items():assert m.sha(HERE/name)==sha
    source_files=[*HERE.glob('*.py'),HERE/'README.md',OUT/'config.json',OUT/'RUNTIME_PLAN.json']
    return dict(plan=PLAN,source={str(q.relative_to(ROOT)):m.sha(q) for q in source_files},
        prior_manifest=m.sha(s.OUT/'manifest.json'),prior_COMPLETE=m.sha(s.OUT/'COMPLETE.json'),prior_review=m.sha(s.OUT/'interpretation/COMPLETE.json'),
        preflight=m.sha(OUT/'preflight/COMPLETE.json'),benchmark=m.sha(OUT/'benchmark/COMPLETE.json'),
        installed_TabM=m.sha(inspect.getfile(b.TabM)),torch=torch.__version__,
        mathematical_check=m.sha(ROOT/'results/tabm_cox_compatibility/20261031_v1/EXACT_EFRON_CPU.json'))


def train(u,c,f,j,k,d,epochs=None,callback=None):
    if m.verify_completed(d):
        model,obj=b.load(d/'neural.pt');assert obj['seed']==seed(f,j,k) and obj['config']==c
        return model,read(d/'audit.json')
    rg.ensure_headroom();d.mkdir(parents=True,exist_ok=True)
    model,trace,audit=b.fit(u['x'],u['tr'].Survival_Days,u['tr'].Death,c,seed(f,j,k),
        u['vx'] if k is not None else None,u['va'].Survival_Days if k is not None else None,u['va'].Death if k is not None else None,
        fixed_epochs=epochs,progress=callback)
    a,v=b.predict_members(model,u['x']),b.predict_members(model,u['vx'])
    b.save(model,d/'neural.pt',u['x'].shape[1],c,seed(f,j,k),s.source_metadata(u,f,j,k))
    np.savez_compressed(d/'predictions.npz',train_members=a,val_members=v,train_ids=u['tr'].patient_id,val_ids=u['va'].patient_id)
    pd.DataFrame(trace).to_csv(d/'training_trace.csv',index=False)
    audit.update(at=m.stamp(),fold=f,imputation=j,inner_fold=k,input_cache_SHA=m.sha(s.r.cache_path(f,j,k)/'COMPLETE.json'))
    m.write_json(d/'audit.json',audit);m.mark_complete(d,['neural.pt','predictions.npz','training_trace.csv','audit.json'])
    return model,audit


def inner(f,j,k,index):
    d=directory('inner',f,j,k)
    if m.verify_completed(d):return
    u=unit(f,j,k);d.mkdir(parents=True,exist_ok=True);rows=[];audits={}
    C,TC=m.c_evaluator(u['va']),m.c_evaluator(u['tr'])
    for c in b.CONFIGS:
        def callback(v):rg.ensure_headroom();m.status(OUT/f'workers/inner_{index}/RUN_STATUS.json',status='running',worker_pid=os.getpid(),fold=f,imputation=j,inner_fold=k,config=c,**v)
        model,audit=train(u,c,f,j,k,d/'fits'/c['config_id'],callback=callback);audits[c['config_id']]=audit
        with np.load(d/'fits'/c['config_id']/'predictions.npz') as z:a=z['train_members'].mean(1);v=z['val_members'].mean(1)
        role='N1_single_Cox' if c['members']==1 else 'N8_shared_Cox';combo='J1_I20_single' if c['members']==1 else 'J8_I20_shared'
        rows.append(dict(role=role,config_id=c['config_id'],alpha=1.,C_index=C(v),train_C=TC(a)))
        for alpha in ALPHAS:
            ta,tv,_=blend(u,a,v,alpha);rows.append(dict(role=combo,config_id=c['config_id'],alpha=alpha,C_index=C(tv),train_C=TC(ta)))
        del model
    for role in ['J1_I20_single','J8_I20_shared']:rows.append(dict(role=role,config_id='control',alpha=0.,C_index=C(u['I20_val']),train_C=TC(u['I20_train'])))
    pd.DataFrame(rows).assign(fold=f,imputation=j,inner_fold=k).to_csv(d/'grid.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),fits=audits,source=s.source_metadata(u,f,j,k),outer_evaluated=False))
    m.mark_complete(d,['grid.csv','audit.json',*[f'fits/{c["config_id"]}/COMPLETE.json' for c in b.CONFIGS]])


def choose():
    assert progress()==dict(inner=500,outer=0);choices={}
    for f in range(1,6):
        q=pd.concat([pd.read_csv(a,float_precision='round_trip') for a in (OUT/f'inner/fold{f:02d}').glob('mi*/grid.csv')],ignore_index=True)
        per=q.groupby(['role','config_id','alpha','inner_fold']).C_index.agg(['mean','count']).reset_index();assert per['count'].eq(20).all()
        means=per.groupby(['role','config_id','alpha'])['mean'].agg(['mean','count']).reset_index();assert means['count'].eq(5).all()
        means['control']=means.config_id.eq('control');means['lr']=means.config_id.map(lambda c:0 if c=='control' else CM[c]['learning_rate'])
        means.to_csv(OUT/f'selection_fold{f:02d}.csv',index=False);choices[str(f)]={}
        for role in ROLES:
            row=means[means.role.eq(role)].sort_values(['mean','control','alpha','lr'],ascending=[False,False,True,True]).iloc[0]
            cid=str(row.config_id);epochs=None
            if cid!='control':
                fits=[read(a)['best_epoch'] for a in (OUT/f'inner/fold{f:02d}').glob(f'mi*/fits/{cid}/audit.json')];assert len(fits)==100
                epochs=int(np.ceil(np.median(fits)))
            choices[str(f)][role]=dict(config_id=cid,config=None if cid=='control' else CM[cid],alpha=float(row.alpha),epochs=epochs,inner_C=float(row['mean']))
    m.write_json(OUT/'FROZEN_CHOICES.json',dict(at=m.stamp(),choices=choices,inner_units=500,inner_NN_fits=2000,outer_evaluated=False))
    event('all_choices_frozen',choices=choices);log('全部内部配置和轮数冻结','只根据内部验证选择，不读取外层成绩',choices,'下一阶段才生成新外层模型及预测。')


def frame(u,f,j,role,score,base):
    q=u['va'][['patient_id','Death','Survival_Days']].copy();q['repeat_id']=1;q['fold']=f;q['imputation']=j;q['score']=score
    surv=s.gpu.cuda_survival(score,base).cpu().numpy()
    for i,t in enumerate(m.COMMON['times']):q[f'survival_{t}d']=surv[:,i]
    return m.normalize(q,role,PLAN['study'],'Official compactTabM exactEfron scalar head;optional train-only standardized I20 fusion')


def outer(f,j,index):
    d=directory('outer',f,j)
    if m.verify_completed(d):return
    u=unit(f,j);d.mkdir(parents=True,exist_ok=True);choices=read(OUT/'FROZEN_CHOICES.json')['choices'][str(f)];models={};rows=[];wrappers={}
    selected={v['config_id']:v for v in choices.values() if v['config_id']!='control'}
    for cid,v in selected.items():
        def callback(w):rg.ensure_headroom();m.status(OUT/f'workers/outer_{index}/RUN_STATUS.json',status='running',worker_pid=os.getpid(),fold=f,imputation=j,config=cid,**w)
        model,_=train(u,CM[cid],f,j,None,d/'fits'/cid,v['epochs'],callback);del model
        with np.load(d/'fits'/cid/'predictions.npz') as z:models[cid]=(z['train_members'].mean(1),z['val_members'].mean(1))
    for role,v in choices.items():
        cid=v['config_id'];stats=None
        if cid=='control':a,tv=u['I20_train'],u['I20_val']
        else:
            a,tv=models[cid]
            if role.startswith('J'):a,tv,stats=blend(u,a,tv,v['alpha'])
        baseline=s.gpu.cuda_breslow(u['tr'].Survival_Days,u['tr'].Death,a)
        frame(u,f,j,role,tv,baseline).to_csv(d/(role+'.csv'),index=False)
        wrappers[role]=dict(choice=v,normalization=stats,baseline={key:value.cpu() for key,value in baseline.items()},source=s.source_metadata(u,f,j,None),neural_file=None if cid=='control' else f'fits/{cid}/neural.pt')
        rows.append(dict(role=role,fold=f,imputation=j,train_C=m.c_evaluator(u['tr'])(a),test_C=m.c_evaluator(u['va'])(tv),control=cid=='control'))
    torch.save(wrappers,d/'wrappers.pt');pd.DataFrame(rows).to_csv(d/'generalization.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),choices=choices,new_neural_models=len(selected),source=s.source_metadata(u,f,j,None)))
    m.mark_complete(d,['wrappers.pt','generalization.csv','audit.json',*[role+'.csv' for role in ROLES],*[f'fits/{cid}/COMPLETE.json' for cid in selected]])


def worker(phase,index):
    path=OUT/f'workers/{phase}_{index}/RUN_STATUS.json'
    try:
        for args in jobs(phase)[index::read(OUT/'RUNTIME_PLAN.json')['workers']]:(inner if phase=='inner' else outer)(*args,index)
        m.status(path,status='complete',worker_pid=os.getpid())
    except BaseException as error:m.status(path,status='failed',worker_pid=os.getpid(),error=repr(error),traceback=traceback.format_exc());raise


def schedule(phase):
    children=[];count=read(OUT/'RUNTIME_PLAN.json')['workers'];ticks=0;previous=None
    try:
        for i in range(count):
            rg.ensure_headroom();d=OUT/f'workers/{phase}_{i}';d.mkdir(parents=True,exist_ok=True)
            with (d/'out.log').open('a') as oo,(d/'err.log').open('a') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'worker','--phase',phase,'--index',str(i)],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
        while any(q.poll() is None for q in children):
            resource=rg.record(OUT/'monitor',ticks%3==0);ticks+=1;pr=progress();status(status='running',phase=phase,progress=pr,worker_launcher_pids=[q.pid for q in children],resource=resource)
            if previous!=pr:event(phase,progress=pr);previous=pr
            if any(q.poll() not in (None,0) for q in children):raise RuntimeError(f'TabM {phase} worker failure')
            time.sleep(2)
        assert all(q.returncode==0 for q in children)
    except BaseException:rg.stop_owned(children);raise


def bench_worker(workers,index):
    root=OUT/f'benchmark/w{workers}';d=root/f'worker{index}';d.mkdir(parents=True,exist_ok=True)
    u=unit(1,4,1);warm,*_=b.fit(u['x'],u['tr'].Survival_Days,u['tr'].Death,b.CONFIGS[0],seed(1,4,1),fixed_epochs=3);del warm
    m.write_json(d/'READY.json',dict(pid=os.getpid()));deadline=time.monotonic()+180
    while not (root/'GO').exists():
        if time.monotonic()>deadline:raise TimeoutError('TabM benchmark barrier')
        rg.ensure_headroom();time.sleep(.05)
    for task in list(range(6))[index::workers]:
        u=unit(task%5+1,4,1);arrays={}
        for c in b.CONFIGS:
            model,trace,audit=b.fit(u['x'],u['tr'].Survival_Days,u['tr'].Death,c,seed(task%5+1,4,1),u['vx'],u['va'].Survival_Days,u['va'].Death,progress=lambda _:rg.ensure_headroom())
            arrays[c['config_id']+'_risk']=b.predict_members(model,u['vx']);arrays[c['config_id']+'_trace']=pd.DataFrame(trace).to_numpy(float)
            m.write_json(root/f'task{task}_{c["config_id"]}.json',audit);del model
        np.savez_compressed(root/f'task{task}.npz',**arrays)
    m.write_json(d/'DONE.json',dict(pid=os.getpid()))


def benchmark():
    root=OUT/'benchmark';root.mkdir(parents=True,exist_ok=True)
    if (root/'SUMMARY.json').exists():raise RuntimeError('Benchmark already recorded')
    assert m.verify_completed(OUT/'preflight');timings=[]
    for workers in PLAN['benchmark_workers']:
        children=[];run=root/f'w{workers}'
        try:
            for i in range(workers):
                d=run/f'worker{i}';d.mkdir(parents=True,exist_ok=True)
                with (d/'out.log').open('w') as oo,(d/'err.log').open('w') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'bench-worker','--workers',str(workers),'--index',str(i)],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
            deadline=time.monotonic()+180
            while not all((run/f'worker{i}/READY.json').exists() for i in range(workers)):
                rg.record(root,True)
                if any(q.poll() not in (None,0) for q in children):raise RuntimeError('TabM benchmark initialization failure')
                if time.monotonic()>deadline:raise TimeoutError('TabM benchmark warmup')
                time.sleep(.2)
            start=time.perf_counter();(run/'GO').write_text(m.stamp())
            while any(q.poll() is None for q in children):rg.record(root,True);time.sleep(.2)
            assert all(q.returncode==0 for q in children);wall=time.perf_counter()-start;error=0.;trace_error=0.;c_error=0.
            for task in range(6):
                C=m.c_evaluator(unit(task%5+1,4,1)['va'])
                with np.load(root/f'w1/task{task}.npz') as a,np.load(run/f'task{task}.npz') as z:
                    for key in a.files:
                        assert a[key].shape==z[key].shape;v=float(abs(a[key]-z[key]).max())
                        if key.endswith('_trace'):trace_error=max(trace_error,v)
                        else:error=max(error,v);c_error=max(c_error,abs(C(a[key].mean(1))-C(z[key].mean(1))))
            with np.load(run/'task0.npz') as a,np.load(run/'task5.npz') as z:
                for key in a.files:np.testing.assert_array_equal(a[key],z[key])
            assert error==trace_error==c_error==0
            timings.append(dict(workers=workers,wall_seconds=wall,clinical_inner_fits=24,prediction_difference=error,loss_trace_difference=trace_error,C_difference=c_error));print(timings[-1],flush=True)
        except BaseException:rg.stop_owned(children);raise
    chosen=min(timings,key=lambda v:v['wall_seconds'])['workers'];m.write_json(root/'SUMMARY.json',dict(at=m.stamp(),timings=timings,selected_workers=chosen,all_checks_passed=True))
    m.mark_complete(root,['SUMMARY.json',*[f'w{w}/task{i}.npz' for w in PLAN['benchmark_workers'] for i in range(6)]])


def prepare():
    assert m.verify_completed(OUT/'preflight') and m.verify_completed(OUT/'benchmark')
    if (OUT/'manifest.json').exists():raise RuntimeError('Already frozen')
    assert not (OUT/'inner').exists() and not (OUT/'outer').exists()
    m.write_json(OUT/'config.json',PLAN);m.write_json(OUT/'RUNTIME_PLAN.json',dict(workers=read(OUT/'benchmark/SUMMARY.json')['selected_workers'],maximum=3))
    m.write_json(OUT/'manifest.json',signature());log('正式方案冻结','从09失败证据检验新的神经信息与参数共享集成',PLAN,'真实CUDA预检和并发一致性通过，未生成新外层结果。')
    status(status='prepared',phase='prepared',progress=progress())


def replay():
    rows=[]
    for phase in ['inner','outer']:
        for args in jobs(phase):
            f,j=args[:2];k=args[2] if phase=='inner' else None;d=directory(phase,f,j,k);assert m.verify_completed(d);u=unit(f,j,k);pred={}
            for fit in (d/'fits').glob('*'):
                assert m.verify_completed(fit);model,obj=b.load(fit/'neural.pt');assert obj['seed']==seed(f,j,k)
                assert obj['metadata']==s.source_metadata(u,f,j,k)
                a,v=b.predict_members(model,u['x']),b.predict_members(model,u['vx'])
                with np.load(fit/'predictions.npz') as z:
                    np.testing.assert_array_equal(z['train_ids'],u['tr'].patient_id);np.testing.assert_array_equal(z['val_ids'],u['va'].patient_id)
                    np.testing.assert_array_equal(a,z['train_members']);np.testing.assert_array_equal(v,z['val_members'])
                pred[fit.name]=(a.mean(1),v.mean(1));del model
            if phase=='outer':
                wrappers=torch.load(d/'wrappers.pt',map_location='cpu',weights_only=False)
                for role,w in wrappers.items():
                    v=w['choice'];cid=v['config_id']
                    a,tv=(u['I20_train'],u['I20_val']) if cid=='control' else pred[cid]
                    if cid!='control' and role.startswith('J'):a,tv,stats=blend(u,a,tv,v['alpha']);assert stats==w['normalization']
                    saved=pd.read_csv(d/(role+'.csv'),float_precision='round_trip').set_index('patient_id').loc[u['va'].patient_id]
                    np.testing.assert_array_equal(tv,saved.score)
                    base=s.gpu.cuda_breslow(u['tr'].Survival_Days,u['tr'].Death,a)
                    for key,value in base.items():np.testing.assert_array_equal(value.cpu(),w['baseline'][key])
                    sv=s.gpu.cuda_survival(tv,{key:value.cuda() for key,value in w['baseline'].items()}).cpu().numpy()
                    np.testing.assert_array_equal(sv,saved[[f'survival_{t}d' for t in m.COMMON['times']]])
            rows.append(dict(phase=phase,fold=f,imputation=j,inner_fold=k or 0,neural_models=len(pred),member_risk_error=0.,outer_survival_error=0.))
            if len(rows)%10==0:rg.record(OUT/'monitor',False);status(status='running',phase='saved_model_replay',units_replayed=len(rows),progress=progress())
    d=OUT/'verification';d.mkdir(exist_ok=True);pd.DataFrame(rows).to_csv(d/'replays.csv',index=False);m.mark_complete(d,['replays.csv'])


def report():
    sr=importlib.util.spec_from_file_location('tabm_cox_frozen_report',HERE/'report_study.py');mod=importlib.util.module_from_spec(sr);sr.loader.exec_module(mod);mod.report_all(sys.modules[__name__])


def run():
    assert signature()==read(OUT/'manifest.json')
    with m.run_lock(OUT):
        try:
            schedule('inner');choose();schedule('outer');status(status='running',phase='saved_model_replay',progress=progress());replay()
            status(status='running',phase='metrics_and_report',progress=progress());report()
        except BaseException as error:status(status='failed',phase='exception',error=repr(error),traceback=traceback.format_exc(),progress=progress());raise


def launch():
    assert signature()==read(OUT/'manifest.json')
    if (OUT/'RUNNING.lock').exists() or m.verify_completed(OUT):raise RuntimeError('Locked or complete')
    rg.ensure_headroom()
    with (OUT/'controller.out.log').open('a',encoding='utf-8') as oo,(OUT/'controller.err.log').open('a',encoding='utf-8') as ee:child=subprocess.Popen([sys.executable,str(__file__),'run'],cwd=ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW)
    m.write_json(OUT/'LAUNCH.json',dict(at=m.stamp(),launcher_pid=child.pid,workers=read(OUT/'RUNTIME_PLAN.json')['workers']));print('TABM_LAUNCHED',child.pid,flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');m.configure();b.configure();parser=argparse.ArgumentParser();parser.add_argument('command',choices=['benchmark','bench-worker','prepare','launch','run','worker'])
    parser.add_argument('--workers',type=int);parser.add_argument('--index',type=int);parser.add_argument('--phase',choices=['inner','outer']);a=parser.parse_args()
    if a.command=='bench-worker':bench_worker(a.workers,a.index)
    elif a.command=='worker':worker(a.phase,a.index)
    else:dict(benchmark=benchmark,prepare=prepare,launch=launch,run=run)[a.command]()

"""Independent curvature change and full-MI inner-selection ablations of G1."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
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

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import curvature_backend as cb
spec=importlib.util.spec_from_file_location('frozen_g1_all20_reference',ROOT/'experiments/cacox_g1_full20/runner.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
p,m,gpu,core=old.p,old.m,old.gpu,old.core
OUT=ROOT/'results/cacox_g1_curvature_study/20261031_v1'
H='H1_curvature';I='I20_selection';ROLES=[H,I];MIS=list(range(1,21));SMALL=[1,2,3];WORKERS=4
COUNTS=[0,75,150,300,600];DEPTHS=[1,2];LEAVES=[25,50]
CAPACITY=json.loads((old.OUT/'FROZEN_CHOICES.json').read_text())['choices']
PLAN=dict(study='cacox_g1_curvature_study_20261031_v1',roles=ROLES,workers=WORKERS,scope='One repeat x five outer folds x 20 MI',
          H1=dict(internal_cells=75,internal_MI=SMALL,outer_models=100,selection=False,capacity=CAPACITY,
                  loss='original n*mean softplus over survival-comparable pairs',
                  local_hessian_bound='2*diag(H)+1e-6',derivative_normalization='divide g and bound by max(1,max(bound))',
                  leaf_constraint='min_child_weight=25 (conservative >=25 sampled rows because all working weights <=1)',
                  limitations='curvature-mass minimum is more conservative than original count minimum; not a pure fixed-partition Newton test'),
          I20=dict(internal_cells=500,internal_MI=MIS,reused_cells=75,new_cells=425,new_grid_fits=1700,
                   depths=DEPTHS,leaves=LEAVES,trees=COUNTS,new_grid_max_rounds=1020000,
                   original_learner_unchanged=True,outer_models=100,
                   selection='maximize mean of five inner-fold means over20 MI, tie fewer trees, shallower, larger leaf'),
          original_offset=.5,tree_eta=.03,subsample=.8,max_bin=32,L1=0,L2=0,gamma=0,
          immutable=old.PLAN['immutable'],neural_training=0,no_outer_tuning=True,no_v2=True,no_sensitivity=True,
          no_automatic_combination=True,all_inner_and_choices_before_outer=True,
          target='pooled C minus RCS >= .02; ordinary Cox >= .02 reported separately; no80/100 condition',
          limitations=old.PLAN['limitations'])

def log(title,purpose,params,location,result,start=None):
    with (OUT/'EXPERIMENTS.md').open('a',encoding='utf-8') as s:
        s.write(f'\n## {title}\n\n- 时间：{start or m.stamp()} → {m.stamp()}\n- 目的：{purpose}\n- 参数及改动：`{json.dumps(params,ensure_ascii=False)}`\n- 结果位置：{location}\n- 结果说明：{result}\n')
def progress():
    return dict(H1_inner=sum(1 for _ in OUT.glob(f'inner/{H}/fold*/mi*/COMPLETE.json')),
                I20_inner=sum(1 for _ in OUT.glob(f'inner/{I}/fold*/mi*/COMPLETE.json')),
                outer_units=sum(1 for _ in OUT.glob('outer/fold*/mi*/COMPLETE.json')))
def status(**kw):m.status(OUT/'RUN_STATUS.json',controller_pid=os.getpid(),**kw)
def event(phase,**kw):
    with (OUT/'stage_events.jsonl').open('a',encoding='utf-8') as s:s.write(json.dumps(dict(at=m.stamp(),phase=phase,**kw))+'\n')
def inner_dir(role,f,j,k):return OUT/f'inner/{role}/fold{f:02d}/mi{j:02d}_inner{k:02d}'
def prefix(depth,leaf,count):return f'd{depth}_l{leaf}_n{count}'

def signature():
    previous=json.loads((old.OUT/'manifest.json').read_text())
    assert old.signature()==previous and m.verify_completed(old.OUT)
    inputs={}
    for f,j,k in itertools.product(range(1,6),MIS,range(1,6)):
        q=p.source_cache(f,j,k)
        for name in ['COMPLETE.json','predictions.npz']:inputs[str((q/name).relative_to(ROOT))]=m.sha(q/name)
    files=[*HERE.glob('*.py'),HERE/'README.md',OUT/'config.json']
    return dict(plan=PLAN,source={str(q.relative_to(ROOT)):m.sha(q) for q in files},full20_G1_manifest=previous,
                full20_G1_complete_hash=m.sha(old.OUT/'COMPLETE.json'),inner_inputs=inputs)

def validate():
    from tabulate import tabulate
    d=OUT/'validation';d.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available()
    t=np.array([10,10,20,30,40,50]);e=np.array([1,0,1,0,1,1]);pair=gpu.comparable_pairs(t,e,'cuda')
    oracle=[(a,b) for a in range(len(t)) for b in range(len(t)) if e[a] and (t[a]<t[b] or (t[a]==t[b] and not e[b]))]
    assert list(zip(pair[0].cpu().tolist(),pair[1].cpu().tolist()))==oracle
    score=torch.tensor([.2,-.4,.6,.3,-.7,.1],device='cuda',dtype=torch.float64,requires_grad=True)
    def loss(s):return len(s)*torch.stack([torch.nn.functional.softplus(s[b]-s[a]) for a,b in oracle]).mean()
    g,h,true_g,diagonal,scale=cb.derivatives(score,pair)
    exact_g=torch.autograd.grad(loss(score),score)[0]
    exact_H=torch.autograd.functional.hessian(loss,score)
    np.testing.assert_allclose(true_g.detach().cpu(),exact_g.cpu(),atol=1e-12,rtol=0)
    np.testing.assert_allclose(diagonal.detach().cpu(),exact_H.diag().cpu(),atol=1e-12,rtol=0)
    upper=torch.diag(2*diagonal+1e-6)
    minimum_eigenvalue=float(torch.linalg.eigvalsh(upper-exact_H).min())
    assert minimum_eigenvalue>=-1e-12
    eps=1e-5;finite=[]
    for index in range(len(score)):
        step=torch.zeros_like(score);step[index]=eps
        finite.append(float((loss(score+step)-loss(score-step))/(2*eps)))
    assert np.max(np.abs(np.array(finite)-exact_g.cpu().numpy()))<1e-8
    shifted=cb.derivatives(score+23,pair)
    np.testing.assert_allclose(shifted[0].detach().cpu(),g.detach().cpu(),atol=1e-12,rtol=0)
    np.testing.assert_allclose(shifted[1].detach().cpu(),h.detach().cpu(),atol=1e-12,rtol=0)
    # Native one-tree update vs independent fixed-leaf Newton sums, all rows sampled.
    rng=np.random.RandomState(1029);x=rng.normal(size=(80,3)).astype(np.float32)
    tt=np.arange(1,81,dtype=float);ee=np.tile([1,0,1,1],20);off=rng.normal(scale=.2,size=80)
    model,found,tele=cb.fit(x,tt,ee,off,x,off,1,2,[1],947,subsample=1.,eta=1.)
    dm=gpu.xgb.QuantileDMatrix(gpu.cp.asarray(x),max_bin=32)
    leaves=model.predict(dm,pred_leaf=True).reshape(-1)
    assert len(np.unique(leaves))==2
    ss=torch.as_tensor(off.astype(np.float32),device='cuda');gg,hh,*_=cb.derivatives(ss,gpu.comparable_pairs(tt,ee,'cuda'))
    expected=np.zeros(80)
    for leaf_id in np.unique(leaves):
        mask=leaves==leaf_id;mask_gpu=torch.as_tensor(mask,device='cuda')
        expected[mask]=-float(gg[mask_gpu].double().sum()/hh[mask_gpu].double().sum())
    native_error=float(np.max(np.abs(found[1][0]-off.astype(np.float32)-expected)))
    assert native_error<2e-6
    # Full real 75-round production fit and repeat; no outer outcomes involved.
    _,tr,va,enc,xx,vx,z=p.load_inner(2,1,1)
    ca,cv=z['c_train_score'],z['c_val_score']
    model,a,real=cb.fit(xx,tr.Survival_Days,tr.Death,.5*ca,vx,.5*cv,2,25,[0,75],102011)
    _,b,_=cb.fit(xx,tr.Survival_Days,tr.Death,.5*ca,vx,.5*cv,2,25,[0,75],102011)
    repeat=max(float(np.max(np.abs(a[75][i]-b[75][i]))) for i in [0,1])
    assert repeat<=1e-6 and m.c_evaluator(va)(a[75][1])==m.c_evaluator(va)(b[75][1])
    model.save_model(d/'curvature.ubj');replay=gpu.xgb.Booster(model_file=d/'curvature.ubj');replay.set_param({'device':'cuda'})
    actual=gpu.cp.asnumpy(replay.inplace_predict(gpu.cp.asarray(vx,dtype=gpu.cp.float32),predict_type='margin',base_margin=gpu.cp.asarray(.5*cv,dtype=gpu.cp.float32))).astype(float)
    np.testing.assert_array_equal(actual,a[75][1])
    before=cb.objective_value(.5*ca,tr.Survival_Days,tr.Death);after=cb.objective_value(a[75][0],tr.Survival_Days,tr.Death)
    assert after<before
    proof=dict(at=m.stamp(),status='passed',gradient_autograd_error=float((true_g-exact_g).abs().max()),
               diagonal_autograd_error=float((diagonal-exact_H.diag()).abs().max()),bound_minus_full_H_min_eigenvalue=minimum_eigenvalue,
               finite_difference_error=float(np.max(np.abs(np.array(finite)-exact_g.cpu().numpy()))),translation_invariance=True,
               native_leaf_Newton_error=native_error,repeat_prediction_error=repeat,repeat_C_difference=0.,saved_replay_error=0.,
               actual_training_leaf_floor=real['minimum_training_leaf_count'],initial_loss=before,final_loss=after,
               cuda_execution=real,report_dependency_available=True)
    m.write_json(d/'VALIDATION.json',proof);m.mark_complete(d,['VALIDATION.json','curvature.ubj'])

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise RuntimeError('Already frozen; never overwrite')
    (OUT/'EXPERIMENTS.md').write_text('# G1二阶更新与全20份内部选型：独立验证\n',encoding='utf-8')
    m.write_json(OUT/'config.json',PLAN);before=signature()
    validate();assert signature()==before
    m.write_json(OUT/'manifest.json',before)
    log('预注册及数学验证','隔离二阶更新与插补选型预算两个改动',PLAN,OUT,'H1固定原容量，75内部诊断；I20复用75并补425内部网格；各100外层，全部内部冻结后才外层。')
    status(status='ready',phase='validated',progress=progress())

def inner(role,f,j,k,index):
    d=inner_dir(role,f,j,k)
    if m.verify_completed(d):return
    d.mkdir(parents=True,exist_ok=True)
    m.status(OUT/f'workers/inner_{index}/RUN_STATUS.json',worker_pid=os.getpid(),status='running',role=role,fold=f,imputation=j,inner_fold=k)
    if role==I and j<=3:
        source=p.OUT/f'inner/fold{f:02d}/mi{j:02d}_inner{k:02d}';assert m.verify_completed(source)
        q=pd.read_csv(source/'grid.csv',float_precision='round_trip');q=q[q.role.eq('G1_cuda_rank')].copy();q['role']=I
        assert len(q)==20;q.to_csv(d/'grid.csv',index=False)
        m.write_json(d/'audit.json',dict(at=m.stamp(),reused_grid=True,source=str(source),source_complete_hash=m.sha(source/'COMPLETE.json')))
        m.mark_complete(d,['grid.csv','audit.json']);return
    _,tr,va,enc,x,vx,z=p.load_inner(f,j,k);ca,cv=z['c_train_score'],z['c_val_score']
    C,TC=m.c_evaluator(va),m.c_evaluator(tr);rows=[];outputs=[];telemetry=[]
    arrays=dict(train_ids=tr.patient_id.to_numpy(),val_ids=va.patient_id.to_numpy())
    if role==H:
        c=CAPACITY[str(f)];count=c['trees'];counts=sorted(set([0,count]+[n for n in [25,75,150,300,600] if n<=count]))
        model,found,tele=cb.fit(x,tr.Survival_Days,tr.Death,.5*ca,vx,.5*cv,c['depth'],c['leaf'],counts,100000+f*1000+j*10+k)
        model.save_model(d/'H1.ubj');outputs.append('H1.ubj');telemetry.append(tele)
        for n,(a,b) in found.items():
            arrays[f'train_{n}']=a;arrays[f'val_{n}']=b
            rows.append(dict(role=role,fold=f,imputation=j,inner_fold=k,depth=c['depth'],leaf=c['leaf'],trees=n,train_C=TC(a),C_index=C(b),
                             train_loss=cb.objective_value(a,tr.Survival_Days,tr.Death),val_loss=cb.objective_value(b,va.Survival_Days,va.Death)))
    else:
        for depth,leaf in itertools.product(DEPTHS,LEAVES):
            model,found,tele=gpu.fit_rank(x,tr.Survival_Days,tr.Death,.5*ca,vx,.5*cv,depth,leaf,COUNTS,100000+f*1000+j*10+k)
            name=f'd{depth}_l{leaf}.ubj';model.save_model(d/name);outputs.append(name);telemetry.append(tele)
            for n,(a,b) in found.items():
                key=prefix(depth,leaf,n);arrays[key+'_train']=a;arrays[key+'_val']=b
                rows.append(dict(role=role,fold=f,imputation=j,inner_fold=k,depth=depth,leaf=leaf,trees=n,train_C=TC(a),C_index=C(b)))
            del model
    pd.DataFrame(rows).to_csv(d/'grid.csv',index=False);np.savez_compressed(d/'predictions.npz',**arrays)
    m.write_json(d/'audit.json',dict(at=m.stamp(),reused_grid=False,no_outer_patients=True,no_neural_training=True,telemetry=telemetry))
    m.mark_complete(d,outputs+['grid.csv','predictions.npz','audit.json'])

def choose():
    assert progress()==dict(H1_inner=75,I20_inner=500,outer_units=0)
    choices={H:CAPACITY,I:{}}
    for f in range(1,6):
        paths=list((OUT/f'inner/{I}/fold{f:02d}').glob('mi*/grid.csv'));assert len(paths)==100
        grid=pd.concat([pd.read_csv(q,float_precision='round_trip') for q in paths],ignore_index=True)
        per_inner=grid.groupby(['inner_fold','depth','leaf','trees']).C_index.agg(['mean','count']).reset_index()
        assert per_inner['count'].eq(20).all()
        means=per_inner.groupby(['depth','leaf','trees'])['mean'].agg(['mean','count']).reset_index()
        assert means['count'].eq(5).all();means.to_csv(OUT/f'selection_fold{f:02d}.csv',index=False)
        best=means.sort_values(['mean','trees','depth','leaf'],ascending=[False,True,True,False]).iloc[0]
        choices[I][str(f)]=dict(depth=int(best.depth),leaf=int(best.leaf),trees=int(best.trees),inner_C=float(best['mean']))
    m.write_json(OUT/'FROZEN_CHOICES.json',dict(at=m.stamp(),choices=choices,internal_cells={H:75,I:500},
                 H1_capacity_prespecified=True,I20_selection_outcomes='inner only',outer_evaluated=False))
    log('内部选型冻结','固定两组全部外层配置',choices,OUT/'FROZEN_CHOICES.json','不把两种改动组合，也不使用任何本轮外层成绩改选。')
    event('choices_frozen',choices=choices)

def predict(model,x,offset,count=None):
    kw={} if count is None else {'iteration_range':(0,count)}
    values=model.inplace_predict(gpu.cp.asarray(x,dtype=gpu.cp.float32),predict_type='margin',base_margin=gpu.cp.asarray(offset,dtype=gpu.cp.float32),**kw)
    assert isinstance(values,gpu.cp.ndarray);return gpu.cp.asnumpy(values).astype(float)

def selected_internal_audit():
    """Replay all selected internal models; reconstruct only missing old grids."""
    frozen=json.loads((OUT/'FROZEN_CHOICES.json').read_text())['choices'];records=[];losses=[]
    for role in ROLES:
        for f,j,k in itertools.product(range(1,6),SMALL if role==H else MIS,range(1,6)):
            d=inner_dir(role,f,j,k);assert m.verify_completed(d)
            _,tr,va,enc,x,vx,z=p.load_inner(f,j,k);ca,cv=z['c_train_score'],z['c_val_score'];c=frozen[role][str(f)]
            n=c['trees'];depth,leaf=c['depth'],c['leaf'];reconstructed=False
            if role==I and j<=3 and n:
                dest=d/'selected_reconstruction';dest.mkdir(exist_ok=True)
                if not m.verify_completed(dest):
                    model,found,tele=gpu.fit_rank(x,tr.Survival_Days,tr.Death,.5*ca,vx,.5*cv,depth,leaf,[n],100000+f*1000+j*10+k)
                    a,b=found[n];model.save_model(dest/'selected.ubj')
                    np.savez_compressed(dest/'predictions.npz',train=a,val=b)
                    m.write_json(dest/'audit.json',dict(at=m.stamp(),choice=c,telemetry=tele))
                    m.mark_complete(dest,['selected.ubj','predictions.npz','audit.json'])
                model=gpu.xgb.Booster(model_file=dest/'selected.ubj');model.set_param({'device':'cuda'})
                with np.load(dest/'predictions.npz') as zz:ea,eb=zz['train'],zz['val']
                reconstructed=True
            elif n:
                name='H1.ubj' if role==H else f'd{depth}_l{leaf}.ubj'
                model=gpu.xgb.Booster(model_file=d/name);model.set_param({'device':'cuda'})
                with np.load(d/'predictions.npz') as zz:
                    keys=(f'train_{n}',f'val_{n}') if role==H else (prefix(depth,leaf,n)+'_train',prefix(depth,leaf,n)+'_val')
                    ea,eb=zz[keys[0]],zz[keys[1]]
            else:ea,eb=.5*ca,.5*cv
            if n:a,b=predict(model,x,.5*ca,n),predict(model,vx,.5*cv,n);del model
            else:a,b=ea,eb
            np.testing.assert_array_equal(a,ea);np.testing.assert_array_equal(b,eb)
            grid=pd.read_csv(d/'grid.csv',float_precision='round_trip');row=grid[grid.depth.eq(depth)&grid.leaf.eq(leaf)&grid.trees.eq(n)]
            assert len(row)==1
            dc=m.c_evaluator(va)(b)-float(row.iloc[0].C_index)
            if abs(dc)>1e-12:raise AssertionError(f'Selected internal C drift {role}/{f}/{j}/{k}: {dc}')
            records.append(dict(role=role,fold=f,imputation=j,inner_fold=k,reconstructed=reconstructed,risk_error=0.,C_drift=dc))
            losses.append(dict(role=role,fold=f,imputation=j,inner_fold=k,trees=n,
                          train_C=m.c_evaluator(tr)(a),val_C=m.c_evaluator(va)(b),
                          train_loss=cb.objective_value(a,tr.Survival_Days,tr.Death),val_loss=cb.objective_value(b,va.Survival_Days,va.Death)))
            if len(records)%25==0:status(status='running',phase='selected_internal_replay',replayed=len(records),progress=progress())
    d=OUT/'verification';d.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(d/'selected_internal_replays.csv',index=False)
    pd.DataFrame(losses).to_csv(d/'selected_internal_losses.csv',index=False)

def reference_source(f,j):return p.OUT/f'outer/fold{f:02d}/mi{j:02d}' if j<=3 else old.OUT/f'outer/fold{f:02d}/mi{j:02d}'
def frame(val,f,j,role,score,survival):
    q=val[['patient_id','Death','Survival_Days']].copy();q['repeat_id']=1;q['fold']=f;q['imputation']=j;q['score']=score
    for c,t in enumerate(m.COMMON['times']):q[f'survival_{t}d']=survival[:,c]
    return m.normalize(q,role,PLAN['study'],'Frozen .5 CACox offset + survival-aware ranking tree')

def outer(f,j,index):
    d=OUT/f'outer/fold{f:02d}/mi{j:02d}'
    if m.verify_completed(d):return
    m.status(OUT/f'workers/outer_{index}/RUN_STATUS.json',worker_pid=os.getpid(),status='running',fold=f,imputation=j)
    choices=json.loads((OUT/'FROZEN_CHOICES.json').read_text())['choices']
    bundle=m.bundle(f);tr,va=bundle.pair(j);enc=m.Encoder(bundle.variables,'robust');x=enc.fit_transform(tr);vx=enc.transform(va)
    net=m.load_original(f,j);ca,cv,_,_=m.scores(net,tr,va);del net
    d.mkdir(parents=True,exist_ok=True);outputs=[];audits=[];general=[]
    for role in ROLES:
        c=choices[role][str(f)];n=c['trees'];depth,leaf=c['depth'],c['leaf'];start=time.perf_counter();tele={};reused=False
        same=all(c[key]==CAPACITY[str(f)][key] for key in ['depth','leaf','trees'])
        if role==I and same:
            source=reference_source(f,j);assert m.verify_completed(source)
            state=torch.load(source/'G1_cuda_rank.pt',map_location='cpu',weights_only=False)
            model=gpu.xgb.Booster(model_file=source/'G1_cuda_rank.ubj');model.set_param({'device':'cuda'})
            a,b=predict(model,x,.5*ca),predict(model,vx,.5*cv);del model
            baseline={k:v.cuda() for k,v in state['baseline'].items()};reused=True
            model_path=str(source/'G1_cuda_rank.ubj')
        else:
            if n:
                learner=cb.fit if role==H else gpu.fit_rank
                model,found,tele=learner(x,tr.Survival_Days,tr.Death,.5*ca,vx,.5*cv,depth,leaf,[n],100000+f*1000+j*10+9)
                a,b=found[n];model.save_model(d/f'{role}.ubj');outputs.append(f'{role}.ubj');model_path=str(d/f'{role}.ubj');del model
            else:a,b=.5*ca,.5*cv;model_path=None
            baseline=gpu.cuda_breslow(tr.Survival_Days,tr.Death,a)
        survival=gpu.cuda_survival(b,baseline).cpu().numpy()
        frame(va,f,j,role,b,survival).to_csv(d/f'{role}.csv',index=False)
        torch.save(dict(choice=c,encoder=enc.__dict__,baseline={k:v.cpu() for k,v in baseline.items()},offset_scale=.5,
                        model_path=model_path,reused=reused),d/f'{role}.pt')
        outputs += [f'{role}.csv',f'{role}.pt']
        general.append(dict(role=role,fold=f,imputation=j,train_C=m.c_evaluator(tr)(a),test_C=m.c_evaluator(va)(b),reused=reused,elapsed_seconds=time.perf_counter()-start))
        audits.append(dict(role=role,choice=c,reused=reused,model_path=model_path,telemetry=tele,no_outer_outcomes_to_learner=True))
    pd.DataFrame(general).to_csv(d/'generalization.csv',index=False)
    m.write_json(d/'audit.json',dict(at=m.stamp(),variables=bundle.variables,roles=audits,no_neural_training=True))
    m.mark_complete(d,outputs+['generalization.csv','audit.json'])

def jobs(phase):
    if phase=='outer':return list(itertools.product(range(1,6),MIS))
    return [(H,f,j,k) for f,j,k in itertools.product(range(1,6),SMALL,range(1,6))]+[(I,f,j,k) for f,j,k in itertools.product(range(1,6),MIS,range(1,6))]

def worker(phase,index):
    path=OUT/f'workers/{phase}_{index}/RUN_STATUS.json'
    try:
        for args in jobs(phase)[index::WORKERS]:
            (inner if phase=='inner' else outer)(*args,index)
            with path.with_name('stage_events.jsonl').open('a',encoding='utf-8') as s:s.write(json.dumps(dict(at=m.stamp(),completed=list(args)))+'\n')
        m.status(path,worker_pid=os.getpid(),status='complete')
    except BaseException as error:
        m.status(path,worker_pid=os.getpid(),status='failed',error=repr(error),traceback=traceback.format_exc());raise

def schedule(phase):
    children=[]
    for index in range(WORKERS):
        d=OUT/f'workers/{phase}_{index}';d.mkdir(parents=True,exist_ok=True)
        with (d/'out.log').open('a',encoding='utf-8') as out,(d/'err.log').open('a',encoding='utf-8') as err:
            q=subprocess.Popen([sys.executable,str(__file__),'worker','--phase',phase,'--index',str(index)],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        children.append(q)
    previous=None
    while any(q.poll() is None for q in children):
        current=progress();status(status='running',phase=phase,worker_launcher_pids=[q.pid for q in children],progress=current)
        if current!=previous:event(phase,progress=current);previous=current
        if any(q.poll() not in (None,0) for q in children):
            for q in children:q.wait()
            raise RuntimeError(f'{phase} worker failed; preserved logs')
        time.sleep(3)
    assert all(q.returncode==0 for q in children)
    event(phase+'_complete',progress=progress())

def verify_outer():
    from sksurv.metrics import concordance_index_censored
    records=[]
    for f,j in itertools.product(range(1,6),MIS):
        d=OUT/f'outer/fold{f:02d}/mi{j:02d}';assert m.verify_completed(d)
        bundle=m.bundle(f);tr,va=bundle.pair(j);net=m.load_original(f,j);ca,cv,_,_=m.scores(net,tr,va);del net
        for role in ROLES:
            state=torch.load(d/f'{role}.pt',map_location='cpu',weights_only=False)
            enc=m.Encoder(bundle.variables,'robust');enc.__dict__.update(state['encoder']);x,vx=enc.transform(tr),enc.transform(va)
            if state['choice']['trees']:
                model=gpu.xgb.Booster(model_file=state['model_path']);model.set_param({'device':'cuda'})
                a,b=predict(model,x,.5*ca),predict(model,vx,.5*cv);del model
            else:a,b=.5*ca,.5*cv
            saved=pd.read_csv(d/f'{role}.csv',float_precision='round_trip').set_index('patient_id').loc[va.patient_id]
            np.testing.assert_array_equal(saved.score.to_numpy(),b)
            survival=gpu.cuda_survival(b,{k:v.cuda() for k,v in state['baseline'].items()}).cpu().numpy()
            np.testing.assert_array_equal(saved[[f'survival_{t}d' for t in m.COMMON['times']]].to_numpy(),survival)
            rebuilt=gpu.cuda_breslow(tr.Survival_Days,tr.Death,a)
            for k in rebuilt:np.testing.assert_allclose(rebuilt[k].cpu(),state['baseline'][k],atol=1e-12,rtol=0)
            c=m.c_evaluator(va)(b)
            assert abs(c-concordance_index_censored(va.Death.to_numpy(bool),va.Survival_Days.to_numpy(float),b,tied_tol=0)[0])<1e-12
            general=pd.read_csv(d/'generalization.csv');row=general[general.role.eq(role)].iloc[0]
            assert abs(row.train_C-m.c_evaluator(tr)(a))<1e-12 and abs(row.test_C-c)<1e-12
            records.append(dict(role=role,fold=f,imputation=j,reused=state['reused'],risk_error=0.,survival_error=0.,independent_C=True,train_only_calibration=True))
        if len(records)%20==0:status(status='running',phase='outer_saved_replay',replayed=len(records),progress=progress())
    d=OUT/'verification';pd.DataFrame(records).to_csv(d/'outer_model_replays.csv',index=False)
    selected=pd.read_csv(d/'selected_internal_replays.csv');assert len(selected)==575
    m.write_json(d/'INDEPENDENT_AUDIT.json',dict(at=m.stamp(),status='passed',inner_selected_replays=575,outer_replays=200,
                 risk_max_error=0.,survival_max_error=0.,independent_C=True,train_only_calibration=True))
    m.mark_complete(d,['selected_internal_replays.csv','selected_internal_losses.csv','outer_model_replays.csv','INDEPENDENT_AUDIT.json'])

def report():
    frames={role:pd.concat([pd.read_csv(q,float_precision='round_trip') for q in OUT.glob(f'outer/fold*/mi*/{role}.csv')],ignore_index=True) for role in ROLES}
    frames['G1_original']=pd.concat([pd.read_csv(q,float_precision='round_trip') for q in old.OUT.glob('outer/fold*/mi*/G1_cuda_rank.csv')],ignore_index=True)
    for name in ['RCS_ridge','ordinary_cox','original_CACox']:frames[name]=pd.read_csv(old.REF/f'predictions_{name}.csv',float_precision='round_trip')
    baseline_metrics=pd.read_csv(old.OUT/'metrics_pooled.csv',float_precision='round_trip')
    metrics=[];ones=[];folds=[];differences=[];summary=dict(at=m.stamp(),scope=PLAN['scope'],models={},comparisons={})
    for name,q in frames.items():
        old.check(q);m.assert_paired(frames['RCS_ridge'],q)
        if name in ROLES:metric=old.pooled(q)
        else:
            key='G1_cuda_rank' if name=='G1_original' else name
            metric=baseline_metrics[baseline_metrics.model_id.eq(key)].drop(columns='model_id').copy()
            assert abs(float(metric.iloc[0].estimate)-np.mean([core.aggregate_metric(q[q.imputation.eq(j)])['estimate'] for j in MIS]))<1e-12
        metric.insert(0,'model_id',name);metrics.append(metric)
        one=m.individual_cindices(q);assert len(one)==100;one.insert(0,'model_id',name);ones.append(one)
        fold=one.groupby('fold').C_index.agg(['mean','min','max']).reset_index();fold.insert(0,'model_id',name);folds.append(fold)
        c=metric.iloc[0];summary['models'][name]=dict(C_index=float(c.estimate),CI95=[float(c.lower),float(c.upper)],mean_100=float(one.C_index.mean()),range_100=[float(one.C_index.min()),float(one.C_index.max())])
    one=pd.concat(ones,ignore_index=True);one.to_csv(OUT/'model_cindex.csv',index=False)
    fold=pd.concat(folds,ignore_index=True);fold.to_csv(OUT/'fold_cindex.csv',index=False)
    metric=pd.concat(metrics,ignore_index=True);metric.to_csv(OUT/'metrics_pooled.csv',index=False)
    paired_rows=[]
    for role in ROLES:
        summary['comparisons'][role]={}
        for ref in ['G1_original','RCS_ridge','ordinary_cox','original_CACox']:
            delta=old.paired(frames[ref],frames[role]);differences.append(dict(model_id=role,reference_id=ref,**delta))
            paired=one[one.model_id.eq(role)].merge(one[one.model_id.eq(ref)],on=['fold','imputation'],suffixes=('_new','_ref'))
            paired['difference']=paired.C_index_new-paired.C_index_ref;paired['reference_id']=ref;paired_rows.append(paired)
            delta['model_counts']=dict(higher=int((paired.difference>0).sum()),equal=int((paired.difference==0).sum()),lower=int((paired.difference<0).sum()),gain_at_least_002=int((paired.difference>=.02).sum()))
            summary['comparisons'][role][ref]=delta
    pd.DataFrame(differences).to_csv(OUT/'paired_differences.csv',index=False);pd.concat(paired_rows).to_csv(OUT/'paired_model_differences.csv',index=False)
    general=pd.concat([pd.read_csv(q) for q in OUT.glob('outer/fold*/mi*/generalization.csv')],ignore_index=True)
    baseline=pd.read_csv(old.OUT/'generalization_each_model.csv').assign(role='G1_original')
    general=pd.concat([general,baseline],ignore_index=True);general['gap']=general.train_C-general.test_C
    general.to_csv(OUT/'generalization_each_model.csv',index=False)
    summary['generalization']=general.groupby('role')[['train_C','test_C','gap']].mean().to_dict('index')
    summary['choices']=json.loads((OUT/'FROZEN_CHOICES.json').read_text())['choices']
    summary['target_vs_rcs_met']={role:summary['comparisons'][role]['RCS_ridge']['estimate']>=.02 for role in ROLES}
    summary['target_vs_ordinary_met']={role:summary['comparisons'][role]['ordinary_cox']['estimate']>=.02 for role in ROLES}
    summary['I20_reused_outer']=int(general[general.role.eq(I)].reused.sum())
    summary['I20_new_outer_fits']=100-summary['I20_reused_outer']
    summary['internal_cells']=dict(H1=75,I20=500)
    m.write_json(OUT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    order=['ordinary_cox','RCS_ridge','original_CACox','G1_original',H,I]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    cs=np.array([summary['models'][n]['C_index'] for n in order]);ci=np.array([summary['models'][n]['CI95'] for n in order])
    axes[0].errorbar(cs,np.arange(6),xerr=[cs-ci[:,0],ci[:,1]-cs],fmt='o',capsize=3)
    axes[0].set(yticks=range(6),yticklabels=order,xlabel='Pooled C-index / conditional 95% CI');axes[0].invert_yaxis()
    axes[1].boxplot([one[one.model_id.eq(n)].C_index for n in order],vert=False)
    axes[1].set(yticks=range(1,7),yticklabels=order,xlabel='100 correlated outer-model C-indices');axes[1].invert_yaxis()
    fig.suptitle('Independent G1 changes: local curvature vs full-20-MI inner selection');fig.tight_layout();fig.savefig(OUT/'cindex_distribution.png',dpi=180);plt.close(fig)
    # Internal-only diagnostic loss curves; no outer trajectory used for choices.
    traces=pd.concat([pd.read_csv(q) for q in OUT.glob(f'inner/{H}/fold*/mi*/grid.csv')],ignore_index=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for f in range(1,6):
        a=traces[traces.fold.eq(f)].groupby('trees')[['train_loss','val_loss']].mean()
        for ax,col in zip(axes,['train_loss','val_loss']):ax.plot(a.index,a[col],marker='.',label=f'Fold {f}')
    axes[0].set(title='H1 internal training softplus loss',xlabel='Trees',ylabel='Mean pair loss')
    axes[1].set(title='H1 internal validation softplus loss',xlabel='Trees',ylabel='Mean pair loss')
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'internal_loss_curves.png',dpi=180);plt.close(fig)
    lines=['# G1二阶更新与全20份内部选型：独立对照','',f'完成时间：{m.stamp()}。','',
           '完整一次五折×20份插补，每方法100外层模型。数据、筛选、插补、划分与CACox参数及权重不变；没有神经重训。',
           'H1仅改排序树的曲率更新，保留原每折容量（75个原3MI内部诊断，不选型）。I20保留原树算法和候选网格，仅将内部选型扩展到500单元。两组没有合并。',
           '', '| 方法 | 汇总C（95%CI） | 100模型均值 | 最小–最大 |','|---|---|---|---|']
    for name in order:
        v=summary['models'][name];lines.append(f"| {name} | {v['C_index']:.5f} ({v['CI95'][0]:.5f}–{v['CI95'][1]:.5f}) | {v['mean_100']:.5f} | {v['range_100'][0]:.5f}–{v['range_100'][1]:.5f} |")
    lines+=['','## 配对差值','', '| 新方法 | 对照 | 差值（95%CI） | 高/同/低 |','|---|---|---|---|']
    for role in ROLES:
        for ref,v in summary['comparisons'][role].items():
            c=v['model_counts'];lines.append(f"| {role} | {ref} | {v['estimate']:+.5f} ({v['lower']:+.5f}–{v['upper']:+.5f}) | {c['higher']}/{c['equal']}/{c['lower']} |")
    lines+=['','## 各折C-index均值','',fold.pivot(index='fold',columns='model_id',values='mean').to_markdown(),
            '', '## AUC','',metric[metric.metric.ne('C_index')][['model_id','metric','estimate','lower','upper']].to_markdown(index=False),
            '', '## 训练测试差距','',json.dumps(summary['generalization'],ensure_ascii=False,indent=2),
            '', '## 冻结配置','',json.dumps(summary['choices'],ensure_ascii=False,indent=2),
            '', '## 数学与范围','',
            'H1采用局部Hessian的对角上界，不是完整Hessian，也不保证全局损失单调或测试改善。梯度和曲率同尺度归一化，使工作权重≤1，保留至少25个采样训练行的保守约束；曲率质量约束可以比原样本数约束更严格，不能将差异全部归因于固定树结构上的Newton更新。',
            '每份插补先按折内可比较对合并C，再Rubin合并20份；配对差值保留相同患者的影响函数。100模型相关，范围不是95%CI。',
            '同队列外层已多轮查看，名单与早停缓存的内部依赖仍在，不是独立无偏确认。无外层选参、无敏感性分析、无自动合并两种改动。',
            f"I20外层复用原G1 {summary['I20_reused_outer']} 个，新拟合 {summary['I20_new_outer_fits']} 个；仅在全部配置相同时复用。H1新拟合100个。内部复用75网格，新拟合1700原算法网格树、75二阶诊断树，并重建缺失的选中原内部模型。", 
            '', '![C-index](cindex_distribution.png)','', '![仅内部loss](internal_loss_curves.png)']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    assert signature()==json.loads((OUT/'manifest.json').read_text())
    assert progress()==dict(H1_inner=75,I20_inner=500,outer_units=100)
    for q in itertools.chain(OUT.glob('inner/*/fold*/mi*/COMPLETE.json'),OUT.glob('outer/fold*/mi*/COMPLETE.json')):assert m.verify_completed(q.parent)
    assert m.verify_completed(OUT/'verification')
    m.write_json(OUT/'FINAL_INTEGRITY.json',dict(status='passed',H1_internal=75,I20_internal=500,I20_reused_grids=75,I20_new_grid_fits=1700,
                selected_internal_replays=575,outer_units=100,outer_models_per_group=100,outer_replays=200,folds=5,imputations=20,skipped=0,
                no_neural_training=True,inputs_code_history_unchanged=True,all_choices_frozen_before_outer=True,
                I20_reused_outer=summary['I20_reused_outer'],I20_new_outer=summary['I20_new_outer_fits']))
    log('两组独立验证完成','比较原G1及匹配RCS',summary,OUT/'FINAL_REPORT.md','不根据外层追加选择或合并。所有100外层单元、两组200结果及保存模型回放通过。')
    m.mark_complete(OUT,['FINAL_REPORT.md','FINAL_INTEGRITY.json','summary.json','metrics_pooled.csv','model_cindex.csv','fold_cindex.csv',
             'paired_differences.csv','paired_model_differences.csv','generalization_each_model.csv','cindex_distribution.png','internal_loss_curves.png',
             'FROZEN_CHOICES.json','verification/INDEPENDENT_AUDIT.json'])
    status(status='complete',phase='complete',progress=progress());event('complete')

def run():
    assert signature()==json.loads((OUT/'manifest.json').read_text())
    with m.run_lock(OUT):
        try:
            schedule('inner');choose();status(status='running',phase='selected_internal_replay',progress=progress());selected_internal_audit()
            schedule('outer');status(status='running',phase='outer_saved_replay',progress=progress());verify_outer()
            status(status='running',phase='metrics_and_report',progress=progress());report()
        except BaseException as error:
            status(status='failed',phase='exception',error=repr(error),traceback=traceback.format_exc(),progress=progress());raise

def launch():
    assert signature()==json.loads((OUT/'manifest.json').read_text())
    if (OUT/'RUNNING.lock').exists() or m.verify_completed(OUT):raise RuntimeError('Locked or complete; inspect instead of duplicate')
    with (OUT/'controller.out.log').open('a',encoding='utf-8') as out,(OUT/'controller.err.log').open('a',encoding='utf-8') as err:
        q=subprocess.Popen([sys.executable,str(__file__),'run'],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    m.write_json(OUT/'LAUNCH.json',dict(at=m.stamp(),launcher_pid=q.pid,workers=WORKERS,device='cuda'))
    print('LAUNCHED',q.pid,flush=True)

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');m.configure()
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','launch','run','worker'])
    parser.add_argument('--phase',choices=['inner','outer']);parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.command=='worker':worker(args.phase,args.index)
    else:{'prepare':prepare,'launch':launch,'run':run}[args.command]()

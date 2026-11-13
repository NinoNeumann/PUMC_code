"""Rebuild the exact original-CACox offset and full-I20 CUDA tree for new repeats.

Scientific functions are imported from the signed original studies. This file
only supplies repeated-fold paths, bounded scheduling, and resumable storage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil
import torch
import xgboost

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/cox_es_rcs_newdata_10x5/20261104_v1'
SOURCE=ROOT/'experiments/cacox_g1_curvature_study/runner.py'
spec=importlib.util.spec_from_file_location('ten_repeat_i20_scientific_source',SOURCE)
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
m,gpu=old.m,old.gpu
sys.path.insert(0,str(ROOT/'experiments/cacox_original_head_study/src'))
from cacox_original_head_study.data import Bundle
from cacox_original_head_study.storage import mark_complete,verify_completed,write_json
from cacox_original_head_study.runtime import sha

DEPTHS=(1,2);LEAVES=(25,50);COUNTS=(0,75,150,300,600)
WORKERS=3


def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):write_json(path,value)
def bundle(r,f):
    path=(OUT/'bundles')/f'repeat{r:02d}_fold{f:02d}'/'main'
    if not verify_completed(path):raise RuntimeError(f'Incomplete bundle {path}')
    return Bundle(path)
def inner_dir(r,f,j,k):return OUT/'i20'/f'repeat{r:02d}'/'inner'/f'fold{f:02d}'/f'mi{j:02d}_inner{k:02d}'
def outer_dir(r,f,j):return OUT/'i20'/f'repeat{r:02d}'/'outer'/f'fold{f:02d}'/f'mi{j:02d}'
def repeat_dir(r):return OUT/'i20'/f'repeat{r:02d}'
def tree_key(depth,leaf):return f'd{depth}_l{leaf}'
def gpu_guard():
    # Resource admission is execution scheduling only. A temporary R MICE
    # memory peak must delay a fit, never change its inputs or hyperparameters.
    last_notice=0.
    while True:
        available=psutil.virtual_memory().available
        free,total=torch.cuda.mem_get_info()
        if available>=5*2**30 and free>=2*2**30:return
        if time.monotonic()-last_notice>=60.:
            print(f'RESOURCE_WAIT free_RAM_GB={available/2**30:.2f} '
                  f'free_VRAM_GB={free/2**30:.2f}',flush=True)
            last_notice=time.monotonic()
        time.sleep(10)


def neural_inner(r,f,j,k,b,tr,va,d):
    target=d/'neural'
    if verify_completed(target):
        return json.loads((target/'audit.json').read_text(encoding='utf8'))
    target.mkdir(parents=True,exist_ok=True);gpu_guard()
    seed=m.COMMON['seed_base']+(f-1)*1000+j*10000+k*100
    start=time.perf_counter()
    fitted,trace,audit=m.fit(tr,b.variables,m.SPEC,m.COMMON,seed,validation=va)
    a,v,*_=m.scores(fitted,tr,va)
    torch.save(dict(state_dict={name:value.detach().cpu() for name,value in fitted.net.state_dict().items()},
        encoder=fitted.encoder.__dict__,spec=fitted.spec,common=fitted.common,
        baseline=fitted.baseline,seed=seed),target/'cacox.pt')
    np.savez_compressed(target/'predictions.npz',train=a,val=v,
                        train_ids=tr.patient_id.to_numpy(),val_ids=va.patient_id.to_numpy())
    trace.to_csv(target/'training_trace.csv',index=False)
    audit.update(repeat_id=r,fold=f,imputation=j,inner_fold=k,
                 input_bundle_SHA=sha(b.directory/'COMPLETE.json'),
                 measured_seconds=time.perf_counter()-start,cuda_train=True)
    write(target/'audit.json',audit)
    mark_complete(target,['cacox.pt','predictions.npz','training_trace.csv','audit.json'])
    del fitted
    torch.cuda.empty_cache()
    return audit


def inner(r,f,j,k):
    d=inner_dir(r,f,j,k)
    if verify_completed(d):return
    b=bundle(r,f);tr,va=b.pair(j,k)
    if (set(tr.patient_id)|set(va.patient_id)) & set(b.raw_assessment.patient_id):
        raise ValueError('Outer assessment patient leaked into I20 inner fit')
    d.mkdir(parents=True,exist_ok=True)
    neural_inner(r,f,j,k,b,tr,va,d)
    with np.load(d/'neural/predictions.npz') as z:
        np.testing.assert_array_equal(z['train_ids'],tr.patient_id)
        np.testing.assert_array_equal(z['val_ids'],va.patient_id)
        ca,cv=z['train'].copy(),z['val'].copy()
    enc=m.Encoder(b.variables,'robust');x=enc.fit_transform(tr);vx=enc.transform(va)
    C,TC=m.c_evaluator(va),m.c_evaluator(tr)
    for depth,leaf in itertools.product(DEPTHS,LEAVES):
        target=d/'trees'/tree_key(depth,leaf)
        if verify_completed(target):continue
        target.mkdir(parents=True,exist_ok=True);gpu_guard()
        started=time.perf_counter()
        model,found,tele=gpu.fit_rank(x,tr.Survival_Days,tr.Death,.5*ca,
                         vx,.5*cv,depth,leaf,COUNTS,100000+f*1000+j*10+k)
        model.save_model(target/'model.ubj')
        arrays={f'n{n}_{side}':score for n,pair in found.items()
                for side,score in zip(('train','val'),pair)}
        np.savez_compressed(target/'predictions.npz',**arrays)
        rows=[dict(repeat_id=r,fold=f,imputation=j,inner_fold=k,depth=depth,
                   leaf=leaf,trees=n,train_C=TC(pair[0]),C_index=C(pair[1]))
              for n,pair in found.items()]
        pd.DataFrame(rows).to_csv(target/'grid.csv',index=False)
        write(target/'audit.json',dict(at=now(),repeat_id=r,fold=f,imputation=j,
              inner_fold=k,depth=depth,leaf=leaf,seed=100000+f*1000+j*10+k,
              telemetry=tele,measured_seconds=time.perf_counter()-started,
              input_neural_SHA=sha(d/'neural/COMPLETE.json'),
              train_device=tele['fit_device'],prediction_device=tele['prediction_device']))
        mark_complete(target,['model.ubj','predictions.npz','grid.csv','audit.json'])
        del model
        torch.cuda.empty_cache()
    all_grids=pd.concat([pd.read_csv(d/'trees'/tree_key(depth,leaf)/'grid.csv')
                         for depth,leaf in itertools.product(DEPTHS,LEAVES)],ignore_index=True)
    if len(all_grids)!=20:raise ValueError('Expected 20 I20 tree grid rows')
    all_grids.to_csv(d/'grid.csv',index=False)
    mark_complete(d,['neural/COMPLETE.json','grid.csv',
      *[f'trees/{tree_key(depth,leaf)}/COMPLETE.json'
        for depth,leaf in itertools.product(DEPTHS,LEAVES)]])


def freeze(r):
    base=repeat_dir(r);target=base/'FROZEN_I20.json'
    if target.exists():return json.loads(target.read_text(encoding='utf8'))['choices']
    if any(not verify_completed(inner_dir(r,f,j,k)) for f,j,k in
           itertools.product(range(1,6),range(1,21),range(1,6))):
        raise RuntimeError(f'Cannot freeze I20: not all 500 inner units complete for repeat {r}')
    choices={}
    for f in range(1,6):
        q=pd.concat([pd.read_csv(inner_dir(r,f,j,k)/'grid.csv',float_precision='round_trip')
                     for j,k in itertools.product(range(1,21),range(1,6))],ignore_index=True)
        per=q.groupby(['inner_fold','depth','leaf','trees']).C_index.agg(['mean','count']).reset_index()
        if not per['count'].eq(20).all():raise ValueError('Missing inner MI grid')
        means=per.groupby(['depth','leaf','trees'])['mean'].agg(['mean','count']).reset_index()
        if not means['count'].eq(5).all():raise ValueError('Missing inner fold grid')
        best=means.sort_values(['mean','trees','depth','leaf'],ascending=[False,True,True,False]).iloc[0]
        choices[str(f)]=dict(depth=int(best.depth),leaf=int(best.leaf),
                             trees=int(best.trees),inner_C=float(best['mean']))
        means.to_csv(base/f'selection_fold{f:02d}.csv',index=False)
    write(target,dict(at=now(),repeat_id=r,choices=choices,inner_units=500,
         no_outer_evaluated=True,criterion='20MI mean per inner fold, then fivefold mean; '
         'tie fewer trees, shallower depth, larger leaf'))
    return choices


def outer(r,f,j):
    d=outer_dir(r,f,j)
    if verify_completed(d):return
    choices=json.loads((repeat_dir(r)/'FROZEN_I20.json').read_text(encoding='utf8'))['choices']
    c=choices[str(f)];b=bundle(r,f);tr,va=b.pair(j)
    epochs=[json.loads((inner_dir(r,f,j,k)/'neural/audit.json').read_text(encoding='utf8'))['best_epoch']
            for k in range(1,6)]
    budget=int(np.median(epochs));seed=m.COMMON['seed_base']+(f-1)*1000+j
    d.mkdir(parents=True,exist_ok=True);gpu_guard();start=time.perf_counter()
    fitted,trace,audit=m.fit(tr,b.variables,m.SPEC,m.COMMON,seed,fixed_epochs=budget)
    ca,cv,*_=m.scores(fitted,tr,va)
    torch.save(dict(state_dict={name:value.detach().cpu() for name,value in fitted.net.state_dict().items()},
        encoder=fitted.encoder.__dict__,spec=fitted.spec,common=fitted.common,
        baseline=fitted.baseline,seed=seed),d/'cacox.pt')
    trace.to_csv(d/'training_trace.csv',index=False)
    del fitted;torch.cuda.empty_cache()
    enc=m.Encoder(b.variables,'robust');x=enc.fit_transform(tr);vx=enc.transform(va)
    count=c['trees'];tree_seed=100000+f*1000+j*10+9
    gpu_guard()
    if count:
        model,found,tele=gpu.fit_rank(x,tr.Survival_Days,tr.Death,.5*ca,
                     vx,.5*cv,c['depth'],c['leaf'],[count],tree_seed)
        aa,vv=found[count]
        model.save_model(d/'model.ubj')
    else:
        aa,vv=.5*ca,.5*cv
        tele=dict(fit_device='cuda',prediction_device='cuda',cuda_gradient_calls=0,
                  no_tree_endpoint=True)
    np.savez_compressed(d/'predictions.npz',I20_train=aa,I20_val=vv,
        CACox_train=ca,CACox_val=cv,train_ids=tr.patient_id.to_numpy(),
        val_ids=va.patient_id.to_numpy())
    write(d/'audit.json',dict(at=now(),repeat_id=r,fold=f,imputation=j,
          variables=b.variables,original_CACox_seed=seed,original_CACox_inner_epochs=epochs,
          original_CACox_refit_epochs=budget,original_CACox_audit=audit,
          tree_choice=c,tree_seed=tree_seed,tree_telemetry=tele,
          measured_seconds=time.perf_counter()-start,
          train_C=m.c_evaluator(tr)(aa),test_C=m.c_evaluator(va)(vv),
          input_bundle_SHA=sha(b.directory/'COMPLETE.json')))
    mark_complete(d,['cacox.pt',*(['model.ubj'] if count else []),
                     'predictions.npz','training_trace.csv','audit.json'])


def worker(phase,r,f,index):
    if phase=='inner':jobs=list(itertools.product(range(1,21),range(1,6)))
    else:jobs=[(j,) for j in range(1,21)]
    for args in jobs[index::WORKERS]:
        (inner if phase=='inner' else outer)(r,f,*args)
        write(OUT/'i20'/f'worker_{phase}_{r:02d}_{f:02d}_{index}.json',
              dict(status='running',at=now(),worker_pid=os.getpid(),phase=phase,
                   repeat_id=r,fold=f,imputation=args[0],inner_fold=args[1] if len(args)>1 else None))


def schedule(phase,r,f):
    logdir=repeat_dir(r)/'logs';logdir.mkdir(parents=True,exist_ok=True)
    children=[]
    try:
        for index in range(WORKERS):
            o=logdir/f'{phase}_fold{f:02d}_worker{index}.out.log'
            e=logdir/f'{phase}_fold{f:02d}_worker{index}.err.log'
            with o.open('a',encoding='utf8') as oo,e.open('a',encoding='utf8') as ee:
                children.append(subprocess.Popen([sys.executable,str(__file__),'worker',
                    '--phase',phase,'--repeat',str(r),'--fold',str(f),'--index',str(index)],
                    cwd=ROOT,stdout=oo,stderr=ee,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        while any(p.poll() is None for p in children):
            completed=sum((inner_dir(r,f,j,k)/'COMPLETE.json').exists() for j,k in itertools.product(range(1,21),range(1,6))) if phase=='inner' else sum((outer_dir(r,f,j)/'COMPLETE.json').exists() for j in range(1,21))
            write(OUT/'I20_RUN_STATUS.json',dict(status='running',at=now(),phase=phase,
                  repeat_id=r,fold=f,completed=completed,total=100 if phase=='inner' else 20,
                  controller_pid=os.getpid(),worker_pids=[p.pid for p in children],
                  free_RAM_GB=psutil.virtual_memory().available/2**30))
            if any(p.poll() not in (None,0) for p in children):
                for p in children:p.wait()
                raise RuntimeError(f'I20 {phase} worker failed r={r} f={f}; inspect {logdir}')
            time.sleep(10)
        if any(p.returncode for p in children):raise RuntimeError(f'I20 {phase} failure r={r} f={f}')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()


def wait_bundle(r,f):
    while not verify_completed((OUT/'bundles')/f'repeat{r:02d}_fold{f:02d}'/'main'):
        status=OUT/'RUN_STATUS.json'
        if status.exists() and json.loads(status.read_text(encoding='utf8')).get('status')=='failed':
            raise RuntimeError(f'Data stage failed before repeat {r} fold {f}')
        time.sleep(15)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['run','worker','audit'])
    parser.add_argument('--phase',choices=['inner','outer'])
    parser.add_argument('--repeat',type=int)
    parser.add_argument('--fold',type=int)
    parser.add_argument('--index',type=int)
    args=parser.parse_args()
    if args.command=='worker':
        m.configure();worker(args.phase,args.repeat,args.fold,args.index);return
    if args.command=='audit':
        print(dict(inner=sum(verify_completed(inner_dir(r,f,j,k)) for r,f,j,k in
            itertools.product(range(1,11),range(1,6),range(1,21),range(1,6))),
            outer=sum(verify_completed(outer_dir(r,f,j)) for r,f,j in
            itertools.product(range(1,11),range(1,6),range(1,21)))))
        return
    scientific=[SOURCE,ROOT/'experiments/cacox_tree_joint_study/runner.py',
        ROOT/'experiments/cacox_cuda_pilot/cuda_models.py',
        ROOT/'experiments/cacox_original_head_study/src/cacox_original_head_study/training.py',
        ROOT/'experiments/cacox_original_head_study/src/cacox_original_head_study/models.py',
        ROOT/'experiments/cacox_original_head_study/src/cacox_original_head_study/_numeric_encoder.py',
        Path(__file__)]
    proof=dict(at=now(),source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in scientific},
        torch=torch.__version__,xgboost=xgboost.__version__,workers=WORKERS,
        input='each new repeat uses its completed 29-variable R-MICE Bundle and selected list',
        original_CACox=m.COMMON,tree_grid=dict(depths=list(DEPTHS),leaves=list(LEAVES),counts=list(COUNTS)),
        internal_selection='mean20MI then mean5 inner folds; all choices frozen before outer')
    manifest=OUT/'I20_MANIFEST.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf8'))
        if {k:v for k,v in saved.items() if k!='at'}!={k:v for k,v in proof.items() if k!='at'}:
            raise RuntimeError('I20 scientific code/config/version changed since launch')
    else:write(manifest,proof)
    lock=OUT/'I20_RUNNING.lock'
    if lock.exists():raise RuntimeError(f'I20 lock exists: {lock}')
    lock.write_text(str(os.getpid()),encoding='ascii')
    try:
        for r in range(1,11):
            for f in range(1,6):
                wait_bundle(r,f)
                if any(not verify_completed(inner_dir(r,f,j,k)) for j,k in
                       itertools.product(range(1,21),range(1,6))):schedule('inner',r,f)
            freeze(r)
            for f in range(1,6):
                if any(not verify_completed(outer_dir(r,f,j)) for j in range(1,21)):
                    schedule('outer',r,f)
            write(repeat_dir(r)/'COMPLETE.json',dict(at=now(),repeat_id=r,
                  inner_units=500,outer_units=100,choice_file_SHA=sha(repeat_dir(r)/'FROZEN_I20.json')))
        write(OUT/'I20_COMPLETE.json',dict(at=now(),new_repeats=10,
                                          new_inner=5000,new_outer=1000))
        write(OUT/'I20_RUN_STATUS.json',dict(status='complete',at=now()))
    except BaseException as error:
        write(OUT/'I20_RUN_STATUS.json',dict(status='failed',at=now(),error=repr(error),
                                             controller_pid=os.getpid()))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()

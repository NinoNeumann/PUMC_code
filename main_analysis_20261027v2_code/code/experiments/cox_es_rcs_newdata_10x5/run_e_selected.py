"""Exact E_selected source fits and finite blend on repeated frozen inputs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
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

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/cox_es_rcs_newdata_10x5/20261104_v1'
source_path=ROOT/'experiments/cacox_heterogeneous_blend_stream/backend.py'
spec=importlib.util.spec_from_file_location('repeated_heterogeneous_blend_scientific',source_path)
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)
b=stream.base
source17,source20=b.source17,b.source20
m,rg=b.m,b.rg
sys.path.insert(0,str(ROOT/'experiments/cacox_original_head_study/src'))
from cacox_original_head_study.data import Bundle
from cacox_original_head_study._numeric_encoder import Encoder
from cacox_original_head_study.storage import mark_complete,verify_completed,write_json
from cacox_original_head_study.runtime import sha

NAM=list(b.NAM)
P_C=next(c for c in source17.legacy.CONDITIONS if c['condition_id']=='P_C')
C_J_SP=next(c for c in source17.CONDITIONS if c['condition_id']=='C_J_SP')
NAM_CONDITIONS=[next(c for c in source20.CONDITIONS if c['condition_id']==name) for name in NAM]
FITS=[('rank_objective',P_C),('joint_score',C_J_SP),
      *[('grouped_additive',c) for c in NAM_CONDITIONS]]
WORKERS=2


def now():return datetime.now(timezone.utc).isoformat()
def source_root(r,name):return OUT/'e_selected_sources'/f'repeat{r:02d}'/name
def bind_repeat(r):
    source17.legacy.OUT=source_root(r,'rank_objective')
    source17.OUT=source_root(r,'joint_score')
    source20.OUT=source_root(r,'grouped_additive')
def group_dir(r,phase,f,k=None):
    return OUT/'e_selected_sources'/f'repeat{r:02d}'/phase/f'fold{f:02d}'/(
        f'inner{k:02d}' if k is not None else 'shared')
def fit_dir(r,phase,f,k,kind,cid):
    return source_root(r,kind)/phase/f'fold{f:02d}'/(
        f'inner{k:02d}' if k is not None else 'shared')/'fits'/cid
def blend_dir(r,phase,f,k=None):return group_dir(r,phase,f,k)
def bundle(r,f):
    path=(OUT/'bundles')/f'repeat{r:02d}_fold{f:02d}'/'main'
    if not verify_completed(path):raise ValueError(f'Incomplete bundle {path}')
    return Bundle(path)
def i20_path(r,f,j,k=None):
    root=OUT/'i20'/f'repeat{r:02d}'
    if k is None:return root/'outer'/f'fold{f:02d}'/f'mi{j:02d}'
    return root/'inner'/f'fold{f:02d}'/f'mi{j:02d}_inner{k:02d}'
def i20_scores(r,f,j,k=None):
    path=i20_path(r,f,j,k)
    if not verify_completed(path):raise ValueError(f'Incomplete I20 risk input {path}')
    if k is None:
        with np.load(path/'predictions.npz') as z:return z['I20_train'].copy(),z['I20_val'].copy()
    choice=json.loads((OUT/'i20'/f'repeat{r:02d}'/'FROZEN_I20.json').read_text(encoding='utf8'))['choices'][str(f)]
    tree=path/'trees'/f"d{choice['depth']}_l{choice['leaf']}"/'predictions.npz'
    with np.load(tree) as z:return z[f"n{choice['trees']}_train"].copy(),z[f"n{choice['trees']}_val"].copy()


def load_group(r,f,k=None):
    bndl=bundle(r,f)
    xs=[];vxs=[];i20a=[];i20v=[];encoders=[];proofs={};anchor=None
    for j in range(1,21):
        tr,va=bndl.pair(j,k)
        enc=Encoder(bndl.variables,'robust')
        x=enc.fit_transform(tr);vx=enc.transform(va)
        aa,vv=i20_scores(r,f,j,k)
        if x.dtype!=np.float32 or vx.dtype!=np.float32 or not np.isfinite(x).all() or not np.isfinite(vx).all():
            raise ValueError('Invalid encoded clinical input')
        if len(aa)!=len(tr) or len(vv)!=len(va):raise ValueError('I20 risk dimensions differ')
        labels=[tr.patient_id.to_numpy(),va.patient_id.to_numpy(),
                tr.Survival_Days.to_numpy(),tr.Death.to_numpy(bool),
                va.Survival_Days.to_numpy(),va.Death.to_numpy(bool)]
        if anchor is None:
            anchor=[z.copy() for z in labels]
            columns=list(enc.columns)
            tr_labels=tr[['patient_id','Death','Survival_Days']].copy()
            va_labels=va[['patient_id','Death','Survival_Days']].copy()
        else:
            for expected,actual in zip(anchor,labels):np.testing.assert_array_equal(actual,expected)
            if columns!=list(enc.columns):raise ValueError('MI column order mismatch')
        xs.append(x.copy());vxs.append(vx.copy());i20a.append(aa);i20v.append(vv)
        encoders.append(enc.__dict__.copy())
        path=i20_path(r,f,j,k);proofs[str(path.relative_to(ROOT))]=sha(path/'COMPLETE.json')
    if k is not None:
        outer_ids=set(bndl.raw_assessment.patient_id)
        if (set(anchor[0])|set(anchor[1])) & outer_ids:raise ValueError('Outer assessment in inner group')
    metadata=dict(repeat_id=r,fold=f,inner_fold=k,MI_views=list(range(1,21)),
        variables=bndl.variables,columns=columns,encoders=encoders,
        training_ids=anchor[0].tolist(),query_ids=anchor[1].tolist(),
        input_complete_SHA=proofs,original_I20_root=str(i20_path(r,f,1,k).parents[2]),
        all_views_same_members_order_columns_outcomes=True,no_new_input_transform=True)
    return dict(x=np.stack(xs),vx=np.stack(vxs),I20_train=np.stack(i20a),I20_val=np.stack(i20v),
        train_time=anchor[2],train_event=anchor[3],val_time=anchor[4],val_event=anchor[5],
        train_ids=anchor[0],val_ids=anchor[1],tr=tr_labels,va=va_labels,metadata=metadata)


def save_fit(backend,g,c,f,k,d,epochs=None):
    if verify_completed(d):return json.loads((d/'audit.json').read_text(encoding='utf8'))
    if psutil.virtual_memory().available<5*2**30:raise MemoryError('Less than 5 GiB free RAM')
    d.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    net,trace,audit=backend.fit(g,c,f,k,fixed_epochs=epochs)
    a=backend.predict_members(net,g['x']);v=backend.predict_members(net,g['vx'])
    backend.save(net,d/'neural.pt',g,c,f,k)
    np.savez_compressed(d/'predictions.npz',train_members=a,val_members=v,
                        train_ids=g['train_ids'],val_ids=g['val_ids'])
    audit.update(repeat_id=g['metadata']['repeat_id'],fold=f,inner_fold=k,
                 metadata=g['metadata'],measured_seconds=time.perf_counter()-start)
    m.write_json(d/'audit.json',audit)
    pd.DataFrame(trace).to_csv(d/'training_trace.csv',index=False)
    mark_complete(d,['neural.pt','predictions.npz','audit.json','training_trace.csv'])
    del net;torch.cuda.empty_cache()
    return audit


def train_group(r,f,k=None,budgets=None):
    bind_repeat(r)
    phase='inner' if k is not None else 'outer'
    g=load_group(r,f,k)
    for kind,c in FITS:
        backend={'rank_objective':source17.legacy,'joint_score':source17,
                 'grouped_additive':source20}[kind]
        epochs=None if k is not None else budgets[c['condition_id']]['epochs']
        save_fit(backend,g,c,f,k,fit_dir(r,phase,f,k,kind,c['condition_id']),epochs)
    d=blend_dir(r,phase,f,k)
    if verify_completed(d):return
    pred,refs=b.sources(g,f,k)
    mixed,stats=b.combine(pred)
    d.mkdir(parents=True,exist_ok=True)
    if k is not None:
        grid=pd.DataFrame(stream.grid(g,mixed,f,k))
        if len(grid)!=17*20:raise ValueError('E_selected 17x20 candidate grid incomplete')
        grid.to_csv(d/'grid.csv',index=False)
        m.write_json(d/'audit.json',dict(at=now(),repeat_id=r,fold=f,inner_fold=k,
               source=g['metadata'],references=refs,training_score_statistics=stats,
               CUDA_inference=True,candidates=17,outer_evaluated=False))
        mark_complete(d,['grid.csv','audit.json'])
    else:
        with torch.no_grad():
            aa=np.stack([mixed[cid][0].cpu().numpy() for cid in b.ORDER])
            vv=np.stack([mixed[cid][1].cpu().numpy() for cid in b.ORDER])
        np.savez_compressed(d/'risks.npz',train=aa,val=vv,
                            train_ids=g['train_ids'],val_ids=g['val_ids'])
        m.write_json(d/'audit.json',dict(at=now(),repeat_id=r,fold=f,
            source=g['metadata'],references=refs,training_score_statistics=stats,
            CUDA_inference=True,candidates=17))
        mark_complete(d,['risks.npz','audit.json'])


def freeze(r):
    root=source_root(r,'blend')
    dest=root/'FROZEN_CHOICES.json'
    if dest.exists():return json.loads(dest.read_text(encoding='utf8'))
    choices={};epochs={}
    for f in range(1,6):
        if any(not verify_completed(blend_dir(r,'inner',f,k)) for k in range(1,6)):
            raise RuntimeError(f'Incomplete source groups before freeze r={r} f={f}')
        q=pd.concat([pd.read_csv(blend_dir(r,'inner',f,k)/'grid.csv',float_precision='round_trip')
                     for k in range(1,6)],ignore_index=True)
        mean=q.groupby(['candidate','inner_fold']).C_index.mean().groupby('candidate').mean()
        values={cid:float(mean[cid]) for cid in b.ORDER};top=max(values.values())
        cid=next(cid for cid in b.ORDER if values[cid]>=top-1e-12)
        choices[str(f)]=dict(candidate=cid,condition=next(z for z in b.CONDITIONS if z['candidate']==cid),
            inner_C=values[cid],all_candidate_C=values)
        epochs[str(f)]={}
        for kind,c in FITS:
            best=[json.loads((fit_dir(r,'inner',f,k,kind,c['condition_id'])/'audit.json').read_text(encoding='utf8'))['best_epoch'] for k in range(1,6)]
            epochs[str(f)][c['condition_id']]=dict(best_inner_epochs=best,
                epochs=int(np.ceil(np.median(best))))
    root.mkdir(parents=True,exist_ok=True)
    m.write_json(dest,dict(at=now(),repeat_id=r,choices=choices,epochs=epochs,
        all_fivefold_before_outer=True,selection='mean20MI per innerfold then mean5fold;1e-12 tie;fixed candidate order'))
    return json.loads(dest.read_text(encoding='utf8'))


def output_fold(r,f):
    bind_repeat(r)
    root=source_root(r,'blend')
    frozen=json.loads((root/'FROZEN_CHOICES.json').read_text(encoding='utf8'))
    d=blend_dir(r,'outer',f)
    if not verify_completed(d):raise ValueError('Missing blended outer risk')
    g=load_group(r,f)
    with np.load(d/'risks.npz') as z:
        aa,vv=z['train'].copy(),z['val'].copy()
    for j in range(1,21):
        target=root/'outer'/f'fold{f:02d}'/f'mi{j:02d}'
        if verify_completed(target):continue
        target.mkdir(parents=True,exist_ok=True)
        selected=frozen['choices'][str(f)]['candidate']
        roles=[*b.ORDER,'E_selected']
        for role in roles:
            cid=selected if role=='E_selected' else role
            index=b.ORDER.index(cid)
            train=aa[index,j-1];val=vv[index,j-1]
            baseline=b.io.parent.old.gpu.cuda_breslow(g['train_time'],g['train_event'],train)
            survival=b.io.parent.old.gpu.cuda_survival(val,baseline).cpu().numpy()
            frame=g['va'].copy()
            frame['repeat_id']=r;frame['fold']=f;frame['imputation']=j;frame['score']=val
            for i,t in enumerate(m.COMMON['times']):frame[f'survival_{t}d']=survival[:,i]
            frame=m.normalize(frame,role,'cox_es_rcs_newdata_10x5_20261104_v1',
                'Frozen17/20 fullrisk convex mixture;training-score alignment;no new blend fits')
            frame.to_csv(target/f'{role}.csv',index=False)
        mark_complete(target,[f'{role}.csv' for role in roles])


def worker(phase,r,index):
    bind_repeat(r)
    if phase=='inner':jobs=list(itertools.product(range(1,6),range(1,6)))
    else:jobs=[(f,None) for f in range(1,6)]
    frozen=None if phase=='inner' else json.loads((source_root(r,'blend')/'FROZEN_CHOICES.json').read_text(encoding='utf8'))
    for f,k in jobs[index::WORKERS]:
        budgets=None if k is not None else frozen['epochs'][str(f)]
        train_group(r,f,k,budgets)
        if k is None:output_fold(r,f)
        m.write_json(OUT/'e_selected_sources'/f'repeat{r:02d}'/
            f'worker_{phase}_{index}.json',dict(at=now(),status='running',phase=phase,
                worker_pid=os.getpid(),repeat_id=r,fold=f,inner_fold=k))


def schedule(phase,r):
    logdir=source_root(r,'blend')/'logs';logdir.mkdir(parents=True,exist_ok=True)
    children=[]
    try:
        for index in range(WORKERS):
            with (logdir/f'{phase}_worker{index}.out.log').open('a',encoding='utf8') as oo,\
                 (logdir/f'{phase}_worker{index}.err.log').open('a',encoding='utf8') as ee:
                children.append(subprocess.Popen([sys.executable,str(__file__),'worker',
                    '--phase',phase,'--repeat',str(r),'--index',str(index)],
                    cwd=ROOT,stdout=oo,stderr=ee,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0))
        while any(p.poll() is None for p in children):
            if phase=='inner':
                completed=sum((blend_dir(r,'inner',f,k)/'COMPLETE.json').exists()
                              for f,k in itertools.product(range(1,6),range(1,6)))
                total=25
            else:
                completed=sum((source_root(r,'blend')/'outer'/f'fold{f:02d}'/f'mi{j:02d}'/'COMPLETE.json').exists()
                              for f,j in itertools.product(range(1,6),range(1,21)))
                total=100
            m.write_json(OUT/'E_SELECTED_RUN_STATUS.json',dict(at=now(),status='running',
                phase=phase,repeat_id=r,completed=completed,total=total,
                controller_pid=os.getpid(),worker_pids=[p.pid for p in children],
                free_RAM_GB=psutil.virtual_memory().available/2**30))
            if any(p.poll() not in (None,0) for p in children):
                for p in children:p.wait()
                raise RuntimeError(f'E_selected {phase} worker failed repeat {r}; see {logdir}')
            time.sleep(10)
        if any(p.returncode for p in children):raise RuntimeError(f'E_selected {phase} worker failure repeat {r}')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()


def wait_i20():
    while not (OUT/'I20_COMPLETE.json').exists():
        status=OUT/'I20_RUN_STATUS.json'
        if status.exists() and json.loads(status.read_text(encoding='utf8')).get('status')=='failed':
            raise RuntimeError(f'I20 source training failed; see {status}')
        m.write_json(OUT/'E_SELECTED_RUN_STATUS.json',dict(at=now(),status='waiting_for_I20',
                     controller_pid=os.getpid()))
        time.sleep(30)


def run_all():
    sources=[Path(__file__),source_path,
        ROOT/'experiments/cacox_heterogeneous_blend/backend.py',
        ROOT/'experiments/cacox_heterogeneous_blend/blend.py',
        ROOT/'experiments/cacox_joint_score_finetune/backend.py',
        ROOT/'experiments/cacox_shared_rank_objective/backend.py',
        ROOT/'experiments/cacox_grouped_additive/backend.py',
        ROOT/'experiments/cacox_tabm_cox/tabm_backend.py',
        ROOT/'experiments/cacox_additive_prototype/model.py',
        ROOT/'experiments/cacox_rank_objective_prototype/pair_loss.py',
        ROOT/'experiments/cacox_smooth_concordance_prototype/losses.py']
    manifest=OUT/'E_SELECTED_MANIFEST.json'
    signed=dict(source_SHA256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        workers=WORKERS,conditions=[c for _,c in FITS],blend_conditions=b.CONDITIONS,
        candidate_order=b.ORDER,source_alpha=source17.ALPHA,
        torch=torch.__version__,input='new repeats: same selected R MICE/I20 group contract',
        first_repeat='new data: all 100 outer outputs refitted')
    if manifest.exists():
        old_manifest=json.loads(manifest.read_text(encoding='utf8'))
        if old_manifest['signed']!=signed:raise RuntimeError('E_selected code/config changed since launch')
    else:m.write_json(manifest,dict(at=now(),signed=signed))
    lock=OUT/'E_SELECTED_RUNNING.lock'
    if lock.exists():raise RuntimeError(f'E_selected lock exists: {lock}')
    lock.write_text(str(os.getpid()),encoding='ascii')
    try:
        wait_i20()
        for r in range(1,11):
            if not (OUT/'i20'/f'repeat{r:02d}'/'COMPLETE.json').exists():
                raise ValueError(f'Incomplete I20 repeat {r}')
            if any(not verify_completed(blend_dir(r,'inner',f,k)) for f,k in
                   itertools.product(range(1,6),range(1,6))):schedule('inner',r)
            freeze(r)
            if any(not verify_completed(source_root(r,'blend')/'outer'/f'fold{f:02d}'/f'mi{j:02d}')
                   for f,j in itertools.product(range(1,6),range(1,21))):schedule('outer',r)
            if any(not verify_completed(source_root(r,'blend')/'outer'/f'fold{f:02d}'/f'mi{j:02d}')
                   for f,j in itertools.product(range(1,6),range(1,21))):
                raise ValueError(f'Incomplete E_selected outer predictions for repeat {r}')
            m.write_json(source_root(r,'blend')/'COMPLETE.json',dict(at=now(),repeat_id=r,
                inner_groups=25,outer_units=100,source_fits=180,
                selected_choice_SHA=sha(source_root(r,'blend')/'FROZEN_CHOICES.json')))
        m.write_json(OUT/'E_SELECTED_COMPLETE.json',dict(at=now(),repeats=10,
            reused_first_repeat_models=0,new_outer_models=1000,
            new_source_groups=300))
        m.write_json(OUT/'E_SELECTED_RUN_STATUS.json',dict(at=now(),status='complete'))
    except BaseException as error:
        m.write_json(OUT/'E_SELECTED_RUN_STATUS.json',dict(at=now(),status='failed',
                     error=repr(error),controller_pid=os.getpid()))
        raise
    finally:lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['audit','group','worker','run'])
    parser.add_argument('--phase',choices=['inner','outer'])
    parser.add_argument('--index',type=int)
    parser.add_argument('--repeat',type=int)
    parser.add_argument('--fold',type=int)
    parser.add_argument('--inner',type=int)
    args=parser.parse_args()
    stream.configure()
    if args.command=='group':
        train_group(args.repeat,args.fold,args.inner)
        print('GROUP_COMPLETE',args.repeat,args.fold,args.inner)
        return
    if args.command=='worker':
        worker(args.phase,args.repeat,args.index)
        return
    if args.command=='audit':
        print(dict(inner_groups=sum(verify_completed(blend_dir(r,'inner',f,k)) for r,f,k in
          itertools.product(range(1,11),range(1,6),range(1,6))),
          outer_groups=sum(verify_completed(blend_dir(r,'outer',f)) for r,f in
          itertools.product(range(1,11),range(1,6)))))
        return
    run_all()


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()

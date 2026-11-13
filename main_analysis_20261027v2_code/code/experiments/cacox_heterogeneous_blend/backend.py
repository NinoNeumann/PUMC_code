"""Read-only17/20 sources and CUDA finite risk mixtures; no optimizer exists."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
source17=module('blend21_frozen_joint_backend',ROOT/'experiments/cacox_joint_score_finetune/backend.py')
source20=module('blend21_frozen_additive_backend',ROOT/'experiments/cacox_grouped_additive/backend.py')
blend=module('blend21_verified_formula',Path(__file__).with_name('blend.py'))
io,m,rg=source17.io,source17.m,source17.rg;OUT=ROOT/'results/cacox_heterogeneous_blend/20261031_v1'
NAM=['A8_NLL','A8_C','A16_NLL','A16_C'];CONTROL='C0_ref17'
CONDITIONS=[dict(candidate=CONTROL,source='C_J_SP',weight17=1.)]+[dict(candidate=f'{name}_w{int(weight*100):03d}',source=name,weight17=weight) for name in NAM for weight in [.75,.5,.25,0.]]
ORDER=[c['candidate'] for c in CONDITIONS];ROLES=[*ORDER,'E_selected']
def read(p):return io.read(p)
def configure():source17.configure()
def source_dir(f,k,name):
    root=source17.OUT if name=='C_J_SP' else source20.OUT
    return root/('inner' if k is not None else 'outer')/f'fold{f:02d}'/(f'inner{k:02d}' if k is not None else 'shared')/'fits'/name
def sources(g,f,k,replay=False):
    predictions={};proofs={}
    for name in ['C_J_SP',*NAM]:
        directory=source_dir(f,k,name);assert m.verify_completed(directory);obj=torch.load(directory/'neural.pt',map_location='cpu',weights_only=False)
        owner=source17 if name=='C_J_SP' else source20
        assert obj['metadata']==g['metadata'] and obj['seed']==io.seed(f,k) and obj['common']==owner.COMMON and obj['config']==owner.CONFIG and obj['condition']['condition_id']==name
        with np.load(directory/'predictions.npz') as z:
            np.testing.assert_array_equal(z['train_ids'],g['train_ids']);np.testing.assert_array_equal(z['val_ids'],g['val_ids']);a,v=z['train_members'].copy(),z['val_members'].copy()
        assert a.shape==(20,len(g['train_ids']),8) and v.shape==(20,len(g['val_ids']),8)
        if replay:
            net,checkpoint=owner.load(directory/'neural.pt');assert next(net.parameters()).is_cuda
            np.testing.assert_array_equal(a,owner.predict_members(net,g['x']));np.testing.assert_array_equal(v,owner.predict_members(net,g['vx']));del net
        if name=='C_J_SP':
            a,v=source17.combine_np(g,a.mean(2),v.mean(2),obj['normalization'],f);normalization=obj['normalization']
        else:
            train=a.mean(2);query=v.mean(2);rows=[io.source.r.mix(g['I20_train'][j],g['I20_val'][j],train[j],query[j],1-source20.ALPHA[f-1]) for j in range(20)]
            a,v=np.stack([z[0] for z in rows]),np.stack([z[1] for z in rows]);normalization=[z[2] for z in rows]
            assert obj['architecture_spec']==source20.training_spec(g,obj['condition'])
        predictions[name]=(a,v);proofs[name]=dict(directory=str(directory.relative_to(ROOT)),COMPLETE_SHA=m.sha(directory/'COMPLETE.json'),neural_SHA=m.sha(directory/'neural.pt'),normalization=normalization)
    return predictions,proofs
def combine(predictions):
    a,v=predictions['C_J_SP'];stats={name:blend.statistics(a,predictions[name][0]) for name in NAM};result={CONTROL:(torch.as_tensor(a,device='cuda'),torch.as_tensor(v,device='cuda'))}
    for c in CONDITIONS[1:]:
        aa,vv=predictions[c['source']];s=stats[c['source']];result[c['candidate']]=(blend.cuda_score(a,aa,c['weight17'],s),blend.cuda_score(v,vv,c['weight17'],s))
    return result,stats
def pair_indices(t,e):
    e=np.asarray(e,bool);t=np.asarray(t);return np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
def CUDA_C(scores,t,e):
    i,j=pair_indices(t,e);assert len(i)>0;left,right=torch.as_tensor(i,device='cuda'),torch.as_tensor(j,device='cuda');scores=torch.stack(scores)
    a,b=scores[...,left],scores[...,right];twice=(2*(a>b).to(torch.int64)+(a==b).to(torch.int64)).sum(-1)
    return twice.cpu().numpy()/(2*len(i))
def grid(g,result,f,k):
    train=CUDA_C([result[c][0] for c in ORDER],g['train_time'],g['train_event']);valid=CUDA_C([result[c][1] for c in ORDER],g['val_time'],g['val_event']);rows=[]
    for i,c in enumerate(ORDER):
        for j in range(20):rows.append(dict(candidate=c,fold=f,inner_fold=k,imputation=j+1,C_index=float(valid[i,j]),train_C=float(train[i,j])))
    return rows
def independent_C(t,e,v):
    i,j=pair_indices(t,e);v=np.asarray(v);return float(((v[i]>v[j])+.5*(v[i]==v[j])).mean())
def native_grid(f,k,name):
    root=source17.OUT if name=='C_J_SP' else source20.OUT
    q=pd.read_csv(root/f'inner/fold{f:02d}/inner{k:02d}/grid.csv',float_precision='round_trip');q=q[q.candidate.eq(name)&q.kind.eq('I20')].sort_values('imputation');assert len(q)==20;return q

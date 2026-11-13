"""Execution-only21 resource fix; signed scientific sources remain read-only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
base=module('stream21_signed_scientific_backend',ROOT/'experiments/cacox_heterogeneous_blend/backend.py')
OUT=ROOT/'results/cacox_heterogeneous_blend/20261031_v2_stream'
io,m,rg,source17,source20=base.io,base.m,base.rg,base.source17,base.source20
blend=base.blend;NAM=base.NAM;CONTROL=base.CONTROL;CONDITIONS=base.CONDITIONS;ORDER=base.ORDER;ROLES=base.ROLES
configure=base.configure;read=base.read;sources=base.sources;combine=base.combine;pair_indices=base.pair_indices;independent_C=base.independent_C;native_grid=base.native_grid;source_dir=base.source_dir
CANDIDATE_BLOCK=1;VIEW_BLOCK=4;PAIR_BLOCK=4096
def CUDA_C(scores,t,e):
    """Exactly sum the same int64 twice-wins with bounded temporary tensors."""
    i,j=pair_indices(t,e);assert len(i)>0
    x=torch.stack(scores);assert x.dtype==torch.float64 and x.ndim==3 and x.shape[1]==20
    left=torch.as_tensor(i,device='cuda');right=torch.as_tensor(j,device='cuda');count=torch.zeros(x.shape[:2],dtype=torch.int64,device='cuda')
    for c in range(x.shape[0]):
        for view in range(0,20,VIEW_BLOCK):
            subset=x[c:c+1,view:view+VIEW_BLOCK];part=torch.zeros(subset.shape[:2],dtype=torch.int64,device='cuda')
            for start in range(0,len(i),PAIR_BLOCK):
                a=subset.index_select(-1,left[start:start+PAIR_BLOCK]);b=subset.index_select(-1,right[start:start+PAIR_BLOCK])
                part.add_((2*(a>b).to(torch.int64)+(a==b).to(torch.int64)).sum(-1))
            count[c:c+1,view:view+VIEW_BLOCK].copy_(part)
    return count.cpu().numpy()/(2*len(i))
def grid(g,result,f,k):
    train=CUDA_C([result[c][0] for c in ORDER],g['train_time'],g['train_event']);valid=CUDA_C([result[c][1] for c in ORDER],g['val_time'],g['val_event']);rows=[]
    for i,c in enumerate(ORDER):
        for j in range(20):rows.append(dict(candidate=c,fold=f,inner_fold=k,imputation=j+1,C_index=float(valid[i,j]),train_C=float(train[i,j])))
    return rows

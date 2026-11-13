"""Local diagonal Hessian majorization for the existing G1 pairwise loss."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cacox_cuda_pilot'))
import cuda_models as base
cp,xgb=base.cp,base.xgb

def derivatives(score,pairs):
    i,j=pairs
    if not len(i):raise ValueError('No comparable survival pairs')
    q=torch.sigmoid(score[j]-score[i]);factor=len(score)/len(i)
    g=torch.zeros_like(score);g.scatter_add_(0,j,q);g.scatter_add_(0,i,-q);g*=factor
    w=q*(1-q)*factor
    diagonal=torch.zeros_like(score);diagonal.scatter_add_(0,i,w);diagonal.scatter_add_(0,j,w)
    bound=2*diagonal+1e-6
    scale=bound.max().clamp_min(1.)
    return (g/scale).contiguous(),(bound/scale).contiguous(),g,diagonal,scale

def objective_value(scores,t,e):
    t=np.asarray(t);e=np.asarray(e).astype(bool);s=np.asarray(scores,float)
    i,j=np.nonzero(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
    return float(np.logaddexp(0.,s[j]-s[i]).mean())

def fit(x,t,e,offset,vx,voffset,depth,leaf,counts,seed,subsample=.8,eta=.03):
    xx,vv=cp.asarray(x,dtype=cp.float32),cp.asarray(vx,dtype=cp.float32)
    oo,ov=cp.asarray(offset,dtype=cp.float32),cp.asarray(voffset,dtype=cp.float32)
    train=xgb.QuantileDMatrix(xx,label=cp.zeros(len(x),dtype=cp.float32),base_margin=oo,max_bin=32)
    pairs=base.comparable_pairs(t,e,'cuda')
    params=dict(device='cuda',tree_method='hist',max_depth=int(depth),min_child_weight=int(leaf),
                eta=float(eta),subsample=float(subsample),sampling_method='uniform',reg_lambda=0.,reg_alpha=0.,
                gamma=0.,max_bin=32,seed=int(seed),nthread=1,disable_default_eval_metric=1)
    model=xgb.Booster(params,[train]);found={};scales=[];mins=[]
    if 0 in counts:found[0]=(np.asarray(offset).copy(),np.asarray(voffset).copy())
    base.sync();start=time.perf_counter()
    for iteration in range(max(counts)):
        pred=oo if iteration==0 else model.inplace_predict(xx,predict_type='margin',base_margin=oo)
        if not isinstance(pred,cp.ndarray):raise RuntimeError('Non-CUDA predictions')
        score=torch.utils.dlpack.from_dlpack(pred)
        g,h,_,_,scale=derivatives(score,pairs)
        if not (g.is_cuda and h.is_cuda):raise RuntimeError('Non-CUDA derivatives')
        if not bool(torch.isfinite(g).all()&torch.isfinite(h).all()&(h>0).all()&(h<=1).all()):
            raise FloatingPointError('Invalid normalized curvature')
        model.boost(train,iteration,grad=cp.from_dlpack(g),hess=cp.from_dlpack(h))
        scales.append(float(scale));mins.append(float(h.min()))
        n=iteration+1
        if n in counts:
            a=model.inplace_predict(xx,predict_type='margin',base_margin=oo)
            b=model.inplace_predict(vv,predict_type='margin',base_margin=ov)
            if not isinstance(a,cp.ndarray) or not isinstance(b,cp.ndarray):raise RuntimeError('Non-CUDA inference')
            found[n]=(cp.asnumpy(a).astype(float),cp.asnumpy(b).astype(float))
    base.sync();elapsed=time.perf_counter()-start
    # Audit on all original training rows, not only the stochastic in-bag rows.
    leaves=model.predict(train,pred_leaf=True,strict_shape=False)
    if leaves.ndim==1:leaves=leaves[:,None]
    sizes=[int(np.unique(leaves[:,k],return_counts=True)[1].min()) for k in range(leaves.shape[1])]
    if min(sizes)<leaf:raise AssertionError('Minimum training-leaf population violated')
    tele=dict(fit_seconds=elapsed,cuda_gradient_calls=max(counts),cuda_train=True,cuda_inference=True,
              hessian='2*diag(true pairwise Hessian)+1e-6, scaled with gradient by max(1,max(bound))',
              scale_min=min(scales),scale_max=max(scales),working_weight_min=min(mins),working_weight_max_at_most=1.,
              minimum_training_leaf_count=min(sizes),minimum_required_leaf_count=int(leaf),
              booster_config=json.loads(model.save_config()))
    return model,found,tele

"""Compact official TabM + exact Efron Cox loss; no preprocessing or splits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import copy, random, time
import numpy as np
import torch
from tabm import TabM

CONFIGS=[dict(config_id=f'k{k}_lr{int(lr*1e6)}',members=k,learning_rate=lr)
         for k in [1,8] for lr in [.0003,.001]]
COMMON=dict(blocks=2,width=64,dropout=.15,weight_decay=.0003,
            optimizer='AdamW',gradient_clip=5.,max_epochs=3000,patience=150,
            min_delta=0.,device='cuda',precision='float32_network_float64_loss')


def configure():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


class Efron:
    """O(n*k) sorted risk sums, with correct same-time risk sets and Efron ties."""
    def __init__(self,t,e,device='cuda'):
        t=np.asarray(t,float);e=np.asarray(e,bool)
        if not np.isfinite(t).all() or not e.any():raise ValueError('Finite time and events required')
        order=np.argsort(-t,kind='stable');st=t[order];se=e[order]
        times=np.unique(st[se]);ends=[];fraction=[]
        for tt in times:
            # All patients at event time belong to its risk set, regardless of row order.
            group=np.flatnonzero((st==tt)&se);m=len(group)
            ends.extend([int(np.flatnonzero(st>=tt)[-1])]*m)
            fraction.extend(np.arange(m,dtype=float)/m)
        unique_groups=[np.flatnonzero((st==tt)&se) for tt in times]
        death_index=np.concatenate(unique_groups);death_group=np.concatenate([np.full(len(g),i) for i,g in enumerate(unique_groups)])
        repeated_group=np.repeat(np.arange(len(times)),[len(g) for g in unique_groups])
        self.order=torch.as_tensor(order,device=device)
        self.event=torch.as_tensor(se,device=device)
        self.ends=torch.as_tensor(ends,device=device)
        self.death_index=torch.as_tensor(death_index,device=device)
        self.death_group=torch.as_tensor(death_group,device=device)
        self.repeated_group=torch.as_tensor(repeated_group,device=device)
        self.fraction=torch.as_tensor(fraction,dtype=torch.float64,device=device)
        self.ngroups=len(times);self.nevents=int(e.sum());self.n=len(t)

    def __call__(self,eta,memberwise=False):
        if eta.ndim==1:eta=eta[:,None]
        v=eta.double()[self.order]
        log_risk=torch.logcumsumexp(v,dim=0)[self.ends]
        # Every death in a tie group has the same full risk set. Scaling each
        # death by that log risk avoids underflow for extreme inter-patient risks.
        fractions=torch.exp(v[self.death_index]-log_risk)
        death_ratio=torch.zeros((self.ngroups,v.shape[1]),dtype=torch.float64,device=v.device)
        death_ratio.index_add_(0,self.death_group,fractions)
        log_correction=torch.log1p(-self.fraction[:,None]*death_ratio[self.repeated_group])
        loss=((log_risk-v[self.death_index])+log_correction).sum(0)/self.nevents
        return loss if memberwise else loss.mean()


def make(features,config,seed,device='cuda'):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed)
    return TabM.make(n_num_features=int(features),d_out=1,n_blocks=COMMON['blocks'],
                     d_block=COMMON['width'],k=config['members'],dropout=COMMON['dropout']).to(device)


def tensor(x,device='cuda'):
    a=np.asarray(x)
    if a.dtype!=np.float32 or not np.isfinite(a).all():raise ValueError('Frozen finite float32 encoded input required')
    return torch.as_tensor(a,device=device)


def output(model,x):return model(x_num=x,x_cat=None).squeeze(-1)


def predict_members(model,x):
    model.eval()
    with torch.no_grad():return output(model,tensor(x,next(model.parameters()).device)).double().cpu().numpy()


def fit(x,t,e,config,seed,vx=None,vt=None,ve=None,fixed_epochs=None,progress=None):
    if fixed_epochs is None and vx is None:raise ValueError('Inner validation or frozen epoch count required')
    if fixed_epochs is not None and not 1<=fixed_epochs<=COMMON['max_epochs']:raise ValueError('Invalid frozen epochs')
    start=time.perf_counter();xt=tensor(x);net=make(x.shape[1],config,seed)
    train_loss=Efron(t,e);val_loss=Efron(vt,ve) if fixed_epochs is None else None
    xv=tensor(vx) if val_loss is not None else None
    optimizer=torch.optim.AdamW(net.parameters(),lr=config['learning_rate'],weight_decay=COMMON['weight_decay'])
    best=float('inf');best_epoch=0;stale=0;state=None;trace=[]
    for epoch in range(1,(fixed_epochs or COMMON['max_epochs'])+1):
        net.train();optimizer.zero_grad(set_to_none=True)
        loss=train_loss(output(net,xt))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite TabM training Cox loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),COMMON['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();row=dict(epoch=epoch,training_member_mean_Efron=float(loss.detach()),gradient_norm_before_clip=float(norm))
        if val_loss is not None:
            net.eval()
            with torch.no_grad():members=output(net,xv);v=float(val_loss(members.mean(1)))
            if not np.isfinite(v):raise FloatingPointError('Nonfinite TabM validation ensemble Cox loss')
            row['validation_ensemble_Efron']=v
            if v<best:
                best=v;best_epoch=epoch;stale=0;state=copy.deepcopy(net.state_dict())
            else:stale+=1
            row.update(best_epoch=best_epoch,stale=stale)
        trace.append(row)
        if progress and (epoch==1 or epoch%50==0):progress(dict(epoch=epoch,best_epoch=best_epoch,stale=stale))
        if val_loss is not None and stale>=COMMON['patience']:break
    if val_loss is not None:net.load_state_dict(state)
    net.eval();torch.cuda.synchronize()
    audit=dict(seed=seed,config=config,common=COMMON,best_epoch=best_epoch if val_loss is not None else epoch,
               stopped_epoch=epoch,cap_hit=val_loss is not None and epoch==COMMON['max_epochs'],
               parameters=sum(p.numel() for p in net.parameters()),fit_seconds=time.perf_counter()-start,
               loss='mean per-member exact Efron;validation Efron of mean member risk',
               optimizer='AdamW decoupled weight decay',full_batch=True,training_patients=len(x),
               training_events=int(np.asarray(e).sum()),cuda_train=next(net.parameters()).is_cuda)
    return net,trace,audit


def save(model,path,features,config,seed,metadata):
    torch.save(dict(state={k:v.cpu() for k,v in model.state_dict().items()},features=features,
                    config=config,seed=seed,common=COMMON,metadata=metadata),path)


def load(path,device='cuda'):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    assert obj['common']==COMMON
    net=make(obj['features'],obj['config'],obj['seed'],device);net.load_state_dict(obj['state']);net.eval()
    return net,obj

"""Iteration15: fixed sharedMI TabM, final mean-risk pair objective only."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,copy,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
previous=module('rankobjective_frozen_seed_backend',ROOT/'experiments/cacox_shared_seed_stop/backend.py')
prototype=module('rankobjective_verified_pair_loss',ROOT/'experiments/cacox_rank_objective_prototype/pair_loss.py')
parent=previous.parent;io,m,rg=previous.io,previous.m,previous.rg
OUT=ROOT/'results/cacox_shared_rank_objective/20261031_v1';COMMON=dict(previous.COMMON);CONFIG=dict(previous.CONFIG)
CONDITIONS=[dict(condition_id=name+'_'+monitor,rank_weight=w,monitor=monitor) for name,w in [('H',.5),('P',1.)] for monitor in ['NLL','C']]
ALPHA=list(previous.ALPHA)
def configure():previous.configure()
def seed(f,k=None):return io.seed(f,k)
def predict_members(net,x):return parent.predict_members(net,x)
def read(p):return io.read(p)
def source_dir(f,k,monitor):
    return previous.source_dir(f,k) if monitor=='NLL' else previous.OUT/('inner' if k is not None else 'outer')/f'fold{f:02d}'/(f'inner{k:02d}' if k is not None else 'shared')/'fits/c_s0'

def fit(group,condition,f,k=None,fixed_epochs=None,progress=None):
    weight=condition['rank_weight'];monitor=condition['monitor'];assert weight in [0.,.5,1.] and monitor in ['NLL','C']
    M,n,p=group['x'].shape;assert M==20
    if fixed_epochs is not None and not 1<=fixed_epochs<=COMMON['max_epochs']:raise ValueError('Invalid frozen epoch budget')
    start=time.perf_counter();net=parent.b.make(p,CONFIG,seed(f,k));initial=parent.initial_digest(net);X=parent.b.tensor(group['x'].reshape(M*n,p))
    cox_loss=parent.b.Efron(group['train_time'],group['train_event']);rank_loss=prototype.MeanRiskPairLoss(group['train_time'],group['train_event'],'cuda')
    if fixed_epochs is None:
        nv=group['vx'].shape[1];VX=parent.b.tensor(group['vx'].reshape(M*nv,p));val_loss=parent.b.Efron(group['val_time'],group['val_event']);C=previous.ValidationC(group)
    optimizer=torch.optim.AdamW(net.parameters(),lr=CONFIG['learning_rate'],weight_decay=COMMON['weight_decay'])
    best=float('inf') if monitor=='NLL' else -1;best_epoch=0;state=None;stale=0;trace=[]
    for epoch in range(1,(fixed_epochs or COMMON['max_epochs'])+1):
        net.train();optimizer.zero_grad(set_to_none=True);eta=parent.b.output(net,X).reshape(M,n,8);cox=cox_loss(eta.permute(1,0,2).reshape(n,M*8))
        rank=rank_loss(eta)
        # Exact endpoints avoid adding zero-gradient paths to the source control.
        objective=cox if weight==0 else rank if weight==1 else (1-weight)*cox+weight*rank
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite shared mean-risk ranking objective')
        objective.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),COMMON['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        v=torch.stack([cox.detach(),rank.detach(),objective.detach(),gn.double()]).cpu().numpy()
        row=dict(epoch=epoch,training_mean_view_member_Efron=float(v[0]),training_mean_view_ensemble_pair_loss=float(v[1]),training_objective=float(v[2]),gradient_norm_before_clip=float(v[3]))
        if fixed_epochs is None:
            net.eval()
            with torch.no_grad():logits=parent.b.output(net,VX).reshape(M,nv,8);value=float(val_loss(logits.double().mean(2).T));count,c=C(logits)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite validation Efron during rank objective fitting')
            row.update(validation_mean_view_ensemble_Efron=value,validation_C_index=c,validation_twice_wins=count,validation_twice_pairs=C.denominator)
            metric=value if monitor=='NLL' else count;better=metric<best if monitor=='NLL' else metric>best
            if better:best=metric;best_epoch=epoch;stale=0;state=copy.deepcopy(net.state_dict())
            else:stale+=1
            row.update(best_epoch=best_epoch,stale=stale)
        trace.append(row)
        if progress and (epoch==1 or epoch%50==0):progress(dict(epoch=epoch,best_epoch=best_epoch,stale=stale))
        if fixed_epochs is None and stale>=COMMON['patience']:break
    if fixed_epochs is None:net.load_state_dict(state)
    net.eval();torch.cuda.synchronize()
    audit=dict(at=m.stamp(),condition=condition,config=CONFIG,common=COMMON,seed=seed(f,k),initial_state_SHA=initial,
        best_epoch=best_epoch if fixed_epochs is None else epoch,stopped_epoch=epoch,cap_hit=fixed_epochs is None and epoch==COMMON['max_epochs'],fit_seconds=time.perf_counter()-start,
        training_patients=n,training_events=int(group['train_event'].sum()),input_views=20,batch_rows=20*n,independent_patient_count=n,
        parameters=sum(t.numel() for t in net.parameters()),cuda_train=next(net.parameters()).is_cuda,fixed_epochs_no_validation_forward=fixed_epochs is not None,
        comparable_pairs=len(rank_loss.left),rank_scope='mean20 separateMI;mean8 member risk perMI;all original Harrell pairs',
        best_validation_value=None if fixed_epochs is not None else best,
        no_new_data_transforms=True,probability_calibration='training Breslow after fitting;not a PH likelihood guarantee for pure rank')
    return net,trace,audit
def save(net,path,group,condition,f,k=None):
    torch.save(dict(state={name:v.detach().cpu() for name,v in net.state_dict().items()},features=group['x'].shape[2],config=CONFIG,seed=seed(f,k),common=COMMON,metadata=group['metadata'],condition=condition),path)
def load(path):return parent.load(path)

"""Iteration20 grouped additive risks; same frozen inputs and mean-risk pairs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,copy,time,random,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj
legacy=module('additive20_frozen_rank_backend',ROOT/'experiments/cacox_shared_rank_objective/backend.py')
architecture=module('additive20_verified_grouped_architecture',ROOT/'experiments/cacox_additive_prototype/model.py')
previous,parent=legacy.previous,legacy.parent;io,m,rg=legacy.io,legacy.m,legacy.rg
OUT=ROOT/'results/cacox_grouped_additive/20261031_v1';ALPHA=list(legacy.ALPHA)
# Explicit NAM parameters; historical TabM width64 is never a NAM width.
COMMON={k:v for k,v in legacy.COMMON.items() if k not in ['width','blocks']}
COMMON.update(hidden_layers=2,architecture='original-variable grouped additive ensemble')
CONFIG=dict(model_family='grouped_additive',members=8,learning_rate=.001,condition_widths=[8,16],hidden_layers=2)
CONDITIONS=[dict(condition_id=f'A{width}_{monitor}',rank_weight=1.,monitor=monitor,width=width) for width in [8,16] for monitor in ['NLL','C']]
def read(p):return io.read(p)
def configure():legacy.configure()
def seed(f,k=None):return io.seed(f,k)
def predict_members(net,x):return parent.predict_members(net,x)
def source_dir(f,k,monitor):return legacy.OUT/('inner' if k is not None else 'outer')/f'fold{f:02d}'/(f'inner{k:02d}' if k is not None else 'shared')/'fits'/('P_'+monitor)
def training_spec(group,condition):
    return dict(features=group['x'].shape[2],width=condition['width'],groups=architecture.groups_from_metadata(group['metadata']),
                columns=group['metadata']['columns'],variables=group['metadata']['variables'],members=8,hidden_layers=2,
                training_X_SHA=hashlib.sha256(group['x'].tobytes()).hexdigest(),clinical_inputs_unchanged=True,
                risk='sum original-variable subnet risks;no additional intercept/interaction/spline branch')
def make_from_spec(spec,actual_seed,device='cuda'):
    random.seed(actual_seed);np.random.seed(actual_seed);torch.manual_seed(actual_seed)
    if device=='cuda':torch.cuda.manual_seed_all(actual_seed)
    net=architecture.GroupedAdditiveEnsemble(spec['features'],spec['groups'],spec['width'],COMMON['dropout'],8).to(device);net.architecture_spec=spec;return net
def fit(group,condition,f,k=None,fixed_epochs=None,progress=None):
    if 'width' not in condition:
        assert condition['rank_weight']==1.;return legacy.fit(group,condition,f,k,fixed_epochs,progress)
    assert condition in CONDITIONS;monitor=condition['monitor'];M,n,p=group['x'].shape;assert M==20
    if fixed_epochs is not None and not 1<=fixed_epochs<=COMMON['max_epochs']:raise ValueError('Invalid frozen epoch budget')
    start=time.perf_counter();spec=training_spec(group,condition);net=make_from_spec(spec,seed(f,k));initial=parent.initial_digest(net);X=parent.b.tensor(group['x'].reshape(M*n,p))
    cox_loss=parent.b.Efron(group['train_time'],group['train_event']);rank_loss=legacy.prototype.MeanRiskPairLoss(group['train_time'],group['train_event'],'cuda')
    if fixed_epochs is None:
        nv=group['vx'].shape[1];VX=parent.b.tensor(group['vx'].reshape(M*nv,p));val_loss=parent.b.Efron(group['val_time'],group['val_event']);C=previous.ValidationC(group)
    optimizer=torch.optim.AdamW(net.parameters(),lr=CONFIG['learning_rate'],weight_decay=COMMON['weight_decay'])
    best=float('inf') if monitor=='NLL' else -1;best_epoch=0;state=None;stale=0;trace=[]
    for epoch in range(1,(fixed_epochs or COMMON['max_epochs'])+1):
        net.train();optimizer.zero_grad(set_to_none=True);eta=parent.b.output(net,X).reshape(M,n,8)
        cox=cox_loss(eta.permute(1,0,2).reshape(n,M*8));rank=rank_loss(eta);objective=rank
        if not torch.isfinite(objective):raise FloatingPointError('Nonfinite grouped additive ranking loss')
        objective.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),COMMON['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        v=torch.stack([cox.detach(),rank.detach(),objective.detach(),gn.double()]).cpu().numpy()
        row=dict(epoch=epoch,training_mean_view_member_Efron=float(v[0]),training_mean_view_ensemble_pair_loss=float(v[1]),training_objective=float(v[2]),gradient_norm_before_clip=float(v[3]))
        if fixed_epochs is None:
            net.eval()
            with torch.no_grad():logits=parent.b.output(net,VX).reshape(M,nv,8);value=float(val_loss(logits.double().mean(2).T));count,c=C(logits)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite grouped additive validation Cox loss')
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
        parameters=sum(t.numel() for t in net.parameters()),active_parameters=net.active_parameter_count(),architecture_spec=spec,
        cuda_train=next(net.parameters()).is_cuda,fixed_epochs_no_validation_forward=fixed_epochs is not None,
        comparable_pairs=len(rank_loss.left),rank_scope='mean20 separateMI;mean8 member risk perMI;original Harrell pairs',
        best_validation_value=None if fixed_epochs is not None else best,no_new_data_transforms=True,
        probability_calibration='training Breslow;not PH likelihood guarantee for pure rank')
    return net,trace,audit
def save(net,path,group,condition,f,k=None):
    if 'width' not in condition:return legacy.save(net,path,group,condition,f,k)
    torch.save(dict(state={name:v.detach().cpu() for name,v in net.state_dict().items()},features=group['x'].shape[2],config=CONFIG,seed=seed(f,k),common=COMMON,metadata=group['metadata'],condition=condition,architecture_spec=net.architecture_spec),path)
def load(path):
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if 'architecture_spec' not in obj:return legacy.load(path)
    assert obj['common']==COMMON and obj['config']==CONFIG
    net=make_from_spec(obj['architecture_spec'],obj['seed']);net.load_state_dict(obj['state']);net.eval();return net,obj

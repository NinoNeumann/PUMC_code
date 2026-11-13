"""Frozen-source TabM fine-tuning on the final fixed-I20 combination score."""
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
legacy=module('jointscore_frozen_rankobjective_backend',ROOT/'experiments/cacox_shared_rank_objective/backend.py')
bounded=module('jointscore_verified_bounded_loss',ROOT/'experiments/cacox_smooth_concordance_prototype/losses.py')
parent=legacy.parent;io,m,rg=legacy.io,legacy.m,legacy.rg
OUT=ROOT/'results/cacox_joint_score_finetune/20261031_v1'
COMMON=dict(legacy.COMMON,max_epochs=500);CONFIG=dict(legacy.CONFIG,learning_rate=.0001);ALPHA=list(legacy.ALPHA)
CONDITIONS=[dict(condition_id=warm+'_'+target,warmstart=warm,target=target,monitor='combined_C',temperature=.1 if target=='J_S01' else 1.) for warm in ['NLL','C'] for target in ['N_SP','J_SP','J_S1','J_S01']]
CONTROLS=['control_NLL','control_C'];ORDER=[*CONTROLS,*[c['condition_id'] for c in CONDITIONS]]
def configure():legacy.configure()
def read(p):return io.read(p)
def seed(f,k=None):return io.seed(f,k)
def source_dir(f,k,warm):
    assert warm in ['NLL','C'];return legacy.OUT/('inner' if k is not None else 'outer')/f'fold{f:02d}'/(f'inner{k:02d}' if k is not None else 'shared')/('fits/P_'+warm)
def normalization(g,train_members):
    a=np.asarray(train_members).mean(2);stats=[]
    for j in range(20):
        stats.append(dict(i20_mean=float(g['I20_train'][j].mean()),i20_sd=max(float(g['I20_train'][j].std()),1e-8),kernel_mean=float(a[j].mean()),kernel_sd=max(float(a[j].std()),1e-8)))
    return stats
def combine_np(g,train,val,stats,f):
    treeweight=1-ALPHA[f-1]
    def join(tree,neural):return np.stack([io.source.r.mix_saved(tree[j],neural[j],treeweight,stats[j]) for j in range(20)])
    return join(g['I20_train'],train),join(g['I20_val'],val)
def combine_torch(raw,tree,stats,f):
    values={key:torch.tensor([s[key] for s in stats],dtype=torch.float64,device=raw.device)[:,None] for key in ['i20_mean','i20_sd','kernel_mean','kernel_sd']};a=ALPHA[f-1]
    return (1-a)*(tree-values['i20_mean'])/values['i20_sd']+a*(raw-values['kernel_mean'])/values['kernel_sd']
def predict_members(net,x):return parent.predict_members(net,x)
def source(g,f,k,warm):
    d=source_dir(f,k,warm);assert m.verify_completed(d);obj=torch.load(d/'neural.pt',map_location='cpu',weights_only=False)
    assert obj['metadata']==g['metadata'] and obj['seed']==seed(f,k) and obj['config']==legacy.CONFIG and obj['common']==legacy.COMMON
    with np.load(d/'predictions.npz') as z:
        np.testing.assert_array_equal(z['train_ids'],g['train_ids']);stats=normalization(g,z['train_members'])
    proof=dict(directory=str(d.relative_to(ROOT)),COMPLETE_SHA=m.sha(d/'COMPLETE.json'),neural_SHA=m.sha(d/'neural.pt'),stats_from='only original sourceNN andI20 training scores;fixed through fine-tuning/inference')
    return obj,stats,proof
def fit(g,condition,f,k=None,fixed_epochs=None,progress=None):
    if fixed_epochs is not None and (not isinstance(fixed_epochs,int) or not 0<=fixed_epochs<=COMMON['max_epochs']):raise ValueError('Invalid frozen additional steps')
    start=time.perf_counter();obj,stats,proof=source(g,f,k,condition['warmstart']);M,n,p=g['x'].shape;assert M==20
    net=parent.b.make(p,CONFIG,seed(f,k));net.load_state_dict(obj['state']);net.eval();initial=parent.initial_digest(net)
    net.joint_details=dict(normalization=stats,source_proof=proof,condition=condition)
    X=parent.b.tensor(g['x'].reshape(M*n,p));tree=torch.tensor(g['I20_train'],dtype=torch.float64,device='cuda')
    pair=legacy.prototype.MeanRiskPairLoss(g['train_time'],g['train_event'],'cuda');smooth=bounded.MeanRiskSigmoid(g['train_time'],g['train_event'],condition['temperature'],'cuda')
    def objective(eta):
        if condition['target']=='N_SP':return pair(eta)
        combined=combine_torch(eta.double().mean(2),tree,stats,f).unsqueeze(2)
        return pair(combined) if condition['target']=='J_SP' else smooth(combined)
    if fixed_epochs is None:
        nv=g['vx'].shape[1];VX=parent.b.tensor(g['vx'].reshape(M*nv,p));vtree=torch.tensor(g['I20_val'],dtype=torch.float64,device='cuda');validator=legacy.previous.ValidationC(g);vcox=parent.b.Efron(g['val_time'],g['val_event'])
    optimizer=torch.optim.AdamW(net.parameters(),lr=CONFIG['learning_rate'],weight_decay=COMMON['weight_decay']);best=-1;best_epoch=0;stale=0;state=copy.deepcopy(net.state_dict());trace=[]
    def evaluate(row,initial_row=False):
        nonlocal best,best_epoch,stale,state
        net.eval()
        with torch.no_grad():
            eta=parent.b.output(net,VX).reshape(M,nv,8);score=combine_torch(eta.double().mean(2),vtree,stats,f);count,C=validator(score.unsqueeze(2));_,NC=validator(eta);NLL=float(vcox(score.T))
        if not np.isfinite(NLL):raise FloatingPointError('Nonfinite combined validation Efron')
        if count>best:best=count;best_epoch=row['epoch'];stale=0;state=copy.deepcopy(net.state_dict())
        elif not initial_row:stale+=1
        row.update(validation_combined_C=C,validation_combined_twice_wins=count,validation_twice_pairs=validator.denominator,validation_NN_C=NC,validation_combined_Efron=NLL,best_epoch=best_epoch,stale=stale)
    with torch.no_grad():initial_loss=float(objective(parent.b.output(net,X).reshape(M,n,8)))
    row=dict(epoch=0,initial_eval_not_optimizer_step=True,training_requested_loss=initial_loss,gradient_norm_before_clip=0.)
    if fixed_epochs is None:evaluate(row,True)
    trace.append(row);initial_validation=dict(row) if fixed_epochs is None else None
    for epoch in range(1,(COMMON['max_epochs'] if fixed_epochs is None else fixed_epochs)+1):
        net.train();optimizer.zero_grad(set_to_none=True);loss=objective(parent.b.output(net,X).reshape(M,n,8))
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite joint-score objective')
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),COMMON['gradient_clip'],error_if_nonfinite=True);optimizer.step();v=torch.stack([loss.detach(),gn.double()]).cpu().numpy()
        row=dict(epoch=epoch,initial_eval_not_optimizer_step=False,training_requested_loss=float(v[0]),gradient_norm_before_clip=float(v[1]))
        if fixed_epochs is None:evaluate(row)
        trace.append(row)
        if progress and (epoch==1 or epoch%50==0):progress(dict(epoch=epoch,best_epoch=best_epoch,stale=stale))
        if fixed_epochs is None and stale>=COMMON['patience']:break
    stopped=trace[-1]['epoch']
    if fixed_epochs is None:net.load_state_dict(state)
    net.eval();torch.cuda.synchronize();audit=dict(at=m.stamp(),condition=condition,config=CONFIG,common=COMMON,fold=f,inner_fold=k,seed=seed(f,k),initial_state_SHA=initial,source_proof=proof,normalization=stats,
        best_epoch=best_epoch if fixed_epochs is None else fixed_epochs,stopped_epoch=stopped,optimizer_steps=stopped,cap_hit=fixed_epochs is None and stopped==COMMON['max_epochs'],fit_seconds=time.perf_counter()-start,
        training_patients=n,training_events=int(g['train_event'].sum()),input_views=20,batch_rows=20*n,independent_patient_count=n,parameters=sum(v.numel() for v in net.parameters()),cuda_train=next(net.parameters()).is_cuda,
        fixed_epochs_no_validation_forward=fixed_epochs is not None,initial_validation=initial_validation,best_validation_value=best if fixed_epochs is None else None,source_input_unchanged=True,output_statistics_frozen=True,initial_epoch0_is_valid_fallback=True)
    return net,trace,audit
def save(net,path,g,c,f,k=None):
    torch.save(dict(state={key:value.detach().cpu() for key,value in net.state_dict().items()},features=g['x'].shape[2],config=CONFIG,common=COMMON,seed=seed(f,k),metadata=g['metadata'],condition=c,**{k:v for k,v in net.joint_details.items() if k!='condition'}),path)
def load(path,device='cuda'):
    obj=torch.load(path,map_location='cpu',weights_only=False);assert obj['common']==COMMON and obj['config']==CONFIG
    net=parent.b.make(obj['features'],CONFIG,obj['seed'],device);net.load_state_dict(obj['state']);net.eval();net.joint_details={key:obj[key] for key in ['normalization','source_proof','condition']};return net,obj

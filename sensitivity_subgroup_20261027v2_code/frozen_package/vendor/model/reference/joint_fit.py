"""Verbatim selected functions for validation only; no historical imports.
Source: experiments/cacox_joint_score_finetune/backend.py
"""

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

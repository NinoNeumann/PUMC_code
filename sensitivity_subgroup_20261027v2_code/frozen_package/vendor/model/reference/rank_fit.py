"""Verbatim selected functions for validation only; no historical imports.
Source: experiments/cacox_shared_rank_objective/backend.py
"""

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

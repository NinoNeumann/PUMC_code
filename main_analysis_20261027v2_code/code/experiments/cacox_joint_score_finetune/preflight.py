"""17full20MI internal fine-tunes plus four zero-step source identity checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('preflight_jointscore_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'preflight';PLAN=b.ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION17_PLAN.md'
def independent_C(g,v):
    t=g['val_time'];e=g['val_event'].astype(bool);i,j=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
    wins=(v[:,i]>v[:,j])+.5*(v[:,i]==v[:,j]);return float(wins.mean()),int(2*wins.sum()),2*20*len(i)
def math_check():
    rng=np.random.default_rng(175);x=rng.normal(size=(20,7,8));tree=rng.normal(size=(20,7));stats=b.normalization(dict(I20_train=tree),x);t=np.array([1.,2.,2.,2.,4.,5.,7.]);e=np.array([1,1,1,0,0,1,0],bool)
    pairs=[(i,j) for i in range(7) for j in range(7) if e[i] and (t[i]<t[j] or (t[i]==t[j] and not e[j]))];rows=[]
    for f in [1,2]:
        expected,_=b.combine_np(dict(I20_train=tree,I20_val=tree),x.mean(2),x.mean(2),stats,f)
        for target,tau in [('J_SP',1.),('J_S1',1.),('J_S01',.1)]:
            values=[];gradient=np.zeros_like(x)
            for mi in range(20):
                for i,j in pairs:
                    z=(expected[mi,j]-expected[mi,i])/tau;p=1/(1+math.exp(-z))
                    val=np.logaddexp(0,z) if target=='J_SP' else p;values.append(val)
                    d=(p if target=='J_SP' else p*(1-p))/tau;g=d*b.ALPHA[f-1]/stats[mi]['kernel_sd']/(20*len(pairs)*8);gradient[mi,j]+=g;gradient[mi,i]-=g
            checks={}
            for device in ['cpu','cuda']:
                eta=torch.tensor(x,device=device,requires_grad=True);score=b.combine_torch(eta.mean(2),torch.tensor(tree,device=device),stats,f)
                ce=float(abs(score.detach().cpu().numpy()-expected).max());loss=(b.legacy.prototype.MeanRiskPairLoss(t,e,device) if target=='J_SP' else b.bounded.MeanRiskSigmoid(t,e,tau,device))(score.unsqueeze(2));loss.backward()
                ve=abs(float(loss.detach())-float(np.mean(values)));ge=float(abs(eta.grad.cpu().numpy()-gradient).max());assert ce<2e-14 and ve<2e-14 and ge<2e-14;checks[device]=dict(combined_value_error=ce,objective_value_error=ve,analytic_gradient_error=ge)
            rows.append(dict(fold_alpha=b.ALPHA[f-1],target=target,**checks))
    # Fixed source output scale remains fixed after network risk variance changes.
    g=dict(I20_train=tree,I20_val=tree);a,_=b.combine_np(g,2*x.mean(2),2*x.mean(2),stats,1);ref,_=b.combine_np(g,2*x.mean(2),2*x.mean(2),b.normalization(g,2*x),1);difference=float(abs(a-ref).max());assert difference>.1
    return dict(checks=rows,fixed_vs_reestimated_output_scale_difference=difference,no_input_preprocessing=True)
def one(name,g,c,f,epochs=None):
    d=OUT/name
    if d.exists():raise RuntimeError('Preserve fit:'+str(d))
    d.mkdir();net,trace,audit=b.fit(g,c,f,1,fixed_epochs=epochs,progress=lambda v:(b.rg.ensure_headroom(),b.m.status(OUT/'RUN_STATUS.json',status='running',task=name,**v)))
    a,v=b.predict_members(net,g['x']),b.predict_members(net,g['vx']);ca,cv=b.combine_np(g,a.mean(2),v.mean(2),audit['normalization'],f);C,count,den=independent_C(g,cv)
    if epochs is None:assert count==audit['best_validation_value']
    b.save(net,d/'neural.pt',g,c,f,1);loaded,obj=b.load(d/'neural.pt');np.testing.assert_array_equal(a,b.predict_members(loaded,g['x']));np.testing.assert_array_equal(v,b.predict_members(loaded,g['vx']))
    order=np.arange(19,-1,-1);np.testing.assert_array_equal(v[order],b.predict_members(loaded,g['vx'][order]));batch=np.stack([b.parent.b.predict_members(loaded,g['vx'][j]) for j in range(20)]);batcherr=float(abs(batch-v).max());assert batcherr<2e-6
    tr=pd.DataFrame(trace);assert tr.epoch.tolist()==list(range(audit['stopped_epoch']+1))
    if epochs is None:
        metric=tr.validation_combined_twice_wins.to_numpy();assert int(np.argmax(metric))==audit['best_epoch'];best=-1;last=0
        for index,value in enumerate(metric):
            if value>best:best=value;last=index
            assert int(tr.best_epoch.iloc[index])==last and int(tr.stale.iloc[index])==index-last
        assert audit['stopped_epoch']==500 or audit['stopped_epoch']-audit['best_epoch']==150
    np.savez_compressed(d/'predictions.npz',train_members=a,val_members=v,train_combined=ca,val_combined=cv,train_ids=g['train_ids'],val_ids=g['val_ids']);tr.to_csv(d/'training_trace.csv',index=False);b.m.write_json(d/'audit.json',dict(**audit,metadata=g['metadata']));b.m.mark_complete(d,['neural.pt','predictions.npz','training_trace.csv','audit.json']);b.rg.record(OUT/'monitor',True)
    row=dict(name=name,fold=f,condition=c,best_additional_epoch=audit['best_epoch'],stopped_epoch=audit['stopped_epoch'],fit_seconds=audit['fit_seconds'],cap_hit=audit['cap_hit'],initial_state_SHA=audit['initial_state_SHA'],C=C,source_proof=audit['source_proof'],saved_reordered_prediction_error=0.,singleMI_batch_error=batcherr,no_new_outer=True)
    print(json.dumps(row),flush=True);return row,tr,a,v
def main():
    b.configure();start=time.perf_counter()
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('Already complete')
    for root in [b.legacy.OUT,b.ROOT/'results/cacox_loss_aggregation/20261031_v1']:
        for d in [root,root/'verification',root/'interpretation',root/'visual_qa']:assert b.m.verify_completed(d)
        for p,h in b.read(root/'manifest.json')['source'].items():assert b.m.sha(b.ROOT/p)==h
    assert b.m.verify_completed(b.ROOT/'results/cacox_smooth_concordance_prototype/20261031_v1');OUT.mkdir(parents=True,exist_ok=True);numeric=math_check();planSHA=b.m.sha(PLAN);rows=[];zero=[];result={}
    for f in [1,4]:
        g=b.io.load_group(f,1);assert not set(g['train_ids'])&set(g['val_ids'])
        for warm in ['NLL','C']:
            c=next(c for c in b.CONDITIONS if c['warmstart']==warm and c['target']=='N_SP');row,tr,a,v=one(f'f{f}_{warm}_zero',g,c,f,0)
            source=b.source_dir(f,1,warm)
            with np.load(source/'predictions.npz') as z:np.testing.assert_array_equal(a,z['train_members']);np.testing.assert_array_equal(v,z['val_members'])
            ca,cv=b.combine_np(g,a.mean(2),v.mean(2),b.normalization(g,a),f)
            for j in range(20):
                aa,vv,_=b.io.source.r.mix(g['I20_train'][j],g['I20_val'][j],a[j].mean(1),v[j].mean(1),1-b.ALPHA[f-1]);np.testing.assert_array_equal(aa,ca[j]);np.testing.assert_array_equal(vv,cv[j])
            zero.append(dict(**row,source_member_and_combined_error=0.,optimizer_steps=0))
        for c in b.CONDITIONS:
            row,tr,a,v=one('f'+str(f)+'_'+c['condition_id'],g,c,f);rows.append(row);result[f,c['condition_id']]=(tr,a,v)
        for warm in ['NLL','C']:
            r=[r for r in rows if r['fold']==f and r['condition']['warmstart']==warm];assert len({r['initial_state_SHA'] for r in r})==1
            initial=[result[f,c['condition_id']][0].iloc[0].validation_combined_C for c in b.CONDITIONS if c['warmstart']==warm];assert len(set(initial))==1
    g=b.io.load_group(1,1);c=next(c for c in b.CONDITIONS if c['condition_id']=='NLL_J_S1');row,tr,a,v=one('f1_NLL_J_S1_repeat',g,c,1);rows.append(row);t,aa,vv=result[1,'NLL_J_S1'];np.testing.assert_array_equal(tr.to_numpy(),t.to_numpy());np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(v,vv)
    assert len(rows)==17 and len(zero)==4 and b.m.sha(PLAN)==planSHA
    proof=dict(at=b.m.stamp(),status='passed',full20MI_additional_fit_calls=17,zero_step_identity_calls=4,new_outer_evaluations=0,checks=rows,zero_step_checks=zero,math=numeric,new_repeat_trace_risk_error=0.,all_warmstart_and_epoch0_checks_passed=True,
        source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',Path(__file__)]},prototype_SHA=b.m.sha(b.ROOT/'experiments/cacox_smooth_concordance_prototype/losses.py'),registered_plan_SHA=planSHA,wall_seconds=time.perf_counter()-start)
    b.m.write_json(OUT/'PREFLIGHT.json',proof);b.m.mark_complete(OUT,['PREFLIGHT.json',*[r['name']+'/COMPLETE.json' for r in rows+zero]]);b.m.status(OUT/'RUN_STATUS.json',status='complete');print(json.dumps({k:v for k,v in proof.items() if k not in ['checks','zero_step_checks']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

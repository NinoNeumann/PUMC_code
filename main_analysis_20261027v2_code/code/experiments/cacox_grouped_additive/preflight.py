"""Full20MI internal-only CUDA checks; never evaluates new outer predictions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('preflight_groupedadditive_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'preflight'
PLAN=b.ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION20_PLAN.md'

def numeric_check():
    proof=b.ROOT/'results/cacox_additive_prototype/20261031_v1';assert b.m.verify_completed(proof);record=b.read(proof/'MATH_CHECK.json')
    for name,h in record['source'].items():assert b.m.sha(b.ROOT/'experiments/cacox_additive_prototype'/name)==h
    return record

def run_one(name,f,k,c):
    target=OUT/name
    if target.exists():raise RuntimeError('Preserve existing preflight artifacts:'+str(target))
    target.mkdir(parents=True);b.rg.ensure_headroom();g=b.io.load_group(f,k);net,trace,audit=b.fit(g,c,f,k)
    a,v=b.predict_members(net,g['x']),b.predict_members(net,g['vx']);b.save(net,target/'neural.pt',g,c,f,k)
    replay,obj=b.load(target/'neural.pt');np.testing.assert_array_equal(a,b.predict_members(replay,g['x']));np.testing.assert_array_equal(v,b.predict_members(replay,g['vx']))
    order=np.arange(19,-1,-1);np.testing.assert_array_equal(b.predict_members(replay,g['vx'][order]),v[order])
    separated=np.stack([b.parent.b.predict_members(replay,g['vx'][j]) for j in range(20)]);batch_error=float(abs(separated-v).max());assert batch_error<2e-6
    means=v.mean(2);t=g['val_time'];e=g['val_event'].astype(bool);i,j=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
    C=float(np.mean((means[:,i]>means[:,j])+.5*(means[:,i]==means[:,j])))
    if c['monitor']=='C':assert abs(C-audit['best_validation_value']/(2*20*len(i)))<1e-12
    if 'width' in c:
        assert obj['architecture_spec']==b.training_spec(g,c)
        assert audit['architecture_spec']['training_X_SHA']==b.training_spec(g,c)['training_X_SHA']
        np.testing.assert_array_equal(g['x'],b.io.load_group(f,k)['x'])
    assert next(net.parameters()).is_cuda and not (set(g['train_ids'])&set(g['val_ids']))
    for p,h in g['metadata']['input_complete_SHA'].items():assert b.m.sha(b.ROOT/p/'COMPLETE.json')==h
    pd.DataFrame(trace).to_csv(target/'training_trace.csv',index=False);b.m.write_json(target/'audit.json',dict(**audit,metadata=g['metadata']));np.savez_compressed(target/'predictions.npz',train_members=a,val_members=v)
    b.m.mark_complete(target,['neural.pt','training_trace.csv','audit.json','predictions.npz']);b.rg.record(OUT/'monitor',True)
    row=dict(task=name,fold=f,inner_fold=k,condition=c,fit_seconds=audit['fit_seconds'],best_epoch=audit['best_epoch'],stopped_epoch=audit['stopped_epoch'],cap_hit=audit['cap_hit'],initial_state_SHA=audit['initial_state_SHA'],saved_risk_error=0.,MI_batch_error=batch_error,MI_axis_reorder_error=0.,internal_C=C,outer_evaluated=False)
    print(json.dumps(row),flush=True);return row,pd.DataFrame(trace),a,v

def main():
    b.configure();start=time.perf_counter();assert PLAN.exists()
    for root in [b.io.OUT,b.previous.OUT,b.legacy.OUT]:
        for sub in ['', 'interpretation','visual_qa']:
            assert b.m.verify_completed(root/sub)
    for d in [b.legacy.OUT]:
        assert b.m.verify_completed(d)
    prototype=b.ROOT/'results/cacox_rank_objective_prototype/20261031_v1';assert b.m.verify_completed(prototype)
    record=b.read(prototype/'MATH_CHECK.json')
    for name,h in record['source'].items():assert b.m.sha(b.ROOT/'experiments/cacox_rank_objective_prototype'/name)==h
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('Preflight already complete')
    OUT.mkdir(parents=True,exist_ok=True);plan_sha=b.m.sha(PLAN);numeric=numeric_check();rows=[];results={};control_errors=[]
    for f in [1,4]:
        for monitor in ['NLL','C']:
            c=dict(condition_id='control_'+monitor,rank_weight=1.,monitor=monitor);row,trace,a,v=run_one(f'f{f}_control_{monitor}',f,1,c);rows.append(row)
            source=b.source_dir(f,1,monitor);old=pd.read_csv(source/'training_trace.csv',float_precision='round_trip')
            common=['epoch','training_mean_view_member_Efron','training_objective','gradient_norm_before_clip','validation_mean_view_ensemble_Efron','best_epoch','stale']
            np.testing.assert_array_equal(trace[common].to_numpy(),old[common].to_numpy())
            with np.load(source/'predictions.npz') as z:np.testing.assert_array_equal(a,z['train_members']);np.testing.assert_array_equal(v,z['val_members'])
            control_errors.append(dict(fold=f,monitor=monitor,trace_and_saved_risk_error=0.))
        for c in b.CONDITIONS:
            row,t,a,v=run_one('f'+str(f)+'_'+c['condition_id'],f,1,c);rows.append(row);results[f,c['condition_id']]=(t,a,v)
        for objective in ['A8','A16']:
            a=results[f,objective+'_NLL'][0];v=results[f,objective+'_C'][0];n=min(len(a),len(v));cols=['training_mean_view_member_Efron','training_mean_view_ensemble_pair_loss','training_objective','gradient_norm_before_clip','validation_mean_view_ensemble_Efron','validation_twice_wins']
            np.testing.assert_array_equal(a.iloc[:n][cols].to_numpy(),v.iloc[:n][cols].to_numpy())
    for cid in ['A8_C','A16_NLL']:
        c=next(c for c in b.CONDITIONS if c['condition_id']==cid);row,t,a,v=run_one('f1_'+cid+'_repeat',1,1,c);rows.append(row);old,aa,vv=results[1,cid]
        np.testing.assert_array_equal(t.to_numpy(),old.to_numpy());np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(v,vv)
    for f in [1,4]:
        assert results[f,'A8_NLL'][0].iloc[0].epoch==1
        aa=b.read(OUT/f'f{f}_A8_NLL/audit.json');bb=b.read(OUT/f'f{f}_A16_NLL/audit.json')
        assert aa['architecture_spec']['groups']==bb['architecture_spec']['groups'] and aa['architecture_spec']['columns']==bb['architecture_spec']['columns']
        assert aa['parameters']<bb['parameters'] and aa['active_parameters']<bb['active_parameters']
    assert len(rows)==14 and b.m.sha(PLAN)==plan_sha
    summary=dict(at=b.m.stamp(),status='passed',new_preflight_NN=14,all_full20MI=True,no_new_outer_predictions=True,checks=rows,CUDA_math=numeric,control_errors=control_errors,
        sameobjective_monitors_common_training_prefix_error=0.,newobjective_repeat_trace_risk_error=0.,source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',Path(__file__)]},
        prototype_pair_SHA=b.m.sha(b.ROOT/'experiments/cacox_rank_objective_prototype/pair_loss.py'),registered_plan_SHA=plan_sha,prototype_architecture_SHA=b.m.sha(b.ROOT/'experiments/cacox_additive_prototype/model.py'),prototype_math_complete_SHA=b.m.sha(b.ROOT/'results/cacox_additive_prototype/20261031_v1/COMPLETE.json'),wall_seconds=time.perf_counter()-start)
    b.m.write_json(OUT/'PREFLIGHT.json',summary);b.m.mark_complete(OUT,['PREFLIGHT.json',*[row['task']+'/COMPLETE.json' for row in rows]])
    print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()

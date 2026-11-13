"""Actual CUDA feasibility only after the preceding study is fully reviewed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tabm_backend as b
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cacox_tabm_cox/20261031_v1'
spec=importlib.util.spec_from_file_location('tabm_frozen_patient_reference',ROOT/'experiments/cacox_pair_patient_sampling/runner.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
m,rg=source.m,source.rg


def seed(f,j,k=None):return 910000+f*1000+j*10+(9 if k is None else k)


def main():
    if not m.verify_completed(source.OUT) or not m.verify_completed(source.OUT/'interpretation'):
        raise RuntimeError('Finish and independently review patient-level study first')
    assert source.signature()==source.read(source.OUT/'manifest.json')
    assert not any(source.read(source.OUT/'summary.json')['minimum_target_met'].values())
    d=OUT/'preflight'
    if (d/'COMPLETE.json').exists():raise RuntimeError('Preflight already recorded')
    d.mkdir(parents=True,exist_ok=True);b.configure();records=[];files=[]
    for f,k_members in [(1,1),(1,8),(4,8)]:
        rg.ensure_headroom();u=source.unit(f,4,1);c=next(v for v in b.CONFIGS if v['members']==k_members and v['learning_rate']==.001)
        stem=f'fold{f:02d}_{c["config_id"]}';target=d/stem;target.mkdir(exist_ok=True)
        def progress(v):
            rg.record(d,True);m.status(d/'RUN_STATUS.json',status='running',worker_pid=os.getpid(),fold=f,config=c,**v)
        net,trace,audit=b.fit(u['x'],u['tr'].Survival_Days,u['tr'].Death,c,seed(f,4,1),u['vx'],u['va'].Survival_Days,u['va'].Death,progress=progress)
        train=b.predict_members(net,u['x']);val=b.predict_members(net,u['vx'])
        meta=source.source_metadata(u,f,4,1);b.save(net,target/'neural.pt',u['x'].shape[1],c,seed(f,4,1),meta)
        loaded,obj=b.load(target/'neural.pt');np.testing.assert_array_equal(val,b.predict_members(loaded,u['vx']))
        order=np.random.RandomState(11).permutation(len(u['vx']));ordered=b.predict_members(loaded,u['vx'][order]);single=np.concatenate([b.predict_members(loaded,row[None]) for row in u['vx']])
        err=max(float(abs(ordered-val[order]).max()),float(abs(single-val).max()));assert err<2e-6
        C=m.c_evaluator(u['va']);assert C(val.mean(1))==C(single.mean(1))==C(ordered.mean(1)[np.argsort(order)])
        repeat_error=0.;trace_error=0.
        if f==1 and k_members==8:
            again,t2,a2=b.fit(u['x'],u['tr'].Survival_Days,u['tr'].Death,c,seed(f,4,1),u['vx'],u['va'].Survival_Days,u['va'].Death,progress=progress)
            repeat_error=float(abs(b.predict_members(again,u['vx'])-val).max());assert repeat_error==0 and a2['best_epoch']==audit['best_epoch']
            np.testing.assert_array_equal(pd.DataFrame(trace),pd.DataFrame(t2));del again
        assert not (set(meta['reference_ids'])&set(meta['query_ids']))
        np.savez_compressed(target/'predictions.npz',train_members=train,val_members=val,train_ids=u['tr'].patient_id,val_ids=u['va'].patient_id)
        pd.DataFrame(trace).to_csv(target/'training_trace.csv',index=False)
        audit.update(saved_replay_error=0.,batch_permutation_error=err,repeat_error=repeat_error,trace_error=trace_error,input_cache_SHA=m.sha(source.r.cache_path(f,4,1)/'COMPLETE.json'))
        m.write_json(target/'audit.json',audit);m.mark_complete(target,['neural.pt','predictions.npz','training_trace.csv','audit.json'])
        records.append(dict(fold=f,imputation=4,inner_fold=1,**audit));files.append(stem+'/COMPLETE.json');del net,loaded
        torch.cuda.empty_cache()
    m.write_json(d/'PREFLIGHT.json',dict(at=m.stamp(),status='passed',checks=records,units='predetermined internal units only',
        previous_study_root_SHA=m.sha(source.OUT/'COMPLETE.json'),previous_review_SHA=m.sha(source.OUT/'interpretation/COMPLETE.json'),
        source_files={p.name:m.sha(p) for p in Path(__file__).parent.glob('*.py')},no_outer_TabM_results=True,no_data_change=True))
    m.mark_complete(d,[*files,'PREFLIGHT.json']);m.status(d/'RUN_STATUS.json',status='complete')
    print(json.dumps(dict(status='passed',records=[{key:a[key] for key in ['fold','config','best_epoch','stopped_epoch','fit_seconds','parameters','batch_permutation_error']} for a in records]),ensure_ascii=False))

if __name__=='__main__':main()

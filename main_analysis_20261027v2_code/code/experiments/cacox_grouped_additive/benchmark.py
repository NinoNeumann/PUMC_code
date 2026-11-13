"""Six full task types perworker;compare normalized throughput,not rawwall."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,argparse,sys,subprocess,time,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('additive20_balanced_benchmark_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'benchmark';NAMES=[c['condition_id'] for c in b.CONDITIONS];RESERVE_PER_WORKER_GiB=2.
CONDITIONS={c['condition_id']:(1 if i%2==0 else 4,1,c) for i,c in enumerate(b.CONDITIONS)}
def tasks(workers):return [name for name in NAMES for _ in range(workers)]
def worker(workers,index):
    b.configure();root=OUT/f'w{workers}';d=root/f'worker{index}';d.mkdir(parents=True,exist_ok=True);g=b.io.load_group(1,1);net,*_=b.fit(g,CONDITIONS[NAMES[0]][2],1,1,fixed_epochs=3);del net
    b.m.write_json(d/'READY.json',dict(worker_pid=os.getpid()));deadline=time.monotonic()+180
    while not (root/'GO').exists():
        if time.monotonic()>deadline:raise TimeoutError('Fast balanced barrier')
        b.rg.ensure_headroom();time.sleep(.05)
    for task in range(len(tasks(workers)))[index::workers]:
        name=tasks(workers)[task];f,k,c=CONDITIONS[name];g=b.io.load_group(f,k)
        def callback(v):b.rg.ensure_headroom();b.m.status(d/'RUN_STATUS.json',status='running',worker_pid=os.getpid(),task=task,task_type=name,condition=c,**v)
        net,trace,audit=b.fit(g,c,f,k,progress=callback);a,v=b.predict_members(net,g['x']),b.predict_members(net,g['vx']);C=b.m.c_evaluator(g['va']);combined=np.stack([b.io.source.r.mix(g['I20_train'][j],g['I20_val'][j],a[j].mean(1),v[j].mean(1),1-b.ALPHA[f-1])[1] for j in range(20)])
        np.savez_compressed(root/f'task{task}.npz',train=a,val=v,trace=pd.DataFrame(trace).to_numpy(float),C=np.array([C(z) for z in combined]));b.m.write_json(root/f'task{task}_audit.json',dict(**audit,task_type=name));del net
    b.m.write_json(d/'DONE.json',dict(worker_pid=os.getpid()));b.m.status(d/'RUN_STATUS.json',status='complete',worker_pid=os.getpid())
def main():
    b.configure();assert b.m.verify_completed(b.OUT/'preflight')
    for name,h in b.read(b.OUT/'preflight/PREFLIGHT.json')['source_files'].items():assert b.m.sha(HERE/name)==h
    plan=b.ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION20_PLAN.md';assert b.m.sha(plan)==b.read(b.OUT/'preflight/PREFLIGHT.json')['registered_plan_SHA']
    for name,h in b.read(b.OUT/'preflight/PREFLIGHT.json')['CUDA_math']['source'].items():assert b.m.sha(b.ROOT/'experiments/cacox_additive_prototype'/name)==h
    if OUT.exists():raise RuntimeError('Preserve previous benchmark artifacts')
    OUT.mkdir();rows=[]
    for workers in [1,2]:
        root=OUT/f'w{workers}';root.mkdir();children=[];start=None;assigned=[{name:[tasks(workers)[t] for t in range(4*workers)[i::workers]].count(name) for name in NAMES} for i in range(workers)]
        assert all(v==dict.fromkeys(NAMES,1) for v in assigned)
        before=b.rg.record(OUT,True);projected=before['physical_used_GiB']+workers*RESERVE_PER_WORKER_GiB
        if projected>b.rg.LIMIT/b.rg.GiB:
            row=dict(workers=workers,status='resource_not_launched',reason='Conservative2GiB perworker reserve exceeds unchanged physicalguard',projected_physical_GiB=projected,actual_before=before,full_shared_NN_fits=0,warmups=0)
            b.m.write_json(root/'RESOURCE_NOT_LAUNCHED.json',row);rows.append(row);print(json.dumps(row),flush=True)
            if workers==1:raise MemoryError('Insufficient headroom for serial benchmark;preserve artifacts')
            continue
        try:
            for i in range(workers):
                b.rg.ensure_headroom();d=root/f'worker{i}';d.mkdir()
                with (d/'out.log').open('w') as oo,(d/'err.log').open('w') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'worker','--workers',str(workers),'--index',str(i)],cwd=b.ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
            deadline=time.monotonic()+180
            while not all((root/f'worker{i}/READY.json').exists() for i in range(workers)):
                b.rg.record(OUT,True)
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Fast benchmark initialization failure')
                if time.monotonic()>deadline:raise TimeoutError('Fast benchmark warmup')
                time.sleep(.2)
            start=time.perf_counter();(root/'GO').write_text(b.m.stamp())
            while any(c.poll() is None for c in children):
                resource=b.rg.record(OUT,True);b.m.status(OUT/'RUN_STATUS.json',status='running',controller_pid=os.getpid(),workers=workers,tasks=4*workers,wrapper_pids=[c.pid for c in children],resource=resource)
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Fast balanced worker failure')
                time.sleep(.2)
            assert all(c.returncode==0 for c in children);wall=time.perf_counter()-start;error=0.
            for task,name in enumerate(tasks(workers)):
                with np.load(OUT/f'w1/task{NAMES.index(name)}.npz') as a,np.load(root/f'task{task}.npz') as z:
                    for key in a.files:assert a[key].shape==z[key].shape;error=max(error,float(abs(a[key]-z[key]).max()))
            assert error==0;row=dict(workers=workers,status='passed',wall_seconds=wall,full_shared_NN_fits=4*workers,seconds_per_fit=wall/(4*workers),fits_per_second=4*workers/wall,
                balanced_types_per_worker=assigned,risk_trace_C_difference=error,warmups=workers)
        except BaseException as e:
            b.rg.stop_owned(children);row=dict(workers=workers,status='rejected',wall_seconds=None if start is None else time.perf_counter()-start,error=repr(e),traceback=traceback.format_exc(),full_shared_NN_fits=len(list(root.glob('task*.npz'))),warmups=workers);b.m.write_json(root/'REJECTED.json',row)
            if workers==1:raise
        rows.append(row);print(json.dumps(row),flush=True)
    passed=[r for r in rows if r['status']=='passed'];chosen=max(passed,key=lambda r:(r['fits_per_second'],-r['workers']))['workers']
    summary=dict(at=b.m.stamp(),timings=rows,selected_workers=chosen,selection_unit='highest passing fullfits_per_second,not absolute wall acrossunequal taskcounts',all_selected_checks_passed=True,
        full_shared_fits_in_passed_cases=sum(r['full_shared_NN_fits'] for r in passed),warmup3epoch_fits=sum(r['warmups'] for r in rows),
        all_tasks_full20MI=True,all_types_once_perworker=True,balanced_types=True,resource_reservation_GiB_perworker=RESERVE_PER_WORKER_GiB,source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',Path(__file__)]},
        preflight_complete_SHA=b.m.sha(b.OUT/'preflight/COMPLETE.json'),registered_plan_SHA=b.m.sha(plan),prototype_architecture_SHA=b.m.sha(b.ROOT/'experiments/cacox_additive_prototype/model.py'))
    b.m.write_json(OUT/'SUMMARY.json',summary);b.m.mark_complete(OUT,['SUMMARY.json',*[f'w{r["workers"]}/task{t}.npz' for r in passed for t in range(r['full_shared_NN_fits'])]])
    b.m.status(OUT/'RUN_STATUS.json',status='complete',controller_pid=os.getpid(),selected_workers=chosen)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','worker']);parser.add_argument('--workers',type=int);parser.add_argument('--index',type=int);a=parser.parse_args()
    if a.command=='worker':worker(a.workers,a.index)
    else:main()

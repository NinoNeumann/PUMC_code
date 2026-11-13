"""All four objective/monitor types, balanced24 real tasks for1/2/3 workers."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,argparse,sys,subprocess,time,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('benchmark_shared_rankobjective_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'benchmark';TASKS=['A','B','C','D','D','C','B','A']*3
CONDITIONS={name:(f,1,next(c for c in b.CONDITIONS if c['condition_id']==cid)) for name,f,cid in [('A',1,'H_NLL'),('B',4,'H_C'),('C',1,'P_NLL'),('D',4,'P_C')]}
def worker(workers,index):
    b.configure();root=OUT/f'w{workers}';d=root/f'worker{index}';d.mkdir(parents=True,exist_ok=True)
    g=b.io.load_group(1,1);net,*_=b.fit(g,CONDITIONS['A'][2],1,1,fixed_epochs=3);del net
    b.m.write_json(d/'READY.json',dict(worker_pid=os.getpid()));deadline=time.monotonic()+180
    while not (root/'GO').exists():
        if time.monotonic()>deadline:raise TimeoutError('Rank objective benchmark barrier')
        b.rg.ensure_headroom();time.sleep(.05)
    for task in list(range(len(TASKS)))[index::workers]:
        f,k,c=CONDITIONS[TASKS[task]];g=b.io.load_group(f,k)
        def progress(v):b.rg.ensure_headroom();b.m.status(d/'RUN_STATUS.json',status='running',worker_pid=os.getpid(),task=task,condition=c,**v)
        net,trace,audit=b.fit(g,c,f,k,progress=progress);a,v=b.predict_members(net,g['x']),b.predict_members(net,g['vx']);C=b.m.c_evaluator(g['va'])
        np.savez_compressed(root/f'task{task}.npz',train=a,val=v,trace=pd.DataFrame(trace).to_numpy(float),C=np.array([C(z.mean(1)) for z in v]));b.m.write_json(root/f'task{task}_audit.json',audit);del net
    b.m.write_json(d/'DONE.json',dict(worker_pid=os.getpid()));b.m.status(d/'RUN_STATUS.json',status='complete',worker_pid=os.getpid())
def main():
    b.configure();assert b.m.verify_completed(b.OUT/'preflight')
    for name,h in b.read(b.OUT/'preflight/PREFLIGHT.json')['source_files'].items():assert b.m.sha(HERE/name)==h
    if (OUT/'SUMMARY.json').exists():raise RuntimeError('Benchmark already recorded')
    OUT.mkdir(exist_ok=True);rows=[]
    for workers in [1,2,3]:
        root=OUT/f'w{workers}';children=[];start=None;assigned=[{name:[TASKS[t] for t in range(len(TASKS))[i::workers]].count(name) for name in CONDITIONS} for i in range(workers)]
        assert all(a==assigned[0] for a in assigned)
        try:
            for i in range(workers):
                b.rg.ensure_headroom();d=root/f'worker{i}';d.mkdir(parents=True,exist_ok=True)
                with (d/'out.log').open('w') as oo,(d/'err.log').open('w') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'worker','--workers',str(workers),'--index',str(i)],cwd=b.ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
            deadline=time.monotonic()+180
            while not all((root/f'worker{i}/READY.json').exists() for i in range(workers)):
                b.rg.record(OUT,True)
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Rank objective benchmark initialization failed')
                if time.monotonic()>deadline:raise TimeoutError('Rank objective warmup')
                time.sleep(.2)
            start=time.perf_counter();(root/'GO').write_text(b.m.stamp())
            while any(c.poll() is None for c in children):
                resource=b.rg.record(OUT,True);b.m.status(OUT/'RUN_STATUS.json',status='running',controller_pid=os.getpid(),workers=workers,wrapper_pids=[c.pid for c in children],resource=resource)
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Rank objective benchmark worker failure')
                time.sleep(.2)
            assert all(c.returncode==0 for c in children);wall=time.perf_counter()-start;error=0.
            for task in range(len(TASKS)):
                with np.load(OUT/f'w1/task{task}.npz') as a,np.load(root/f'task{task}.npz') as z:
                    for key in a.files:assert a[key].shape==z[key].shape;error=max(error,float(abs(a[key]-z[key]).max()))
                with np.load(root/f'task{TASKS.index(TASKS[task])}.npz') as a,np.load(root/f'task{task}.npz') as z:
                    for key in a.files:np.testing.assert_array_equal(a[key],z[key])
            assert error==0;row=dict(workers=workers,status='passed',wall_seconds=wall,full_shared_NN_fits=len(TASKS),balanced_types_per_worker=assigned,risk_trace_C_difference=error)
        except BaseException as e:
            b.rg.stop_owned(children);row=dict(workers=workers,status='rejected',wall_seconds=None if start is None else time.perf_counter()-start,error=repr(e),traceback=traceback.format_exc());b.m.write_json(root/'REJECTED.json',row)
            if workers==1:raise
        rows.append(row);print(json.dumps(row),flush=True)
    passed=[r for r in rows if r['status']=='passed'];chosen=min(passed,key=lambda r:r['wall_seconds'])['workers']
    b.m.write_json(OUT/'SUMMARY.json',dict(at=b.m.stamp(),timings=rows,selected_workers=chosen,all_selected_checks_passed=True,full_shared_fits_in_passed_cases=len(TASKS)*len(passed),warmup3epoch_fits=sum(r['workers'] for r in rows),
        all_tasks_full20MI=True,balanced_types=True,task_types=CONDITIONS,source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',Path(__file__)]},preflight_complete_SHA=b.m.sha(b.OUT/'preflight/COMPLETE.json')))
    b.m.mark_complete(OUT,['SUMMARY.json',*[f'w{r["workers"]}/task{t}.npz' for r in passed for t in range(len(TASKS))]]);b.m.status(OUT/'RUN_STATUS.json',status='complete',controller_pid=os.getpid(),selected_workers=chosen)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','worker']);parser.add_argument('--workers',type=int);parser.add_argument('--index',type=int);a=parser.parse_args()
    if a.command=='worker':worker(a.workers,a.index)
    else:main()

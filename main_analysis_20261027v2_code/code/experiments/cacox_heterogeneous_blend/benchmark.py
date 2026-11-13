"""Full cached-model group evaluations; normalized throughput, zero fits."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import importlib.util,argparse,subprocess,sys,time,json,traceback
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('heterogeneous21_benchmark_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'benchmark';NAMES=['fold01','fold04'];FOLDS={'fold01':1,'fold04':4};RESERVE=2.
def tasks(workers):return [name for name in NAMES for _ in range(workers)]
def evaluate(f):
    g=b.io.load_group(f,1);pred,proof=b.sources(g,f,1);result,stats=b.combine(pred);grid=b.grid(g,result,f,1)
    arrays={cid+suffix:result[cid][i].cpu().numpy() for cid in b.ORDER for i,suffix in enumerate(['_train','_val'])}
    arrays['C']=np.array([[row['train_C'],row['C_index']] for row in grid]);return arrays
def worker(workers,index):
    b.configure();root=OUT/f'w{workers}';d=root/f'worker{index}';d.mkdir(parents=True,exist_ok=True)
    evaluate(1);b.m.write_json(d/'READY.json',dict(worker_pid=os.getpid()));deadline=time.monotonic()+180
    while not (root/'GO').exists():
        if time.monotonic()>deadline:raise TimeoutError('Cached mixture benchmark barrier')
        b.rg.ensure_headroom();time.sleep(.05)
    for task in range(len(tasks(workers)))[index::workers]:
        name=tasks(workers)[task];b.rg.ensure_headroom();b.m.status(d/'RUN_STATUS.json',status='running',worker_pid=os.getpid(),task=task,group=name)
        arrays=evaluate(FOLDS[name]);np.savez_compressed(root/f'task{task}.npz',**arrays)
    b.m.status(d/'RUN_STATUS.json',status='complete',worker_pid=os.getpid());b.m.write_json(d/'DONE.json',dict(worker_pid=os.getpid()))
def main():
    b.configure();assert b.m.verify_completed(b.OUT/'preflight')
    pre=b.read(b.OUT/'preflight/PREFLIGHT.json')
    for name,h in pre['source_files'].items():assert b.m.sha(HERE/name)==h
    plan=b.ROOT/'results/cacox_autonomous_optimization/20261031_v1/ITERATION21_PLAN.md';assert b.m.sha(plan)==pre['registered_plan_SHA']
    if OUT.exists():raise RuntimeError('Preserve benchmark')
    OUT.mkdir();rows=[]
    for workers in [1,2]:
        root=OUT/f'w{workers}';root.mkdir();children=[];start=None
        before=b.rg.record(OUT,True);projected=before['physical_used_GiB']+workers*RESERVE
        if projected>b.rg.LIMIT/b.rg.GiB:
            row=dict(workers=workers,status='resource_not_launched',reason='Conservative2GiB perworker reserve exceeds unchanged guard',projected_physical_GiB=projected,group_evaluations=0,warmup_group_evaluations=0);rows.append(row);b.m.write_json(root/'RESOURCE_NOT_LAUNCHED.json',row)
            if workers==1:raise MemoryError(row['reason'])
            continue
        try:
            for i in range(workers):
                b.rg.ensure_headroom();d=root/f'worker{i}';d.mkdir()
                with (d/'out.log').open('w') as oo,(d/'err.log').open('w') as ee:children.append(subprocess.Popen([sys.executable,str(__file__),'worker','--workers',str(workers),'--index',str(i)],cwd=b.ROOT,stdout=oo,stderr=ee,creationflags=subprocess.CREATE_NO_WINDOW))
            deadline=time.monotonic()+180
            while not all((root/f'worker{i}/READY.json').exists() for i in range(workers)):
                b.rg.record(OUT,True)
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Cached mixture warmup failed')
                if time.monotonic()>deadline:raise TimeoutError('Cached mixture warmup timeout')
                time.sleep(.2)
            start=time.perf_counter();(root/'GO').write_text(b.m.stamp())
            while any(c.poll() is None for c in children):
                resource=b.rg.record(OUT,True);b.m.status(OUT/'RUN_STATUS.json',status='running',controller_pid=os.getpid(),workers=workers,wrapper_pids=[c.pid for c in children],resource=resource)
                if any(c.poll() not in (None,0) for c in children):raise RuntimeError('Cached mixture worker failure')
                time.sleep(.2)
            assert all(c.returncode==0 for c in children);wall=time.perf_counter()-start;error=0.
            for task,name in enumerate(tasks(workers)):
                with np.load(OUT/f'w1/task{NAMES.index(name)}.npz') as a,np.load(root/f'task{task}.npz') as z:
                    assert a.files==z.files
                    for key in a.files:np.testing.assert_array_equal(a[key],z[key])
            row=dict(workers=workers,status='passed',wall_seconds=wall,group_evaluations=2*workers,seconds_per_group=wall/(2*workers),groups_per_second=2*workers/wall,warmup_group_evaluations=workers,all_types_once_perworker=True,risks_and_C_difference=error,new_model_fits=0)
        except BaseException as e:
            b.rg.stop_owned(children);row=dict(workers=workers,status='rejected',wall_seconds=None if start is None else time.perf_counter()-start,error=repr(e),traceback=traceback.format_exc(),group_evaluations=len(list(root.glob('task*.npz'))),warmup_group_evaluations=workers);b.m.write_json(root/'REJECTED.json',row)
            if workers==1:raise
        rows.append(row);print(json.dumps(row),flush=True)
    passed=[r for r in rows if r['status']=='passed'];selected=max(passed,key=lambda r:(r['groups_per_second'],-r['workers']))['workers']
    summary=dict(at=b.m.stamp(),timings=rows,selected_workers=selected,selection_unit='highest passing groups_per_second;different counts cannot compare rawwall',all_selected_checks_passed=True,group_evaluations=sum(r['group_evaluations'] for r in passed),warmup_group_evaluations=sum(r['warmup_group_evaluations'] for r in rows),new_model_fits=0,registered_plan_SHA=b.m.sha(plan),preflight_complete_SHA=b.m.sha(b.OUT/'preflight/COMPLETE.json'),source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',HERE/'blend.py',Path(__file__)]})
    b.m.write_json(OUT/'SUMMARY.json',summary);b.m.mark_complete(OUT,['SUMMARY.json',*[f'w{r["workers"]}/task{t}.npz' for r in passed for t in range(r['group_evaluations'])]]);b.m.status(OUT/'RUN_STATUS.json',status='complete',controller_pid=os.getpid(),selected_workers=selected)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['run','worker']);parser.add_argument('--workers',type=int);parser.add_argument('--index',type=int);args=parser.parse_args()
    if args.command=='worker':worker(args.workers,args.index)
    else:main()

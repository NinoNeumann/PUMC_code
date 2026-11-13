"""Portable CACox ES sensitivity analysis launcher. No RCS-ridge fitting."""
from __future__ import annotations
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
CODE=ROOT/'code'
sys.path.insert(0,str(CODE))
from portable_hash import sha256, verify_complete
from verify_prediction import verify_prediction
BRANCHES=('complete_case','egfr_observed','renal_observed','known_subtype','no_egpa')
ACTIVE={}
STOP=False

def stamp():return datetime.now(timezone.utc).isoformat()
def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)
def rscript(explicit=None):
    value=explicit or os.environ.get('CACOX_RSCRIPT') or shutil.which('Rscript')
    if value:
        found=shutil.which(value) or (str(Path(value).resolve()) if Path(value).is_file() else None)
        if found:return found
    candidates=[]
    if os.name=='nt':
        for base in [os.environ.get('ProgramFiles','C:/Program Files'),os.environ.get('LOCALAPPDATA','')]:
            if base:
                candidates.extend(Path(base).glob('R/R-*/bin/Rscript.exe'))
                candidates.extend(Path(base).glob('R/R-*/bin/x64/Rscript.exe'))
    if candidates:
        return str(sorted(candidates,key=lambda p:str(p),reverse=True)[0])
    raise RuntimeError('Rscript was not found. Install R, or pass --rscript "C:/Program Files/R/R-4.5.2/bin/Rscript.exe".')
def runroot(value):
    p=Path(value or os.environ.get('CACOX_RUN_ROOT','run'))
    p=(ROOT/p).resolve() if not p.is_absolute() else p.resolve()
    if not p.is_relative_to(ROOT):raise ValueError('Keep --run-root inside the extracted package for portable signatures.')
    parts=p.relative_to(ROOT).parts
    if not parts or parts[0] in {'input','code','vendor','.Rlib','.venv'}:
        raise ValueError('Use a separate output folder such as run or run_trial; never the package root or an input/code/library folder.')
    return p
def environment(args,output,R):
    return dict(os.environ,CACOX_PORTABLE_ROOT=str(ROOT),CACOX_RUN_ROOT=str(output),
                CACOX_R_LIBS=os.environ.get('CACOX_R_LIBS',str(ROOT/'.Rlib')),
                CACOX_PYTHON=sys.executable,CACOX_RSCRIPT=R,PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                VECLIB_MAXIMUM_THREADS='1',PYTORCH_MPS_FAST_MATH='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
def invoke(args,env,log=None):
    if log:
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a',encoding='utf-8') as stream:
            result=subprocess.run(args,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    else:result=subprocess.run(args,cwd=ROOT,env=env)
    if result.returncode:raise RuntimeError(f'Command failed ({result.returncode}); see {log or "output above"}')
def verify_package():
    p=ROOT/'PACKAGE_MANIFEST.json'
    if not p.exists():return
    record=json.loads(p.read_text(encoding='utf-8'))
    for name,digest in record['files'].items():
        file=ROOT/name
        if not file.is_file() or sha256(file)!=digest:raise ValueError(f'Package file changed or missing: {name}')
def preflight(args,output,R,env):
    verify_package()
    import torch
    names=('numpy','pandas','scipy','scikit-learn','tabm','rtdl_num_embeddings','xgboost','psutil','torch')
    versions={n:importlib.metadata.version(n) for n in names}
    device=args.device
    if device=='auto':device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Check NVIDIA driver and CUDA PyTorch installation; CPU fallback is not automatic when --device cuda is requested.')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    x=torch.ones(2,device=device,requires_grad=True)
    (x.square().sum()).backward()
    if not torch.equal(x.grad.cpu(),torch.full((2,),2.)):raise RuntimeError('Device gradient probe failed')
    if device=='cuda':torch.cuda.synchronize()
    info=dict(at=stamp(),python=sys.version.split()[0],platform=sys.platform,versions=versions,device=device,
              gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
              torch_cuda=torch.version.cuda,neural_device_test='passed',
              scientific_budget='unchanged; max_epochs 3000/3000/500; patience 150')
    args.device=device
    invoke([R,'--vanilla',str(CODE/'check_environment.R')],env)
    version=subprocess.run([R,'--version'],env=env,capture_output=True,text=True,check=True)
    info['R_version']=(version.stdout+version.stderr).strip()
    with (output/'environment_R.csv').open(encoding='utf-8-sig',newline='') as stream:
        info['R_packages']={row['package']:row['version'] for row in csv.DictReader(stream)}
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'environment_python.json',info)
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    return info
def key(i):return f'repeat{(i-1)//5+1:02d}_fold{(i-1)%5+1:02d}'
def fold_dir(output,b,i):return output/'branches'/b/key(i)
def data_ready(output,b,i):
    p=fold_dir(output,b,i)
    return (p/'selection_frequency_29variables.csv').is_file() and all(
        (p/k/'imputation.rds').is_file() for k in ['outer',*[f'inner_{j:02d}' for j in range(1,6)]])
def model_complete(output,b,i):
    p=fold_dir(output,b,i)/'cacox_es'
    if not (p/'COMPLETE.json').exists():return False
    verify_prediction(p)
    return True
def parse_ordinals(s):
    if ':' in s:
        lo,hi=map(int,s.split(':'));out=list(range(lo,hi+1))
    else:out=[int(x) for x in s.split(',')]
    if not out or len(set(out))!=len(out) or not set(out)<=set(range(1,51)):raise ValueError('Use unique ordinals 1..50')
    return out
def stop_children():
    global STOP
    STOP=True
    import psutil
    for proc in list(ACTIVE.values()):
        try:
            parent=psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            parent.terminate()
        except psutil.NoSuchProcess:pass
def signal_stop(signum,frame):stop_children()
def one(stage,b,i,args,output,R,env):
    log=output/'logs'/stage/b/f'{key(i)}.log';log.parent.mkdir(parents=True,exist_ok=True)
    command=([R,'--vanilla',str(CODE/'data_fold.R'),b,str(i)] if stage=='data' else
             [sys.executable,str(CODE/'cacox_fold.py'),b,str(i),'--device',args.device,'--threads','1','--rscript',R])
    start=time.monotonic()
    with log.open('a',encoding='utf-8') as stream:
        stream.write(f'\nSTART {stamp()} {stage} {b} {i}\n');stream.flush()
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        token=(stage,b,i);ACTIVE[token]=proc
        try:rc=proc.wait()
        finally:ACTIVE.pop(token,None)
        stream.write(f'\nEXIT {rc} {stamp()}\n')
    good=rc==0 and (data_ready(output,b,i) if stage=='data' else model_complete(output,b,i))
    return dict(stage=stage,branch=b,ordinal=i,success=good,exit_code=rc,
                seconds=time.monotonic()-start,log=log.relative_to(ROOT).as_posix())
def acquire(output):
    import psutil
    handle=(output/'launcher.lock').open('a+b')
    try:
        handle.seek(0)
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('This run folder already has a live launcher. Use status or stop.')
    write_json(output/'RUNNING.json',dict(pid=os.getpid(),created=psutil.Process().create_time(),at=stamp()))
    return handle
def freeze_execution(output,info,args):
    scientific=[*CODE.glob('*.py'),*CODE.glob('*.R'),*ROOT.joinpath('vendor').rglob('*.py'),
                *ROOT.joinpath('vendor').rglob('*.R'),*ROOT.joinpath('input').glob('*')]
    config=dict(device=args.device,threads=1,versions=info['versions'],python=info['python'],
                platform=info['platform'],gpu=info['gpu'],torch_cuda=info['torch_cuda'],
                R_version=info['R_version'],R_packages=info['R_packages'],
                sources={p.relative_to(ROOT).as_posix():sha256(p) for p in sorted(scientific) if p.is_file()})
    p=output/'EXECUTION_CONFIG.json'
    if p.exists() and json.loads(p.read_text())!=config:
        raise RuntimeError('Code, data, device or environment differs from this run. Use a new --run-root; do not mix checkpoints across machines/devices.')
    write_json(p,config)
def pipeline(args,output,R,env,info):
    global STOP
    output.mkdir(parents=True,exist_ok=True)
    lock=acquire(output)
    try:
        stop=output/'STOP'
        if stop.exists():
            if not args.resume:raise RuntimeError('STOP exists. Use --resume to continue intentionally.')
            stop.unlink()
        freeze_execution(output,info,args)
        invoke([R,'--vanilla',str(CODE/'initialize.R')],env)
        bs=args.branches.split(',');ids=parse_ordinals(args.ordinals)
        if len(set(bs))!=len(bs) or not set(bs)<=set(BRANCHES):raise ValueError('Invalid branches')
        jobs=[(b,i) for i in ids for b in bs]
        pending_data=[j for j in jobs if not data_ready(output,*j)]
        pending_train=[j for j in jobs if not model_complete(output,*j)] if args.command!='data' else []
        finished=[];running={};failed=False;last=None
        state=dict(planned_folds=len(jobs),device=args.device,active=[],results=[])
        signal.signal(signal.SIGINT,signal_stop);signal.signal(signal.SIGTERM,signal_stop)
        with ThreadPoolExecutor(max_workers=args.data_workers) as dp,ThreadPoolExecutor(max_workers=1) as tp:
            while pending_data or pending_train or running:
                if stop.exists():stop_children()
                if not STOP and not failed:
                    n=sum(stage=='data' for stage,_,_ in running.values())
                    while pending_data and n<args.data_workers:
                        b,i=pending_data.pop(0);f=dp.submit(one,'data',b,i,args,output,R,env);running[f]=('data',b,i);n+=1
                    if not any(stage=='cacox' for stage,_,_ in running.values()):
                        for b,i in pending_train:
                            if data_ready(output,b,i):
                                pending_train.remove((b,i));f=tp.submit(one,'cacox',b,i,args,output,R,env)
                                running[f]=('cacox',b,i);break
                for f in list(running):
                    if f.done():
                        stage,b,i=running.pop(f)
                        try:res=f.result()
                        except Exception as error:res=dict(stage=stage,branch=b,ordinal=i,success=False,error=str(error))
                        finished.append(res);print(json.dumps(res,ensure_ascii=False),flush=True)
                        if not res['success']:failed=True
                state=dict(at=stamp(),status='STOPPING' if STOP else 'FAILED' if failed else 'RUNNING',
                    device=args.device,planned_folds=len(jobs),pending_data=len(pending_data),pending_cacox=len(pending_train),
                    active=[dict(stage=s,branch=b,ordinal=i) for s,b,i in running.values()],results=finished)
                write_json(output/'progress.json',state)
                summary=(state['status'],len(pending_data),len(pending_train),len(running))
                if summary!=last:print(stamp(),summary,flush=True);last=summary
                if (STOP or failed) and not running:break
                if not running and not pending_data and pending_train and not any(data_ready(output,*j) for j in pending_train):
                    raise RuntimeError('Missing input data; inspect data logs.')
                if pending_data or pending_train or running:time.sleep(2)
        if STOP or failed:
            write_json(output/'progress.json',dict(state,status='STOPPED' if STOP else 'FAILED',at=stamp()))
            raise RuntimeError('Stopped or failed. Completed stages remain saved; inspect progress.json/logs before resuming.')
        if args.command=='all' and set(bs)==set(BRANCHES) and ids==list(range(1,51)):
            invoke([R,'--vanilla',str(CODE/'subgroup_evaluation.R')],env)
            invoke([R,'--vanilla',str(CODE/'aggregate_sensitivity.R')],env)
        write_json(output/'progress.json',dict(state,status='COMPLETE',at=stamp()))
    finally:
        marker=output/'RUNNING.json'
        if marker.exists():marker.unlink()
        lock.close()
def status(output):
    rows=[dict(branch=b,planned=50,data_ready=sum(data_ready(output,b,i) for i in range(1,51)),
               cacox_complete=sum((fold_dir(output,b,i)/'cacox_es/COMPLETE.json').exists() for i in range(1,51))) for b in BRANCHES]
    print(json.dumps(dict(at=stamp(),run_root=output.relative_to(ROOT).as_posix(),folds=rows,
                         note='Counts use completion markers; full file hashes are verified on resume and formal aggregation.'),ensure_ascii=False,indent=2))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['install-r','check','init','subgroups','all','data','aggregate','status','stop'])
    p.add_argument('--rscript');p.add_argument('--run-root');p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='cuda')
    p.add_argument('--data-workers',type=int,default=2);p.add_argument('--branches',default=','.join(BRANCHES))
    p.add_argument('--ordinals',default='1:50');p.add_argument('--resume',action='store_true')
    a=p.parse_args();output=runroot(a.run_root)
    if a.command=='status':status(output);return
    if a.command=='stop':output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch();print('Stop requested.');return
    if not 1<=a.data_workers<=8:p.error('--data-workers must be 1..8; 2 is recommended initially')
    R=rscript(a.rscript);env=environment(a,output,R)
    if a.command=='install-r':invoke([R,'--vanilla',str(CODE/'install_R_packages.R')],env);return
    if a.command in ('subgroups','aggregate'):
        invoke([R,'--vanilla',str(CODE/('subgroup_evaluation.R' if a.command=='subgroups' else 'aggregate_sensitivity.R'))],env);return
    info=preflight(a,output,R,env)
    if a.command=='check':return
    if a.command=='init':invoke([R,'--vanilla',str(CODE/'initialize.R')],env);return
    pipeline(a,output,R,env,info)
if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:stop_children();sys.exit(130)
    except Exception as error:print(f'ERROR: {error}',file=sys.stderr);sys.exit(1)

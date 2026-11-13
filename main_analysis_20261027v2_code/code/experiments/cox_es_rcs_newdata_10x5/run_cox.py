"""Exact archived ordinary Cox fits on all 10x5x20 matched outer inputs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/cox_e_selected_10x5/20261101_v1'
OLD = ROOT / 'results/ordinary_cox_matched_comparison/20261030_v1'
RSCRIPT = Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')
WORKERS = 4


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    os.replace(temp,path)


def folder(r,k):
    return OUT/'ordinary_cox'/f'repeat{r:02d}_fold{k:02d}'


def result_file(r,k,j):
    return folder(r,k)/f'mi{j:02d}'/'prediction.csv'


def fit_one(r,k,j):
    destination=result_file(r,k,j)
    if destination.exists() and (destination.parent/'COMPLETE.txt').exists():
        return dict(reused=True,seconds=0,imputation=j)
    if psutil.virtual_memory().available<5*2**30:
        raise MemoryError('Less than 5 GiB free; refusing new Cox process')
    log=destination.parent/'fit.log';log.parent.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    command=[str(RSCRIPT),'--vanilla',str(HERE/'cox_fit.R'),str(ROOT),str(OUT),str(r),str(k),str(j)]
    with log.open('w',encoding='utf8') as output:
        proc=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    if proc.returncode or not destination.exists() or not (destination.parent/'COMPLETE.txt').exists():
        raise RuntimeError(f'Cox failed repeat={r} fold={k} MI={j}; see {log}')
    q=pd.read_csv(destination)
    if len(q)!=228 or q.patient_id.duplicated().any() or not q.score.notna().all():
        raise ValueError(f'Invalid ordinary Cox predictions: {destination}')
    return dict(reused=False,seconds=time.perf_counter()-started,imputation=j,
                prediction_SHA256=sha(destination))


def wait_ready(r,k):
    fold=OUT/f'repeat{r:02d}_fold{k:02d}'
    while not (fold/'selected_variables.txt').exists():
        status=OUT/'RUN_STATUS.json'
        if status.exists():
            try:
                state=json.loads(status.read_text(encoding='utf8'))
            except (PermissionError,json.JSONDecodeError):
                # Windows can briefly deny a concurrent status-file read.
                # Wait for the next poll; fitted Cox inputs are unaffected.
                time.sleep(1)
                continue
            if state.get('status')=='failed':
                raise RuntimeError(f'Data stage failed before repeat={r} fold={k}; see {status}')
        time.sleep(10)
    if not (fold/'MICE_VALIDATION.json').exists():
        raise RuntimeError(f'Missing MICE validation: {fold}')


def audit_fold(r,k):
    source=OLD/f'repeat01_fold{k:02d}/predictions_all20.csv' if r==1 else None
    q=pd.read_csv(source if source else folder(r,k)/'predictions_all20.csv')
    expected={'repeat_id':r,'fold':k,'imputation':20,'patient_id':228}
    if len(q)!=4560 or q[['repeat_id','fold','imputation','patient_id']].duplicated().any():
        raise ValueError(f'Bad Cox fold rows r={r} f={k}')
    if q.repeat_id.nunique()!=1 or int(q.repeat_id.iloc[0])!=r or q.fold.nunique()!=1 or int(q.fold.iloc[0])!=k:
        raise ValueError(f'Bad Cox fold identity r={r} f={k}')
    if q.imputation.nunique()!=expected['imputation'] or q.patient_id.nunique()!=expected['patient_id']:
        raise ValueError(f'Bad Cox fold coverage r={r} f={k}')
    for j,z in q.groupby('imputation'):
        if len(z)!=228 or not z.score.notna().all():raise ValueError(f'Bad Cox MI r={r} f={k} j={j}')
    return dict(rows=len(q),prediction_SHA256=sha(source if source else folder(r,k)/'predictions_all20.csv'))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'COX_MANIFEST.json'
    sources=[HERE/'cox_fit.R',HERE/'run_cox.py',
             HERE.parent/'ordinary_cox_matched_comparison/upstream_code/cox29_reliable_fit.R',
             HERE.parent/'ordinary_cox_matched_comparison/upstream_code/cox29_fitting_functions.R']
    signatures={str(p.relative_to(ROOT)):sha(p) for p in sources}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['source_SHA256']!=signatures:
            raise RuntimeError('Cox code changed after first fit')
    else:write(manifest,dict(at=now(),source_SHA256=signatures,workers=WORKERS,
                             first_repeat='verified read-only exact cached fits',new_fits=900))
    lock=OUT/'COX_RUNNING.lock'
    if lock.exists():raise RuntimeError(f'Cox lock exists: {lock}')
    lock.write_text(str(os.getpid()),encoding='ascii')
    try:
        for r in range(1,11):
            for k in range(1,6):
                if r==1:
                    q=audit_fold(r,k)
                    write(folder(r,k)/'REUSED_REFERENCE.json',dict(at=now(),source=str((OLD/f'repeat01_fold{k:02d}/predictions_all20.csv').relative_to(ROOT)),**q))
                    continue
                if (folder(r,k)/'FOLD_COMPLETE.json').exists():
                    audit_fold(r,k)
                    continue
                wait_ready(r,k)
                jobs=[]
                with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                    futures=[pool.submit(fit_one,r,k,j) for j in range(1,21)]
                    for future in as_completed(futures):
                        jobs.append(future.result())
                        write(OUT/'COX_RUN_STATUS.json',dict(status='running',at=now(),pid=os.getpid(),
                              repeat_id=r,fold=k,completed_MI=len(jobs),total_MI=20))
                q=pd.concat([pd.read_csv(result_file(r,k,j),float_precision='round_trip')
                             for j in range(1,21)],ignore_index=True)
                q.to_csv(folder(r,k)/'predictions_all20.csv',index=False)
                proof=audit_fold(r,k)
                write(folder(r,k)/'FOLD_COMPLETE.json',dict(at=now(),jobs=jobs,**proof))
        write(OUT/'COX_COMPLETE.json',dict(at=now(),repeats=10,folds=50,models=1000,
                                          new_models=900,reused_models=100))
        write(OUT/'COX_RUN_STATUS.json',dict(status='complete',at=now()))
    except BaseException as error:
        write(OUT/'COX_RUN_STATUS.json',dict(status='failed',at=now(),error=repr(error),pid=os.getpid()))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()

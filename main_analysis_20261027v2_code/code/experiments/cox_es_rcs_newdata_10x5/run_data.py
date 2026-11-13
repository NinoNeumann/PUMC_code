"""Resumable, bounded R-MICE and Boruta/LASSO preparation for repeats 2-10."""
from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments/cacox_original_head_study/src'))
from cacox_original_head_study.data import Bundle
from cacox_original_head_study.schema29 import CANDIDATES
from cacox_original_head_study.storage import mark_complete, verify_completed

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/cox_es_rcs_newdata_10x5/20261104_v1'
RSCRIPT = Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')
SCRIPT = HERE / 'r_stage.R'
R_LIBRARY = ROOT / '.r_library_selector_windows'
SOURCE = ROOT / 'results/cacox_tree_selected29_integrated_20261028_v1'
MAX_WORKERS = {'impute': 4, 'select': 6}
MIN_FREE_GB = 5.0
MAX_OWNED_GB = 8.0


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf8')
    os.replace(temporary, path)


def unit(r: int, k: int) -> Path:
    return OUT / f'repeat{r:02d}_fold{k:02d}'


def memory() -> dict:
    free = psutil.virtual_memory().available / 2**30
    owned = 0.0
    for process in psutil.process_iter(['name', 'cmdline', 'memory_info']):
        try:
            cmd = ' '.join(process.info['cmdline'] or [])
            if 'cox_es_rcs_newdata_10x5/r_stage.R' in cmd.replace('\\', '/'):
                owned += process.info['memory_info'].rss / 2**30
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return dict(free_GB=free, owned_R_GB=owned)


def await_memory() -> None:
    started = time.monotonic()
    while True:
        reading = memory()
        if reading['free_GB'] >= MIN_FREE_GB and reading['owned_R_GB'] < MAX_OWNED_GB:
            return
        if time.monotonic() - started > 900:
            raise MemoryError(f'Memory headroom unavailable for 15 minutes: {reading}')
        time.sleep(5)


def stage_done(stage: str, r: int, k: int, ident: int) -> bool:
    d = unit(r, k)
    if stage == 'impute':
        part = d / ('outer' if ident == 0 else f'inner_{ident:02d}')
        target = part / 'imputation.rds'
        return target.exists() and target.stat().st_size > 1000 and (part / 'mice_chain_diagnostic.csv').exists()
    target = d / f'imputation_{ident:02d}' / 'combined_selection.csv'
    if not target.exists():
        return False
    frame = pd.read_csv(target)
    return len(frame) == 29 and set(['variable', 'union_selected', 'lasso_lambda_min']).issubset(frame)


def validate_mice(r: int, k: int) -> dict:
    outcomes = []
    for ident in range(6):
        if not stage_done('impute', r, k, ident):
            raise RuntimeError(f'Missing MICE r={r} f={k} part={ident}')
        part = unit(r, k) / ('outer' if ident == 0 else f'inner_{ident:02d}')
        if (part / 'mice_logged_events.csv').exists():
            raise RuntimeError(f'MICE logged events r={r} f={k} part={ident}')
        diagnostic = pd.read_csv(part / 'mice_chain_diagnostic.csv')
        high = diagnostic.loc[diagnostic.classical_Rhat > 1.1]
        if len(high):
            raise RuntimeError(f'MICE Rhat>1.1 r={r} f={k} part={ident}: '
                               f'{high[["variable", "classical_Rhat"]].to_dict("records")}')
        outcomes.append(dict(partition=ident, max_Rhat=float(diagnostic.classical_Rhat.max())))
    return dict(parts=outcomes, max_Rhat=max(p['max_Rhat'] for p in outcomes))


def run_one(stage: str, r: int, k: int, ident: int) -> dict:
    if stage_done(stage, r, k, ident):
        return dict(stage=stage, repeat_id=r, fold=k, ident=ident, reused=True, seconds=0)
    await_memory()
    directory = unit(r, k) / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    logfile = directory / f'{stage}_{ident:02d}.log'
    command = [str(RSCRIPT), '--vanilla', str(SCRIPT), str(ROOT), str(OUT),
               stage, str(r), str(k), str(ident)]
    started = time.perf_counter()
    with logfile.open('w', encoding='utf8') as output:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        result = proc.wait()
    if result != 0 or not stage_done(stage, r, k, ident):
        raise RuntimeError(f'{stage} failed r={r} f={k} id={ident} exit={result}; see {logfile}')
    return dict(stage=stage, repeat_id=r, fold=k, ident=ident, reused=False,
                seconds=time.perf_counter() - started, log=str(logfile.relative_to(ROOT)))


def batch(stage: str, r: int, k: int, identifiers: range) -> list[dict]:
    jobs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS[stage]) as pool:
        futures = {pool.submit(run_one, stage, r, k, ident): ident for ident in identifiers}
        for future in as_completed(futures):
            jobs.append(future.result())
            atomic_json(OUT / 'RUN_STATUS.json', dict(status='running', stage=stage,
                       repeat_id=r, fold=k, completed_in_stage=len(jobs), total_in_stage=len(identifiers),
                       controller_pid=os.getpid(), resource=memory(), at=now()))
    return sorted(jobs, key=lambda row: row['ident'])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'audit'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    identity = {str(path.relative_to(ROOT)): sha(path) for path in [
        SCRIPT, HERE / 'run_data.py', HERE / 'export_bundle.R',
        HERE / 'prepare_input.R', HERE / 'export_input_excel.py',
        SOURCE / 'upstream_source/data_imputation_functions.R',
        SOURCE / 'code/leaf_sum_prediction.R', SOURCE / 'code/selection29_fast.R',
        OUT / 'input/config.rds',
        OUT / 'input/raw_typed.rds',
        OUT / 'input/20261027v2.xlsx',
        OUT / 'input/20261027v2_text.csv',
        OUT / 'input/outer_assignments.csv']}
    manifest = OUT / 'DATA_MANIFEST.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['source_SHA256'] != identity:
            raise RuntimeError('Source/config/data hash changed since data stage began')
    else:
        atomic_json(manifest, dict(at=now(), source_SHA256=identity, repeats=10, folds=5,
                    m=20, maxit=60, workers=MAX_WORKERS, min_free_GB=MIN_FREE_GB,
                    max_owned_GB=MAX_OWNED_GB, selection='Boruta non-rejected UNION '
                    'Cox-LASSO lambda.min, >=10/20; each new development fold'))
    if args.command == 'audit':
        print(json.dumps(dict(prepared=sum((unit(r,k)/'prepared.rds').exists()
                                        for r in range(1,11) for k in range(1,6)),
             imputed=sum(stage_done('impute',r,k,i) for r in range(1,11) for k in range(1,6) for i in range(6)),
             selected=sum(stage_done('select',r,k,i) for r in range(1,11) for k in range(1,6) for i in range(1,21))),
             ensure_ascii=False))
        return
    lock = OUT / 'DATA_RUNNING.lock'
    if lock.exists():
        raise RuntimeError(f'Lock exists: {lock}; inspect owner before recovery')
    lock.write_text(str(os.getpid()), encoding='ascii')
    try:
        with (OUT / 'DATA_EVENTS.jsonl').open('a', encoding='utf8') as events:
            for r in range(1,11):
                for k in range(1,6):
                    d = unit(r,k)
                    if not (d/'prepared.rds').exists():
                        log=d/'logs/prepare.log'
                        log.parent.mkdir(parents=True,exist_ok=True)
                        with log.open('w',encoding='utf8') as output:
                            result=subprocess.run([str(RSCRIPT),'--vanilla',str(SCRIPT),str(ROOT),str(OUT),
                                'prepare',str(r),str(k)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                        if result.returncode or not (d/'prepared.rds').exists():
                            raise RuntimeError(f'Prepare failed r={r} f={k}; see {log}')
                    if not all(stage_done('impute',r,k,i) for i in range(6)):
                        jobs = batch('impute',r,k,range(6))
                        events.write(json.dumps(dict(at=now(),stage='impute',repeat_id=r,fold=k,jobs=jobs))+'\n')
                        events.flush()
                    diagnostic = validate_mice(r,k)
                    atomic_json(d/'MICE_VALIDATION.json',diagnostic)
                    if not all(stage_done('select',r,k,i) for i in range(1,21)):
                        jobs = batch('select',r,k,range(1,21))
                        events.write(json.dumps(dict(at=now(),stage='select',repeat_id=r,fold=k,jobs=jobs))+'\n')
                        events.flush()
                    if not (d/'selected_variables.txt').exists():
                        log = d/'logs/summary.log'
                        with log.open('w',encoding='utf8') as output:
                            result = subprocess.run([str(RSCRIPT),'--vanilla',str(SCRIPT),str(ROOT),str(OUT),
                                'summary',str(r),str(k)],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
                        if result.returncode != 0:
                            raise RuntimeError(f'Selection summary failed: {log}')
                    selected = (d/'selected_variables.txt').read_text(encoding='utf8').splitlines()
                    bundle = OUT/'bundles'/f'repeat{r:02d}_fold{k:02d}'/'main'
                    if not verify_completed(bundle):
                        export_log=d/'logs/export_bundle.log'
                        with export_log.open('w',encoding='utf8') as output:
                            result=subprocess.run([str(RSCRIPT),'--vanilla',str(HERE/'export_bundle.R'),
                                str(ROOT),str(OUT),str(r),str(k)],cwd=ROOT,
                                stdout=output,stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise RuntimeError(f'Bundle export failed: {export_log}')
                        atomic_json(bundle/'bundle.json',dict(m=20,maxit=60,
                            candidate_variables=CANDIDATES,selected_variables=selected,
                            repeat_id=r,fold=k,backend='upstream R MICE 3.19.0, 20x60',
                            selection_rule='Boruta non-rejected UNION Cox-LASSO lambda.min; >=10/20'))
                        loaded=Bundle(bundle)
                        if loaded.variables!=selected:raise ValueError('Exported variable mismatch')
                        for mi in [1,10,20]:loaded.inner_pairs(mi);loaded.pair(mi)
                        files=[str(p.relative_to(bundle)) for p in bundle.rglob('*')
                               if p.is_file() and p.name!='COMPLETE.json']
                        mark_complete(bundle,files)
                    if not verify_completed(bundle):raise RuntimeError(f'Bundle verification failed: {bundle}')
                    events.write(json.dumps(dict(at=now(),stage='fold_ready',repeat_id=r,fold=k,
                                                  selected=selected,max_Rhat=diagnostic['max_Rhat']))+'\n')
                    events.flush()
                    atomic_json(OUT/'RUN_STATUS.json',dict(status='running',stage='fold_ready',
                                repeat_id=r,fold=k,selected_count=len(selected),
                                completed_new_folds=sum((unit(a,b)/'selected_variables.txt').exists()
                                    for a in range(1,11) for b in range(1,6)),
                                controller_pid=os.getpid(),resource=memory(),at=now()))
        atomic_json(OUT/'DATA_COMPLETE.json',dict(at=now(),prepared=50,new_folds=50,
                    imputation_partitions=300,selection_MI_jobs=1000))
        atomic_json(OUT/'RUN_STATUS.json',dict(status='data_complete',at=now()))
    except BaseException as error:
        atomic_json(OUT/'RUN_STATUS.json',dict(status='failed',at=now(),error=repr(error),
                    controller_pid=os.getpid(),resource=memory()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()

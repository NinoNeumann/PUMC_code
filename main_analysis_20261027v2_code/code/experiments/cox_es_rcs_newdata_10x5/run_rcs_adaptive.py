"""Frozen 10 x 5 x 20 RCS + ridge-Cox outer fits on existing R MICE data."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
INPUT = ROOT / 'results/cox_es_rcs_newdata_10x5/20261104_v1'
OUTPUT = ROOT / 'results/rcs_ridge_newdata_10x5/20261104_v1'
OLD = ROOT / 'results/cacox_vs_rcs_selected29_optimization_20261028_v1'
RSCRIPT = Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')
WORKERS = 2
KEY = ['repeat_id', 'fold', 'imputation', 'patient_id']
PROB = ['survival_365d', 'survival_730d', 'survival_1095d']


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')
    os.replace(tmp, path)


def keys(r: int, f: int):
    return f'repeat{r:02d}_fold{f:02d}'


def qualified_mice_path(fold: Path, part: str) -> Path:
    base = fold / part
    for iteration in (120, 100, 80):
        candidate = base / 'adaptive' / f'iter_{iteration:03d}' / 'imputation.rds'
        if candidate.is_file():
            return candidate
    return base / 'imputation.rds'


def validation_path(fold: Path) -> Path:
    adaptive = fold / 'MICE_VALIDATION_ADAPTIVE.json'
    return adaptive if adaptive.is_file() else fold / 'MICE_VALIDATION.json'


def result_path(r: int, f: int):
    return OUTPUT / 'folds' / keys(r, f) / 'predictions_all20.csv'


def verify_fold(r: int, f: int):
    path = result_path(r, f)
    if any(not (path.parent / f'model_{j:02d}.rds').is_file() for j in range(1,21)):
        raise ValueError(f'Missing saved RCS fitted objects {r},{f}')
    q = pd.read_csv(path)
    reference = pd.read_csv(INPUT / 'bundles' / keys(r, f) / 'main' / 'raw_assessment.csv')
    if len(q) != 4560 or len(reference) != 228:
        raise ValueError(f'Incorrect fold rows {r},{f}')
    if q[KEY].duplicated().any() or reference.patient_id.duplicated().any():
        raise ValueError(f'Duplicate outer predictions {r},{f}')
    if set(q.repeat_id) != {r} or set(q.fold) != {f}:
        raise ValueError(f'Incorrect RCS fold identity {r},{f}')
    if sorted(q.imputation.unique().tolist()) != list(range(1, 21)):
        raise ValueError(f'Incomplete MI numbers {r},{f}')
    q = q.sort_values(KEY).reset_index(drop=True)
    for _, part in q.groupby('imputation'):
        part = part.sort_values('patient_id').reset_index(drop=True)
        expected = reference.sort_values('patient_id').reset_index(drop=True)
        if not np.array_equal(part.patient_id, expected.patient_id) or not np.array_equal(
                part[['Death', 'Survival_Days']], expected[['Death', 'Survival_Days']]):
            raise ValueError(f'RCS/test-bundle patient pairing mismatch {r},{f}')
    if not np.isfinite(q[['score_rcs_ridge', *PROB]].to_numpy()).all():
        raise ValueError(f'Nonfinite predictions {r},{f}')
    if ((q[PROB] < 0) | (q[PROB] > 1)).any().any():
        raise ValueError(f'Invalid survival probabilities {r},{f}')
    return {'rows': len(q), 'models': 20, 'sha256': digest(path)}


def source_manifest():
    files = [HERE / 'fit_fold_adaptive.R', HERE / 'ridge_functions_adaptive.R',
             HERE / 'run_rcs_adaptive.py',
             OLD / 'run_matched_rcs_ridge.R',
             INPUT / 'DATA_MANIFEST.json', INPUT / 'ADAPTIVE_MICE_QC_MANIFEST.json',
             INPUT / 'DATA_COMPLETE.json',
             INPUT / 'input/raw_typed.rds', INPUT / 'input/20261027v2.xlsx']
    files += sorted((OLD / 'windows_rcs_ridge_code').glob('*.R'))
    for r in range(1, 11):
        for f in range(1, 6):
            fold = INPUT / keys(r, f)
            files += [fold / name for name in ('prepared.rds', 'selected_variables.txt',
                      'selection_frequency_29variables.csv')]
            files.append(validation_path(fold))
            for part in ['outer', *(f'inner_{j:02d}' for j in range(1, 6))]:
                files.append(fold / part / 'imputation.rds')
                effective = qualified_mice_path(fold, part)
                if effective != fold / part / 'imputation.rds':
                    files.append(effective)
    if any(not p.is_file() for p in files):
        raise FileNotFoundError([str(p) for p in files if not p.is_file()][:10])
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def audit_inputs():
    for name in ('DATA_COMPLETE.json',):
        if not (INPUT / name).is_file():
            raise ValueError(f'Incomplete frozen input: {name}')
    for r in range(1,11):
        for f in range(1,6):
            fold=INPUT/keys(r,f)
            for name in ('prepared.rds','selected_variables.txt',
                         'outer/imputation.rds'):
                if not (fold/name).is_file():raise ValueError(f'Missing RCS input {fold/name}')
            if not validation_path(fold).is_file():
                raise ValueError(f'Missing RCS validation {fold}')
            if not (INPUT/'bundles'/keys(r,f)/'main/COMPLETE.json').is_file():
                raise ValueError(f'Missing completed bundle {r},{f}')


def fit_fold(r: int, f: int):
    key = keys(r, f)
    target = OUTPUT / 'folds' / key
    if (target / 'COMPLETE.json').exists():
        proof = verify_fold(r, f)
        stored = json.loads((target / 'COMPLETE.json').read_text(encoding='utf8'))
        if proof['sha256'] != stored['prediction_SHA256']:
            raise ValueError(f'Completed fold changed: {key}')
        return {'key': key, 'reused': True, **proof}
    while psutil.virtual_memory().available < 5 * 2**30:
        time.sleep(10)
    target.mkdir(parents=True, exist_ok=True)
    log = target / 'fit.log'
    start = time.perf_counter()
    with log.open('a', encoding='utf8') as stream:
        code = subprocess.run([str(RSCRIPT), '--vanilla', str(HERE / 'fit_fold_adaptive.R'),
              str(ROOT), str(INPUT), str(OUTPUT), key], cwd=ROOT,
              stdout=stream, stderr=subprocess.STDOUT,
              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0).returncode
    if code or not (target / 'COMPLETE.txt').exists():
        raise RuntimeError(f'RCS fit failed {key}; see {log}')
    proof = verify_fold(r, f)
    write(target / 'COMPLETE.json', {'at': now(), 'prediction_SHA256': proof['sha256'],
          'elapsed_seconds': time.perf_counter() - start, 'models': 20})
    return {'key': key, 'reused': False, **proof}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT / 'RUN_STATUS.json'
    lock = OUTPUT / 'RUNNING.lock'
    if lock.exists():
        raise RuntimeError(f'Existing RCS runner lock: {lock}')
    while not (INPUT / 'DATA_COMPLETE.json').is_file():
        source_status=INPUT/'RUN_STATUS.json'
        if source_status.exists():
            try:
                state=json.loads(source_status.read_text(encoding='utf8'))
                if state.get('status')=='failed':
                    raise RuntimeError(f'Data stage failed before RCS fit: {state}')
            except (PermissionError,json.JSONDecodeError):
                pass
        write(status, {'at':now(),'status':'waiting_for_data','controller_pid':os.getpid()})
        time.sleep(30)
    manifest = OUTPUT / 'MANIFEST.json'
    audit_inputs()
    sources = source_manifest()
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['source_SHA256'] != sources:
            raise ValueError('Input or source changed since RCS run started')
    else:
        write(manifest, {'at': now(), 'source_SHA256': sources, 'workers': WORKERS,
                         'target': '10 repeats x 5 folds x 20 MI', 'old_models_reused': 0})
    lock.write_text(str(os.getpid()), encoding='ascii')
    try:
        jobs = [(r, f) for r in range(1, 11) for f in range(1, 6)]
        done = []
        write(status, {'at': now(), 'status': 'running', 'controller_pid': os.getpid(),
                       'completed_new_folds': 0, 'total_new_folds': 50, 'completed_new_models': 0})
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(fit_fold, r, f): (r, f) for r, f in jobs}
            for future in as_completed(futures):
                result = future.result()
                done.append(result)
                write(status, {'at': now(), 'status': 'running', 'controller_pid': os.getpid(),
                               'completed_new_folds': len(done), 'total_new_folds': 50,
                               'completed_new_models': 20 * len(done), 'last_fold': result['key']})
        write(OUTPUT / 'FIT_COMPLETE.json', {'at': now(), 'repeats': 10, 'folds': 50,
              'models': 1000, 'new_models': 1000, 'reused_models': 0,
              'fold_SHA256': {z['key']: z['sha256'] for z in done}})
        write(status, {'at': now(), 'status': 'fits_complete', 'controller_pid': os.getpid(),
                       'completed_new_folds': 50, 'completed_new_models': 1000})
    except BaseException as error:
        write(status, {'at': now(), 'status': 'failed', 'controller_pid': os.getpid(), 'error': repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

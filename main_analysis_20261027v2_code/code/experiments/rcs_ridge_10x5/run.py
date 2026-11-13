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
INPUT = ROOT / 'results/cox_e_selected_10x5/20261101_v1'
OUTPUT = ROOT / 'results/rcs_ridge_10x5/20261102_v1'
OLD = ROOT / 'results/cacox_vs_rcs_selected29_optimization_20261028_v1'
OLD_COX = ROOT / 'results/ordinary_cox_matched_comparison/20261030_v1'
RSCRIPT = Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')
WORKERS = 4
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


def old_path(f: int):
    return OLD / 'matched_rcs_ridge' / keys(1, f) / 'predictions_all20.csv'


def result_path(r: int, f: int):
    return old_path(f) if r == 1 else OUTPUT / 'folds' / keys(r, f) / 'predictions_all20.csv'


def cox_path(r: int, f: int):
    if r == 1:
        return OLD_COX / keys(1, f) / 'predictions_all20.csv'
    return INPUT / 'ordinary_cox' / keys(r, f) / 'predictions_all20.csv'


def verify_fold(r: int, f: int):
    path = result_path(r, f)
    q = pd.read_csv(path)
    cox = pd.read_csv(cox_path(r, f))
    if len(q) != 4560 or len(cox) != 4560:
        raise ValueError(f'Incorrect fold rows {r},{f}')
    if q[KEY].duplicated().any() or cox[KEY].duplicated().any():
        raise ValueError(f'Duplicate outer predictions {r},{f}')
    if set(q.repeat_id) != {r} or set(q.fold) != {f}:
        raise ValueError(f'Incorrect RCS fold identity {r},{f}')
    if sorted(q.imputation.unique().tolist()) != list(range(1, 21)):
        raise ValueError(f'Incomplete MI numbers {r},{f}')
    q = q.sort_values(KEY).reset_index(drop=True)
    cox = cox.sort_values(KEY).reset_index(drop=True)
    if not q[KEY].equals(cox[KEY]) or not np.array_equal(q[['Death', 'Survival_Days']], cox[['Death', 'Survival_Days']]):
        raise ValueError(f'RCS/Cox patient pairing mismatch {r},{f}')
    if not np.isfinite(q[['score_rcs_ridge', *PROB]].to_numpy()).all():
        raise ValueError(f'Nonfinite predictions {r},{f}')
    if ((q[PROB] < 0) | (q[PROB] > 1)).any().any():
        raise ValueError(f'Invalid survival probabilities {r},{f}')
    return {'rows': len(q), 'models': 20, 'sha256': digest(path)}


def source_manifest():
    files = [HERE / 'fit_fold.R', HERE / 'audit_first.R', HERE / 'run.py',
             OLD / 'run_matched_rcs_ridge.R',
             INPUT / 'DATA_MANIFEST.json', INPUT / 'DATA_COMPLETE.json',
             INPUT / 'FINAL_INPUT_INTEGRITY.json',
             OLD / 'final_comparison' / 'COMPLETE.json']
    files += sorted((OLD / 'windows_rcs_ridge_code').glob('*.R'))
    for f in range(1, 6):
        files += [old_path(f), cox_path(1, f)]
    for r in range(2, 11):
        for f in range(1, 6):
            fold = INPUT / keys(r, f)
            files += [fold / name for name in ('prepared.rds', 'selected_variables.txt',
                      'selection_frequency_29variables.csv', 'MICE_VALIDATION.json',
                      'outer/imputation.rds', *(f'inner_{j:02d}/imputation.rds' for j in range(1, 6)))]
    if any(not p.is_file() for p in files):
        raise FileNotFoundError([str(p) for p in files if not p.is_file()][:10])
    return {str(p.relative_to(ROOT)): digest(p) for p in files}


def audit_inputs():
    for name in ('COMPLETE.json', 'DATA_COMPLETE.json', 'COX_COMPLETE.json',
                 'I20_COMPLETE.json', 'E_SELECTED_COMPLETE.json', 'FINAL_INPUT_INTEGRITY.json'):
        if not (INPUT / name).is_file():
            raise ValueError(f'Incomplete frozen input: {name}')
    integrity = json.loads((INPUT / 'FINAL_INPUT_INTEGRITY.json').read_text(encoding='utf8'))
    if (integrity['folds'], integrity['imputations'], integrity['missing']) != (50, 20, 0):
        raise ValueError('Frozen input integrity failed')
    audit = subprocess.run([str(RSCRIPT), '--vanilla', str(HERE / 'audit_first.R'), str(ROOT), str(INPUT)],
                           cwd=ROOT, capture_output=True, text=True, encoding='utf8', errors='replace')
    (OUTPUT / 'first_repeat_input_audit.log').write_text(audit.stdout + audit.stderr, encoding='utf8')
    if audit.returncode or audit.stdout.count('FIRST_REPEAT_INPUT_IDENTICAL') != 5:
        raise RuntimeError('First repeat split audit failed')
    reused = {keys(1, f): verify_fold(1, f) for f in range(1, 6)}
    write(OUTPUT / 'FIRST_REPEAT_REUSED.json', {'at': now(), 'folds': reused, 'models': 100,
          'source': str(OLD / 'matched_rcs_ridge'), 'rule': 'exact same splits, read-only predictions'})


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
        code = subprocess.run([str(RSCRIPT), '--vanilla', str(HERE / 'fit_fold.R'),
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
    manifest = OUTPUT / 'MANIFEST.json'
    audit_inputs()
    sources = source_manifest()
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf8'))['source_SHA256'] != sources:
            raise ValueError('Input or source changed since RCS run started')
    else:
        write(manifest, {'at': now(), 'source_SHA256': sources, 'workers': WORKERS,
                         'target': '10 repeats x 5 folds x 20 MI', 'old_models_reused': 100})
    lock.write_text(str(os.getpid()), encoding='ascii')
    try:
        jobs = [(r, f) for r in range(2, 11) for f in range(1, 6)]
        done = []
        write(status, {'at': now(), 'status': 'running', 'controller_pid': os.getpid(),
                       'completed_new_folds': 0, 'total_new_folds': 45, 'completed_new_models': 0})
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(fit_fold, r, f): (r, f) for r, f in jobs}
            for future in as_completed(futures):
                result = future.result()
                done.append(result)
                write(status, {'at': now(), 'status': 'running', 'controller_pid': os.getpid(),
                               'completed_new_folds': len(done), 'total_new_folds': 45,
                               'completed_new_models': 20 * len(done), 'last_fold': result['key']})
        write(OUTPUT / 'FIT_COMPLETE.json', {'at': now(), 'repeats': 10, 'folds': 50,
              'models': 1000, 'new_models': 900, 'reused_models': 100,
              'fold_SHA256': {z['key']: z['sha256'] for z in done}})
        write(status, {'at': now(), 'status': 'fits_complete', 'controller_pid': os.getpid(),
                       'completed_new_folds': 45, 'completed_new_models': 900})
    except BaseException as error:
        write(status, {'at': now(), 'status': 'failed', 'controller_pid': os.getpid(), 'error': repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()

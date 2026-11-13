"""Resume the new-data run with prespecified MICE convergence extensions.

Each partition starts at the frozen 60 iterations. Only a partition with
classical R-hat >1.1 continues its same 20 chains to 80, 100, then 120.
Existing 60-iteration files and all completed fold bundles remain unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys

import pandas as pd

import run_data as original
import run_data_8process as sched
from run_data_6process import check_frozen_inputs

HERE = Path(__file__).resolve().parent
STAGE = HERE / 'r_stage_adaptive.R'
EXPORT = HERE / 'export_bundle_adaptive.R'
EXTEND = HERE / 'extend_mice_adaptive.R'
AUDIT = original.OUT / 'ADAPTIVE_MICE_QC_MANIFEST.json'
THRESHOLD = 1.1
MAX_ITER = 120


def part_path(repeat: int, fold: int, ident: int) -> Path:
    return original.unit(repeat, fold) / ('outer' if ident == 0 else f'inner_{ident:02d}')


def effective(part: Path) -> tuple[Path, int]:
    for iteration in (120, 100, 80):
        candidate = part / 'adaptive' / f'iter_{iteration:03d}' / 'imputation.rds'
        if candidate.is_file():
            return candidate, iteration
    return part / 'imputation.rds', 60


def diagnostic(part: Path) -> tuple[float, list[dict]]:
    path, iteration = effective(part)
    if not path.is_file() or path.stat().st_size <= 1000:
        raise RuntimeError(f'Missing MICE object: {path}')
    table_path = path.parent / 'mice_chain_diagnostic.csv'
    if not table_path.is_file():
        raise RuntimeError(f'Missing MICE diagnostic: {table_path}')
    if (path.parent / 'mice_logged_events.csv').exists() or (part / 'mice_logged_events.csv').exists():
        raise RuntimeError(f'MICE logged events: {part}')
    table = pd.read_csv(table_path)
    if not {'variable', 'parameter', 'classical_Rhat', 'iterations_used', 'chains'} <= set(table):
        raise RuntimeError(f'Invalid MICE diagnostic columns: {table_path}')
    if not (table['chains'].eq(20).all() and table['iterations_used'].eq(iteration // 2).all()):
        raise RuntimeError(f'Invalid MICE diagnostic dimensions: {table_path}')
    high = table.loc[table.classical_Rhat > THRESHOLD,
                     ['variable', 'parameter', 'classical_Rhat']].to_dict('records')
    return float(table.classical_Rhat.max()), high


def validate_mice(repeat: int, fold: int, *, extend: bool) -> dict:
    outcomes = []
    for ident in range(6):
        if not original.stage_done('impute', repeat, fold, ident):
            raise RuntimeError(f'Missing base MICE r={repeat} f={fold} part={ident}')
        part = part_path(repeat, fold, ident)
        maximum, high = diagnostic(part)
        while high and extend:
            _, current = effective(part)
            if current >= MAX_ITER:
                break
            target = current + 20
            logfile = original.unit(repeat, fold) / 'logs' / f'extend_{ident:02d}_to_{target:03d}.log'
            sched.r_command([str(original.RSCRIPT), '--vanilla', str(EXTEND),
                             str(original.ROOT), str(original.OUT), str(repeat),
                             str(fold), str(ident), str(target)], logfile)
            maximum, high = diagnostic(part)
            original.atomic_json(part / 'adaptive' / f'iter_{target:03d}' / 'QC_RESULT.json',
                                 dict(at=original.now(), threshold=THRESHOLD, iterations=target,
                                      max_Rhat=maximum, exceeding=high,
                                      base_sha256=original.sha(part / 'imputation.rds')))
            sched.event('adaptive_mice_qc', repeat, fold, partition=ident,
                        iterations=target, max_Rhat=maximum, exceeding=high)
        if high:
            raise RuntimeError(f'MICE Rhat>{THRESHOLD} after up to {effective(part)[1]} '
                               f'iterations r={repeat} f={fold} part={ident}: {high}')
        outcomes.append(dict(partition=ident, iterations=effective(part)[1], max_Rhat=maximum))
    return dict(parts=outcomes, max_Rhat=max(row['max_Rhat'] for row in outcomes))


def completed_count() -> int:
    ready = 0
    for repeat in range(1, 11):
        for fold in range(1, 6):
            directory = original.unit(repeat, fold)
            selected = directory / 'selected_variables.txt'
            if not selected.is_file():
                continue
            validate_mice(repeat, fold, extend=False)
            if not all(original.stage_done('select', repeat, fold, mi) for mi in range(1, 21)):
                raise RuntimeError(f'Incomplete selection in completed fold {repeat}/{fold}')
            bundle = original.OUT / 'bundles' / f'repeat{repeat:02d}_fold{fold:02d}' / 'main'
            if not original.verify_completed(bundle):
                raise RuntimeError(f'Invalid completed bundle {repeat}/{fold}')
            ready += 1
    return ready


def process_fold(repeat: int, fold: int) -> None:
    directory = original.unit(repeat, fold)
    bundle = original.OUT / 'bundles' / f'repeat{repeat:02d}_fold{fold:02d}' / 'main'
    selected_path = directory / 'selected_variables.txt'
    if selected_path.exists():
        validate_mice(repeat, fold, extend=False)
        if not all(original.stage_done('select', repeat, fold, mi) for mi in range(1, 21)):
            raise RuntimeError(f'Incomplete selected fold {repeat}/{fold}')
        if not original.verify_completed(bundle):
            raise RuntimeError(f'Invalid completed bundle {repeat}/{fold}')
        return

    sched.status('prepare', repeat, fold)
    if not (directory / 'prepared.rds').is_file():
        sched.r_command([str(original.RSCRIPT), '--vanilla', str(STAGE), str(original.ROOT),
                         str(original.OUT), 'prepare', str(repeat), str(fold)],
                        directory / 'logs/prepare_adaptive.log')
    if not (directory / 'prepared.rds').is_file():
        raise RuntimeError(f'Missing prepared fold {repeat}/{fold}')

    sched.status('impute', repeat, fold)
    if not all(original.stage_done('impute', repeat, fold, i) for i in range(6)):
        sched.event('impute', repeat, fold,
                    jobs=original.batch('impute', repeat, fold, range(6)))
    qc = validate_mice(repeat, fold, extend=True)
    original.atomic_json(directory / 'MICE_VALIDATION_ADAPTIVE.json', qc)

    sched.status('select', repeat, fold)
    if not all(original.stage_done('select', repeat, fold, i) for i in range(1, 21)):
        sched.event('select', repeat, fold,
                    jobs=original.batch('select', repeat, fold, range(1, 21)))
    if not selected_path.is_file():
        sched.r_command([str(original.RSCRIPT), '--vanilla', str(STAGE), str(original.ROOT),
                         str(original.OUT), 'summary', str(repeat), str(fold)],
                        directory / 'logs/summary_adaptive.log')
    selected = selected_path.read_text(encoding='utf8').splitlines()
    if not selected:
        raise RuntimeError(f'Empty selected list {repeat}/{fold}')

    sched.status('export', repeat, fold)
    if not original.verify_completed(bundle):
        sched.r_command([str(original.RSCRIPT), '--vanilla', str(EXPORT), str(original.ROOT),
                         str(original.OUT), str(repeat), str(fold)],
                        directory / 'logs/export_bundle_adaptive.log')
        original.atomic_json(bundle / 'bundle.json', dict(
            m=20, maxit=60, base_maxit=60, adaptive_maxit=MAX_ITER,
            effective_iterations={str(x['partition']): x['iterations'] for x in qc['parts']},
            candidate_variables=original.CANDIDATES, selected_variables=selected,
            repeat_id=repeat, fold=fold,
            backend='upstream R MICE 3.19.0; same 20 chains with adaptive convergence QC',
            selection_rule='Boruta non-rejected UNION Cox-LASSO lambda.min; >=10/20'))
        loaded = original.Bundle(bundle)
        if loaded.variables != selected:
            raise RuntimeError(f'Exported variables differ {repeat}/{fold}')
        for mi in (1, 10, 20):
            loaded.inner_pairs(mi)
            loaded.pair(mi)
        files = [str(path.relative_to(bundle)) for path in bundle.rglob('*')
                 if path.is_file() and path.name != 'COMPLETE.json']
        original.mark_complete(bundle, files)
    if not original.verify_completed(bundle):
        raise RuntimeError(f'Invalid bundle {repeat}/{fold}')
    sched.event('fold_ready', repeat, fold, selected=selected, max_Rhat=qc['max_Rhat'],
                effective_iterations=qc['parts'])
    sched.status('fold_ready', repeat, fold)


def main() -> None:
    if sys.argv[1:] not in (['audit'], ['run']):
        raise SystemExit('Usage: run_data_adaptive.py audit|run')
    check_frozen_inputs()
    sources = {str(p.relative_to(original.ROOT)): original.sha(p)
               for p in (STAGE, EXPORT, EXTEND, Path(__file__),
                         HERE / 'run_data_8process.py')}
    signature = dict(original_data_manifest_sha256=original.sha(original.OUT / 'DATA_MANIFEST.json'),
                     adaptive_sources=sources, threshold=THRESHOLD,
                     iterations=[60, 80, 100, 120], chains=20,
                     max_R_processes=sched.R_PROCESSES, folds_in_flight=sched.FOLDS_IN_FLIGHT)
    before = completed_count()
    if sys.argv[1:] == ['audit']:
        print(json.dumps(dict(completed_folds=before, **signature)))
        return
    if AUDIT.exists():
        old = json.loads(AUDIT.read_text(encoding='utf8'))
        if old['signature'] != signature:
            raise RuntimeError('Adaptive protocol source or input changed')
    else:
        original.atomic_json(AUDIT, dict(at=original.now(), signature=signature,
                                         completed_folds_verified_before_resume=before))
    lock = original.OUT / 'DATA_RUNNING.lock'
    if lock.exists():
        raise RuntimeError(f'Active or stale data lock: {lock}')
    lock.write_text(str(os.getpid()), encoding='ascii')
    original.MAX_WORKERS = sched.PER_FOLD_WORKERS.copy()
    original.SCRIPT = STAGE
    original.run_one = sched.bounded_run_one
    original.atomic_json = sched.locked_atomic_json
    try:
        pending = [(r, k) for r in range(1, 11) for k in range(1, 6)
                   if not (original.unit(r, k) / 'selected_variables.txt').exists()]
        for offset in range(0, len(pending), sched.FOLDS_IN_FLIGHT):
            pair = pending[offset:offset+sched.FOLDS_IN_FLIGHT]
            with ThreadPoolExecutor(max_workers=len(pair)) as pool:
                futures = {pool.submit(process_fold, *key): key for key in pair}
                for future in as_completed(futures):
                    future.result()
        if completed_count() != 50:
            raise RuntimeError('Data integrity count is not 50')
        original.atomic_json(original.OUT / 'DATA_COMPLETE.json', dict(
            at=original.now(), prepared=50, new_folds=50, imputation_partitions=300,
            selection_MI_jobs=1000, scheduler='adaptive_mice_8process_twofold',
            adaptive_manifest_sha256=original.sha(AUDIT)))
        original.atomic_json(original.OUT / 'RUN_STATUS.json',
                             dict(status='data_complete', at=original.now()))
    except BaseException as error:
        original.atomic_json(original.OUT / 'RUN_STATUS.json', dict(
            status='failed', scheduler='adaptive_mice_8process_twofold',
            error=repr(error), controller_pid=os.getpid(),
            resource=original.memory(), at=original.now()))
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf8')
    main()

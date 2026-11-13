from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
import time

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent.parent
sys.path.insert(0, str(OUT / 'snapshot/src'))
from pmc_rebuttal.schema import CANDIDATES
from pmc_rebuttal.storage import mark_complete, verify_completed

RSCRIPT = Path(r'C:\Users\NinoNeumann\AppData\Local\Programs\R\R-4.4.2\bin\Rscript.exe')


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf8')


def one(fold, j):
    started = time.perf_counter()
    log = OUT / f'logs/selection_fold{fold:02d}_mi{j:02d}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w', encoding='utf8') as stream:
        run = subprocess.run([str(RSCRIPT), '--vanilla', str(OUT / 'code/select_one.R'),
                              str(OUT), str(fold), str(j)], cwd=PROJECT,
                             stdout=stream, stderr=subprocess.STDOUT)
    return dict(imputation=j, exit_code=run.returncode, seconds=time.perf_counter()-started,
                log=str(log))


def summarize(fold, jobs, seconds):
    key = f'repeat01_fold{fold:02d}'
    folder = OUT / 'selection' / key
    frames = [pd.read_csv(folder / f'imputation_{j:02d}/combined_selection.csv') for j in range(1, 21)]
    validations = [pd.read_csv(folder / f'imputation_{j:02d}/verification.csv').iloc[0] for j in range(1, 21)]
    freq = (pd.concat(frames).groupby('variable', sort=False)
            .agg(boruta_non_rejected=('boruta_non_rejected', 'sum'),
                 lasso_lambda_min=('lasso_lambda_min', 'sum'),
                 union_selected=('union_selected', 'sum'))
            .reindex(CANDIDATES).reset_index())
    freq['included'] = freq.union_selected >= 10
    selected = freq.loc[freq.included, 'variable'].tolist()
    upstream = pd.read_csv(OUT / 'upstream_cache/cv10_20261027v2' / key /
                           'selection_frequency_29variables.csv')
    same_frequency = bool(freq.set_index('variable').union_selected.equals(
        upstream.set_index('variable').union_selected))
    same_selected = selected == upstream.loc[upstream.union_selected >= 10, 'variable'].tolist()
    union_disagreements = 0
    for j, frame in enumerate(frames, 1):
        prior = pd.read_csv(OUT / 'upstream_cache/cv10_20261027v2' / key /
                            f'imputation_{j:02d}/combined_selection.csv')
        union_disagreements += int((frame.union_selected != prior.union_selected).sum())
    freq.to_csv(folder / 'selection_frequency_29variables.csv', index=False)
    (folder / 'selected_variables.txt').write_text('\n'.join(selected)+'\n', encoding='utf8')
    bundle = OUT / 'bundles' / key / 'main'
    write(bundle / 'bundle.json', dict(m=20, maxit=60, candidate_variables=CANDIDATES,
         selected_variables=selected, backend='upstream_R_MICE_cache_R_Boruta29_LASSO_leafsum',
         selection_rule='Boruta non-rejected UNION Cox-LASSO lambda.min; frequency >=10/20',
         upstream_repeat=1, upstream_fold=fold))
    files = [str(p.relative_to(bundle)) for p in bundle.rglob('*')
             if p.is_file() and p.name != 'COMPLETE.json']
    mark_complete(bundle, files)
    assert verify_completed(bundle)
    report = dict(fold=fold, wall_seconds=seconds, workers=8, jobs=jobs,
                  selected_variables=selected, selected_count=len(selected),
                  same_selected_as_upstream=same_selected,
                  same_frequency_as_upstream=same_frequency,
                  union_disagreements=union_disagreements,
                  raw_boruta_matches=sum(bool(v.raw_boruta_match_upstream) for v in validations),
                  completed_utc=datetime.now(timezone.utc).isoformat())
    write(folder / 'timing_summary.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'jobs'}, ensure_ascii=False), flush=True)


def main():
    for fold in range(1, 6):
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            jobs = list(pool.map(lambda j: one(fold, j), range(1, 21)))
        if any(j['exit_code'] for j in jobs):
            raise RuntimeError(f'Selection worker failure in fold {fold}: {jobs}')
        summarize(fold, jobs, time.perf_counter()-started)


if __name__ == '__main__':
    main()

"""Reconstruct frozen E_selected Breslow curves and score daily 0–1095 d IBS."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'results/cox_e_selected_10x5/20261101_v1'
OLD = ROOT / 'results/cacox_heterogeneous_blend/20261031_v2_stream'
OUT = ROOT / 'results/ibs_0_3y_20261103_v1'
BASE = ROOT / 'results/rcs_ridge_10x5/20261102_v1/fold_first_cindex_20261103'
ORDER = ['C0_ref17'] + [f'{name}_w{weight}' for name in
          ('A8_NLL', 'A8_C', 'A16_NLL', 'A16_C')
          for weight in ('075', '050', '025', '000')]
GRID = np.arange(1096, dtype=np.float64)
TIMES = (365, 730, 1095)

sys.path.insert(0, str(ROOT / 'experiments/cacox_original_head_study/src'))
from cacox_original_head_study.probability import km, km_at


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    frame.to_csv(temp, index=False, float_format='%.17g')
    os.replace(temp, path)


def source_paths(r: int, f: int):
    if r == 1:
        base = OLD
        bundle = ROOT / 'results/cacox_tree_selected29_integrated_20261028_v1/bundles' / f'repeat{r:02d}_fold{f:02d}' / 'main'
        group = base / 'outer' / f'fold{f:02d}' / 'shared'
        predictions = group
        choice = base / 'FROZEN_CHOICES.json'
    else:
        base = INPUT / 'e_selected_sources' / f'repeat{r:02d}'
        bundle = INPUT / 'bundles' / f'repeat{r:02d}_fold{f:02d}' / 'main'
        group = base / 'outer' / f'fold{f:02d}' / 'shared'
        predictions = base / 'blend' / 'outer' / f'fold{f:02d}'
        choice = base / 'blend' / 'FROZEN_CHOICES.json'
    return bundle / 'raw_development.csv', bundle / 'raw_assessment.csv', group / 'risks.npz', choice, predictions


def one_fold(r: int, f: int, archived: pd.DataFrame):
    raw_path, assessment_path, risk_path, choice_path, predictions = source_paths(r, f)
    if not all(p.is_file() for p in (raw_path, assessment_path, risk_path, choice_path)):
        raise FileNotFoundError((r, f, raw_path, assessment_path, risk_path, choice_path))
    dev = pd.read_csv(raw_path).set_index('patient_id', verify_integrity=True)
    selected = json.loads(choice_path.read_text(encoding='utf8'))['choices'][str(f)]['candidate']
    ci = ORDER.index(selected)
    with np.load(risk_path) as store:
        if r==1:
            train = store[f'{selected}_train'].copy()
            val = store[f'{selected}_val'].copy()
            train_ids = pd.read_csv(raw_path).patient_id.to_numpy()
            val_ids = pd.read_csv(assessment_path).patient_id.to_numpy()
        else:
            train = store['train'][ci].copy()
            val = store['val'][ci].copy()
            train_ids = store['train_ids'].copy()
            val_ids = store['val_ids'].copy()
    if train.shape != (20, 912) or val.shape != (20, 228):
        raise ValueError(f'Invalid frozen risk dimensions r={r} f={f}')
    dev = dev.loc[train_ids]
    t_train = dev.Survival_Days.to_numpy(np.float64)
    e_train = dev.Death.to_numpy(np.int64)
    censor = km(t_train, 1-e_train)
    g_grid = km_at(censor, GRID)
    if not np.all(g_grid > 0):
        raise ValueError(f'IPCW positivity failure r={r} f={f}')
    event_times, death_counts = np.unique(t_train[e_train==1], return_counts=True)
    risk_mask = t_train[None, :] >= event_times[:, None]
    rows = []
    source_files = [raw_path, assessment_path, risk_path, choice_path]
    for j in range(1,21):
        old_path = predictions / f'mi{j:02d}' / 'E_selected.csv'
        old = pd.read_csv(old_path).set_index('patient_id', verify_integrity=True).loc[val_ids]
        source_files.append(old_path)
        score_train, score_val = train[j-1], val[j-1]
        score_diff = np.max(np.abs(old.score.to_numpy(np.float64)-score_val))
        if score_diff > 1e-10:
            raise ValueError(f'Frozen score mismatch r={r} f={f} j={j}: {score_diff}')
        shift = np.median(score_train)
        increments = death_counts / (risk_mask @ np.exp(score_train-shift))
        cumhaz = np.cumsum(increments)
        ch = np.r_[0., cumhaz][np.searchsorted(event_times, GRID, side='right')]
        s = np.exp(-np.exp(score_val-shift)[:, None]*ch[None, :])
        old_s = old[[f'survival_{t}d' for t in TIMES]].to_numpy(np.float64)
        survival_diff = np.max(np.abs(s[:, list(TIMES)]-old_s))
        if survival_diff > 1e-10:
            raise ValueError(f'Archived survival mismatch r={r} f={f} j={j}: {survival_diff}')
        test_time = old.Survival_Days.to_numpy(np.float64)
        test_event = old.Death.to_numpy(np.int64)
        g_before = km_at(censor, test_time, left=True)
        if np.any(g_before[(test_event==1)&(test_time<=1095)] <= 0):
            raise ValueError(f'IPCW event positivity failure r={r} f={f} j={j}')
        dead = (test_time[:,None] <= GRID[None,:]) & (test_event[:,None] == 1)
        alive = test_time[:,None] > GRID[None,:]
        terms = np.zeros_like(s)
        np.divide(s*s, g_before[:,None], out=terms, where=dead)
        alive_terms = ((1-s)*(1-s))/g_grid[None,:]
        terms += np.where(alive, alive_terms, 0.)
        brier = terms.mean(axis=0)
        for t in TIMES:
            ref = archived.loc[(archived.repeat_id==r)&(archived.fold==f)&
                (archived.imputation==j)&(archived.horizon_days==t)&
                (archived.model=='E_selected'), 'Brier_IPCW']
            if len(ref)!=1 or abs(brier[t]-float(ref.iloc[0]))>1e-10:
                raise ValueError(f'Brier archive mismatch r={r} f={f} j={j} t={t}: {brier[t]}, {ref}')
        rows.append(dict(repeat_id=r, fold=f, imputation=j, model='E_selected',
            IBS_0_1095d_daily=float(np.trapezoid(brier, GRID)/1095),
            Brier_365d=float(brier[365]), Brier_730d=float(brier[730]),
            Brier_1095d=float(brier[1095]), max_S_3point_diff=float(survival_diff),
            max_score_diff=float(score_diff), candidate=selected))
    result = pd.DataFrame(rows)
    target = OUT / f'repeat{r:02d}_fold{f:02d}' / 'e_ibs.csv'
    atomic_csv(result, target)
    proof = dict(repeat_id=r, fold=f, models=20, grid='daily 0..1095 inclusive',
        source_sha256={str(p.relative_to(ROOT)):file_hash(p) for p in source_files},
        result_sha256=file_hash(target), max_S_3point_diff=float(result.max_S_3point_diff.max()),
        max_Brier_archive_diff_lt_1e10=True)
    final = target.with_name('E_COMPLETE.json')
    temp = final.with_name(final.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(proof,indent=2),encoding='utf8')
    os.replace(temp,final)
    return result


def main():
    archived = pd.read_csv(BASE/'brier_fold_imputation.csv')
    args = sys.argv[1:]
    jobs = [(int(args[0]),int(args[1]))] if args else [(r,f) for r in range(1,11) for f in range(1,6)]
    for r,f in jobs:
        target = OUT/f'repeat{r:02d}_fold{f:02d}'/'E_COMPLETE.json'
        if target.exists():
            proof = json.loads(target.read_text(encoding='utf8'))
            csv = target.with_name('e_ibs.csv')
            if csv.exists() and file_hash(csv)==proof['result_sha256']:
                print(f'IBS_E_REUSED repeat{r:02d}_fold{f:02d}',flush=True)
                continue
            raise ValueError(f'Completed E IBS corrupted r={r} f={f}')
        result = one_fold(r,f,archived)
        print(f'IBS_E repeat{r:02d}_fold{f:02d} mean={result.IBS_0_1095d_daily.mean():.9f} max_archive_diff={result.max_S_3point_diff.max():.3g}',flush=True)


if __name__ == '__main__':
    main()

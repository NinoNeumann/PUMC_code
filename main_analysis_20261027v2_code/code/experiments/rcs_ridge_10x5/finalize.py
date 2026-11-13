"""Verify all 1000 RCS fits and summarize matched RCS, Cox and E_selected."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results/rcs_ridge_10x5/20261102_v1'
INPUT = ROOT / 'results/cox_e_selected_10x5/20261101_v1'
RSCRIPT = Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')
sys.path.insert(0, str(ROOT / 'experiments/cox_e_selected_10x5'))
import final_metrics as prior
sys.path.insert(0, str(ROOT / 'experiments/cacox_original_head_study/src'))
from cacox_original_head_study import metrics_core as core
from cacox_original_head_study.storage import mark_complete, verify_completed
from cacox_original_head_study.runtime import sha

KEY = ['repeat_id', 'fold', 'imputation', 'patient_id']
TIMES = [365, 730, 1095]
MODELS = ['ordinary_cox', 'E_selected', 'rcs_ridge_cox']
PAIRS = [('rcs_ridge_cox', 'ordinary_cox'), ('rcs_ridge_cox', 'E_selected')]


def run():
    complete = OUT / 'FIT_COMPLETE.json'
    if not complete.is_file():
        raise ValueError('All 50 RCS folds must be completed before final inference')
    source = json.loads(complete.read_text(encoding='utf8'))
    if (source['folds'], source['models']) != (50, 1000):
        raise ValueError('Unexpected fit coverage')
    if not (INPUT / 'COMPLETE.json').is_file():
        raise ValueError('Matched Cox/E_selected input is incomplete')
    target = OUT / 'final'
    target.mkdir(parents=True, exist_ok=True)

    frames = prior.load_all(10)
    rcs_parts = []
    models = []
    for r in range(1, 11):
        for f in range(1, 6):
            key = f'repeat{r:02d}_fold{f:02d}'
            path = (ROOT / 'results/cacox_vs_rcs_selected29_optimization_20261028_v1/matched_rcs_ridge' / key / 'predictions_all20.csv') if r == 1 else OUT / 'folds' / key / 'predictions_all20.csv'
            if r > 1:
                proof = json.loads((OUT / 'folds' / key / 'COMPLETE.json').read_text(encoding='utf8'))
                if proof['prediction_SHA256'] != sha(path):
                    raise ValueError(f'Prediction changed: {key}')
            q = pd.read_csv(path)
            if len(q) != 4560 or q[KEY].duplicated().any():
                raise ValueError(f'Incomplete RCS fold: {key}')
            for j, z in q.groupby('imputation'):
                if len(z) != 228:
                    raise ValueError(f'Incomplete RCS MI: {key} {j}')
            q = q.rename(columns={'score_rcs_ridge': 'score'})
            for t in TIMES:
                q[f'risk_{t}d'] = 1 - q[f'survival_{t}d']
            rcs_parts.append(q)
    frames['rcs_ridge_cox'] = pd.concat(rcs_parts, ignore_index=True)
    aligned = {m: q.sort_values(KEY).reset_index(drop=True) for m, q in frames.items()}
    anchor = aligned['ordinary_cox']
    for m, q in aligned.items():
        if len(q) != 228000 or not q[KEY].equals(anchor[KEY]) or not np.array_equal(q[['Death', 'Survival_Days']], anchor[['Death', 'Survival_Days']]):
            raise ValueError(f'Unmatched test patients/outcomes: {m}')
        if not np.isfinite(q[['score', *(f'survival_{t}d' for t in TIMES)]].to_numpy()).all():
            raise ValueError(f'Nonfinite predictions: {m}')
        models.append(m)
    paired_scores = anchor[KEY + ['Death', 'Survival_Days']].copy()
    for m in MODELS:
        paired_scores[f'score_{m}'] = aligned[m].score.to_numpy()
    paired_scores.to_csv(target / 'paired_scores.csv', index=False)

    per = []
    pooled = []
    paired = []
    for horizon in [None, *TIMES]:
        metric = 'C_index' if horizon is None else f'AUC_{horizon}d'
        column = 'score' if horizon is None else f'risk_{horizon}d'
        estimates = {}
        for m in MODELS:
            values = []
            for j in range(1, 21):
                z = core.aggregate_metric(frames[m].loc[frames[m].imputation.eq(j)], column, horizon)
                per.append(dict(model=m, metric=metric, imputation=j,
                                estimate=z['estimate'], variance=z['variance']))
                values.append(z)
            estimates[m] = values
            pooled.append(dict(model=m, metric=metric, **core.rubin(
                [z['estimate'] for z in values], [z['variance'] for z in values], True)))
        for left, right in PAIRS:
            differences = []
            variances = []
            for a, b in zip(estimates[left], estimates[right]):
                ids = a['contributions'].keys() | b['contributions'].keys()
                differences.append(a['estimate'] - b['estimate'])
                variances.append(sum((a['contributions'].get(i, 0.) - b['contributions'].get(i, 0.))**2 for i in ids))
            paired.append(dict(comparison=f'{left}_minus_{right}', metric=metric,
                               **core.rubin(differences, variances)))
        print('METRIC', metric, {m: estimates[m][0]['estimate'] for m in MODELS}, flush=True)
    pd.DataFrame(per).to_csv(target / 'metrics_by_imputation.csv', index=False)
    pd.DataFrame(pooled).to_csv(target / 'metrics_pooled.csv', index=False)
    pd.DataFrame(paired).to_csv(target / 'paired_differences.csv', index=False)
    prob = prior.probability(frames, 10)
    prob.to_csv(target / 'probability_by_model.csv', index=False)
    prob.groupby(['model', 'horizon_days'])[['Brier_IPCW', 'predicted_mean_risk', 'observed_KM_risk', 'calibration_mean_gap']].mean().reset_index().to_csv(target / 'probability_summary.csv', index=False)

    with (target / 'pvalues_R.log').open('w', encoding='utf8') as log:
        process = subprocess.run([str(RSCRIPT), '--vanilla', str(HERE / 'pvalues.R'), str(ROOT), str(OUT)],
                                 cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'R paired inference failed: {target / "pvalues_R.log"}')
    r = pd.read_csv(target / 'cindex_and_pvalues.csv')
    p = pd.DataFrame(pooled)
    for m in MODELS:
        x = float(r.loc[r.model.eq(m), 'estimate'].iloc[0])
        y = float(p.loc[p.model.eq(m) & p.metric.eq('C_index'), 'estimate'].iloc[0])
        if abs(x - y) > 1e-10:
            raise ValueError(f'R/Python C-index mismatch for {m}: {x} versus {y}')

    cindex_rows = []
    for m in MODELS:
        q = aligned[m]
        for (repeat_id, fold, imputation), z in q.groupby(['repeat_id', 'fold', 'imputation']):
            c = core.aggregate_metric(z, 'score', None)['estimate']
            cindex_rows.append(dict(model=m, repeat_id=repeat_id, fold=fold,
                                    imputation=imputation, C_index=c))
    detail = pd.DataFrame(cindex_rows)
    if len(detail) != 3000:
        raise ValueError('Model C-index distribution must contain 3000 rows')
    detail.to_csv(target / 'model_cindex.csv', index=False)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    labels = ['Ordinary Cox', 'E_selected', 'RCS + ridge-Cox']
    ax[0].boxplot([detail.loc[detail.model.eq(m), 'C_index'] for m in MODELS],
                  tick_labels=labels, showfliers=False)
    ax[0].set_ylabel('Outer-test Harrell C-index')
    ax[0].grid(axis='y', alpha=.2)
    pivot = detail.pivot(index=['repeat_id', 'fold', 'imputation'], columns='model', values='C_index')
    for left, right in PAIRS:
        delta = pivot[left] - pivot[right]
        ax[1].hist(delta, bins=30, alpha=.65, label=f'RCS − {right}')
    ax[1].axvline(0, color='black', linewidth=1)
    ax[1].set_xlabel('Paired model C-index difference')
    ax[1].set_ylabel('Count')
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(target / 'cindex_distribution.png')
    plt.close(fig)

    summary = {'at': datetime.now(timezone.utc).isoformat(), 'folds': 50,
               'imputations': 20, 'models_per_method': 1000,
               'reused_rcs_models': 100, 'new_rcs_models': 900,
               'matched_outer_test_patients': 1140, 'R_python_C_agree': True}
    (target / 'SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    report = ['# RCS + ridge-Cox full 10 × 5 × 20 comparison', '',
              'Same 1140 patients, 29 candidate variables, frozen outer folds, R MICE imputations and Boruta + LASSO selected lists as the completed ordinary Cox / E_selected study.',
              'First repeat RCS models reused read-only after exact split audit; 45 new folds refit using the archived nine-lambda ridge grid and five internal folds × 20 imputations.', '',
              '## Outer-test C-index', '']
    report += ['| Method | C-index (95% CI) | 1000-model mean (min–max) |',
               '|---|---:|---:|']
    for m in MODELS:
        z = r.loc[r.model.eq(m)].iloc[0]
        vals = detail.loc[detail.model.eq(m), 'C_index']
        report.append(f'| {m} | {z.estimate:.5f} ({z.CI95_lower:.5f}–{z.CI95_upper:.5f}) | {vals.mean():.5f} ({vals.min():.5f}–{vals.max():.5f}) |')
    report += ['', '## Paired C-index differences', '',
               '| Comparison | Difference (95% CI) | Rubin p | Conditional p without Rubin |',
               '|---|---:|---:|---:|']
    for left, right in PAIRS:
        z = r.loc[r.model.eq(f'{left}_minus_{right}')].iloc[0]
        report.append(f'| {left} − {right} | {z.estimate:.5f} ({z.CI95_lower:.5f}–{z.CI95_upper:.5f}) | {z.p_Rubin:.6g} | {z.p_conditional_normal_no_Rubin:.6g} |')
    report += ['', '## AUC and IPCW Brier', '',
               '| Method | AUC 365d | AUC 730d | AUC 1095d | Brier 365d | Brier 730d | Brier 1095d |',
               '|---|---:|---:|---:|---:|---:|---:|']
    brier = pd.read_csv(target / 'probability_summary.csv')
    for m in MODELS:
        auc = [p.loc[p.model.eq(m) & p.metric.eq(f'AUC_{t}d'), 'estimate'].iloc[0] for t in TIMES]
        bs = [brier.loc[brier.model.eq(m) & brier.horizon_days.eq(t), 'Brier_IPCW'].iloc[0] for t in TIMES]
        report.append('| ' + m + ' | ' + ' | '.join(f'{x:.5f}' for x in [*auc, *bs]) + ' |')
    report += ['', 'All intervals and p values are conditional on the same repeatedly used cohort, the frozen splits and fitted model definitions. The 1000 imputation models are correlated and do not constitute 1000 independent patients or external validation.', '']
    (target / 'FINAL_REPORT.md').write_text('\n'.join(report), encoding='utf8')
    products = ['paired_scores.csv', 'metrics_by_imputation.csv', 'metrics_pooled.csv',
                'paired_differences.csv', 'probability_by_model.csv', 'probability_summary.csv',
                'pvalues_R.log', 'cindex_and_pvalues.csv', 'model_cindex.csv',
                'cindex_distribution.png', 'SUMMARY.json', 'FINAL_REPORT.md']
    mark_complete(target, products)
    if not verify_completed(target):
        raise ValueError('Final output checksum verification failed')
    (OUT / 'COMPLETE.json').write_text(json.dumps({'at': datetime.now(timezone.utc).isoformat(),
        'folds': 50, 'models': 1000, 'new_models': 900, 'reused_models': 100,
        'final_sha256': sha(target / 'COMPLETE.json')}, indent=2), encoding='utf8')
    print('FINAL_COMPLETE', target, flush=True)


if __name__ == '__main__':
    run()

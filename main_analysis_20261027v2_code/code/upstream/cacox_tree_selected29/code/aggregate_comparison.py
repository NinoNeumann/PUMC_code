from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OUT / 'snapshot/src'))
sys.path.insert(0, str(OUT / 'runner_scripts'))
from pmc_rebuttal import metrics
from pmc_rebuttal.storage import verify_completed
from performance_gpu_candidates import optimized_metrics

optimized_metrics(metrics)
RESULT = OUT / 'comparison'
RESULT.mkdir(exist_ok=True)


def common(frame):
    return frame[['repeat_id', 'fold', 'imputation', 'patient_id',
                  'Death', 'Survival_Days']].copy()


def read_predictions():
    rows = []
    selected = []
    for fold in range(1, 6):
        key = f'repeat01_fold{fold:02d}'
        bundle = OUT / 'bundles' / key / 'main'
        gpu = OUT / 'gpu_runs' / f'{key}_attempt01' / key / 'main'
        upstream = (OUT / 'upstream_cache/cv10_20261027v2_rcs_ridge' / key /
                    'models/predictions_all20.csv')
        if not verify_completed(bundle) or not verify_completed(gpu):
            raise ValueError(f'Incomplete fold {fold}')
        model = pd.read_csv(gpu / 'predictions_all20.csv', float_precision='round_trip')
        reference = pd.read_csv(upstream, float_precision='round_trip')
        keys = ['repeat_id', 'fold', 'imputation', 'patient_id']
        z = model.merge(reference, on=keys, validate='one_to_one',
                        suffixes=('_cacox', '_upstream'), indicator=True)
        if (len(z) != 228*20 or not z._merge.eq('both').all()
                or not np.array_equal(z.Death_cacox, z.Death_upstream)
                or not np.array_equal(z.Survival_Days_cacox, z.Survival_Days_upstream)):
            raise ValueError(f'Prediction pairing failed in fold {fold}')
        rows.append(z)
        local_selection = json.loads((bundle / 'bundle.json').read_text(encoding='utf8'))
        prior = pd.read_csv(OUT / 'upstream_cache/cv10_20261027v2' / key /
                            'selection_frequency_29variables.csv')
        upstream_selection = prior.loc[prior.union_selected >= 10, 'variable'].tolist()
        selected.append(dict(fold=fold, variables=local_selection['selected_variables'],
                             upstream_variables=upstream_selection,
                             match=local_selection['selected_variables'] == upstream_selection))
    all_rows = pd.concat(rows, ignore_index=True)
    if len(all_rows) != 1140*20 or all_rows.groupby('imputation').patient_id.nunique().tolist() != [1140]*20:
        raise ValueError('Five outer folds do not cover 1140 patients per imputation')
    return all_rows, selected


def make_model_frame(joined, score_column):
    z = joined[['repeat_id', 'fold', 'imputation', 'patient_id',
                'Death_cacox', 'Survival_Days_cacox', score_column]].copy()
    z.columns = ['repeat_id', 'fold', 'imputation', 'patient_id',
                 'Death', 'Survival_Days', 'score']
    return z


joined, selections = read_predictions()
scores = {'Ordinary Cox': 'score_cox', 'RCS + ridge-Cox': 'score_rcs_ridge',
          'CACox + selected variables': 'score'}
models = {label: make_model_frame(joined, column) for label, column in scores.items()}

model_rows = []
fold_rows = []
for name, frame in models.items():
    for fold, group in frame.groupby('fold', sort=True):
        estimates, variances = [], []
        for j, z in group.groupby('imputation', sort=True):
            point = metrics.harrell(z.Survival_Days, z.Death, z.score)
            estimates.append(point['estimate'])
            variances.append(point['variance'])
            model_rows.append(dict(model=name, fold=int(fold), imputation=int(j),
                                   C_index=point['estimate'],
                                   comparable_pairs=point['comparable_pairs']))
        fold_rows.append(dict(model=name, fold=int(fold),
                              **metrics.rubin(estimates, variances, True)))
model_table = pd.DataFrame(model_rows)
fold_table = pd.DataFrame(fold_rows)
model_table.to_csv(RESULT / 'model_cindex.csv', index=False)
fold_table.to_csv(RESULT / 'fold_cindex.csv', index=False)

# Independently reproduce every archived upstream fold C before comparing CACox.
upstream_fold = pd.read_csv(OUT / 'upstream_cache/cv10_20261027v2_rcs_ridge/summary/C_index_by_fold.csv')
for name, metric_name in [('Ordinary Cox', 'C_cox'), ('RCS + ridge-Cox', 'C_rcs_ridge')]:
    for fold in range(1, 6):
        actual = float(fold_table[(fold_table.model == name) & (fold_table.fold == fold)].estimate.iloc[0])
        expected = float(upstream_fold[(upstream_fold.ordinal == fold) &
                                       (upstream_fold.metric == metric_name)].estimate.iloc[0])
        if abs(actual - expected) > 1e-10:
            raise ValueError(f'Upstream metric not reproduced: {name} fold {fold}')

all_metrics = []
for name, frame in models.items():
    each, pooled = metrics.evaluate_predictions(frame, m=20, times=(365, 730, 1095))
    each.insert(0, 'model', name)
    pooled.insert(0, 'model', name)
    each.to_csv(RESULT / f'metrics_each_imputation_{name.replace(" ", "_").replace("+", "plus")}.csv', index=False)
    all_metrics.append(pooled)
pooled_table = pd.concat(all_metrics, ignore_index=True)
pooled_table.to_csv(RESULT / 'pooled_metrics.csv', index=False)

comparisons = []
for baseline in ('Ordinary Cox', 'RCS + ridge-Cox'):
    each, pooled = metrics.paired_difference(models[baseline],
                                             models['CACox + selected variables'],
                                             m=20, times=())
    each.insert(0, 'baseline', baseline)
    each.to_csv(RESULT / f'paired_each_imputation_{baseline.replace(" ", "_").replace("+", "plus")}.csv', index=False)
    row = pooled.iloc[0].to_dict()
    row['baseline'] = baseline
    row['p_value_two_sided'] = float(2 * student_t.sf(abs(row['estimate']/row['SE']), row['df']))
    comparisons.append(row)
paired_table = pd.DataFrame(comparisons)
paired_table.to_csv(RESULT / 'paired_cindex.csv', index=False)

overview = []
for name, z in model_table.groupby('model', sort=False):
    overview.append(dict(model=name, mean_100=float(z.C_index.mean()),
                         minimum_100=float(z.C_index.min()), maximum_100=float(z.C_index.max()),
                         above_077=int((z.C_index > .77).sum()), total_models=len(z)))
overview_table = pd.DataFrame(overview)
overview_table.to_csv(RESULT / 'model_distribution_summary.csv', index=False)

palette = {'Ordinary Cox': '#4B6577', 'RCS + ridge-Cox': '#BA814D',
           'CACox + selected variables': '#237B73'}
fig, ax = plt.subplots(figsize=(9, 5.2))
order = list(scores)
series = [model_table.loc[model_table.model == name, 'C_index'].to_numpy() for name in order]
violin = ax.violinplot(series, positions=np.arange(1, 4), showmeans=False,
                       showmedians=False, showextrema=False)
for body, name in zip(violin['bodies'], order):
    body.set_facecolor(palette[name]); body.set_alpha(.22); body.set_edgecolor(palette[name])
box = ax.boxplot(series, positions=np.arange(1, 4), widths=.22, patch_artist=True,
                 showfliers=False, medianprops={'color':'#202020'})
for patch, name in zip(box['boxes'], order):
    patch.set_facecolor(palette[name]); patch.set_alpha(.72)
ax.axhline(.77, color='#AD3C3C', linestyle='--', linewidth=1.2, label='C-index = 0.77')
ax.set_xticks([1, 2, 3], order)
ax.set_ylabel('Outer-test Harrell C-index')
ax.set_title('Same 5 folds and 20 imputations per fold (100 model evaluations)')
ax.set_ylim(min(z.min() for z in series)-.02, max(z.max() for z in series)+.02)
ax.grid(axis='y', alpha=.18)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig(RESULT / 'cindex_distribution.png', dpi=200)
fig.savefig(RESULT / 'cindex_distribution.pdf')
plt.close(fig)

summary = dict(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
               cohort_n=1140, outer_folds=5, imputations_per_fold=20, model_evaluations_per_method=100,
               upstream_repeat=1, inputs='Identical outer folds and archived R MICE imputation caches',
               selected_variables=selections,
               all_selected_variables_match_upstream=all(z['match'] for z in selections),
               overview=overview, paired_cindex=comparisons,
               method='Within-fold comparable-pair-weighted Harrell C; patient clustered IJ and Rubin pooling across 20 MI',
               upstream_metric_reproduced=True,
               caveat='Same score/metric inputs as upstream; CACox is a new fit. Conditional CI excludes model-development uncertainty.')
(RESULT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf8')
print(json.dumps({'overview': overview, 'paired_cindex': comparisons,
                  'all_selected_variables_match_upstream': summary['all_selected_variables_match_upstream']},
                 ensure_ascii=False), flush=True)

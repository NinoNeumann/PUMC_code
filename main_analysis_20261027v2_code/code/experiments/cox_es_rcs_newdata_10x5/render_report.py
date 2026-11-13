"""Render the audited 10x5x20 ordinary Cox versus E_selected results."""
from datetime import datetime,timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cox_e_selected_10x5/20261101_v1'
FINAL=OUT/'final'

def fmt(x):return f'{float(x):.5f}'

def main():
    if not (FINAL/'COMPLETE.json').exists():raise RuntimeError('Metrics not complete')
    summary=pd.read_csv(OUT/'final_cindex_and_pvalues.csv').set_index('model')
    models=pd.read_csv(OUT/'final_model_cindex.csv')
    pooled=pd.read_csv(FINAL/'metrics_pooled.csv')
    probs=pd.read_csv(FINAL/'probability_summary.csv')
    if len(models)!=2000 or set(models.model)!={'ordinary_cox','E_selected'}:
        raise ValueError('Expected exactly 1000 models per method')
    df=models.pivot(index=['repeat_id','fold','imputation'],columns='model',values='C_index')
    if len(df)!=1000 or df.isna().any().any():raise ValueError('Unpaired C-index distribution')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    axes[0].boxplot([df.ordinary_cox,df.E_selected],tick_labels=['Ordinary Cox','E_selected'],
                    showmeans=True,meanprops={'marker':'D','markerfacecolor':'#1e6381','markeredgecolor':'#1e6381'})
    axes[0].set_ylabel('Outer test C-index')
    axes[0].set_title('Matched 10 × 5 × 20 models')
    axes[0].grid(axis='y',alpha=.2)
    delta=df.E_selected-df.ordinary_cox
    axes[1].hist(delta,bins=32,color='#387c97',edgecolor='white')
    axes[1].axvline(0,color='black',lw=1)
    axes[1].axvline(float(summary.loc['E_selected_minus_cox','estimate']),color='#c85e37',lw=2,
                    label='pooled difference')
    axes[1].set_xlabel('E_selected − ordinary Cox C-index')
    axes[1].set_ylabel('Matched model pairs')
    axes[1].legend(frameon=False)
    fig.savefig(FINAL/'cindex_distribution.png',dpi=180)
    plt.close(fig)
    rows=['| Method | Pooled C-index (95% CI) | 1000 model mean (min–max) |',
          '|---|---:|---:|']
    for key,label in [('ordinary_cox','Ordinary Cox'),('E_selected','E_selected')]:
        r=summary.loc[key];v=df[key]
        rows.append(f'| {label} | {fmt(r.estimate)} ({fmt(r.CI95_lower)}–{fmt(r.CI95_upper)}) | {fmt(v.mean())} ({fmt(v.min())}–{fmt(v.max())}) |')
    d=summary.loc['E_selected_minus_cox']
    rows.append(f'| Difference: E_selected − Cox | {fmt(d.estimate)} ({fmt(d.CI95_lower)}–{fmt(d.CI95_upper)}) | {fmt(delta.mean())} ({fmt(delta.min())}–{fmt(delta.max())}) |')
    auc=['| Model | 365 d AUC (95% CI) | 730 d AUC (95% CI) | 1095 d AUC (95% CI) |',
         '|---|---:|---:|---:|']
    for key in ['ordinary_cox','E_selected']:
        part=[]
        for t in [365,730,1095]:
            z=pooled.loc[pooled.model.eq(key)&pooled.metric.eq(f'AUC_{t}d')].iloc[0]
            part.append(f'{fmt(z.estimate)} ({fmt(z.lower)}–{fmt(z.upper)})')
        auc.append('| '+key+' | '+' | '.join(part)+' |')
    brier=['| Model | 365 d Brier | 730 d Brier | 1095 d Brier |','|---|---:|---:|---:|']
    for key in ['ordinary_cox','E_selected']:
        part=[fmt(probs.loc[probs.model.eq(key)&probs.horizon_days.eq(t),'Brier_IPCW'].iloc[0]) for t in [365,730,1095]]
        brier.append('| '+key+' | '+' | '.join(part)+' |')
    selection=[]
    for r in range(2,11):
        for f in range(1,6):
            p=OUT/f'repeat{r:02d}_fold{f:02d}'/'selected_variables.txt'
            if not p.exists():raise ValueError(f'Missing selection {p}')
            selection.append(dict(repeat_id=r,fold=f,variables=len(p.read_text(encoding='utf8').splitlines())))
    pd.DataFrame(selection).to_csv(FINAL/'selected_variable_counts.csv',index=False)
    note=f'''# Ordinary Cox versus E_selected: 10 × 5 × 20 repeated experiment

Completed at {datetime.now(timezone.utc).astimezone().isoformat()}.

The same 1140 patients, 29 candidate variables, outer partitions and 20 R MICE draws were used for both methods. The first fivefold repeat reuses checked read-only results. For the other 45 outer folds, the fixed Boruta plus LASSO rule was rerun on each development set; no outer test outcomes entered selection. E_selected selected one of its 17 frozen mixture candidates using five internal folds × 20 imputations before predicting each outer fold.

## Outer test C-index

{chr(10).join(rows)}

Paired 1000-model distribution: {int((delta>0).sum())}/1000 pairs favor E_selected; {int((delta>=.02).sum())}/1000 pairs improve by at least 0.02. Model pairs share patients, folds, imputations and selected variables, so they are not independent observations.

For the pooled C-index difference, upstream-style patient-clustered Rubin p = {float(d.p_Rubin):.6g}; conditional patient-clustered normal p without Rubin = {float(d.p_conditional_normal_no_Rubin):.6g}. Rubin 95% CI = {fmt(d.CI95_lower)}–{fmt(d.CI95_upper)}; non-Rubin conditional 95% CI = {fmt(d.conditional_CI95_lower_no_Rubin)}–{fmt(d.conditional_CI95_upper_no_Rubin)}. Both refer to this cohort and the frozen fitted procedures; repeated use of the cohort and prior exploration prevents an independent external-validation claim.

## Time-specific AUC

{chr(10).join(auc)}

## IPCW Brier score

{chr(10).join(brier)}

The Brier estimate uses the outer development set for the censoring distribution. Prediction tables and all per-imputation estimates are retained under `final/`, with the source model outputs in `ordinary_cox/` and `e_selected_sources/`. The chart is `final/cindex_distribution.png`.
'''
    (FINAL/'FINAL_REPORT.md').write_text(note,encoding='utf8')
    print('REPORT_COMPLETE',FINAL/'FINAL_REPORT.md')

if __name__=='__main__':main()

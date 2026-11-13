"""Paired 10-repeat inference and probability metrics on frozen outer predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'results/cox_e_selected_10x5/20261101_v1'
OLD_C=ROOT/'results/ordinary_cox_matched_comparison/20261030_v1'
OLD_E=ROOT/'results/cacox_heterogeneous_blend/20261031_v2_stream'
sys.path.insert(0,str(ROOT/'experiments/cacox_original_head_study/src'))
from cacox_original_head_study import metrics_core as core
from cacox_original_head_study.probability import km,km_at
from cacox_original_head_study.storage import write_json,mark_complete,verify_completed
from cacox_original_head_study.runtime import sha

KEY=['repeat_id','fold','imputation','patient_id']
TIMES=[365,730,1095]
RSCRIPT=Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')


def source(r,f,j,model):
    if model=='ordinary_cox':
        if r==1:
            path=OLD_C/f'repeat01_fold{f:02d}'/'predictions_all20.csv'
            return pd.read_csv(path).query('imputation == @j').copy()
        return pd.read_csv(OUT/'ordinary_cox'/f'repeat{r:02d}_fold{f:02d}'/f'mi{j:02d}'/'prediction.csv')
    if r==1:
        path=OLD_E/'outer'/f'fold{f:02d}'/'shared'/f'mi{j:02d}'/'E_selected.csv'
    else:
        path=OUT/'e_selected_sources'/f'repeat{r:02d}'/'blend'/'outer'/f'fold{f:02d}'/f'mi{j:02d}'/'E_selected.csv'
    return pd.read_csv(path)


def load_all(repeats):
    rows={'ordinary_cox':[],'E_selected':[]}
    for r,f,j in itertools.product(range(1,repeats+1),range(1,6),range(1,21)):
        for model in rows:
            q=source(r,f,j,model)
            if len(q)!=228 or q[KEY].duplicated().any() or not np.isfinite(q[['score',*[f'survival_{t}d' for t in TIMES]]].to_numpy()).all():
                raise ValueError(f'Invalid {model} r{r}f{f}j{j}')
            if set(q.repeat_id)!={r} or set(q.fold)!={f} or set(q.imputation)!={j}:
                raise ValueError(f'Incorrect unit identity {model} r{r}f{f}j{j}')
            for t in TIMES:q[f'risk_{t}d']=1-q[f'survival_{t}d']
            rows[model].append(q)
    frames={model:pd.concat(parts,ignore_index=True) for model,parts in rows.items()}
    a=frames['ordinary_cox'].sort_values(KEY).reset_index(drop=True)
    b=frames['E_selected'].sort_values(KEY).reset_index(drop=True)
    if not a[KEY].equals(b[KEY]) or not np.array_equal(a[['Death','Survival_Days']],b[['Death','Survival_Days']]):
        raise ValueError('Outer test patients/outcomes mismatch between arms')
    if len(a)!=repeats*1140*20:
        raise ValueError('Incomplete repeated predictions')
    return frames


def metric_tables(frames):
    per=[];pooled=[];paired=[]
    for horizon in [None,*TIMES]:
        metric='C_index' if horizon is None else f'AUC_{horizon}d'
        column='score' if horizon is None else f'risk_{horizon}d'
        by_model={}
        for model,frame in frames.items():
            values=[]
            for j in range(1,21):
                q=core.aggregate_metric(frame.loc[frame.imputation.eq(j)],column,horizon)
                per.append(dict(model=model,metric=metric,imputation=j,
                                estimate=q['estimate'],variance=q['variance']))
                values.append(q)
            by_model[model]=values
            pooled.append(dict(model=model,metric=metric,**core.rubin(
                [z['estimate'] for z in values],[z['variance'] for z in values],True)))
        estimates=[];variances=[]
        for a,b in zip(by_model['ordinary_cox'],by_model['E_selected']):
            ids=a['contributions'].keys()|b['contributions'].keys()
            estimates.append(b['estimate']-a['estimate'])
            variances.append(sum((b['contributions'].get(i,0.)-a['contributions'].get(i,0.))**2 for i in ids))
        paired.append(dict(comparison='E_selected_minus_ordinary_cox',metric=metric,
                           **core.rubin(estimates,variances)))
        print('METRIC',metric,'Cox',pooled[-2]['estimate'],'E_selected',pooled[-1]['estimate'],flush=True)
    return pd.DataFrame(per),pd.DataFrame(pooled),pd.DataFrame(paired)


def probability(frames,repeats):
    records=[]
    for r,f in itertools.product(range(1,repeats+1),range(1,6)):
        bpath=(ROOT/'results/cacox_tree_selected29_integrated_20261028_v1/bundles' if r==1 else OUT/'bundles')/f'repeat{r:02d}_fold{f:02d}'/'main'
        dev=pd.read_csv(bpath/'raw_development.csv')
        censor=km(dev.Survival_Days,1-dev.Death)
        for model,frame in frames.items():
            part=frame.loc[frame.repeat_id.eq(r)&frame.fold.eq(f)]
            for j,z in part.groupby('imputation'):
                observed=km(z.Survival_Days,z.Death)
                time=z.Survival_Days.to_numpy(float);event=z.Death.to_numpy(int)
                g_before=km_at(censor,time,True)
                for horizon in TIMES:
                    survival=z[f'survival_{horizon}d'].to_numpy(float)
                    dead=(time<=horizon)&(event==1);alive=time>horizon
                    g=float(km_at(censor,horizon))
                    if g<=0 or np.any(g_before[dead]<=0):raise ValueError(f'No censor support r{r}f{f}j{j}')
                    terms=np.zeros(len(z))
                    terms[dead]=survival[dead]**2/g_before[dead]
                    terms[alive]=(1-survival[alive])**2/g
                    predicted=float((1-survival).mean());actual=float(1-km_at(observed,horizon))
                    records.append(dict(model=model,repeat_id=r,fold=f,imputation=int(j),horizon_days=horizon,
                        Brier_IPCW=float(terms.mean()),predicted_mean_risk=predicted,
                        observed_KM_risk=actual,calibration_mean_gap=predicted-actual))
    return pd.DataFrame(records)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['smoke','final'])
    args=parser.parse_args()
    repeats=1 if args.mode=='smoke' else 10
    if repeats==10:
        for name in ['COX_COMPLETE.json','E_SELECTED_COMPLETE.json','DATA_COMPLETE.json','I20_COMPLETE.json']:
            if not (OUT/name).exists():raise ValueError(f'Missing required completion {name}')
    target=OUT/('validation/first_repeat_metrics' if repeats==1 else 'final')
    target.mkdir(parents=True,exist_ok=True)
    frames=load_all(repeats)
    per,pooled,paired=metric_tables(frames)
    probability_rows=probability(frames,repeats)
    per.to_csv(target/'metrics_by_imputation.csv',index=False)
    pooled.to_csv(target/'metrics_pooled.csv',index=False)
    paired.to_csv(target/'paired_differences.csv',index=False)
    probability_rows.to_csv(target/'probability_by_model.csv',index=False)
    probability_rows.groupby(['model','horizon_days'])[['Brier_IPCW','predicted_mean_risk','observed_KM_risk','calibration_mean_gap']].mean().reset_index().to_csv(target/'probability_summary.csv',index=False)
    if repeats==10:
        with (target/'pvalues_R.log').open('w',encoding='utf8') as log:
            result=subprocess.run([str(RSCRIPT),'--vanilla',str(HERE/'final_pvalues.R'),str(ROOT),str(OUT)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('R upstream paired p-value computation failed; see pvalues_R.log')
        r=pd.read_csv(OUT/'final_cindex_and_pvalues.csv')
        for model in ['ordinary_cox','E_selected']:
            x=r.loc[r.model.eq(model),'estimate'].iloc[0]
            y=pooled.loc[pooled.model.eq(model)&pooled.metric.eq('C_index'),'estimate'].iloc[0]
            if abs(x-y)>1e-10:raise ValueError(f'R/Python C-index mismatch for {model}: {x}, {y}')
        match=frames['ordinary_cox'][KEY].sort_values(KEY).reset_index(drop=True)
        write_json(target/'SUMMARY.json',dict(at=datetime.now(timezone.utc).isoformat(),
            repeats=10,folds=50,imputations=20,models_per_arm=1000,patients=1140,
            paired_rows_per_arm=len(match),R_and_Python_C_agree=True,
            p_scope='Patient-clustered within same cohort and frozen splits/models; Rubin and conditional no-Rubin'))
        mark_complete(target,['metrics_by_imputation.csv','metrics_pooled.csv','paired_differences.csv',
                              'probability_by_model.csv','probability_summary.csv','pvalues_R.log','SUMMARY.json'])
    else:mark_complete(target,['metrics_by_imputation.csv','metrics_pooled.csv','paired_differences.csv',
                               'probability_by_model.csv','probability_summary.csv'])
    print('METRICS_COMPLETE',target,flush=True)

if __name__=='__main__':main()

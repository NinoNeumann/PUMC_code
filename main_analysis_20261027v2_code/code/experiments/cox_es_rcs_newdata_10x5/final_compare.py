"""Paired outer-test summary for the two wholly refitted new-data arms."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'results/cox_es_rcs_newdata_10x5/20261104_v1'
RCS=ROOT/'results/rcs_ridge_newdata_10x5/20261104_v1'
DEST=DATA/'comparison'
KEY=['repeat_id','fold','imputation','patient_id']
TIMES=(365,730,1095)
import sys
sys.path.insert(0,str(ROOT/'experiments/cacox_original_head_study/src'))
from cacox_original_head_study import metrics_core as core
from cacox_original_head_study.probability import km,km_at
from cacox_original_head_study.storage import mark_complete,verify_completed,write_json
from cacox_original_head_study.runtime import sha


def now():return datetime.now(timezone.utc).isoformat()


def wait_for_inputs():
    status=DATA/'COMPARISON_RUN_STATUS.json'
    while not ((DATA/'E_SELECTED_COMPLETE.json').exists() and (RCS/'FIT_COMPLETE.json').exists()):
        for path in [DATA/'RUN_STATUS.json',DATA/'I20_RUN_STATUS.json',
                     DATA/'E_SELECTED_RUN_STATUS.json',RCS/'RUN_STATUS.json']:
            if path.exists():
                try:state=json.loads(path.read_text(encoding='utf8'))
                except (PermissionError,json.JSONDecodeError):continue
                if state.get('status')=='failed':raise RuntimeError(f'Upstream failed: {path}: {state}')
        write_json(status,dict(at=now(),status='waiting_for_both_arms',controller_pid=os.getpid()))
        time.sleep(30)


def load_predictions():
    frames={'CACox_ES':[],'RCS_ridge_Cox':[]}
    for r in range(1,11):
        for f in range(1,6):
            key=f'repeat{r:02d}_fold{f:02d}'
            bundle=DATA/'bundles'/key/'main'
            if not verify_completed(bundle):raise ValueError(f'Incomplete training bundle {key}')
            rcsdir=RCS/'folds'/key
            proof=json.loads((rcsdir/'COMPLETE.json').read_text(encoding='utf8'))
            rcsfile=rcsdir/'predictions_all20.csv'
            if sha(rcsfile)!=proof['prediction_SHA256']:
                raise ValueError(f'RCS prediction changed {key}')
            q=pd.read_csv(rcsfile)
            if len(q)!=4560:raise ValueError(f'RCS fold count {key}')
            q=q.rename(columns={'score_rcs_ridge':'score'})
            frames['RCS_ridge_Cox'].append(q)
            for j in range(1,21):
                e=DATA/'e_selected_sources'/f'repeat{r:02d}'/'blend'/'outer'/f'fold{f:02d}'/f'mi{j:02d}'
                if not verify_completed(e):raise ValueError(f'Incomplete E_selected {key} mi{j:02d}')
                frames['CACox_ES'].append(pd.read_csv(e/'E_selected.csv'))
    result={name:pd.concat(parts,ignore_index=True).sort_values(KEY).reset_index(drop=True)
            for name,parts in frames.items()}
    anchor=result['CACox_ES']
    for name,q in result.items():
        if len(q)!=228000 or q[KEY].duplicated().any() or not q[KEY].equals(anchor[KEY]):
            raise ValueError(f'Incomplete or unpaired predictions for {name}')
        if not np.array_equal(q[['Death','Survival_Days']],anchor[['Death','Survival_Days']]):
            raise ValueError(f'Outcome mismatch for {name}')
        if not np.isfinite(q[['score',*(f'survival_{t}d' for t in TIMES)]].to_numpy()).all():
            raise ValueError(f'Nonfinite predictions for {name}')
        for t in TIMES:q[f'risk_{t}d']=1-q[f'survival_{t}d']
    return result


def metrics(frames):
    per=[];pooled=[];paired=[]
    for horizon in [None,*TIMES]:
        label='C_index' if horizon is None else f'AUC_{horizon}d'
        col='score' if horizon is None else f'risk_{horizon}d'
        estimates={}
        for name,q in frames.items():
            values=[]
            for j in range(1,21):
                z=core.aggregate_metric(q.loc[q.imputation.eq(j)],col,horizon)
                per.append(dict(method=name,metric=label,imputation=j,
                                estimate=z['estimate'],variance=z['variance']))
                values.append(z)
            estimates[name]=values
            pooled.append(dict(method=name,metric=label,**core.rubin(
                [z['estimate'] for z in values],[z['variance'] for z in values],True)))
        differences=[];variances=[]
        for e,c in zip(estimates['CACox_ES'],estimates['RCS_ridge_Cox']):
            ids=e['contributions'].keys()|c['contributions'].keys()
            differences.append(e['estimate']-c['estimate'])
            variances.append(sum((e['contributions'].get(i,0.)-
                                  c['contributions'].get(i,0.))**2 for i in ids))
        z=core.rubin(differences,variances)
        paired.append(dict(comparison='CACox_ES_minus_RCS_ridge_Cox',metric=label,**z,
            p_Rubin=float(2*student_t.sf(abs(z['estimate']/z['SE']),z['df'])),
            p_conditional_normal_no_Rubin=float(2*norm.sf(abs(z['estimate'])/
                np.sqrt(z['within_variance'])))))
        print('METRIC',label,[(x['method'],x['estimate']) for x in pooled if x['metric']==label],flush=True)
    return pd.DataFrame(per),pd.DataFrame(pooled),pd.DataFrame(paired)


def probability(frames):
    rows=[]
    for r in range(1,11):
        for f in range(1,6):
            key=f'repeat{r:02d}_fold{f:02d}'
            dev=pd.read_csv(DATA/'bundles'/key/'main/raw_development.csv')
            censor=km(dev.Survival_Days,1-dev.Death)
            for name,frame in frames.items():
                subset=frame.loc[frame.repeat_id.eq(r)&frame.fold.eq(f)]
                for j,z in subset.groupby('imputation'):
                    observed=km(z.Survival_Days,z.Death)
                    time=z.Survival_Days.to_numpy(float);event=z.Death.to_numpy(int)
                    before=km_at(censor,time,True)
                    for t in TIMES:
                        survival=z[f'survival_{t}d'].to_numpy(float)
                        dead=(time<=t)&(event==1);alive=time>t
                        g=float(km_at(censor,t))
                        if g<=0 or np.any(before[dead]<=0):
                            raise ValueError(f'No censor support {key} mi{j} day{t}')
                        terms=np.zeros(len(z))
                        terms[dead]=survival[dead]**2/before[dead]
                        terms[alive]=(1-survival[alive])**2/g
                        rows.append(dict(method=name,repeat_id=r,fold=f,imputation=int(j),
                            horizon_days=t,Brier_IPCW=float(terms.mean()),
                            predicted_mean_risk=float((1-survival).mean()),
                            observed_KM_risk=float(1-km_at(observed,t))))
    return pd.DataFrame(rows)


def run():
    lock=DATA/'COMPARISON_RUNNING.lock'
    if lock.exists():raise RuntimeError(f'Comparison controller lock exists: {lock}')
    lock.write_text(str(os.getpid()),encoding='ascii')
    try:
        wait_for_inputs()
        write_json(DATA/'COMPARISON_RUN_STATUS.json',dict(at=now(),status='verifying',controller_pid=os.getpid()))
        DEST.mkdir(parents=True,exist_ok=True)
        frames=load_predictions()
        write_json(DATA/'COMPARISON_RUN_STATUS.json',dict(at=now(),status='calculating_metrics',controller_pid=os.getpid()))
        per,pooled,paired=metrics(frames)
        prob=probability(frames)
        per.to_csv(DEST/'metrics_by_imputation.csv',index=False)
        pooled.to_csv(DEST/'metrics_pooled.csv',index=False)
        paired.to_csv(DEST/'paired_differences.csv',index=False)
        prob.to_csv(DEST/'probability_by_model.csv',index=False)
        prob.groupby(['method','horizon_days'])[['Brier_IPCW','predicted_mean_risk','observed_KM_risk']].mean().reset_index().to_csv(DEST/'probability_summary.csv',index=False)
        detail=[]
        for name,frame in frames.items():
            for (r,f,j),q in frame.groupby(['repeat_id','fold','imputation']):
                detail.append(dict(method=name,repeat_id=r,fold=f,imputation=j,
                    C_index=core.aggregate_metric(q,'score',None)['estimate']))
        detail=pd.DataFrame(detail)
        if len(detail)!=2000:raise ValueError('Expected 1000 model C-index values per arm')
        detail.to_csv(DEST/'model_cindex.csv',index=False)
        paired_scores=frames['CACox_ES'][KEY+['Death','Survival_Days']].copy()
        for name,frame in frames.items():paired_scores[f'score_{name}']=frame.score.to_numpy()
        paired_scores.to_csv(DEST/'paired_scores.csv',index=False)
        fig,ax=plt.subplots(figsize=(7,4.4),dpi=160)
        labels=['CACox (ES)','RCS + ridge-Cox']
        ax.boxplot([detail.loc[detail.method.eq(name),'C_index'] for name in frames],
                   tick_labels=labels,showfliers=False)
        ax.set_ylabel('Outer-test Harrell C-index');ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(DEST/'cindex_distribution.png');plt.close(fig)
        c=pooled.loc[pooled.metric.eq('C_index')]
        d=paired.loc[paired.metric.eq('C_index')].iloc[0]
        lines=['# New-data CACox (ES) versus RCS + ridge-Cox','',
               '20261027v2.xlsx; 1140 patients; 29 predictors; 10 repeats × 5 outer folds × 20 R MICE datasets. All 50 folds and all 1000 outer models per arm were newly fitted.','',
               '| Method | Pooled C-index (95% CI) | 1000-model mean (min–max) |',
               '|---|---:|---:|']
        for name,label in [('CACox_ES','CACox (ES)'),('RCS_ridge_Cox','RCS + ridge-Cox')]:
            z=c.loc[c.method.eq(name)].iloc[0];vals=detail.loc[detail.method.eq(name),'C_index']
            lines.append(f'| {label} | {z.estimate:.5f} ({z.lower:.5f}–{z.upper:.5f}) | {vals.mean():.5f} ({vals.min():.5f}–{vals.max():.5f}) |')
        lines += ['',f'Paired C-index difference, CACox (ES) minus RCS: {d.estimate:.5f} ({d.lower:.5f}–{d.upper:.5f}); Rubin p={d.p_Rubin:.6g}; conditional p without Rubin={d.p_conditional_normal_no_Rubin:.6g}.','',
                  'These intervals and p values are conditional on the same cohort, repeated splits and fitted models; this is not an independent external validation.','']
        (DEST/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
        write_json(DEST/'SUMMARY.json',dict(at=now(),repeats=10,folds=50,imputations=20,
            models_per_arm=1000,patients=1140,new_input_SHA256=sha(DATA/'input/20261027v2.xlsx')))
        outputs=['metrics_by_imputation.csv','metrics_pooled.csv','paired_differences.csv',
            'probability_by_model.csv','probability_summary.csv','model_cindex.csv',
            'paired_scores.csv','cindex_distribution.png','FINAL_REPORT.md','SUMMARY.json']
        mark_complete(DEST,outputs)
        if not verify_completed(DEST):raise ValueError('Final comparison output verification failed')
        write_json(DATA/'COMPARISON_RUN_STATUS.json',dict(at=now(),status='complete',controller_pid=os.getpid()))
        print('COMPARISON_COMPLETE',DEST,flush=True)
    except BaseException as error:
        write_json(DATA/'COMPARISON_RUN_STATUS.json',dict(at=now(),status='failed',error=repr(error),controller_pid=os.getpid()))
        raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':run()

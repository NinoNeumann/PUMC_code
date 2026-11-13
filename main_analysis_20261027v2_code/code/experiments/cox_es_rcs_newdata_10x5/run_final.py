"""Wait for both matched arms, verify 50 folds, then run final inference."""
from datetime import datetime,timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cox_e_selected_10x5/20261101_v1'
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/cacox_original_head_study/src'))
from cacox_original_head_study.storage import write_json,mark_complete,verify_completed
from cacox_original_head_study.runtime import sha

def now():return datetime.now(timezone.utc).isoformat()
def status(state,**extra):write_json(OUT/'FINAL_RUN_STATUS.json',dict(at=now(),status=state,controller_pid=os.getpid(),**extra))

def readiness():
    for name in ['DATA_COMPLETE.json','COX_COMPLETE.json','I20_COMPLETE.json','E_SELECTED_COMPLETE.json']:
        if not (OUT/name).exists():return False
    return True

def check_upstream():
    for name in ['RUN_STATUS.json','COX_RUN_STATUS.json','I20_RUN_STATUS.json','E_SELECTED_RUN_STATUS.json']:
        p=OUT/name
        if p.exists():
            s=json.loads(p.read_text(encoding='utf8'))
            if s.get('status')=='failed':raise RuntimeError(f'Upstream failed: {name}: {s}')

def verify():
    old_e=ROOT/'results/cacox_heterogeneous_blend/20261031_v2_stream'
    if not (old_e/'COMPLETE.json').exists() or not (old_e/'FINAL_INTEGRITY.json').exists():
        raise ValueError('Historical E_selected first repeat lacks integrity proof')
    for r,f in itertools.product(range(1,11),range(1,6)):
        fold=OUT/f'repeat{r:02d}_fold{f:02d}'
        if not (fold/'prepared.rds').exists():raise ValueError(f'Missing prepared split {r},{f}')
        if r==1:
            if not (OUT/'ordinary_cox'/f'repeat01_fold{f:02d}'/'REUSED_REFERENCE.json').exists():
                raise ValueError(f'Missing Cox first-repeat reference proof f={f}')
            for j in range(1,21):
                ref=old_e/'outer'/f'fold{f:02d}'/'shared'/f'mi{j:02d}'
                if not verify_completed(ref) or not (ref/'E_selected.csv').exists():
                    raise ValueError(f'Missing E_selected historical reference f={f} j={j}')
            continue
        for name in ['MICE_VALIDATION.json','selected_variables.txt','selection_frequency_29variables.csv']:
            if not (fold/name).exists():raise ValueError(f'Missing {name} r={r} f={f}')
        bundle=OUT/'bundles'/f'repeat{r:02d}_fold{f:02d}'/'main'
        if not verify_completed(bundle):raise ValueError(f'Incomplete bundle r={r} f={f}')
        if not (OUT/'ordinary_cox'/f'repeat{r:02d}_fold{f:02d}'/'FOLD_COMPLETE.json').exists():
            raise ValueError(f'Incomplete Cox fold r={r} f={f}')
        for j in range(1,21):
            cox=OUT/'ordinary_cox'/f'repeat{r:02d}_fold{f:02d}'/f'mi{j:02d}'
            if not (cox/'COMPLETE.txt').exists():raise ValueError(f'Missing Cox r={r} f={f} j={j}')
            i20=OUT/'i20'/f'repeat{r:02d}'/'outer'/f'fold{f:02d}'/f'mi{j:02d}'
            if not verify_completed(i20):raise ValueError(f'Missing I20 r={r} f={f} j={j}')
            e=OUT/'e_selected_sources'/f'repeat{r:02d}'/'blend'/'outer'/f'fold{f:02d}'/f'mi{j:02d}'
            if not verify_completed(e) or not (e/'E_selected.csv').exists():
                raise ValueError(f'Missing E_selected r={r} f={f} j={j}')
        if not (OUT/'e_selected_sources'/f'repeat{r:02d}'/'blend'/'FROZEN_CHOICES.json').exists():
            raise ValueError(f'Missing frozen E_selected choices r={r}')
    return dict(repeats=10,folds=50,imputations=20,outer_models_each=1000,
                first_repeat_reused_each=100,new_models_each=900,
                missing=0)

def main():
    lock=OUT/'FINAL_RUNNING.lock'
    if lock.exists():raise RuntimeError(f'Final controller already launched: {lock}')
    lock.write_text(str(os.getpid()),encoding='ascii')
    try:
        while not readiness():
            check_upstream();status('waiting_for_both_arms')
            time.sleep(30)
        status('verifying_all_units')
        integrity=verify()
        write_json(OUT/'FINAL_INPUT_INTEGRITY.json',dict(at=now(),**integrity))
        final=OUT/'final'
        if not verify_completed(final):
            status('calculating_metrics')
            with (OUT/'final_metrics.out.log').open('a',encoding='utf8') as oo,\
                 (OUT/'final_metrics.err.log').open('a',encoding='utf8') as ee:
                proc=subprocess.run([sys.executable,str(HERE/'final_metrics.py'),'final'],
                    cwd=ROOT,stdout=oo,stderr=ee)
            if proc.returncode:raise RuntimeError('Final metrics failed; see final_metrics.err.log')
        status('rendering_report')
        with (OUT/'report.out.log').open('a',encoding='utf8') as oo,\
             (OUT/'report.err.log').open('a',encoding='utf8') as ee:
            proc=subprocess.run([sys.executable,str(HERE/'render_report.py')],
                cwd=ROOT,stdout=oo,stderr=ee)
        if proc.returncode:raise RuntimeError('Final report failed; see report.err.log')
        products=['final/COMPLETE.json','final/FINAL_REPORT.md','final/cindex_distribution.png',
                  'final/metrics_pooled.csv','final/paired_differences.csv',
                  'final_cindex_and_pvalues.csv','final_model_cindex.csv','FINAL_INPUT_INTEGRITY.json']
        write_json(OUT/'COMPLETE.json',dict(at=now(),**integrity,
            product_SHA256={name:sha(OUT/name) for name in products},
            p_methods=['upstream-style patient-clustered Rubin','conditional normal without Rubin'],
            completed=True))
        status('complete',report=str(final/'FINAL_REPORT.md'))
    except BaseException as error:
        status('failed',error=repr(error))
        raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()

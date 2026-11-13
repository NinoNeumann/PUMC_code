"""Run and audit frozen RCS refits solely to reconstruct 0–1095 d curves."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/ibs_0_3y_20261103_v1'
ARCHIVE=ROOT/'results/rcs_ridge_10x5/20261102_v1/fold_first_cindex_20261103/brier_fold_imputation.csv'
SCRIPT=ROOT/'experiments/rcs_ridge_10x5/ibs_0_3y_refit_fold.R'
RSCRIPT=Path('C:/Users/NinoNeumann/AppData/Local/Programs/R/R-4.4.2/bin/Rscript.exe')
INPUT=ROOT/'results/cox_e_selected_10x5/20261101_v1'

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()

def check(r,f,archive):
    dest=OUT/f'repeat{r:02d}_fold{f:02d}'
    csv=dest/'rcs_ibs.csv'
    z=pd.read_csv(csv)
    if len(z)!=20 or list(z.imputation)!=list(range(1,21)) or set(z.model)!={'rcs_ridge_cox'}:
        raise ValueError(f'RCS unit count r={r} f={f}')
    if not (np.isfinite(z.IBS_0_1095d_daily).all() and z.IBS_0_1095d_daily.between(0,1).all()):
        raise ValueError(f'Invalid RCS IBS r={r} f={f}')
    if z.max_S_3point_diff.max()>1e-7 or z.max_LP_diff.max()>1e-7:
        raise ValueError(f'RCS archived 3-point prediction mismatch r={r} f={f}')
    a=archive.loc[(archive.repeat_id==r)&(archive.fold==f)&archive.model.eq('rcs_ridge_cox')]
    for t in (365,730,1095):
        q=a.loc[a.horizon_days==t].sort_values('imputation')
        if len(q)!=20 or np.max(np.abs(q.Brier_IPCW.to_numpy()-z[f'Brier_{t}d'].to_numpy()))>1e-10:
            raise ValueError(f'RCS Brier archive mismatch r={r} f={f} t={t}')
    return dict(repeat_id=r,fold=f,models=20,grid='daily 0..1095 inclusive',
        script_sha256=sha(SCRIPT),result_sha256=sha(csv),
        max_S_3point_diff=float(z.max_S_3point_diff.max()),
        max_LP_diff=float(z.max_LP_diff.max()),
        max_Brier_archive_diff_lt_1e10=True)

def one(r,f,archive):
    key=f'repeat{r:02d}_fold{f:02d}'
    dest=OUT/key
    done=dest/'R_COMPLETE.json'
    if done.exists():
        saved=json.loads(done.read_text(encoding='utf8'))
        current=check(r,f,archive)
        if current!=saved:raise ValueError(f'Completed RCS IBS changed {key}')
        return key,True
    if not (dest/'rcs_ibs.csv').exists():
        dest.mkdir(parents=True,exist_ok=True)
        call=[str(RSCRIPT),'--vanilla',str(SCRIPT),str(ROOT),str(INPUT),str(dest),key]
        p=subprocess.run(call,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace')
        (dest/'rcs_refit.log').write_text(p.stdout+p.stderr,encoding='utf8')
        if p.returncode:
            raise RuntimeError(f'RCS refit failed {key}: {p.stderr[-1200:]}')
    proof=check(r,f,archive)
    temp=done.with_name(done.name+f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(proof,indent=2),encoding='utf8')
    os.replace(temp,done)
    return key,False

def main():
    archive=pd.read_csv(ARCHIVE)
    start=time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(one,r,f,archive) for r in range(1,11) for f in range(1,6)]
        for future in as_completed(futures):
            key,reused=future.result()
            print('RCS_IBS_REUSED' if reused else 'RCS_IBS_COMPLETE',key,flush=True)
    print('RCS_IBS_ALL_COMPLETE',round(time.time()-start,2),flush=True)

if __name__=='__main__':main()

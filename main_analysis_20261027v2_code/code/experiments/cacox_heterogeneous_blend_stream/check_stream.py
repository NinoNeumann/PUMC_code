"""CPU integer oracle and all cached21 clinical preflight C; no dense GPU call."""
import importlib.util,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('stream21_math_backend',HERE/'backend.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
OUT=b.OUT/'math'
def main():
    if OUT.exists():raise RuntimeError('Preserve stream math proof')
    b.configure();b.rg.ensure_headroom();torch.cuda.reset_peak_memory_stats();rng=np.random.default_rng(3123);x=rng.integers(-2,3,size=(17,20,7)).astype(np.float64);t=np.array([1.,2.,2.,3.,4.,4.,6.]);e=np.array([1,1,0,1,1,0,0],bool)
    i,j=b.pair_indices(t,e);oracle=((2*(x[...,i]>x[...,j])+(x[...,i]==x[...,j])).sum(-1))/(2*len(i));actual=b.CUDA_C([torch.tensor(a,device='cuda') for a in x],t,e);np.testing.assert_array_equal(actual,oracle)
    old=b.ROOT/'results/cacox_heterogeneous_blend/20261031_v1';records=[]
    for f in [1,4]:
        g=b.io.load_group(f,1);q=pd.read_csv(old/f'preflight/fold{f:02d}/grid.csv',float_precision='round_trip');d=old/f'preflight/fold{f:02d}';assert b.m.verify_completed(d)
        with np.load(d/'risks.npz') as z:
            for kind,timekey,eventkey,column in [('train','train_time','train_event','train_C'),('val','val_time','val_event','C_index')]:
                actual=b.CUDA_C([torch.as_tensor(z[c+'_'+kind],device='cuda') for c in b.ORDER],g[timekey],g[eventkey]);expected=np.stack([q[q.candidate.eq(c)].sort_values('imputation')[column].to_numpy() for c in b.ORDER]);np.testing.assert_array_equal(actual,expected)
        records.append(dict(fold=f,candidate_MI_rows=340,all_train_validation_C_difference=0.));b.rg.ensure_headroom()
    peak=torch.cuda.max_memory_allocated()/2**20;OUT.mkdir(parents=True);resource=b.rg.record(OUT/'monitor',True)
    record=dict(at=b.m.stamp(),status='passed',synthetic_ties_and_event_censor_same_time_integer_oracle=True,cached_clinical_preflight_rows=680,checks=records,source_model_fits=0,new_outer_predictions=0,actual_peak_allocated_GPU_MiB=peak,blocks=dict(candidates=1,MI_views=4,pairs=4096),base21_manifest_SHA=b.m.sha(old/'manifest.json'),source_files={p.name:b.m.sha(p) for p in [HERE/'backend.py',Path(__file__)]},resource=resource)
    b.m.write_json(OUT/'MATH_CHECK.json',record);b.m.mark_complete(OUT,['MATH_CHECK.json']);print(json.dumps(record),flush=True)
if __name__=='__main__':main()

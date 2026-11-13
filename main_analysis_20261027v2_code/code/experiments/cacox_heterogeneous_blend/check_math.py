"""Synthetic CUDA/independent scalar oracle;zero clinical data or fitting."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,hashlib
from pathlib import Path
import numpy as np
import torch
from blend import statistics,numpy_score,cuda_score
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'results/cacox_heterogeneous_blend/20261031_v1/math'
def main():
    if OUT.exists():raise RuntimeError('Preserve synthetic evidence')
    torch.use_deterministic_algorithms(True);rng=np.random.default_rng(3122)
    a=rng.normal(size=(20,7));b=rng.normal(2,3,size=(20,7));v=rng.normal(size=(20,5));w=rng.normal(-2,4,size=(20,5));originals=[x.copy() for x in [a,b,v,w]]
    stats=statistics(a,b);checks=[]
    for weight in [0.,.25,.5,.75,1.]:
        expected=np.empty_like(v)
        for m in range(20):
            for n in range(5):
                shifted=(w[m,n]-sum(b[m])/7)/(sum((b[m]-sum(b[m])/7)**2)/7)**.5*(sum((a[m]-sum(a[m])/7)**2)/7)**.5+sum(a[m])/7
                expected[m,n]=v[m,n] if weight==1. else shifted if weight==0. else weight*v[m,n]+(1-weight)*shifted
        cpu=numpy_score(v,w,weight,stats);gpu=cuda_score(v,w,weight,stats).cpu().numpy();error=max(float(abs(cpu-expected).max()),float(abs(gpu-expected).max()));assert error<2e-12
        if weight==1.:np.testing.assert_array_equal(gpu,v)
        if weight==0.:
            for m in range(20):np.testing.assert_array_equal(np.sign(gpu[m,:,None]-gpu[m,None,:]),np.sign(w[m,:,None]-w[m,None,:]))
        # Query values cannot alter the stored training statistics.
        assert stats==statistics(a,b);checks.append(dict(weight=weight,scalar_CPU_CUDA_error=error))
    constant=statistics(np.ones((20,7)),np.zeros((20,7)));assert all(s['control_sd']==s['other_sd']==1e-8 for s in constant)
    assert np.isfinite(cuda_score(v,w,.5,constant).cpu().numpy()).all()
    for x,y in zip([a,b,v,w],originals):np.testing.assert_array_equal(x,y)
    OUT.mkdir(parents=True);record=dict(status='passed',checks=checks,source_control_bitwise_identity=True,other_endpoint_positive_affine_ranking=True,constant_training_SD_guard=True,training_only_statistics=True,input_unchanged=True,clinical_data_read=False,clinical_fit_calls=0,optimizer_steps=0,source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'blend.py',Path(__file__)]})
    p=OUT/'MATH_CHECK.json';p.write_text(json.dumps(record,indent=2),encoding='utf-8');(OUT/'COMPLETE.json').write_text(json.dumps(dict(outputs={'MATH_CHECK.json':hashlib.sha256(p.read_bytes()).hexdigest()}),indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
if __name__=='__main__':main()

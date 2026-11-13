"""CPU independent scalar oracle/gradient and finite-temperature counterexample."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,math,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from losses import MeanRiskSigmoid
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cacox_smooth_concordance_prototype/20261031_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    if OUT.exists():raise RuntimeError('Preserve checks')
    assert not torch.cuda.is_initialized();rng=np.random.default_rng(418);x=rng.normal(size=(3,7,4));t=np.array([1.,2.,2.,2.,4.,5.,7.]);e=np.array([1,1,1,0,0,1,0],bool)
    pairs=[(i,j) for i in range(7) for j in range(7) if e[i] and (t[i]<t[j] or (t[i]==t[j] and not e[j]))];rows=[]
    for tau in [.1,1.]:
        tensor=torch.tensor(x,requires_grad=True);loss=MeanRiskSigmoid(t,e,tau)(tensor);loss.backward();grad=np.zeros_like(x);values=[];score=x.mean(2)
        for mi in range(3):
            for i,j in pairs:
                z=(score[mi,j]-score[mi,i])/tau;p=1/(1+math.exp(-z));values.append(p);g=p*(1-p)/(tau*3*len(pairs)*4);grad[mi,j]+=g;grad[mi,i]-=g
        ve=abs(float(loss)-float(np.mean(values)));ge=float(abs(tensor.grad.numpy()-grad).max());assert ve<1e-14 and ge<1e-14
        shifted=x+rng.normal(size=(3,1,1));assert abs(float(MeanRiskSigmoid(t,e,tau)(torch.tensor(shifted)))-float(loss))<1e-14
        reordered=x[:,:,[3,1,0,2]];assert abs(float(MeanRiskSigmoid(t,e,tau)(torch.tensor(reordered)))-float(loss))<1e-14
        perturb=rng.normal(size=x.shape);perturb-=perturb.mean(2,keepdims=True);assert abs(float(MeanRiskSigmoid(t,e,tau)(torch.tensor(x+perturb)))-float(loss))<1e-14
        rows.append(dict(temperature=tau,value_error=ve,analytic_gradient_error=ge,shift_member_permutation_zeromean_perturbation_invariant=True))
    t3=np.array([1.,2.,3.]);e3=np.ones(3,bool);f=MeanRiskSigmoid(t3,e3,1.);a=np.array([.02,.01,0.]);b=np.array([10.,0.,1.]);pa=np.array([[0,1],[0,2],[1,2]])
    def C(v):return float(np.mean((v[pa[:,0]]>v[pa[:,1]])+.5*(v[pa[:,0]]==v[pa[:,1]])))
    la=float(f(torch.tensor(a[None,:,None])));lb=float(f(torch.tensor(b[None,:,None])));assert la>lb and C(a)>C(b)
    scale=float(f(torch.tensor(100*a[None,:,None])));assert scale!=la and C(a)==C(100*a)
    assert not torch.cuda.is_initialized();OUT.mkdir(parents=True);record=dict(at=datetime.now().astimezone().isoformat(),status='passed',clinical_patients=0,optimizer_steps=0,CUDA_initialized=False,checks=rows,
        counterexample=dict(temperature=1.,scores_A=a.tolist(),scores_B=b.tolist(),C_A=C(a),C_B=C(b),loss_A=la,loss_B=lb,lower_smooth_loss_can_have_lower_discrete_C=True,positive_rescale_loss=scale,positive_rescale_C=C(100*a)),
        interpretation='finite-temperature bounded smooth surrogate is not guaranteed value-monotone in discrete C;nonconvex and can saturate on very wrong pairs;no clinical experiment or efficacy claim',source={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('losses.py')]})
    p=OUT/'MATH_CHECK.json';p.write_text(json.dumps(record,indent=2),encoding='utf-8');(OUT/'COMPLETE.json').write_text(json.dumps(dict(outputs={'MATH_CHECK.json':sha(p)}),indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
if __name__=='__main__':main()

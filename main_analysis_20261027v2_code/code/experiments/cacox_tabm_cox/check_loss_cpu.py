"""Independent mathematical checks before any real TabM-Cox experiment."""
import json
from pathlib import Path
import numpy as np
import torch
from tabm_backend import Efron


def oracle(eta,t,e):
    if eta.ndim==1:eta=eta[:,None]
    ans=[]
    for h in range(eta.shape[1]):
        value=eta[:,h].double();parts=[]
        for tt in np.unique(t[e]):
            death=(t==tt)&e;m=int(death.sum());risk=t>=tt
            for j in range(m):
                a=risk.astype(float)-j/m*death
                live=a>0
                parts.append(torch.logsumexp(value[live]+torch.tensor(np.log(a[live])),0))
        ans.append((torch.stack(parts).sum()-value[e].sum())/int(e.sum()))
    return torch.stack(ans).mean()


def main():
    t=np.array([1,1,1,2,2,3,5,7.]);e=np.array([1,1,0,1,0,0,1,1],bool)
    torch.manual_seed(42);records=[]
    for k in [1,8]:
        for scale in [1.,1000.]:
            eta=(torch.randn(len(t),k,dtype=torch.float64)*scale).requires_grad_();a=Efron(t,e,'cpu')(eta);b=oracle(eta,t,e)
            ga=torch.autograd.grad(a,eta)[0];gb=torch.autograd.grad(b,eta)[0]
            error=float(abs(a-b).detach());grad=float(abs(ga-gb).max());assert error<1e-10 and grad<1e-10
            p=np.array([2,0,1,4,3,7,6,5]);v=Efron(t[p],e[p],'cpu')(eta.detach()[p]);shift=Efron(t,e,'cpu')(eta.detach()+10000)
            assert abs(float(v-a))<1e-10 and abs(float(shift-a))<1e-9
            records.append(dict(members=k,risk_scale=scale,value_error=error,gradient_error=grad,
                                tie_order_error=abs(float(v-a)),shift_error=abs(float(shift-a))))
    dst=Path(__file__).resolve().parents[2]/'results/tabm_cox_compatibility/20261031_v1/EXACT_EFRON_CPU.json'
    if dst.exists():raise RuntimeError('Loss check already recorded')
    dst.write_text(json.dumps(dict(status='passed',checks=records,clinical_fits=0,CUDA_work=False),indent=2),encoding='utf-8')
    print(dst)

if __name__=='__main__':main()

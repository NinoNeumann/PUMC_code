"""Independent synthetic CPU oracle; zero clinical data and zero fitting."""
import json,hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from pair_loss import MeanRiskPairLoss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cacox_rank_objective_prototype/20261031_v1'

def oracle(eta,time,event):
    M,n,K=eta.shape;r=eta.mean(2);total=0.;gradient=np.zeros_like(eta);count=0
    for i in range(n):
        for j in range(n):
            if event[i] and (time[i]<time[j] or (time[i]==time[j] and not event[j])):
                count+=1
                for view in range(M):
                    d=r[view,j]-r[view,i];total+=np.logaddexp(0.,d);prob=1./(1.+np.exp(-d))
                    gradient[view,i,:]-=prob/K;gradient[view,j,:]+=prob/K
    return total/(M*count),gradient/(M*count),count
def main():
    if (OUT/'COMPLETE.json').exists():raise RuntimeError('Already checked')
    torch.set_num_threads(1);rng=np.random.default_rng(619);time=np.array([1.,2.,2.,3.,5.,5.,8.]);event=np.array([1,1,0,0,1,1,0],bool)
    a=rng.normal(size=(3,len(time),8));v=torch.tensor(a,requires_grad=True);loss=MeanRiskPairLoss(time,event)(v);loss.backward();value,gradient,count=oracle(a,time,event)
    value_error=abs(float(loss.detach())-value);gradient_error=float(abs(v.grad.numpy()-gradient).max());assert value_error<1e-14 and gradient_error<1e-14
    # Independent central differences exercise the actual autograd route.
    eps=1e-5;finite=[]
    for index in [(0,0,0),(1,2,4),(2,6,7)]:
        plus=a.copy();minus=a.copy();plus[index]+=eps;minus[index]-=eps;g=(oracle(plus,time,event)[0]-oracle(minus,time,event)[0])/(2*eps)
        finite.append(abs(g-gradient[index]))
    assert max(finite)<1e-10
    perm=np.array([6,2,4,0,5,1,3]);memberperm=np.array([7,3,5,1,0,2,4,6]);shifts=rng.normal(size=(3,1,1))
    assert abs(float(MeanRiskPairLoss(time[perm],event[perm])(torch.tensor(a[:,perm,:])))-value)<1e-14
    assert abs(float(MeanRiskPairLoss(time,event)(torch.tensor(a[[2,0,1]][:,:,memberperm])))-value)<1e-14
    assert abs(float(MeanRiskPairLoss(time,event)(torch.tensor(a+shifts)))-value)<1e-14
    assert abs(gradient.sum(axis=(1,2))).max()<1e-14
    other=a.copy();other[0]*=2;_,g2,_=oracle(other,time,event);np.testing.assert_array_equal(g2[1:],gradient[1:])
    short=MeanRiskPairLoss([1.,2.],[1,0]);good=float(short(torch.tensor([[[2.],[-2.]]])));bad=float(short(torch.tensor([[[-2.],[2.]]])));assert good<bad
    expected_ties=bool(((time[1]==time[2]) and event[1] and not event[2]));assert expected_ties and not bool(event[4] and time[4]<time[5])
    try:MeanRiskPairLoss([1.,2.],[0,0])
    except ValueError:pass
    else:raise AssertionError('No-pair input must be rejected')
    record=dict(at=datetime.now().astimezone().isoformat(),status='passed',scope='synthetic CPU mathematical prototype,not clinical experiment',clinical_patients=0,optimizer_steps=0,CUDA_calls=0,
        synthetic_MI=3,synthetic_patients=7,members=8,comparable_pairs=count,value_error=value_error,analytic_gradient_error=gradient_error,finite_difference_error=max(finite),
        perMI_shift_and_patient_member_MI_permutation_invariant=True,MI_gradients_independent=True,correct_risk_sign=True,actual_training_risk= 'member logits FP64 mean withinMI',
        new_prediction_or_performance_pooling=False,limitations='softplus is a smooth ranking surrogate,not actual C;lower training loss does not guarantee test C improvement;members receive mean-risk gradients',
        source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('pair_loss.py')]})
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'MATH_CHECK.json';path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(outputs={'MATH_CHECK.json':hashlib.sha256(path.read_bytes()).hexdigest()}),indent=2),encoding='utf-8');print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()

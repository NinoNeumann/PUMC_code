"""Synthetic scalar-network oracle and CUDA compatibility; no clinical inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from model import GroupedAdditiveEnsemble,groups_from_metadata
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=ROOT/'results/cacox_additive_prototype/20261031_v1'

def oracle(net,x):
    members=[]
    for k in range(8):
        values=[]
        for j,group in enumerate(net.groups):
            h=F.relu(x[:,group]@net.w1[j,k,:len(group),:]+net.b1[j,k])
            h=F.relu(h@net.w2[j,k]+net.b2[j,k]);values.append(h@net.head[j,k])
        members.append(torch.stack(values,1).sum(1))
    return torch.stack(members,1).unsqueeze(-1)

def main():
    if OUT.exists():raise RuntimeError('Preserve prototype proof')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(3120);groups=[[0],[1,2,3],[4]];metadata=dict(columns=['Age','Antibody=1','Antibody=2','Antibody=3','CRP'],variables=['Age','Antibody','CRP'])
    assert groups_from_metadata(metadata)==groups
    rows=[]
    for width in [8,16]:
        net=GroupedAdditiveEnsemble(5,groups,width,dropout=0.).double();reference=copy.deepcopy(net)
        x=torch.randn(7,5,dtype=torch.float64);original=x.clone();xx=x.clone().requires_grad_();xr=x.clone().requires_grad_()
        a=net(xx);b=oracle(reference,xr);value=float(abs(a-b).max());assert value<1e-12
        probe=torch.randn_like(a);(a*probe).sum().backward();(b*probe).sum().backward();grad=float(abs(xx.grad-xr.grad).max())
        for (name,p),(other,q) in zip(net.named_parameters(),reference.named_parameters()):
            assert name==other and p.grad is not None and q.grad is not None;grad=max(grad,float(abs(p.grad-q.grad).max()))
        assert grad<1e-12;assert torch.count_nonzero(net.w1.grad*(1-net.padding_mask[:,None,:,None]))==0
        torch.testing.assert_close(x,original,rtol=0,atol=0)
        # A variable's contribution never reads any other variable's values.
        with torch.no_grad():
            shifted=x.clone();shifted[:,groups[1]]+=3.
            partial=lambda z,j,k:(F.relu(F.relu(z[:,groups[j]]@net.w1[j,k,:len(groups[j]),:]+net.b1[j,k])@net.w2[j,k]+net.b2[j,k])*net.head[j,k]).sum(1)
            for j in [0,2]:torch.testing.assert_close(partial(x,j,0),partial(shifted,j,0),rtol=0,atol=0)
        cuda=copy.deepcopy(net).float().cuda().eval();cpu=copy.deepcopy(net).float().eval();z=x.float()
        with torch.no_grad():v=cuda(z.cuda()).cpu();c=cpu(z)
        cpu_cuda=float(abs(v-c).max());assert cpu_cuda<2e-6 and v.shape==(7,8,1)
        torch.manual_seed(1020);cuda.train();cuda.dropout=.15;before=copy.deepcopy(cuda.state_dict())
        optim=torch.optim.AdamW(cuda.parameters(),lr=.001,weight_decay=.0003);out=cuda(z.cuda()).squeeze(-1).double().mean(1)
        # Synthetic comparable-pair-like objective exercises the complete graph.
        loss=F.softplus(out[1:]-out[:-1]).mean();loss.backward();torch.nn.utils.clip_grad_norm_(cuda.parameters(),5.,error_if_nonfinite=True);optim.step()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in cuda.parameters())
        cuda.eval()
        with torch.no_grad():first=cuda(z.cuda()).cpu()
        restored=GroupedAdditiveEnsemble(5,groups,width,.15).cuda();restored.load_state_dict(cuda.state_dict());restored.eval()
        with torch.no_grad():torch.testing.assert_close(first,restored(z.cuda()).cpu(),rtol=0,atol=0)
        assert any(not torch.equal(before[name],value) for name,value in cuda.state_dict().items())
        rows.append(dict(width=width,CPU_value_error=value,input_and_parameter_gradient_error=grad,CPU_CUDA_forward_error=cpu_cuda,
            parameters=sum(p.numel() for p in net.parameters()),active_parameters=net.active_parameter_count(),masked_padding_gradient_zero=True,
            all_original_columns_and_variable_groups_retained=True,no_cross_variable_connections=True,CUDA_backward_and_restore=True))
    OUT.mkdir(parents=True);record=dict(status='passed',checks=rows,synthetic_patients=7,clinical_data_read=False,clinical_fit_calls=0,
        synthetic_CUDA_optimizer_steps=2,not_a_clinical_efficacy_result=True,source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'model.py',Path(__file__)]})
    p=OUT/'MATH_CHECK.json';p.write_text(json.dumps(record,indent=2),encoding='utf-8');(OUT/'COMPLETE.json').write_text(json.dumps(dict(outputs={'MATH_CHECK.json':hashlib.sha256(p.read_bytes()).hexdigest()}),indent=2),encoding='utf-8');print(json.dumps(record),flush=True)
if __name__=='__main__':main()

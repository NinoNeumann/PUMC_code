"""Research prototype only:bounded Harrell-pair sigmoid on sameMI mean risk."""
import numpy as np
import torch
class MeanRiskSigmoid:
    def __init__(self,t,e,temperature=1.,device='cpu'):
        t=np.asarray(t);e=np.asarray(e,bool);i,j=np.where(e[:,None]&((t[:,None]<t[None,:])|((t[:,None]==t[None,:])&~e[None,:])))
        if not len(i):raise ValueError('No comparable pairs')
        if not np.isfinite(temperature) or temperature<=0:raise ValueError('Invalid fixed temperature')
        self.left=torch.as_tensor(i,device=device);self.right=torch.as_tensor(j,device=device);self.temperature=float(temperature)
    def __call__(self,eta):
        if eta.ndim!=3:raise ValueError('MI,patient,member tensor required')
        score=eta.double().mean(2)
        return torch.sigmoid((score[:,self.right]-score[:,self.left])/self.temperature).mean()

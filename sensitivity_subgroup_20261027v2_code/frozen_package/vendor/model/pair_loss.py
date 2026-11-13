"""Prototype only: separate-MI mean-member survival-pair logistic objective.

No model, patient sampling, imputation or input transforms. This module is not
registered as a clinical experiment. Positive log risk denotes earlier death.
"""
import numpy as np
import torch
import torch.nn.functional as F

class MeanRiskPairLoss:
    def __init__(self,time,event,device='cpu'):
        time=np.asarray(time,dtype=float);event=np.asarray(event,dtype=bool)
        if time.ndim!=1 or event.shape!=time.shape or not np.isfinite(time).all():raise ValueError('Finite one-dimensional outcomes required')
        left,right=np.where(event[:,None]&((time[:,None]<time[None,:])|((time[:,None]==time[None,:])&~event[None,:])))
        if not len(left):raise ValueError('At least one Harrell-comparable pair required')
        self.left=torch.as_tensor(left,device=device);self.right=torch.as_tensor(right,device=device);self.n=len(time)
    def __call__(self,member_logits):
        if member_logits.ndim!=3 or member_logits.shape[1]!=self.n:raise ValueError('Expected MI x originalpatients x members')
        risk=member_logits.double().mean(2)
        return F.softplus(risk[:,self.right]-risk[:,self.left]).mean()

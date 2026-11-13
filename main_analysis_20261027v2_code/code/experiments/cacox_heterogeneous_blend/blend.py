"""Train-score affine alignment and fixed convex weights; no input transforms."""
import numpy as np
import torch

def statistics(control_train,other_train):
    assert control_train.ndim==other_train.ndim==2 and control_train.shape==other_train.shape and control_train.shape[0]==20
    return [dict(control_mean=float(a.mean()),control_sd=max(float(a.std()),1e-8),other_mean=float(b.mean()),other_sd=max(float(b.std()),1e-8)) for a,b in zip(control_train,other_train)]

def numpy_score(control,other,weight,stats):
    if weight not in [0.,.25,.5,.75,1.]:raise ValueError('Registered finite weight required')
    if weight==1.:return np.asarray(control).copy()
    aligned=np.stack([(b-s['other_mean'])/s['other_sd']*s['control_sd']+s['control_mean'] for b,s in zip(other,stats)])
    return aligned if weight==0. else weight*control+(1-weight)*aligned

def cuda_score(control,other,weight,stats):
    if weight not in [0.,.25,.5,.75,1.]:raise ValueError('Registered finite weight required')
    a=torch.as_tensor(control,dtype=torch.float64,device='cuda');b=torch.as_tensor(other,dtype=torch.float64,device='cuda')
    if weight==1.:return a
    values={key:torch.tensor([s[key] for s in stats],dtype=torch.float64,device='cuda')[:,None] for key in ['control_mean','control_sd','other_mean','other_sd']}
    aligned=(b-values['other_mean'])/values['other_sd']*values['control_sd']+values['control_mean']
    return aligned if weight==0. else weight*a+(1-weight)*aligned

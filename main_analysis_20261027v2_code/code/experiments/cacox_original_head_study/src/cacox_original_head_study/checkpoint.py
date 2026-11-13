"""New checkpoints use weights-only loading; original reference checkpoints remain read-only."""
from pathlib import Path
import os
import numpy as np
import torch
from ._numeric_encoder import Encoder
from .models import make_model
from .time_grid import TimeGrid
from .training import FittedModel


def save_checkpoint(path,fitted,audit):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    baseline=None if fitted.baseline is None else {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in fitted.baseline.items()}
    obj=dict(format_version=1,architecture="original_CrossAttentionCox",state_dict={k:v.detach().cpu() for k,v in fitted.net.state_dict().items()},
             encoder=fitted.encoder.__dict__,spec=fitted.spec,common=fitted.common,
             grid=None if fitted.grid is None else fitted.grid.cuts.tolist(),baseline=baseline,audit=audit)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp");torch.save(obj,temp);os.replace(temp,path)


def load_checkpoint(path,device="cpu"):
    obj=torch.load(path,map_location="cpu",weights_only=True)
    if obj["format_version"]!=1 or obj["architecture"]!="original_CrossAttentionCox":raise ValueError("Wrong checkpoint architecture")
    common=obj["common"]|{"device":device};enc=Encoder(obj["encoder"]["variables"],common["scale"]);enc.__dict__.update(obj["encoder"])
    net=make_model(len(enc.columns),obj["spec"],common,0);net.load_state_dict(obj["state_dict"])
    grid=None if obj["grid"] is None else TimeGrid(np.asarray(obj["grid"],float))
    baseline=None if obj["baseline"] is None else {k:np.asarray(v,float) if isinstance(v,list) else v for k,v in obj["baseline"].items()}
    return FittedModel(net,enc,obj["spec"],common,grid,baseline),obj["audit"]

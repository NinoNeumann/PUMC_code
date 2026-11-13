"""Original backbone, raw encoded predictors, fixed Adam settings and head-specific loss."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from ._numeric_encoder import Encoder
from .models import make_model, feature_digest
from .original_network import cox_ph_loss_sorted
from .losses.discrete import logistic_hazard_nll
from .time_grid import TimeGrid
from .prediction import breslow, survival_from_baseline, discrete_predictions


def original_loss(frame,device):
    time=torch.as_tensor(frame.Survival_Days.to_numpy(float),dtype=torch.float32,device=device)
    order=time.argsort(descending=True)
    event=torch.as_tensor(frame.Death.to_numpy(float),dtype=torch.float32,device=device)[order]
    return lambda score:cox_ph_loss_sorted(score.reshape(-1)[order],event)


@dataclass
class FittedModel:
    net: object
    encoder: Encoder
    spec: dict
    common: dict
    grid: object=None
    baseline: object=None

    def predict(self,frame):
        self.net.eval()
        x=torch.as_tensor(self.encoder.transform(frame),device=next(self.net.parameters()).device)
        with torch.no_grad():out=self.net(x).cpu().numpy().astype(float)
        if self.spec["kind"]=="cox":
            score=out.reshape(-1)
            return score,survival_from_baseline(score,self.baseline,self.common["times"])
        return discrete_predictions(out,self.grid,self.common["times"],self.common["rmst_horizon"])


def fit(development,variables,spec,common,seed,validation=None,fixed_epochs=None,progress=None):
    if validation is None and fixed_epochs is None:raise ValueError("Inner validation or frozen epochs required")
    if fixed_epochs is not None and not 1<=fixed_epochs<=common["max_epochs"]:raise ValueError("Invalid epoch budget")
    device=common["device"];encoder=Encoder(variables,common["scale"])
    x=torch.as_tensor(encoder.fit_transform(development),device=device)
    net=make_model(x.shape[1],spec,common,seed);initial=feature_digest(net)
    grid=None;audit={"train_n":len(development),"train_events":int(development.Death.sum()),
                     "architecture":"original_CrossAttentionCox","spline_branch":False}
    if spec["kind"]=="discrete":
        grid=TimeGrid.fit(development.Survival_Days,development.Death,spec["bins"],common["rmst_horizon"])
        labels=grid.labels(development.Survival_Days,development.Death,device)
        observed=labels[0].sum(0);events=labels[1].sum(0)
        probability=torch.clamp(events/torch.clamp(observed,min=1),1e-5,1-1e-5)
        with torch.no_grad():net.head[-1].bias.copy_(torch.logit(probability))
        train_loss=lambda logits:logistic_hazard_nll(logits,labels)
        audit["train_grid"]=grid.diagnostics(development.Survival_Days,development.Death)
    else:train_loss=original_loss(development,device)
    if validation is not None:
        if set(development.patient_id)&set(validation.patient_id):raise ValueError("Train/validation overlap")
        vx=torch.as_tensor(encoder.transform(validation),device=device)
        if grid is None:validation_loss=original_loss(validation,device)
        else:
            vlabels=grid.labels(validation.Survival_Days,validation.Death,device)
            validation_loss=lambda logits:logistic_hazard_nll(logits,vlabels)
            audit["validation_grid"]=grid.diagnostics(validation.Survival_Days,validation.Death)
    optimizer=torch.optim.Adam(net.parameters(),lr=common["learning_rate"],weight_decay=common["weight_decay"])
    best=float("inf");best_epoch=0;stale=0;state=None;trace=[]
    for epoch in range(1,(fixed_epochs or common["max_epochs"])+1):
        net.train();optimizer.zero_grad(set_to_none=True);loss=train_loss(net(x))
        if not torch.isfinite(loss):raise FloatingPointError("Nonfinite training loss")
        loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),common["gradient_clip"],error_if_nonfinite=True);optimizer.step()
        row={"epoch":epoch,"train_loss":float(loss.detach().cpu())}
        if validation is not None:
            net.eval()
            with torch.no_grad():value=float(validation_loss(net(vx)).cpu())
            if not np.isfinite(value):raise FloatingPointError("Nonfinite validation loss")
            row["validation_loss"]=value
            if value<best:
                best,best_epoch,stale=value,epoch,0
                state={k:v.detach().clone() for k,v in net.state_dict().items()}
            else:stale+=1
        trace.append(row)
        if progress and (epoch==1 or epoch%50==0):progress(dict(epoch=epoch,best_epoch=best_epoch,stale=stale))
        if validation is not None and stale>=common["patience"]:break
    if validation is not None:net.load_state_dict(state)
    net.eval()
    with torch.no_grad():raw=net(x).cpu().numpy().astype(float)
    baseline=breslow(development.Survival_Days,development.Death,raw.reshape(-1)) if grid is None else None
    fitted=FittedModel(net,encoder,dict(spec),dict(common),grid,baseline)
    summary=dict(best_epoch=best_epoch if validation is not None else epoch,stopped_epoch=epoch,seed=seed,
                 initial_feature_sha256=initial,parameters=sum(p.numel() for p in net.parameters()),
                 feature_parameters=sum(p.numel() for n,p in net.named_parameters() if not n.startswith("head.6.")),
                 head_parameters=sum(p.numel() for p in net.head[-1].parameters()),effective_common=dict(common),**audit)
    return fitted,pd.DataFrame(trace),summary

"""Training-censor-distribution IPCW probability diagnostics."""
import json
import numpy as np
import pandas as pd
from .evaluation import evaluate, paired_difference, individual_cindices, assert_complete, assert_paired
from .runtime import stamp, sha
from .storage import write_json, mark_complete

def km(time,event):
    q=pd.DataFrame({"time":np.asarray(time),"event":np.asarray(event)})
    groups=q.groupby("time").agg(n=("event","size"),d=("event","sum"))
    risk=len(q)-groups.n.cumsum().shift(1,fill_value=0)
    return groups.index.to_numpy(float),np.r_[1.,np.cumprod(1-groups.d.to_numpy()/risk.to_numpy())]


def km_at(curve,time,left=False):
    t,s=curve
    return s[np.searchsorted(t,np.asarray(time),side="left" if left else "right")]


def probability_checks(study,frames):
    records=[]
    for fold in range(1,6):
        dev=pd.read_csv(study.bundle_path(fold)/"raw_development.csv")
        censor=km(dev.Survival_Days,1-dev.Death)
        for model,frame in frames.items():
            for j,z in frame[frame.fold.eq(fold)].groupby("imputation"):
                observed=km(z.Survival_Days,z.Death)
                time=z.Survival_Days.to_numpy(float); event=z.Death.to_numpy(int)
                g_before=km_at(censor,time,True)
                for horizon in study.config["common"]["times"]:
                    probability=z[f"survival_{horizon}d"].to_numpy(float)
                    dead=(time<=horizon)&(event==1); alive=time>horizon
                    g=float(km_at(censor,horizon))
                    if g<=0 or np.any(g_before[dead]<=0): raise ValueError("IPCW positivity failure")
                    terms=np.zeros(len(z))
                    terms[dead]=probability[dead]**2/g_before[dead]
                    terms[alive]=(1-probability[alive])**2/g
                    predicted=float((1-probability).mean()); actual=float(1-km_at(observed,horizon))
                    records.append({"model_id":model,"fold":fold,"imputation":int(j),"horizon_days":horizon,
                                    "Brier_IPCW":float(terms.mean()),"predicted_mean_risk":predicted,
                                    "observed_KM_risk":actual,"calibration_mean_gap":predicted-actual})
    return pd.DataFrame(records)



"""Read-only Brier, three-point IBS, DCA, calibration and risk-strata outputs."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auc_roc_fold_first import ROOT, OUT, DEST, E_STUDY, MODELS, TIMES, e_file, rcs_file, sha
sys.path.insert(0,str(ROOT/"experiments/cacox_original_head_study/src"))
from cacox_original_head_study.probability import km,km_at

THRESHOLDS=np.round(np.arange(.01,.301,.01),2)
KEY=["repeat_id","fold","imputation","patient_id"]


def development_file(r:int,f:int)->Path:
    parent=(ROOT/"results/cacox_tree_selected29_integrated_20261028_v1/bundles"
            if r==1 else E_STUDY/"bundles")
    return parent/f"repeat{r:02d}_fold{f:02d}"/"main/raw_development.csv"


def process_fold(r:int,f:int):
    dev=pd.read_csv(development_file(r,f))
    if len(dev)!=912:
        raise ValueError(f"Incomplete development set {r}/{f}")
    censor=km(dev.Survival_Days,1-dev.Death)
    rcs=pd.read_csv(rcs_file(r,f))
    if len(rcs)!=4560 or rcs[KEY].duplicated().any():
        raise ValueError(f"Incomplete RCS predictions {r}/{f}")
    risks=[]
    curves=[]
    for j in range(1,21):
        b=rcs.loc[rcs.imputation.eq(j)].sort_values("patient_id").reset_index(drop=True)
        e=pd.read_csv(e_file(r,f,j)).sort_values("patient_id").reset_index(drop=True)
        if len(b)!=228 or len(e)!=228 or not b[KEY].equals(e[KEY]):
            raise ValueError(f"Unpaired model predictions {r}/{f}/{j}")
        if not np.array_equal(b[["Death","Survival_Days"]],e[["Death","Survival_Days"]]):
            raise ValueError(f"Unpaired outcomes {r}/{f}/{j}")
        q=b[KEY+["Death","Survival_Days"]].copy()
        time=b.Survival_Days.to_numpy(float)
        event=b.Death.to_numpy(int)
        for model,frame in zip(MODELS,(e,b)):
            for horizon in TIMES:
                risk=1-frame[f"survival_{horizon}d"].to_numpy(float)
                if not np.isfinite(risk).all() or np.any((risk<0)|(risk>1)):
                    raise ValueError(f"Invalid {model} risk {r}/{f}/{j}/{horizon}")
                q[f"risk_{model}_{horizon}d"]=risk
        risks.append(q)
        for horizon in TIMES:
            dead=(time<=horizon)&(event==1)
            alive=time>horizon
            g_before=np.asarray(km_at(censor,time,True),float)
            g=float(km_at(censor,horizon))
            if g<=0 or np.any(g_before[dead]<=0):
                raise ValueError(f"No censoring support {r}/{f}/{horizon}")
            case_weight=np.zeros(228,float)
            control_weight=np.zeros(228,float)
            case_weight[dead]=1/g_before[dead]
            control_weight[alive]=1/g
            odds=THRESHOLDS/(1-THRESHOLDS)
            all_net=(case_weight.sum()-odds*control_weight.sum())/228
            for model in MODELS:
                risk=q[f"risk_{model}_{horizon}d"].to_numpy(float)
                selected=(risk[:,None]>=THRESHOLDS[None,:]).astype(float)
                tp=case_weight@selected/228
                fp=control_weight@selected/228
                benefit=tp-odds*fp
                curves.extend(dict(repeat_id=r,fold=f,imputation=j,model=model,
                    horizon_days=horizon,threshold=float(p),net_benefit=float(n),
                    treat_all_net_benefit=float(a),treat_none_net_benefit=0.,
                    selection_fraction=float(s))
                    for p,n,a,s in zip(THRESHOLDS,benefit,all_net,selected.mean(axis=0)))
    return r,f,pd.concat(risks,ignore_index=True),pd.DataFrame(curves)


def survival_at(time,event,horizon):
    curve=km(time,event)
    return float(km_at(curve,horizon))


def run():
    if not (OUT/"final/COMPLETE.json").is_file() or not (DEST/"AUC_ROC_COMPLETE.json").is_file():
        raise ValueError("Full matched comparison and earlier summaries must be complete")
    figures=DEST/"figures"
    figures.mkdir(exist_ok=True)
    source=pd.read_csv(OUT/"final/probability_by_model.csv")
    brier=source.loc[source.model.isin(MODELS)].copy()
    if len(brier)!=6000 or brier.duplicated(["repeat_id","fold","imputation","model","horizon_days"]).any():
        raise ValueError("Incomplete archived Brier scores")
    brier.to_csv(DEST/"brier_fold_imputation.csv",index=False)
    fold_brier=(brier.groupby(["model","repeat_id","fold","horizon_days"],as_index=False)
                .agg(Brier_IPCW=("Brier_IPCW","mean")))
    if len(fold_brier)!=300:
        raise ValueError("Incomplete fold-first Brier results")
    fold_brier.to_csv(DEST/"brier_fold_pooled.csv",index=False)
    overall_brier=(brier.groupby(["model","horizon_days"],as_index=False)
                   .agg(Brier_IPCW=("Brier_IPCW","mean")))
    archived=pd.read_csv(OUT/"final/probability_summary.csv")
    matched=overall_brier.merge(archived[["model","horizon_days","Brier_IPCW"]],
        on=["model","horizon_days"],suffixes=("","_archived"),validate="one_to_one")
    if len(matched)!=6 or np.max(np.abs(matched.Brier_IPCW-matched.Brier_IPCW_archived))>1e-12:
        raise ValueError("Archived Brier score mismatch")
    overall_brier.to_csv(DEST/"brier_overall.csv",index=False)
    wide=brier.pivot(index=["model","repeat_id","fold","imputation"],
                     columns="horizon_days",values="Brier_IPCW")
    ibs=(.25*wide[365]+.5*wide[730]+.25*wide[1095]).rename("IBS_365_1095_3point_approx")
    ibs=ibs.reset_index()
    if len(ibs)!=2000:
        raise ValueError("Incomplete three-point IBS approximation")
    ibs.to_csv(DEST/"ibs_3point_fold_imputation.csv",index=False)
    fold_ibs=ibs.groupby(["model","repeat_id","fold"],as_index=False)["IBS_365_1095_3point_approx"].mean()
    fold_ibs.to_csv(DEST/"ibs_3point_fold_pooled.csv",index=False)
    overall_ibs=ibs.groupby("model",as_index=False)["IBS_365_1095_3point_approx"].mean()
    overall_ibs.to_csv(DEST/"ibs_3point_overall.csv",index=False)

    tasks=[(r,f) for r in range(1,11) for f in range(1,6)]
    results=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(process_fold,r,f):(r,f) for r,f in tasks}
        for future in as_completed(futures):
            results.append(future.result())
            print("FOLD",*futures[future],"DONE",len(results),"/ 50",flush=True)
    results.sort(key=lambda x:(x[0],x[1]))
    pred=pd.concat([x[2] for x in results],ignore_index=True)
    dca=pd.concat([x[3] for x in results],ignore_index=True)
    if len(pred)!=228000 or len(dca)!=180000 or pred.duplicated(KEY).any():
        raise ValueError("Incomplete DCA or patient-level prediction records")
    if not pred.groupby("patient_id").size().eq(200).all():
        raise ValueError("Each patient should have 10 repeats x 20 MI predictions")
    dca.to_csv(DEST/"dca_fold_imputation.csv",index=False)
    fold_dca=(dca.groupby(["model","repeat_id","fold","horizon_days","threshold"],as_index=False)
              .agg(net_benefit=("net_benefit","mean"),
                   treat_all_net_benefit=("treat_all_net_benefit","mean"),
                   selection_fraction=("selection_fraction","mean")))
    fold_dca.to_csv(DEST/"dca_fold_pooled.csv",index=False)
    overall_dca=(fold_dca.groupby(["model","horizon_days","threshold"],as_index=False)
                 .agg(net_benefit=("net_benefit","mean"),
                      treat_all_net_benefit=("treat_all_net_benefit","mean"),
                      selection_fraction=("selection_fraction","mean")))
    overall_dca.to_csv(DEST/"dca_overall.csv",index=False)

    # One out-of-fold averaged risk per original patient. Group boundaries use
    # predictions only; no outcomes or outer-test performance select cutoffs.
    outcome=pred.groupby("patient_id").agg(Death=("Death","first"),
                                         Survival_Days=("Survival_Days","first"))
    if (pred.groupby("patient_id").Death.nunique()>1).any() or \
       (pred.groupby("patient_id").Survival_Days.nunique()>1).any():
        raise ValueError("Patient outcomes vary across repeats")
    score_columns=[f"risk_{model}_{t}d" for model in MODELS for t in TIMES]
    patient=outcome.join(pred.groupby("patient_id")[score_columns].mean())
    patient.to_csv(DEST/"patient_out_of_fold_risk.csv")

    calib_rows=[]
    for model in MODELS:
        for horizon in TIMES:
            risk=patient[f"risk_{model}_{horizon}d"]
            quintile=pd.qcut(risk,5,labels=False,duplicates="raise")+1
            for bin_id in range(1,6):
                z=patient.loc[quintile.eq(bin_id)]
                calib_rows.append(dict(model=model,horizon_days=horizon,quintile=bin_id,
                    patients=len(z),predicted_survival=float(1-risk.loc[z.index].mean()),
                    observed_KM_survival=survival_at(z.Survival_Days,z.Death,horizon),
                    observed_events=int(z.Death.sum()),
                    min_predicted_risk=float(risk.loc[z.index].min()),
                    max_predicted_risk=float(risk.loc[z.index].max())))
    calibration=pd.DataFrame(calib_rows)
    if len(calibration)!=30 or not calibration.groupby(["model","horizon_days"]).patients.sum().eq(1140).all():
        raise ValueError("Incomplete survival calibration groups")
    calibration.to_csv(DEST/"survival_calibration.csv",index=False)

    strata=[]
    km_rows=[]
    colors=("#218b64","#e6a342","#bb3e39")
    for model in MODELS:
        risk=patient[f"risk_{model}_1095d"]
        group=pd.qcut(risk,3,labels=["Low","Intermediate","High"],duplicates="raise")
        for label in ("Low","Intermediate","High"):
            z=patient.loc[group.eq(label)]
            t,s=km(z.Survival_Days,z.Death)
            keep=t<=1095
            times=np.r_[0.,t[keep]]
            values=np.r_[1.,s[1:][keep]]
            if times[-1]<1095:
                times=np.r_[times,1095.]
                values=np.r_[values,values[-1]]
            strata.append(dict(model=model,stratum=label,patients=len(z),
                events=int(z.Death.sum()),min_1095d_risk=float(risk.loc[z.index].min()),
                max_1095d_risk=float(risk.loc[z.index].max()),
                mean_1095d_risk=float(risk.loc[z.index].mean()),
                KM_survival_365d=survival_at(z.Survival_Days,z.Death,365),
                KM_survival_730d=survival_at(z.Survival_Days,z.Death,730),
                KM_survival_1095d=survival_at(z.Survival_Days,z.Death,1095)))
            km_rows.extend(dict(model=model,stratum=label,time_days=float(x),
                                KM_survival=float(y)) for x,y in zip(times,values))
    strata=pd.DataFrame(strata)
    strata.to_csv(DEST/"risk_strata_summary.csv",index=False)
    km_rows=pd.DataFrame(km_rows)
    km_rows.to_csv(DEST/"risk_strata_km_points.csv",index=False)

    names={"E_selected":"E_selected","rcs_ridge_cox":"RCS + ridge-Cox"}
    model_colors={"E_selected":"#1765b3","rcs_ridge_cox":"#d47a15"}
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True,layout="constrained")
    for ax,horizon in zip(axes,TIMES):
        ax.plot([0,1],[0,1],color="gray",linestyle="--",linewidth=1)
        for model in MODELS:
            z=calibration.loc[calibration.model.eq(model)&calibration.horizon_days.eq(horizon)]
            ax.plot(z.predicted_survival,z.observed_KM_survival,marker="o",
                    color=model_colors[model],label=names[model])
        ax.set(title=f"{horizon} days",xlabel="Mean predicted survival",xlim=(0.5,1),ylim=(0.5,1))
        ax.grid(alpha=.2);ax.legend(loc="lower right",fontsize=8,frameon=False)
    axes[0].set_ylabel("Observed KM survival")
    fig.suptitle("Survival calibration: five prediction-based groups per method")
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"survival_calibration.{suffix}",dpi=220)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8,5.2),layout="constrained")
    for model in MODELS:
        z=overall_brier.loc[overall_brier.model.eq(model)].sort_values("horizon_days")
        ax.plot(z.horizon_days,z.Brier_IPCW,marker="o",linewidth=2,
                color=model_colors[model],label=names[model])
    ax.set(xlabel="Time since baseline (days)",ylabel="IPCW Brier score (lower is better)",
           title="Outer-test Brier scores at the three saved horizons",xticks=TIMES)
    ax.grid(alpha=.25);ax.legend(frameon=False)
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"brier_by_time.{suffix}",dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.5),sharex=True,sharey=True,layout="constrained")
    for ax,horizon in zip(axes,TIMES):
        for model in MODELS:
            z=overall_dca.loc[overall_dca.model.eq(model)&overall_dca.horizon_days.eq(horizon)]
            ax.plot(z.threshold,z.net_benefit,color=model_colors[model],linewidth=2,label=names[model])
        base=overall_dca.loc[overall_dca.model.eq(MODELS[0])&overall_dca.horizon_days.eq(horizon)]
        ax.plot(base.threshold,base.treat_all_net_benefit,color="#666666",linestyle="--",label="Treat all")
        ax.axhline(0,color="black",linewidth=1,label="Treat none")
        ax.set(title=f"{horizon} days",xlabel="Threshold event probability",xlim=(.01,.30))
        ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False)
    axes[0].set_ylabel("IPCW net benefit")
    fig.suptitle("Exploratory decision curves: 1%–30% thresholds")
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"dca_3_horizons.{suffix}",dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharex=True,sharey=True,layout="constrained")
    for ax,model in zip(axes,MODELS):
        for label,color in zip(("Low","Intermediate","High"),colors):
            z=km_rows.loc[km_rows.model.eq(model)&km_rows.stratum.eq(label)]
            ax.step(z.time_days,z.KM_survival,where="post",color=color,linewidth=2,label=label)
        ax.set(title=names[model],xlabel="Time since baseline (days)",xlim=(0,1095),ylim=(.5,1.01))
        ax.grid(alpha=.2);ax.legend(frameon=False)
    axes[0].set_ylabel("Kaplan–Meier survival")
    fig.suptitle("Risk strata from averaged out-of-fold 1095-day predictions (n=1140)")
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"risk_stratification_km.{suffix}",dpi=220)
    plt.close(fig)

    summary={"at_utc":datetime.now(timezone.utc).isoformat(),
        "models":list(MODELS),"folds":50,"imputations_per_fold":20,
        "brier_source_reconciled":True,
        "IBS_definition":"trapezoidal mean of IPCW Brier at 365,730,1095d over [365,1095]; three-point approximation, not dense-grid IBS",
        "DCA_definition":"outer-test IPCW event/control counts; censoring KM fitted on outer development set; thresholds 0.01 to 0.30 exploratory",
        "calibration_definition":"1140 unique patients; mean of 200 out-of-fold predictions per patient; method-specific prediction quintiles; observed KM survival",
        "risk_strata_definition":"method-specific tertiles of averaged out-of-fold 1095d risk; descriptive KM, no outcome-based threshold tuning",
        "source_sha256":{str(p.relative_to(ROOT)):sha(p) for p in
            (OUT/"final/probability_by_model.csv",OUT/"final/COMPLETE.json")},
        "outputs":{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob("*"))
                   if p.is_file() and p.name!="SUPPLEMENT_COMPLETE.json"}}
    (DEST/"SUPPLEMENT_COMPLETE.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
    print("COMPLETE Brier, approximate IBS, DCA, survival calibration and risk-stratified KM")
    print(overall_brier.to_string(index=False))
    print(overall_ibs.to_string(index=False))


if __name__=="__main__":
    run()

"""Read-only fold-first AUC and time-dependent ROC audit of the completed study."""
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
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rcs_ridge_10x5/20261102_v1"
DEST = OUT / "fold_first_cindex_20261103"
OLD_RCS = ROOT / "results/cacox_vs_rcs_selected29_optimization_20261028_v1"
OLD_E = ROOT / "results/cacox_heterogeneous_blend/20261031_v2_stream"
E_STUDY = ROOT / "results/cox_e_selected_10x5/20261101_v1"
sys.path.insert(0, str(ROOT / "experiments/cacox_original_head_study/src"))
from cacox_original_head_study import metrics_core as core

TIMES = (365, 730, 1095)
MODELS = ("E_selected", "rcs_ridge_cox")
FPR = np.linspace(0, 1, 201)
KEY = ["repeat_id", "fold", "imputation", "patient_id"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def rcs_file(r: int, f: int) -> Path:
    key = f"repeat{r:02d}_fold{f:02d}"
    return ((OLD_RCS / "matched_rcs_ridge" / key / "predictions_all20.csv")
            if r == 1 else (OUT / "folds" / key / "predictions_all20.csv"))


def e_file(r: int, f: int, j: int) -> Path:
    if r == 1:
        return OLD_E / "outer" / f"fold{f:02d}" / "shared" / f"mi{j:02d}" / "E_selected.csv"
    return (E_STUDY / "e_selected_sources" / f"repeat{r:02d}" / "blend" /
            "outer" / f"fold{f:02d}" / f"mi{j:02d}" / "E_selected.csv")


def calculate_fold(r: int, f: int):
    rcs = pd.read_csv(rcs_file(r, f))
    if len(rcs) != 4560 or rcs[KEY].duplicated().any():
        raise ValueError(f"Incomplete RCS fold {r}/{f}")
    rows = []
    curves = np.empty((20, 2, 3, 201), dtype=float)
    influences = np.empty((20, 2, 3, 228), dtype=float)
    ids = np.empty((20, 228), dtype=int)
    for j in range(1, 21):
        b = rcs.loc[rcs.imputation.eq(j)].sort_values("patient_id").reset_index(drop=True)
        e = pd.read_csv(e_file(r, f, j)).sort_values("patient_id").reset_index(drop=True)
        if len(b) != 228 or len(e) != 228 or not b[KEY].equals(e[KEY]):
            raise ValueError(f"Unmatched patients {r}/{f}/{j}")
        if not np.array_equal(b[["Death", "Survival_Days"]], e[["Death", "Survival_Days"]]):
            raise ValueError(f"Unmatched outcomes {r}/{f}/{j}")
        ids[j-1] = b.patient_id.to_numpy(int)
        time = b.Survival_Days.to_numpy(float)
        event = b.Death.to_numpy(int)
        for m, frame in enumerate((e, b)):
            for k, horizon in enumerate(TIMES):
                risk = 1 - frame[f"survival_{horizon}d"].to_numpy(float)
                if not np.isfinite(risk).all() or np.any((risk < 0) | (risk > 1)):
                    raise ValueError(f"Invalid risk {r}/{f}/{j}/{MODELS[m]}/{horizon}")
                z = core.ipcw_auc(time, event, risk, horizon)
                curves[j-1, m, k] = z["roc"]
                influences[j-1, m, k] = z["influence"]
                rows.append(dict(repeat_id=r, fold=f, imputation=j,
                                 model=MODELS[m], horizon_days=horizon,
                                 AUC=z["estimate"], within_variance=z["variance"],
                                 cases=z["cases"], controls=z["controls"]))
    return r, f, pd.DataFrame(rows), curves, influences, ids


def summary_from_fold(z: pd.DataFrame):
    d = {}
    for m in MODELS:
        x = z.loc[z.model.eq(m)].sort_values("imputation")
        if len(x) != 20 or x.imputation.tolist() != list(range(1, 21)):
            raise ValueError("Missing fold AUC imputation")
        d[m] = core.rubin(x.AUC, x.within_variance, bounded=True)
    return d


def run():
    if not (OUT / "final/COMPLETE.json").is_file() or not (E_STUDY / "COMPLETE.json").is_file():
        raise ValueError("Matched full experiments are incomplete")
    DEST.mkdir(parents=True, exist_ok=True)
    figures = DEST / "figures"
    figures.mkdir(exist_ok=True)
    work = [(r, f) for r in range(1, 11) for f in range(1, 6)]
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(calculate_fold, r, f): (r, f) for r, f in work}
        for future in as_completed(futures):
            results.append(future.result())
            print("FOLD", *futures[future], "DONE", len(results), "/ 50", flush=True)
    results.sort(key=lambda x: (x[0], x[1]))
    per = pd.concat([x[2] for x in results], ignore_index=True)
    if len(per) != 6000:
        raise ValueError("Expected 6000 model/horizon/imputation AUC records")
    all_curves = np.stack([x[3] for x in results])
    patient_ids = np.stack([x[5] for x in results])
    if all_curves.shape != (50, 20, 2, 3, 201):
        raise ValueError("Unexpected ROC coverage")

    # Pairing and within-imputation variance for each fold-specific difference.
    fold_rows = []
    for idx, (r, f, q, _, inf, _) in enumerate(results):
        for k, horizon in enumerate(TIMES):
            z = q.loc[q.horizon_days.eq(horizon)]
            by_model = summary_from_fold(z)
            e = z.loc[z.model.eq("E_selected")].sort_values("imputation")
            b = z.loc[z.model.eq("rcs_ridge_cox")].sort_values("imputation")
            delta = e.AUC.to_numpy() - b.AUC.to_numpy()
            variance = np.sum((inf[:, 0, k] - inf[:, 1, k])**2, axis=1)
            diff = core.rubin(delta, variance)
            p = 2*student_t.sf(abs(diff["estimate"]/diff["SE"]), diff["df"])
            fold_rows.append(dict(repeat_id=r, fold=f, horizon_days=horizon,
                imputations=20,
                AUC_E_selected=by_model["E_selected"]["estimate"],
                E_CI95_lower=by_model["E_selected"]["lower"],
                E_CI95_upper=by_model["E_selected"]["upper"],
                AUC_rcs_ridge_cox=by_model["rcs_ridge_cox"]["estimate"],
                RCS_CI95_lower=by_model["rcs_ridge_cox"]["lower"],
                RCS_CI95_upper=by_model["rcs_ridge_cox"]["upper"],
                difference_E_minus_RCS=diff["estimate"],
                difference_CI95_lower=diff["lower"],
                difference_CI95_upper=diff["upper"],
                fold_Rubin_p=p,
                cases=int(e.cases.iloc[0]),controls=int(e.controls.iloc[0])))
    folds = pd.DataFrame(fold_rows)

    # Reconstruct archived overall AUC and paired variance from patient-level
    # influences. Repeated test appearances are clustered by original patient.
    source_ids = np.sort(pd.read_csv(OUT / "final/paired_scores.csv",usecols=["patient_id"]).patient_id.unique())
    if len(source_ids) != 1140:
        raise ValueError("Unexpected cohort patient count")
    estimates = np.empty((20, 2, 3))
    variances = np.empty((20, 2, 3))
    diff_variances = np.empty((20, 3))
    for j in range(20):
        infl = np.zeros((2, 3, 1140), dtype=float)
        for _, _, _, _, by_patient, ids in results:
            positions = np.searchsorted(source_ids, ids[j])
            if not np.array_equal(source_ids[positions],ids[j]):
                raise ValueError("Patient identity mismatch")
            infl[:, :, positions] += by_patient[j]/50
        for m, model in enumerate(MODELS):
            for k, horizon in enumerate(TIMES):
                estimates[j,m,k] = per.loc[(per.imputation.eq(j+1)) &
                    (per.model.eq(model)) & (per.horizon_days.eq(horizon)),"AUC"].mean()
                variances[j,m,k] = np.sum(infl[m,k]**2)
        diff_variances[j] = np.sum((infl[0]-infl[1])**2,axis=1)

    archived = pd.read_csv(OUT / "final/metrics_pooled.csv")
    archived_mi = pd.read_csv(OUT / "final/metrics_by_imputation.csv")
    archived_diff = pd.read_csv(OUT / "final/paired_differences.csv")
    overall_rows = []
    for k, horizon in enumerate(TIMES):
        for m, model in enumerate(MODELS):
            q = core.rubin(estimates[:,m,k],variances[:,m,k],bounded=True)
            old = archived.loc[archived.model.eq(model) & archived.metric.eq(f"AUC_{horizon}d")].iloc[0]
            if max(abs(q[key]-old[key]) for key in ("estimate","SE","lower","upper")) > 1e-10:
                raise ValueError(f"Overall AUC mismatch: {model} {horizon}")
            old_mi = archived_mi.loc[archived_mi.model.eq(model) &
                archived_mi.metric.eq(f"AUC_{horizon}d")].sort_values("imputation")
            if len(old_mi)!=20 or np.max(np.abs(old_mi.estimate.to_numpy()-estimates[:,m,k]))>1e-10:
                raise ValueError(f"Per-MI AUC mismatch: {model} {horizon}")
            overall_rows.append(dict(horizon_days=horizon,model=model,AUC=q["estimate"],
                CI95_lower=q["lower"],CI95_upper=q["upper"],SE=q["SE"]))
        difference = core.rubin(estimates[:,0,k]-estimates[:,1,k],diff_variances[:,k])
        old = archived_diff.loc[archived_diff.comparison.eq("rcs_ridge_cox_minus_E_selected") &
            archived_diff.metric.eq(f"AUC_{horizon}d")].iloc[0]
        if abs(difference["estimate"]+old.estimate)>1e-10 or abs(difference["SE"]-old.SE)>1e-10:
            raise ValueError(f"Overall paired AUC mismatch: {horizon}")
        overall_rows.append(dict(horizon_days=horizon,model="E_selected_minus_RCS",
            AUC=difference["estimate"],CI95_lower=difference["lower"],
            CI95_upper=difference["upper"],SE=difference["SE"]))
    overall = pd.DataFrame(overall_rows)

    mean_curves = all_curves.mean(axis=(0,1))
    fold_curves = all_curves.mean(axis=1)
    roc_rows = []
    fold_roc_rows = []
    checks = []
    for m, model in enumerate(MODELS):
        for k, horizon in enumerate(TIMES):
            curve = mean_curves[m,k]
            auc_from_curve = float(np.trapezoid(curve,FPR))
            auc_from_estimates = float(overall.loc[(overall.model.eq(model)) &
                (overall.horizon_days.eq(horizon)),"AUC"].iloc[0])
            checks.append(dict(model=model,horizon_days=horizon,
                reported_AUC=auc_from_estimates,integral_of_201_point_mean_ROC=auc_from_curve,
                absolute_difference=abs(auc_from_estimates-auc_from_curve)))
            roc_rows.extend(dict(model=model,horizon_days=horizon,FPR=float(x),TPR=float(y))
                            for x,y in zip(FPR,curve))
            for idx,(r,f,*_) in enumerate(results):
                fold_roc_rows.extend(dict(repeat_id=r,fold=f,model=model,horizon_days=horizon,
                    FPR=float(x),TPR=float(y)) for x,y in zip(FPR,fold_curves[idx,m,k]))
    checks = pd.DataFrame(checks)
    if checks.absolute_difference.max()>0.005:
        raise ValueError("The 201-point interpolated ROC does not reconcile with AUC")

    per.to_csv(DEST/"auc_fold_imputation.csv",index=False)
    folds.to_csv(DEST/"auc_fold_pooled.csv",index=False)
    overall.to_csv(DEST/"auc_overall.csv",index=False)
    pd.DataFrame(roc_rows).to_csv(DEST/"roc_pooled_points.csv",index=False)
    pd.DataFrame(fold_roc_rows).to_csv(DEST/"roc_fold_pooled_points.csv",index=False)
    checks.to_csv(DEST/"roc_auc_reconciliation.csv",index=False)
    np.savez_compressed(DEST/"roc_all_units.npz",TPR=all_curves,FPR=FPR,
        repeat_id=np.array([x[0] for x in results]),
        fold=np.array([x[1] for x in results]),
        model=np.array(MODELS),horizon_days=np.array(TIMES))

    colors={"E_selected":"#1765b3","rcs_ridge_cox":"#d47a15"}
    names={"E_selected":"E_selected","rcs_ridge_cox":"RCS + ridge-Cox"}
    fig,ax=plt.subplots(figsize=(8,5.2),layout="constrained")
    for model in MODELS:
        x=overall.loc[overall.model.eq(model)].sort_values("horizon_days")
        ax.errorbar(x.horizon_days,x.AUC,
            yerr=[x.AUC-x.CI95_lower,x.CI95_upper-x.AUC],
            marker="o",capsize=4,linewidth=2,color=colors[model],label=names[model])
    ax.set(xlabel="Time since baseline (days)",ylabel="Time-dependent AUC",
           title="Outer-test AUC across 10 repeats x 5 folds x 20 imputations",
           xticks=TIMES,ylim=(0.65,0.86))
    ax.grid(alpha=.25);ax.legend(frameon=False)
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"auc_by_time.{suffix}",dpi=220)
    plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharex=True,sharey=True,layout="constrained")
    for k,(ax,horizon) in enumerate(zip(axes,TIMES)):
        ax.plot([0,1],[0,1],color="gray",linestyle="--",linewidth=1)
        for m,model in enumerate(MODELS):
            value=overall.loc[(overall.model.eq(model)) &
                (overall.horizon_days.eq(horizon)),"AUC"].iloc[0]
            ax.plot(FPR,mean_curves[m,k],color=colors[model],linewidth=2,
                    label=f"{names[model]} (AUC {value:.3f})")
        ax.set(title=f"{horizon} days",xlabel="False positive rate",xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.2);ax.legend(loc="lower right",fontsize=8,frameon=False)
    axes[0].set_ylabel("True positive rate (IPCW)")
    fig.suptitle("Outer-test time-dependent ROC: mean of 50 folds x 20 imputations")
    for suffix in ("png","pdf"):
        fig.savefig(figures/f"roc_3_horizons.{suffix}",dpi=220)
    plt.close(fig)

    manifest={"at_utc":datetime.now(timezone.utc).isoformat(),
        "source":str(OUT/"final"),"folds":50,"imputations_per_fold":20,
        "methods":list(MODELS),"horizons_days":list(TIMES),
        "max_roc_auc_201_grid_difference":float(checks.absolute_difference.max()),
        "archived_auc_and_paired_difference_reconciled":True,
        "outputs":{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob("*"))
                   if p.is_file() and p.name!="AUC_ROC_COMPLETE.json"}}
    (DEST/"AUC_ROC_COMPLETE.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("COMPLETE: 50 fold-level AUC summaries and 3 paired ROC curves")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    run()

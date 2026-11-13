"""Validate and fingerprint the read-only final evaluation products."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from auc_roc_fold_first import OUT, DEST, MODELS, TIMES, sha


def main():
    original=json.loads((OUT/"final/COMPLETE.json").read_text(encoding="utf8"))
    for name,digest in original["outputs"].items():
        if sha(OUT/"final"/name)!=digest:
            raise ValueError(f"Archived original output changed: {name}")
    expected={
        "fold_imputation_cindex.csv":1000,
        "fold_pooled_cindex.csv":50,
        "auc_fold_imputation.csv":6000,
        "auc_fold_pooled.csv":150,
        "auc_overall.csv":9,
        "roc_pooled_points.csv":1206,
        "roc_fold_pooled_points.csv":60300,
        "roc_auc_reconciliation.csv":6,
        "brier_fold_imputation.csv":6000,
        "brier_fold_pooled.csv":300,
        "brier_overall.csv":6,
        "ibs_3point_fold_imputation.csv":2000,
        "ibs_3point_fold_pooled.csv":100,
        "ibs_3point_overall.csv":2,
        "dca_fold_imputation.csv":180000,
        "dca_fold_pooled.csv":9000,
        "dca_overall.csv":180,
        "patient_out_of_fold_risk.csv":1140,
        "survival_calibration.csv":30,
        "risk_strata_summary.csv":6,
    }
    for name,rows in expected.items():
        if len(pd.read_csv(DEST/name))!=rows:
            raise ValueError(f"Unexpected rows: {name}")
    auc=pd.read_csv(DEST/"auc_fold_imputation.csv")
    if not auc.groupby(["model","horizon_days","repeat_id","fold"]).size().eq(20).all():
        raise ValueError("Incomplete AUC MI coverage")
    check=pd.read_csv(DEST/"roc_auc_reconciliation.csv")
    if check.absolute_difference.max()>.00025:
        raise ValueError("ROC/AUC interpolation mismatch")
    brier=pd.read_csv(DEST/"brier_overall.csv")
    prior=pd.read_csv(OUT/"final/probability_summary.csv")
    z=brier.merge(prior,on=["model","horizon_days"],suffixes=("","_old"))
    if len(z)!=6 or np.max(np.abs(z.Brier_IPCW-z.Brier_IPCW_old))>1e-12:
        raise ValueError("Brier mismatch")
    ibs=pd.read_csv(DEST/"ibs_3point_overall.csv").set_index("model")
    for model in MODELS:
        q=brier.loc[brier.model.eq(model)].set_index("horizon_days").Brier_IPCW
        expected_ibs=.25*q[365]+.5*q[730]+.25*q[1095]
        if abs(expected_ibs-ibs.loc[model,"IBS_365_1095_3point_approx"])>1e-12:
            raise ValueError("Approximate IBS formula mismatch")
    strata=pd.read_csv(DEST/"risk_strata_summary.csv")
    if not strata.groupby("model").patients.sum().eq(1140).all():
        raise ValueError("Risk strata patient counts")
    dca=pd.read_csv(DEST/"dca_overall.csv")
    if set(dca.horizon_days)!=set(TIMES) or not dca.groupby(["model","horizon_days"]).size().eq(30).all():
        raise ValueError("Incomplete DCA thresholds")
    expected_figs=[f"figures/{name}.{suffix}" for name in
        ("auc_by_time","roc_3_horizons","brier_by_time",
         "survival_calibration","dca_3_horizons","risk_stratification_km")
        for suffix in ("png","pdf")]
    if any(not (DEST/p).is_file() for p in expected_figs):
        raise ValueError("Missing figure")
    completion=("AUC_ROC_COMPLETE.json","SUPPLEMENT_COMPLETE.json","FINAL_INTEGRITY.json")
    auc_names=["fold_imputation_cindex.csv","fold_pooled_cindex.csv","overall_comparison.csv",
               "auc_fold_imputation.csv","auc_fold_pooled.csv","auc_overall.csv",
               "roc_all_units.npz","roc_auc_reconciliation.csv",
               "roc_fold_pooled_points.csv","roc_pooled_points.csv",
               *[p for p in expected_figs if "auc_by_time" in p or "roc_3_horizons" in p]]
    supplement_names=[*expected.keys(),"dca_overall.csv","risk_strata_km_points.csv",
                      *[p for p in expected_figs if not ("auc_by_time" in p or "roc_3_horizons" in p)]]
    for marker,names in ((completion[0],auc_names),(completion[1],supplement_names)):
        path=DEST/marker
        record=json.loads(path.read_text(encoding="utf8"))
        record["outputs"]={name:sha(DEST/name) for name in sorted(set(names))}
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf8")
    final={"at_utc":datetime.now(timezone.utc).isoformat(),"status":"passed",
           "archived_original_hashes_verified":True,
           "counts":expected,
           "max_roc_auc_grid_difference":float(check.absolute_difference.max()),
           "outputs":{str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob("*"))
                      if p.is_file() and p.name not in completion}}
    (DEST/completion[2]).write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf8")
    print("FINAL_INTEGRITY PASSED",len(final["outputs"]),"deliverables checked")


if __name__=="__main__":
    main()

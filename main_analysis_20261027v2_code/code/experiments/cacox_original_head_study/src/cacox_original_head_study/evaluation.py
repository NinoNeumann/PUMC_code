"""Explicit metric adapters; Harrell score and time-specific AUC are distinct."""
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from . import metrics_core as core

KEY = ["repeat_id", "fold", "imputation", "patient_id"]
TIMES = (365, 730, 1095)


def normalize(frame, model_id, study_id, score_definition):
    q = frame.rename(columns={"score_rcs_ridge": "score"}).copy()
    required = KEY + ["Death", "Survival_Days", "score"] + [f"survival_{t}d" for t in TIMES]
    if not set(required).issubset(q.columns) or q[KEY].duplicated().any():
        raise ValueError("Prediction schema/keys invalid")
    if not np.isfinite(q[["score", *[f"survival_{t}d" for t in TIMES]]].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions")
    s = q[[f"survival_{t}d" for t in TIMES]].to_numpy(float)
    if np.any(s < 0) or np.any(s > 1) or np.any(np.diff(s, axis=1) > 1e-10):
        raise ValueError("Survival probability range/order invalid")
    for t in TIMES:
        q[f"risk_{t}d"] = 1 - q[f"survival_{t}d"]
    q["model_id"], q["run_id"], q["score_definition"] = model_id, study_id, score_definition
    return q.sort_values(KEY).reset_index(drop=True)


def assert_complete(frame):
    if (len(frame) != 1140 * 20 or frame[KEY].duplicated().any()
            or set(frame.imputation) != set(range(1,21)) or set(frame.fold) != set(range(1,6))
            or not frame.groupby(["fold","imputation"]).size().eq(228).all()
            or not frame.groupby("imputation").patient_id.nunique().eq(1140).all()):
        raise ValueError("Expected complete five folds x20 imputations")


def assert_paired(main, alternative):
    a, b = main.sort_values(KEY).reset_index(drop=True), alternative.sort_values(KEY).reset_index(drop=True)
    if not a[KEY].equals(b[KEY]) or not np.array_equal(a[["Death","Survival_Days"]], b[["Death","Survival_Days"]]):
        raise ValueError("Patient/partition/imputation/outcome pairing mismatch")


def evaluate(frame, times=TIMES):
    per, pooled = [], []
    for horizon in (None, *times):
        metric = "C_index" if horizon is None else f"AUC_{horizon}d"
        column = "score" if horizon is None else f"risk_{horizon}d"
        results = []
        for j in range(1,21):
            z = frame[frame.imputation.eq(j)]
            result = core.aggregate_metric(z, score_column=column, horizon=horizon)
            results.append(result)
            per.append({"metric":metric,"imputation":j,"estimate":result["estimate"],"variance":result["variance"]})
        pooled.append({"metric":metric,**core.rubin([r["estimate"] for r in results], [r["variance"] for r in results], True)})
    return pd.DataFrame(per), pd.DataFrame(pooled)


def paired_difference(main, alternative, times=TIMES):
    assert_paired(main, alternative)
    per, pooled = [], []
    for horizon in (None, *times):
        metric = "C_index" if horizon is None else f"AUC_{horizon}d"
        column = "score" if horizon is None else f"risk_{horizon}d"
        estimates, variances = [], []
        for j in range(1,21):
            a = core.aggregate_metric(main[main.imputation.eq(j)], column, horizon)
            b = core.aggregate_metric(alternative[alternative.imputation.eq(j)], column, horizon)
            ids = a["contributions"].keys() | b["contributions"].keys()
            variance = sum((b["contributions"].get(i,0.) - a["contributions"].get(i,0.))**2 for i in ids)
            delta = b["estimate"] - a["estimate"]
            estimates.append(delta); variances.append(variance)
            per.append({"metric":metric,"imputation":j,"reference":a["estimate"],"alternative":b["estimate"],
                        "difference":delta,"variance_difference":variance})
        result = core.rubin(estimates, variances)
        result["p_value_two_sided"] = float(2 * student_t.sf(abs(result["estimate"] / result["SE"]), result["df"])) if result["SE"] else None
        pooled.append({"metric":metric,**result})
    return pd.DataFrame(per), pd.DataFrame(pooled)


def individual_cindices(frame):
    records = []
    for (fold,j), z in frame.groupby(["fold","imputation"]):
        c = core.harrell(z.Survival_Days, z.Death, z.score)
        records.append({"fold":int(fold),"imputation":int(j),"C_index":c["estimate"]})
    return pd.DataFrame(records)

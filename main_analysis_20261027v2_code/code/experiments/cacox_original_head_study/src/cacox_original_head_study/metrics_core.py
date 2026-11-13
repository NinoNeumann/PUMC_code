"""Conditional prediction-based inference; never labelled optimism correction.

IPCW AUC follows cv50_time_auc.R. C uses within-fold event/comparable pairs
and a patient-level ratio influence function; numerical parity is checked
against saved R outputs in the test suite before reporting equivalence.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def harrell(time, event, score):
    time, event, score = map(np.asarray, (time, event, score))
    if not (len(time) == len(event) == len(score)) or not np.isfinite(score).all():
        raise ValueError("Invalid survival vectors")
    # R survival treats censoring at t as occurring after deaths at t.
    w = (event[:, None] == 1) & ((time[:, None] < time[None, :]) |
        ((time[:, None] == time[None, :]) & (event[None, :] == 0)))
    den = int(w.sum())
    if den == 0:
        raise ValueError("No comparable pairs")
    h = w * ((score[:, None] > score[None, :]) + .5 * (score[:, None] == score[None, :]))
    value = float(h.sum() / den)
    residual = h - value * w
    influence = (residual.sum(axis=0) + residual.sum(axis=1)) / den
    return {"estimate": value, "variance": float(influence @ influence),
            "influence": influence, "comparable_pairs": den}


def _auc_point(time, event, score, horizon, compute_roc=True):
    time, event, score = map(np.asarray, (time, event, score))
    cases = np.flatnonzero((time <= horizon) & (event == 1))
    controls = np.flatnonzero(time > horizon)
    if not len(cases) or not len(controls):
        raise ValueError("AUC lacks cases/controls")
    ct = np.sort(np.unique(time[(event == 0) & (time <= horizon)]))
    g = np.cumprod([1 - np.sum((time == u) & (event == 0)) / np.sum(time >= u) for u in ct])
    left = np.array([g[np.searchsorted(ct, time[i], side="left") - 1]
                     if np.searchsorted(ct, time[i], side="left") else 1. for i in cases])
    if np.any(left <= 0) or (len(g) and g[-1] <= 0):
        raise ValueError("Censoring KM lacks support")
    weights = 1 / left
    h = (score[cases, None] > score[None, controls]) + .5 * (score[cases, None] == score[None, controls])
    value = float(np.sum(weights * h.sum(axis=1)) / (weights.sum() * len(controls)))
    if not compute_roc:
        return value, None, len(cases), len(controls)
    # Rank-block ROC: same endpoint/interpolation rule as the R source.
    wc, wn = np.zeros(len(time)), np.zeros(len(time))
    wc[cases] = weights / weights.sum()
    wn[controls] = 1 / len(controls)
    order = np.argsort(-score, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(time) - 1]
    fp = np.r_[0., np.cumsum(wn[order])[ends]]
    tp = np.r_[0., np.cumsum(wc[order])[ends]]
    grid = np.linspace(0, 1, 201)
    roc = []
    for x in grid:
        eq = np.flatnonzero(np.abs(fp - x) < 1e-12)
        if len(eq):
            roc.append(float(tp[eq].max()))
        else:
            lo = np.flatnonzero(fp < x)[-1]
            hi = np.flatnonzero(fp > x)[0]
            roc.append(float(tp[lo] + (tp[hi] - tp[lo]) * (x - fp[lo]) / (fp[hi] - fp[lo])))
    return value, np.asarray(roc), len(cases), len(controls)


def ipcw_auc(time, event, score, horizon):
    time, event, score = map(np.asarray, (time, event, score))
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite risk")
    if np.sum((time <= horizon) & (event == 1)) < 2 or np.sum(time > horizon) < 2:
        raise ValueError("At least two cases/controls required for delete-one inference")
    value, roc, cases, controls = _auc_point(time, event, score, horizon)
    deleted = []
    for i in range(len(time)):
        keep = np.arange(len(time)) != i
        deleted.append(_auc_point(time[keep], event[keep], score[keep], horizon, compute_roc=False)[0])
    deleted = np.asarray(deleted)
    influence = (deleted.mean() - deleted) * np.sqrt((len(time) - 1) / len(time))
    return {"estimate": value, "variance": float(influence @ influence),
            "influence": influence, "roc": roc, "cases": cases, "controls": controls}


def rubin(estimates, variances, bounded=False):
    q, u = np.asarray(estimates, float), np.asarray(variances, float)
    if len(q) < 2 or len(q) != len(u) or not np.isfinite(q).all() or not np.isfinite(u).all() or np.any(u < 0):
        raise ValueError("Rubin pooling requires >=2 finite estimates and nonnegative variances")
    m = len(q)
    within, between = float(u.mean()), float(q.var(ddof=1))
    extra = (1 + 1 / m) * between
    total = within + extra
    df = float((m - 1) * (1 + within / extra) ** 2) if extra > 0 else np.inf
    cut = student_t.ppf(.975, df)
    low, high = q.mean() - cut * np.sqrt(total), q.mean() + cut * np.sqrt(total)
    if bounded:
        low, high = max(0., low), min(1., high)
    return {"estimate": float(q.mean()), "SE": float(np.sqrt(total)), "lower": float(low),
            "upper": float(high), "df": df, "within_variance": within,
            "between_variance": between, "total_variance": total,
            "interval_scope": "Conditional on fitted models, selection, splits; not optimism corrected"}


def aggregate_metric(d: pd.DataFrame, score_column="score", horizon=None):
    """Within-fold rankings; cluster repeated contributions by original patient ID."""
    groups = list(d.groupby(["repeat_id", "fold"], sort=True))
    records = []
    for _, z in groups:
        if z.patient_id.duplicated().any():
            raise ValueError("Duplicate patient within a fold")
        args = (z.Survival_Days.to_numpy(), z.Death.to_numpy(), z[score_column].to_numpy())
        result = harrell(*args) if horizon is None else ipcw_auc(*args, horizon)
        weight = result["comparable_pairs"] if horizon is None else len(z)
        records.append((z, result, weight))
    total = sum(w for _, _, w in records)
    contributions = {}
    estimate = 0.
    for z, result, w in records:
        fraction = w / total
        estimate += fraction * result["estimate"]
        for patient, v in zip(z.patient_id, result["influence"]):
            contributions[patient] = contributions.get(patient, 0.) + fraction * v
    return {"estimate": estimate, "variance": sum(v * v for v in contributions.values()),
            "contributions": contributions}


def evaluate_predictions(d: pd.DataFrame, m=20, times=(365, 730, 1095)):
    if set(d.imputation) != set(range(1, m + 1)) or d.duplicated(["repeat_id", "fold", "imputation", "patient_id"]).any():
        raise ValueError("Incomplete/duplicate imputation predictions")
    rows, pooled = [], []
    for horizon in [None, *times]:
        name = "C_index" if horizon is None else f"AUC_{horizon}d"
        results = []
        for j in range(1, m + 1):
            result = aggregate_metric(d[d.imputation == j], horizon=horizon)
            rows.append({"metric": name, "imputation": j, "estimate": result["estimate"], "variance": result["variance"]})
            results.append(result)
        pooled.append({"metric": name, **rubin([r["estimate"] for r in results], [r["variance"] for r in results], True)})
    return pd.DataFrame(rows), pd.DataFrame(pooled)


def paired_difference(main, alternative, m=20, times=(365, 730, 1095)):
    keys = ["repeat_id", "fold", "imputation", "patient_id"]
    if main.duplicated(keys).any() or alternative.duplicated(keys).any():
        raise ValueError("Duplicate paired predictions")
    paired = alternative.merge(main, on=keys, suffixes=("_alternative", "_main"), how="left", validate="one_to_one", indicator=True)
    if not paired._merge.eq("both").all():
        raise ValueError("Alternative patients absent from main")
    for c in ["Death", "Survival_Days"]:
        if not np.array_equal(paired[f"{c}_alternative"], paired[f"{c}_main"]):
            raise ValueError("Paired outcome mismatch")
    a = paired[keys].copy()
    a["Death"], a["Survival_Days"] = paired.Death_main, paired.Survival_Days_main
    per, pooled = [], []
    for horizon in [None, *times]:
        metric = "C_index" if horizon is None else f"AUC_{horizon}d"
        qs, us = [], []
        for j in range(1, m + 1):
            z = a[a.imputation == j].copy()
            z["score"] = paired.loc[z.index, "score_main"]
            main_result = aggregate_metric(z, horizon=horizon)
            z["score"] = paired.loc[z.index, "score_alternative"]
            other = aggregate_metric(z, horizon=horizon)
            ids = main_result["contributions"].keys() | other["contributions"].keys()
            variance = sum((other["contributions"].get(i, 0.) - main_result["contributions"].get(i, 0.)) ** 2 for i in ids)
            difference = other["estimate"] - main_result["estimate"]
            qs.append(difference)
            us.append(variance)
            per.append({"metric": metric, "imputation": j, "main_on_matched_patients": main_result["estimate"],
                "alternative": other["estimate"], "difference_alternative_minus_main": difference, "variance_difference": variance})
        pooled.append({"metric": metric, "comparison": "alternative minus main on identical patients", **rubin(qs, us)})
    return pd.DataFrame(per), pd.DataFrame(pooled)


def roc_coordinates(d, m=20, times=(365, 730, 1095)):
    records = []
    for horizon in times:
        mean = np.zeros(201)
        for j in range(1, m + 1):
            z = d[d.imputation == j]
            if not len(z):
                raise ValueError("Missing imputation")
            for _, g in z.groupby(["repeat_id", "fold"]):
                _, curve, _, _ = _auc_point(g.Survival_Days.to_numpy(), g.Death.to_numpy(), g.score.to_numpy(), horizon)
                mean += len(g) / len(z) / m * curve
        records.extend({"days": horizon, "FPR": x, "TPR": y} for x, y in zip(np.linspace(0, 1, 201), mean))
    return pd.DataFrame(records)

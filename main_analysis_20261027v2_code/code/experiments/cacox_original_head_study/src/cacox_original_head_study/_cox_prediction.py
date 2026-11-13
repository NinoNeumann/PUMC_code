"""Unchanged baseline-hazard and survival conversion."""
import numpy as np
from scipy.special import logsumexp

def breslow(time, event, scores):
    time, event, scores = map(np.asarray, (time, event, scores))
    shift = float(np.median(scores))
    times = np.sort(np.unique(time[event == 1]))
    if not len(times):
        raise ValueError("No development events for baseline hazard")
    increments = [np.sum((time == t) & (event == 1)) * np.exp(-logsumexp(scores[time >= t] - shift)) for t in times]
    return {"event_times": times, "cumulative_hazard": np.cumsum(increments), "shift": shift}

def survival_from_baseline(scores, baseline, times):
    times = np.asarray(times, float)
    idx = np.searchsorted(baseline["event_times"], times, side="right")
    ch = np.r_[0., baseline["cumulative_hazard"]][idx]
    with np.errstate(over="raise", invalid="raise"):
        survival = np.exp(-np.outer(np.exp(np.asarray(scores) - baseline["shift"]), ch))
    if not np.isfinite(survival).all():
        raise FloatingPointError("Invalid survival probabilities")
    return survival

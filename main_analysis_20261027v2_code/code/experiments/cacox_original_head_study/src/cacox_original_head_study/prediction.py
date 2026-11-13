"""Cox conversion and piecewise constant hazard interpolation for discrete heads."""
import numpy as np
from ._cox_prediction import breslow, survival_from_baseline


def discrete_predictions(logits, grid, times=(365., 730., 1095.), horizon=1095.):
    logits = np.asarray(logits, dtype=float)
    times = np.asarray(times, float)
    if not np.isfinite(logits).all() or logits.ndim != 2 or logits.shape[1] != grid.bins:
        raise ValueError("Invalid hazard logits")
    if horizon > grid.cuts[-1] or np.any(times > grid.cuts[-1]) or np.any(times < 0):
        raise ValueError("Prediction horizon outside training grid")
    widths = np.diff(grid.cuts)
    full_hazard = np.logaddexp(0., logits)
    rates = full_hazard / widths[None, :]
    exposure = np.clip(times[:, None] - grid.cuts[None, :-1], 0., widths[None, :])
    survival = np.exp(-(rates @ exposure.T))
    rmst_width = np.clip(horizon - grid.cuts[:-1], 0., widths)
    previous_survival = np.exp(-np.column_stack([np.zeros(len(logits)), np.cumsum(full_hazard[:, :-1], axis=1)]))
    integral = np.broadcast_to(rmst_width, rates.shape).copy()
    np.divide(-np.expm1(-rates * rmst_width[None, :]), rates, out=integral, where=rates > 1e-12)
    rmst = np.sum(previous_survival * integral, axis=1)
    if (not np.isfinite(rmst).all() or np.any(rmst < 0) or np.any(rmst > horizon + 1e-8)
            or np.any(np.diff(survival, axis=1) > 1e-10)):
        raise FloatingPointError("Invalid survival curve/RMST")
    return -rmst, survival

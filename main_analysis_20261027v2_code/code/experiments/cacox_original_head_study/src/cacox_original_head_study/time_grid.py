"""Training-fitted interval boundaries and explicit continuous-time rounding."""
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class TimeGrid:
    cuts: np.ndarray

    @classmethod
    def fit(cls, time, event, bins, required_horizon=1095.):
        time, event = np.asarray(time, float), np.asarray(event, int)
        if not np.isfinite(time).all() or np.any(time <= 0) or not np.isin(event, [0, 1]).all() or event.sum() == 0:
            raise ValueError("Invalid survival targets for time grid")
        internal = np.quantile(time[event == 1], np.arange(1, bins) / bins)
        cuts = np.r_[0., internal, time.max()]
        if np.any(np.diff(cuts) <= 0):
            raise ValueError("Repeated/non-increasing time boundaries; candidate is invalid")
        if cuts[-1] < required_horizon:
            raise ValueError("Training follow-up does not support the fixed risk horizon")
        return cls(cuts)

    @property
    def bins(self):
        return len(self.cuts) - 1

    def labels(self, time, event, device="cpu"):
        time, event = np.asarray(time, float), np.asarray(event, int)
        if not np.isfinite(time).all() or np.any(time <= 0) or not np.isin(event, [0, 1]).all():
            raise ValueError("Invalid targets")
        event = (event == 1) & (time <= self.cuts[-1])
        index = np.minimum(np.searchsorted(self.cuts[1:], time, side="left"), self.bins - 1)
        target = np.zeros((len(time), self.bins), np.float32)
        target[np.flatnonzero(event), index[event]] = 1
        # Deaths: fully survived preceding bins, then one event bin.
        # Censors: only intervals completed by censoring are observed.
        mask = np.where(event[:, None],
                        np.arange(self.bins)[None, :] <= index[:, None],
                        self.cuts[None, 1:] <= time[:, None]).astype(np.float32)
        return torch.as_tensor(mask, device=device), torch.as_tensor(target, device=device)

    def diagnostics(self, time, event):
        mask, target = self.labels(time, event)
        return {"bins": self.bins, "cuts": self.cuts.tolist(),
                "n": len(time), "events_encoded": int(target.sum()),
                "beyond_grid": int(np.sum(np.asarray(time) > self.cuts[-1])),
                "no_complete_observed_interval": int((mask.sum(1) == 0).sum())}

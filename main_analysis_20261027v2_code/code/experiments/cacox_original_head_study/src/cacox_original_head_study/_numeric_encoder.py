"""Training-only numeric scaling and fixed legal categorical contrasts."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema29 import CONTINUOUS, LEVELS


class Encoder:
    def __init__(self, variables, scale="robust"):
        if scale not in {"robust", "standard"}:
            raise ValueError("scale must be explicitly robust or standard")
        self.variables = list(variables)
        self.scale = scale
        self.centers = {}
        self.scales = {}
        self.columns = []

    def fit(self, d: pd.DataFrame):
        if d[self.variables].isna().any().any():
            raise ValueError("Imputation must precede encoding")
        self.columns = []
        for c in self.variables:
            if c in CONTINUOUS:
                x = d[c].to_numpy(float)
                self.centers[c] = float(np.median(x)) if self.scale == "robust" else float(np.mean(x))
                s = float(np.quantile(x, .75) - np.quantile(x, .25)) if self.scale == "robust" else float(np.std(x))
                self.scales[c] = s if s > 0 else 1.
                self.columns.append(c)
            else:
                self.columns.extend(f"{c}={v}" for v in LEVELS[c][1:])
        return self

    def transform(self, d: pd.DataFrame):
        if not self.columns:
            raise ValueError("Encoder not fitted")
        arrays = []
        for c in self.variables:
            if d[c].isna().any():
                raise ValueError(f"Unimputed {c}")
            if c in CONTINUOUS:
                arrays.append((d[c].to_numpy(float) - self.centers[c]) / self.scales[c])
            else:
                values = d[c].astype(str)
                if not set(values).issubset(LEVELS[c]):
                    raise ValueError(f"Illegal {c} level")
                arrays.extend((values == v).to_numpy(float) for v in LEVELS[c][1:])
        x = np.column_stack(arrays).astype("float32")
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite model input")
        return x

    def fit_transform(self, d):
        return self.fit(d).transform(d)

"""Portable preprocessing contract shared by Python training and an upstream backend.

Every completed partition carries RAW provenance, IDs and explicit imputation
indices. No guessing pairings by file order or treating RDS as raw Excel.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .schema29 import CANDIDATES, LEVELS, clean_frame


class Bundle:
    def __init__(self, directory, m=20):
        self.directory = Path(directory)
        self.m = m
        self.meta = json.loads((self.directory / "bundle.json").read_text())
        if self.meta["m"] != m or self.meta["maxit"] != 60:
            raise ValueError("Expected original 20 imputations and 60 MICE iterations")
        self.variables = list(self.meta["selected_variables"])
        candidates = self.meta["candidate_variables"]
        if (not candidates or len(candidates) != len(set(candidates))
                or not set(candidates).issubset(CANDIDATES)):
            raise ValueError("Invalid candidate variables")
        if (not self.variables or len(self.variables) != len(set(self.variables))
                or not set(self.variables).issubset(candidates)):
            raise ValueError("Invalid selected variables")
        types = {v: "string" for v in candidates if v in LEVELS}
        self.raw_dev = clean_frame(pd.read_csv(self.directory / "raw_development.csv", dtype=types), candidates)
        self.raw_assessment = clean_frame(pd.read_csv(self.directory / "raw_assessment.csv", dtype=types), candidates)
        if set(self.raw_dev.patient_id) & set(self.raw_assessment.patient_id):
            raise ValueError("Outer development/assessment overlap")

    def pair(self, j, inner=None):
        if j not in range(1, self.m + 1):
            raise ValueError("Invalid imputation")
        p = self.directory / ("outer" if inner is None else f"inner_{inner:02d}")
        candidates = self.meta["candidate_variables"]
        types = {v: "string" for v in candidates if v in LEVELS}
        dev = clean_frame(pd.read_csv(p / f"development_{j:02d}.csv", dtype=types), candidates)
        val = clean_frame(pd.read_csv(p / f"assessment_{j:02d}.csv", dtype=types), candidates)
        if dev[candidates].isna().any().any() or val[candidates].isna().any().any():
            raise ValueError("Uncompleted imputation")
        if set(dev.patient_id) & set(val.patient_id):
            raise ValueError("Partition overlap")
        raw = pd.concat([self.raw_dev, self.raw_assessment]).set_index("patient_id")
        for frame in [dev, val]:
            if not set(frame.patient_id).issubset(raw.index):
                raise ValueError("Unknown patient")
            matched = raw.loc[frame.patient_id]
            if not np.array_equal(frame.Death.to_numpy(), matched.Death.to_numpy()) or not np.array_equal(frame.Survival_Days.to_numpy(), matched.Survival_Days.to_numpy()):
                raise ValueError("Outcomes changed by preprocessing")
            for c in candidates:
                mask = matched[c].notna().to_numpy()
                a, b = frame[c].to_numpy()[mask], matched[c].to_numpy()[mask]
                if not np.array_equal(a, b):
                    raise ValueError(f"Observed {c} changed by preprocessing")
        if inner is None:
            if set(dev.patient_id) != set(self.raw_dev.patient_id) or set(val.patient_id) != set(self.raw_assessment.patient_id):
                raise ValueError("Outer membership changed")
        elif set(dev.patient_id) | set(val.patient_id) != set(self.raw_dev.patient_id):
            raise ValueError("Inner split must partition outer development only")
        return dev, val

    def inner_pairs(self, j):
        pairs = [self.pair(j, k) for k in range(1, 6)]
        validation_ids = pd.concat([val.patient_id for _, val in pairs], ignore_index=True)
        if validation_ids.duplicated().any() or set(validation_ids) != set(self.raw_dev.patient_id):
            raise ValueError("Five inner validation folds must cover development patients exactly once")
        return pairs

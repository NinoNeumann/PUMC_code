"""Explicit verified 29-variable contract, including ESR."""
from __future__ import annotations
import numpy as np
import pandas as pd

CONTINUOUS = ['Age', 'Hb', 'PLT', 'eGFR', 'ESR', 'CRP']

CANDIDATES = ['Gender', 'Age', 'Type', 'Antibody', 'Coronary_Heart_Disease', 'Stroke', 'Allergy_History', 'Smoking_History', 'Hyperlipidemia', 'Diabetes_Mellitus', 'Hypertension', 'Hb', 'PLT', 'eGFR', 'ESR', 'CRP', 'ANA', 'Systemic_Involvement', 'Musculoskeletal_Involvement', 'Cutaneous_Involvement', 'Ocular_Involvement', 'ENT_Involvement', 'Cardiac_Involvement', 'Gastrointestinal_Involvement', 'Pulmonary_Involvement', 'Renal_Involvement', 'Neurological_Involvement', 'Glucocorticoid_Use', 'cDMARDs_Use']

LEVELS = {c: ["no", "yes"] for c in CANDIDATES if c not in CONTINUOUS}

LEVELS.update(Gender=["1", "2"], Type=["EGPA", "GPA", "MPA"],
              Antibody=["0", "1", "2", "3"], Smoking_History=["never", "former", "current"],
              ANA=["negative", "positive"])

MISSING = {"", "na", "n/a", "nan", "null", "none", "unknown", "unkown", "未知", "缺失"}

def clean_frame(frame: pd.DataFrame, variables=CANDIDATES) -> pd.DataFrame:
    """Require known columns, legal levels, complete outcomes, stable row IDs."""
    required = ["Death", "Survival_Days", *variables]
    absent = set(required) - set(frame.columns)
    if absent:
        raise ValueError(f"Missing required fields: {sorted(absent)}; do not replace with 18-variable data")
    d = frame.copy()
    for c in required:
        text = d[c].astype("string").str.strip()
        d[c] = text.mask(text.str.lower().isin(MISSING))
    for c in ["Death", "Survival_Days", *[v for v in variables if v in CONTINUOUS]]:
        d[c] = pd.to_numeric(d[c], errors="raise").astype(float)
        if np.isinf(d[c]).any():
            raise ValueError(f"Nonfinite {c}")
    for c in variables:
        if c in LEVELS:
            values = d[c].str.upper() if c == "Type" else d[c].str.lower()
            invalid = set(values.dropna()) - set(LEVELS[c])
            if invalid:
                raise ValueError(f"Unknown {c} levels: {invalid}")
            d[c] = pd.Categorical(values, categories=LEVELS[c])
    if d[["Death", "Survival_Days"]].isna().any().any():
        raise ValueError("Missing outcomes")
    if not d.Death.isin([0, 1]).all() or not (d.Survival_Days > 0).all():
        raise ValueError("Invalid event/time")
    if "patient_id" not in d:
        d["patient_id"] = np.arange(1, len(d) + 1)
    if d.patient_id.isna().any() or d.patient_id.duplicated().any():
        raise ValueError("Patient IDs must be unique and complete")
    return d

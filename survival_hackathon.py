#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moffitt Survival Baseline — Hackathon Starter (Research only)
-------------------------------------------------------------
- Loads HuggingFace dataset: Lab-Rasool/hackathon
- Detects survival time + event columns
- Trains:
    (A) Cox Proportional Hazards (lifelines)
    (B) Random Survival Forest (optional if scikit-survival installed)
- Cross-validation
- C-index + calibration plots
-------------------------------------------------------------
Research only. Not for clinical use.
"""

import argparse
import os
import re
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

# Try to import scikit-survival (optional)
_has_sksurv = True
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import (
        concordance_index_censored,
        integrated_brier_score,
        cumulative_dynamic_auc
    )
    from sksurv.util import Surv
except Exception:
    _has_sksurv = False

warnings.filterwarnings("ignore")


# -----------------------------
# Detect time / event columns
# -----------------------------
def _pick_first_present(candidates: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
        for dc in df.columns:
            if dc.lower() == c.lower():
                return dc
    return None


def detect_time_event(df: pd.DataFrame) -> Tuple[str, str]:
    time_candidates = [
        "overall_survival_days", "os_days", "days_to_death",
        "survival_days", "days_to_last_followup"
    ]
    event_candidates = [
        "overall_survival_event", "os_event", "event",
        "death_event", "status", "dead", "deceased"
    ]
    t_col = _pick_first_present(time_candidates, df)
    e_col = _pick_first_present(event_candidates, df)
    if t_col is None or e_col is None:
        raise ValueError("Couldn't detect survival columns — rename in dataset if needed.")
    return t_col, e_col


def normalize_event_column(ecol: str, df: pd.DataFrame) -> pd.Series:
    s = df[ecol]
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)
    sval = s.astype(str).str.lower()
    if "alive" in sval.values or "dead" in sval.values:
        return (sval == "dead").astype(int)
    return (~s.isna()).astype(int)


# -----------------------------
# Feature filtering
# -----------------------------
def basic_feature_drop(df: pd.DataFrame, tcol: str, ecol: str) -> pd.DataFrame:
    leakage_regex = re.compile("days|death|followup|os|time|date|survival", re.I)
    keep = [
        c for c in df.columns
        if c not in (tcol, ecol) and not leakage_regex.search(c)
    ]
    trimmed = df[keep].copy()
    # Drop free-text fields
    for c in list(trimmed.columns):
        if trimmed[c].dtype == object:
            if trimmed[c].astype(str).str.len().mean() > 200:
                trimmed.drop(columns=[c], inplace=True)
    return trimmed


def get_preprocessor(dfX: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in dfX.columns if pd.api.types.is_numeric_dtype(dfX[c])]
    cat_cols = [c for c in dfX.columns if c not in num_cols]

    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ]), cat_cols)
    ])


# -----------------------------
# Plots
# -----------------------------
def km_plot(time, event, outpath):
    km = KaplanMeierFitter()
    km.fit(time, event)
    km.plot_survival_function()
    plt.title("Kaplan–Meier: Overall Survival")
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--horizons", nargs="+", type=int, default=[180, 365])
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    print("Loading dataset...")
    ds = load_dataset("Lab-Rasool/hackathon")
    df = ds[list(ds.keys())[0]].to_pandas()

    tcol, ecol = detect_time_event(df)
    print(f"Detected: time = {tcol}, event = {ecol}")

    time = df[tcol].astype(float)
    event = normalize_event_column(ecol, df)

    mask = time > 0
    df = df.loc[mask].reset_index(drop=True)
    time = time.loc[mask]
    event = event.loc[mask]

    X = basic_feature_drop(df, tcol, ecol)
    pre = get_preprocessor(X)

    km_plot(time, event, "outputs/km_overall.png")

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    cidx_cox = []

    print("Running cross-validation...")
    y = event.values

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ttr, tte = time.iloc[train_idx], time.iloc[test_idx]
        etr, ete = y[train_idx], y[test_idx]

        Xtr_proc = pre.fit_transform(Xtr)
        Xte_proc = pre.transform(Xte)
        feat_cols = [f"x{i}" for i in range(Xtr_proc.shape[1])]

        df_tr = pd.DataFrame(Xtr_proc, columns=feat_cols)
        df_tr["T"] = ttr
        df_tr["E"] = etr

        cph = CoxPHFitter(penalizer=1.0)
        cph.fit(df_tr, duration_col="T", event_col="E")

        df_te = pd.DataFrame(Xte_proc, columns=feat_cols)
        risk = cph.predict_partial_hazard(df_te).values.ravel()

        cidx = concordance_index(ete, tte, risk)
        cidx_cox.append(cidx)
        print(f"Fold {fold_i}: C-index = {cidx:.3f}")

    print("\n=== RESULTS ===")
    print(f"C-index (Cox) mean: {np.mean(cidx_cox):.3f}")

    print("Done! Check outputs/ folder.")


if __name__ == "__main__":
    main()

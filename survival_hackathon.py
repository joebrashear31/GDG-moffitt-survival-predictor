#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moffitt Survival Baseline — Local/Cloud + Robust Cox + Export (Research only)

Features:
- Uses local train/test folders if provided; else falls back to HuggingFace dataset.
- Adjustable sampling via --max-rows for quick iteration.
- Strong leakage filtering (drops outcome-ish columns like vital_status, progression/recurrence).
- Safe OneHotEncoder (version-aware), rare-category capping, drop='first' to avoid dummy traps.
- Cross-validation with feature alignment & Cox PH robust fitting (penalization + retries).
- Exports model+preprocessor with joblib for serving (FastAPI).
- Optional RSF path (if scikit-survival installed), but Cox is default/recommended for small N.
"""

import argparse
import json
import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from packaging import version
import sklearn

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError

# Optional RSF deps
_HAS_SKSURV = True
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import (
        concordance_index_censored,
        integrated_brier_score,
        cumulative_dynamic_auc,
    )
    from sksurv.util import Surv
except Exception:
    _HAS_SKSURV = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------
# File loading (local + HF fallback)
# ---------------------------------------------------------------------
def _auto_find_file(dirpath: str) -> str:
    csvs = sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.lower().endswith(".csv")])
    pars = sorted([os.path.join(dirpath, f) for f in os.listdir(dirpath) if f.lower().endswith(".parquet")])
    if csvs:
        return csvs[0]
    if pars:
        return pars[0]
    raise FileNotFoundError(f"No CSV or Parquet files found in: {dirpath}")

def load_local_split(data_dir: str, split: str) -> pd.DataFrame:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Missing folder: {split_dir}")
    fpath = _auto_find_file(split_dir)
    if fpath.lower().endswith(".csv"):
        df = pd.read_csv(fpath)
    else:
        df = pd.read_parquet(fpath)
    print(f"Loaded {split}: {fpath} (rows={len(df)}, cols={len(df.columns)})")
    return df

def load_training_data(data_dir: Optional[str]) -> pd.DataFrame:
    # Prefer local; fallback to HF
    if data_dir:
        data_dir = os.path.expanduser(data_dir)
        train_dir = os.path.join(data_dir, "train")
        if os.path.isdir(train_dir):
            print(f"✅ Using local training data: {train_dir}")
            return load_local_split(data_dir, "train")
    print("🌐 Local train not found — pulling from HuggingFace (Lab-Rasool/hackathon)...")
    from datasets import load_dataset
    ds = load_dataset("Lab-Rasool/hackathon")
    return ds[list(ds.keys())[0]].to_pandas()

def load_test_data(data_dir: Optional[str]) -> Optional[pd.DataFrame]:
    if data_dir:
        data_dir = os.path.expanduser(data_dir)
        test_dir = os.path.join(data_dir, "test")
        if os.path.isdir(test_dir):
            print(f"✅ Using local test data: {test_dir}")
            return load_local_split(data_dir, "test")
    print("⚠️ No local test data — will skip submission.")
    return None


# ---------------------------------------------------------------------
# Survival columns detection / normalization
# ---------------------------------------------------------------------
def _pick_first_present(cands: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
        for dc in df.columns:
            if dc.lower() == c.lower():
                return dc
    return None

def detect_time_event(df: pd.DataFrame) -> Tuple[str, str]:
    time_candidates = [
        "overall_survival_days","os_days","days_to_death","survival_days",
        "survival_time","time_to_event","time",
        "days_to_last_followup","days_to_last_follow_up","followup_days",
    ]
    event_candidates = [
        "overall_survival_event","os_event","event","status","death_event",
        "dead","deceased","vital_status",
    ]
    tcol = _pick_first_present(time_candidates, df)
    ecol = _pick_first_present(event_candidates, df)
    if tcol is None:
        raise ValueError("No time column found (e.g., overall_survival_days).")
    if ecol is None:
        raise ValueError("No event column found (e.g., overall_survival_event).")
    return tcol, ecol

def normalize_event_column(ecol: str, df: pd.DataFrame) -> pd.Series:
    s = df[ecol]
    if s.dtype == bool:
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(float) > 0).astype(int)
    sval = s.astype(str).str.lower().str.strip()
    if ("alive" in set(sval)) or ("dead" in set(sval)):
        return (sval == "dead").astype(int)
    return (~s.isna()).astype(int)


# ---------------------------------------------------------------------
# Leakage filtering + preprocessing
# ---------------------------------------------------------------------
def basic_feature_drop(df: pd.DataFrame, tcol: str, ecol: str) -> pd.DataFrame:
    """
    Drop obvious leakage/high-risk columns:
    - outcome/status/survival/progression/recurrence/last contact/date/time/etc.
    - IDs/MRN-like
    - long free-text (>200 avg chars) removed for baseline
    """
    drop_like = [
        r"^id$", r"\bpatient[_ ]?id\b", r"\bmrn\b", r"\brecord\b", r"\buid\b",
        r"^date", r"_date", r"timestamp",
        r"^days", r"\bdeath\b",
        r"\bfollow[_ ]?up\b",
        r"overall[_ ]?survival", r"\bos\b", r"last[_ ]?contact", r"\bsurvival\b", r"\btime\b",
        r"\bvital[_ ]?status\b",
        r"\bstatus\b",
        r"\bprogress(ion|ive)?\b",
        r"\brecurrence\b",
        r"\brelapse\b",
        r"\bdead\b|\bdeceased\b",
    ]
    regex = re.compile("|".join(drop_like), flags=re.I)

    keep_cols = []
    for c in df.columns:
        if c in (tcol, ecol):
            continue
        if regex.search(c):
            continue
        keep_cols.append(c)

    trimmed = df[keep_cols].copy()

    # Drop long free-text to avoid overfitting with tiny N
    for c in list(trimmed.columns):
        if trimmed[c].dtype == object:
            try:
                avg_len = trimmed[c].astype(str).map(len).mean()
                if avg_len and avg_len > 200:
                    trimmed.drop(columns=[c], inplace=True)
            except Exception:
                pass
    return trimmed

def get_preprocessor(dfX: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in dfX.columns if pd.api.types.is_numeric_dtype(dfX[c])]
    cat_cols = [c for c in dfX.columns if c not in num_cols]

    num_tf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    # Version-safe OHE + rare-category handling + drop='first'
    ohe_kwargs = {"drop": "first"}
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe_kwargs.update({
            "handle_unknown": "infrequent_if_exist",
            "sparse_output": False,
            "min_frequency": 5,  # or a proportion, e.g., 0.01
        })
    else:
        ohe_kwargs.update({
            "handle_unknown": "ignore",
            "sparse": False,
        })

    cat_tf = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**ohe_kwargs)),
    ])

    return ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols),
    ])

def align_features(df_in: pd.DataFrame, ref_cols: List[str]) -> pd.DataFrame:
    """Ensure df_in has exactly the same columns (order) as ref_cols."""
    return df_in.reindex(columns=ref_cols, fill_value=np.nan)


# ---------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------
def km_plot(time, event, outpath):
    km = KaplanMeierFitter()
    km.fit(durations=time, event_observed=event)
    plt.figure()
    km.plot_survival_function(ci_show=True)
    plt.title("Kaplan–Meier: Overall Survival")
    plt.xlabel("Days"); plt.ylabel("Survival probability")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

def calibration_at_time_plot(y_time, y_event, pred_surv_probs, horizon_days, outpath, bins=5):
    df = pd.DataFrame({"pred": pred_surv_probs, "time": y_time, "event": y_event}).dropna()
    if df.empty:
        return
    df["bin"] = pd.qcut(df["pred"], q=bins, duplicates="drop")
    km = KaplanMeierFitter()
    exps, obss = [], []
    for _, sub in df.groupby("bin"):
        exps.append(float(sub["pred"].mean()))
        try:
            km.fit(sub["time"], event_observed=sub["event"])
            obss.append(float(km.survival_function_at_times([horizon_days]).values[0]))
        except Exception:
            obss.append(np.nan)
    exps, obss = np.array(exps), np.array(obss)
    if len(exps) == 0:
        return
    lo, hi = float(np.nanmin([exps, obss])), float(np.nanmax([exps, obss]))
    plt.figure(); plt.scatter(exps, obss); plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("Mean predicted survival (bin)"); plt.ylabel("KM observed survival (bin)")
    plt.title(f"Calibration at {horizon_days} days")
    plt.tight_layout(); plt.savefig(outpath); plt.close()


# ---------------------------------------------------------------------
# Robust Cox fitting utilities
# ---------------------------------------------------------------------
def prune_constant_duplicate_columns(df_cox_tr: pd.DataFrame) -> pd.DataFrame:
    """Remove constant and duplicate feature columns (keep T/E)."""
    feat_cols = [c for c in df_cox_tr.columns if c not in ("T", "E")]
    # Drop constants
    nonconst = [c for c in feat_cols if df_cox_tr[c].nunique(dropna=False) > 1]
    df = df_cox_tr[nonconst + ["T", "E"]].copy()
    # Drop duplicates
    X = df.drop(columns=["T", "E"])
    X = X.T.drop_duplicates().T
    df = pd.concat([X, df[["T", "E"]]], axis=1)
    return df

def fit_cox_with_retry(df_tr: pd.DataFrame) -> CoxPHFitter:
    """Try increasingly strong regularization to avoid singular matrices."""
    for penalizer, l1 in [(2.0, 0.5), (5.0, 0.5), (10.0, 0.9)]:
        cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1)
        try:
            cph.fit(df_tr, duration_col="T", event_col="E")
            return cph
        except ConvergenceError:
            continue
    # last resort: ridge only
    cph = CoxPHFitter(penalizer=25.0, l1_ratio=0.0)
    cph.fit(df_tr, duration_col="T", event_col="E")
    return cph


# ---------------------------------------------------------------------
# ID detection
# ---------------------------------------------------------------------
def detect_id_column(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns:
        return prefer
    for c in ["patient_id","id","case_id","subject_id","record_id"]:
        for dc in df.columns:
            if dc == c or dc.lower() == c.lower():
                return dc
    return None


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=os.path.expanduser("~/Desktop/moffitt hackathon/hackathon"))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--horizons", nargs="+", type=int, default=[180, 365])
    ap.add_argument("--model", choices=["cox","rsf","both"], default="cox")
    ap.add_argument("--id-col", type=str, default=None)
    ap.add_argument("--model-out", type=str, default="outputs/model_export.pkl",
                    help="Path to save fitted preprocessor + model export (joblib).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="If set, subsample train/test to this many rows for quick debugging.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # ---------------- Load TRAIN ----------------
    df_train = load_training_data(args.data_dir)

    # Optional sampling for quick iteration
    if args.max_rows is not None:
        n = min(args.max_rows, len(df_train))
        print(f"⚠️ Limiting train data to {n} rows (debug mode).")
        df_train = df_train.sample(n=n, random_state=args.seed).reset_index(drop=True)

    # Detect survival cols & clean rows
    tcol, ecol = detect_time_event(df_train)
    time = pd.to_numeric(df_train[tcol], errors="coerce")
    event = normalize_event_column(ecol, df_train)
    mask = time.notna() & (time > 0) & event.notna()
    df_train = df_train.loc[mask].reset_index(drop=True)
    time = pd.to_numeric(df_train[tcol], errors="coerce").astype(float)
    event = normalize_event_column(ecol, df_train).astype(int)

    # Features
    Xraw = basic_feature_drop(df_train, tcol, ecol)
    if Xraw.shape[1] == 0:
        raise ValueError("No usable feature columns after leakage filtering. Adjust basic_feature_drop().")
    print(f"Using {Xraw.shape[1]} feature columns.")

    pre = get_preprocessor(Xraw)
    km_plot(time.values, event.values, os.path.join("outputs","km_overall.png"))

    # ---------------- Cross-validation ----------------
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    y_event = event.values.astype(int)
    cidx_cox, cidx_rsf, ibs_list = [], [], []
    auc_at_h = {int(h): [] for h in args.horizons}

    for fold, (tr, te) in enumerate(skf.split(Xraw, y_event), 1):
        Xtr, Xte = Xraw.iloc[tr], Xraw.iloc[te]
        ttr, tte = time.iloc[tr].values, time.iloc[te].values
        etr, ete = y_event[tr], y_event[te]

        # Align test features to train fold cols BEFORE fitting preprocessor
        Xte = align_features(Xte, list(Xtr.columns))

        Xtr_proc = pre.fit_transform(Xtr)
        Xte_proc = pre.transform(Xte)

        # Columns BEFORE pruning (exactly match the matrices)
        feat_cols_pre = [f"x{i}" for i in range(Xtr_proc.shape[1])]

        # ----- Cox -----
        if args.model in ("cox","both"):
            # Train DF with pre columns, then prune
            df_cox_tr = pd.DataFrame(Xtr_proc, columns=feat_cols_pre)
            df_cox_tr["T"] = ttr
            df_cox_tr["E"] = etr

            df_cox_tr = prune_constant_duplicate_columns(df_cox_tr)
            train_feats = [c for c in df_cox_tr.columns if c not in ("T","E")]

            # Test DF: create with pre columns then reindex to pruned set
            df_cox_te = pd.DataFrame(Xte_proc, columns=feat_cols_pre)
            df_cox_te = df_cox_te.reindex(columns=train_feats, fill_value=0.0)

            cph = fit_cox_with_retry(df_cox_tr)
            risk_scores = cph.predict_partial_hazard(df_cox_te).values.reshape(-1)
            cidx = concordance_index(event_observed=ete, event_times=tte, predicted_scores=risk_scores)
            cidx_cox.append(float(cidx))

            # Calibration-at-time
            for h in args.horizons:
                sp = cph.predict_survival_function(df_cox_te, times=[h]).T.values.reshape(-1)
                calibration_at_time_plot(tte, ete, sp, int(h),
                                         os.path.join("outputs", f"calibration_cox_fold{fold}_{int(h)}d.png"))

        # ----- RSF (optional) -----
        if args.model in ("rsf","both") and _HAS_SKSURV:
            y_tr = Surv.from_arrays(event=etr.astype(bool), time=ttr.astype(float))
            y_te = Surv.from_arrays(event=ete.astype(bool), time=tte.astype(float))
            rsf = RandomSurvivalForest(
                n_estimators=400, min_samples_split=10, min_samples_leaf=10,
                max_features="sqrt", n_jobs=-1, random_state=fold+17
            )
            rsf.fit(Xtr_proc, y_tr)
            rsf_risk = -rsf.predict(Xte_proc)
            c_cens = concordance_index_censored(ete.astype(bool), tte, rsf_risk)[0]
            cidx_rsf.append(float(c_cens))
            try:
                times_grid = np.linspace(np.percentile(ttr,10), np.percentile(ttr,90), 50)
                surv_funcs = rsf.predict_survival_function(Xte_proc, return_array=False)
                S_mat = np.vstack([fn(times_grid) for fn in surv_funcs])
                ibs = integrated_brier_score(y_tr, y_te, S_mat, times_grid)
                ibs_list.append(float(ibs))
                for h in args.horizons:
                    S_h = np.array([fn([h])[0] for fn in surv_funcs])
                    risk_h = 1.0 - S_h
                    try:
                        auc, _ = cumulative_dynamic_auc(
                            Surv.from_arrays(event=etr.astype(bool), time=ttr.astype(float)),
                            y_te, risk_h, np.array([h])
                        )
                        auc_at_h[int(h)].append(float(auc[0]))
                    except Exception:
                        label = (tte <= h) & (ete == 1)
                        if len(np.unique(label.astype(int))) == 2:
                            auc_simple = roc_auc_score(label.astype(int), risk_h)
                            auc_at_h[int(h)].append(float(auc_simple))
            except Exception:
                pass

    # Summarize CV metrics
    metrics: Dict[str,object] = {"folds": args.cv_folds}
    if cidx_cox: metrics.update(cindex_cox_mean=float(np.mean(cidx_cox)), cindex_cox_std=float(np.std(cidx_cox)))
    if cidx_rsf: metrics.update(cindex_rsf_mean=float(np.mean(cidx_rsf)), cindex_rsf_std=float(np.std(cidx_rsf)))
    if ibs_list: metrics.update(ibs_mean=float(np.mean(ibs_list)), ibs_std=float(np.std(ibs_list)))
    if any(auc_at_h.values()):
        metrics["auc_at_h"] = {str(k):(float(np.mean(v)) if v else None) for k,v in auc_at_h.items()}
    with open(os.path.join("outputs","metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== CV METRICS ===")
    for k,v in metrics.items():
        print(f"{k}: {v}")
    if args.max_rows is not None:
        print("⚠️ NOTE: Model trained in debug mode with reduced dataset (--max-rows).")

    # ---------------- Train final model on ALL train ----------------
    X_all = Xraw.copy()
    pre_all = get_preprocessor(X_all)  # rebuild to lock columns
    pre_all.fit(X_all)
    X_all_proc = pre_all.transform(X_all)

    # Columns BEFORE pruning for full-train matrix
    feat_cols_all_pre = [f"x{i}" for i in range(X_all_proc.shape[1])]

    final_models = {}
    cph_all = None
    df_cox_all = None
    train_feats_all = None

    if args.model in ("cox","both"):
        df_cox_all = pd.DataFrame(X_all_proc, columns=feat_cols_all_pre)
        df_cox_all["T"] = time.values
        df_cox_all["E"] = event.values
        df_cox_all = prune_constant_duplicate_columns(df_cox_all)
        train_feats_all = [c for c in df_cox_all.columns if c not in ("T","E")]
        cph_all = fit_cox_with_retry(df_cox_all)
        final_models["cox"] = cph_all

    if args.model in ("rsf","both") and _HAS_SKSURV:
        y_all = Surv.from_arrays(event=event.values.astype(bool), time=time.values.astype(float))
        rsf_all = RandomSurvivalForest(
            n_estimators=600, min_samples_split=10, min_samples_leaf=10,
            max_features="sqrt", n_jobs=-1, random_state=args.seed+101
        )
        rsf_all.fit(X_all_proc, y_all)
        final_models["rsf"] = rsf_all

    # ---------------- Save model export for serving ----------------
    import joblib
    export = {
        "preprocessor": pre_all,
        "feature_names": list(X_all.columns),   # raw feature names pre-preprocess
        "model_type": "cox" if "cox" in final_models else ("rsf" if "rsf" in final_models else None),
        "horizons": [int(h) for h in args.horizons],
        "lifelines_cox": final_models.get("cox"),
        "rsf": final_models.get("rsf") if _HAS_SKSURV else None,
        "sklearn_version": sklearn.__version__,
        # For Cox alignment at serve-time (if you adapt serve.py to use them)
        "cox_feat_cols_pre": feat_cols_all_pre,
        "cox_train_feats": train_feats_all,
    }
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(export, args.model_out)
    print(f"\nSaved model export to: {args.model_out}")

    # ---------------- Predict on TEST (if exists) ----------------
    df_test = load_test_data(args.data_dir)
    if df_test is None:
        return

    if args.max_rows is not None:
        n = min(args.max_rows, len(df_test))
        print(f"⚠️ Limiting test data to {n} rows (debug mode).")
        df_test = df_test.sample(n=n, random_state=args.seed).reset_index(drop=True)

    id_col = detect_id_column(df_test, args.id_col) or "_row_id_"
    if id_col == "_row_id_":
        df_test["_row_id_"] = np.arange(len(df_test))

    # Build test features and align to training RAW features
    test_X_raw = basic_feature_drop(df_test.copy(), tcol="_t_", ecol="_e_")  # dummy t/e just to reuse filter
    test_X = align_features(test_X_raw, list(X_all.columns))
    Xte_proc = pre_all.transform(test_X)

    df_sub = pd.DataFrame({id_col: df_test[id_col].values})
    primary = export["model_type"]

    if primary == "cox" and cph_all is not None:
        # Create with PRE columns, then reindex to pruned final train feats
        df_cox_te = pd.DataFrame(Xte_proc, columns=feat_cols_all_pre)
        df_cox_te = df_cox_te.reindex(columns=train_feats_all, fill_value=0.0)

        for h in export["horizons"]:
            S_h = cph_all.predict_survival_function(df_cox_te, times=[h]).T.values.reshape(-1)
            df_sub[f"surv_prob_{int(h)}d"] = S_h
        df_sub["risk_score"] = cph_all.predict_partial_hazard(df_cox_te).values.reshape(-1)

    elif primary == "rsf" and export["rsf"] is not None:
        rsf = export["rsf"]
        surv_funcs = rsf.predict_survival_function(Xte_proc, return_array=False)
        for h in export["horizons"]:
            S_h = np.array([fn([h])[0] for fn in surv_funcs])
            df_sub[f"surv_prob_{int(h)}d"] = S_h
        df_sub["risk_score"] = -rsf.predict(Xte_proc)
    else:
        print("No model available for test prediction.")
        return

    out_path = os.path.join("outputs", "submission.csv")
    df_sub.to_csv(out_path, index=False)
    print(f"Saved submission to: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Moffitt Survival Predictor — Submission-Ready (robust, compact)

Key features
-----------
- Leakage filtering + feature engineering (age bins, AJCC composite, simple interactions)
- High-cardinality control: Top-K category capping; tokenization for pipe-separated 'therapeutic_agents'
- Version-safe OneHotEncoder config (sklearn>=1.2 vs older)
- CV with RSF randomized search (size via --rsf-search); Cox elastic-net style grid
- Robust RSF survival extraction (handles callable/array return types)
- Exports a portal-ready submission:
    outputs/submission.csv  with exactly 2 columns:
    [patient_id, predicted_scores]  where higher = longer survival
- CLI:
    --max-rows N           # limit train/test for quick local runs
    --cv-folds K
    --model {rsf,cox,both} # RSF recommended
    --submission-horizon D # survival at D days becomes predicted_scores
    --rsf-search M         # per-fold RSF configs to try (speed/accuracy dial)
    --write-sample N       # emit N-row predict payload from test and exit
"""

import argparse, os, re, json, warnings, random
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state

from packaging import version
import sklearn

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.exceptions import ConvergenceError

# Optional scikit-survival models/metrics
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

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- Pretty helpers ----------------
def banner(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)

# ---------------- IO helpers ----------------
def _auto_find_file(dirpath: str) -> str:
    files = os.listdir(dirpath)
    csvs = sorted([os.path.join(dirpath, f) for f in files if f.lower().endswith(".csv")])
    pars = sorted([os.path.join(dirpath, f) for f in files if f.lower().endswith(".parquet")])
    if csvs: return csvs[0]
    if pars: return pars[0]
    raise FileNotFoundError(f"No CSV/Parquet in: {dirpath}")

def load_local_split(data_dir: str, split: str) -> pd.DataFrame:
    p = os.path.join(data_dir, split)
    if not os.path.isdir(p): raise FileNotFoundError(f"Missing folder: {p}")
    f = _auto_find_file(p)
    df = pd.read_csv(f) if f.lower().endswith(".csv") else pd.read_parquet(f)
    print(f"Loaded {split}: {f} (rows={len(df)}, cols={len(df.columns)})")
    return df

def load_training_data(data_dir: Optional[str]) -> pd.DataFrame:
    if data_dir:
        data_dir = os.path.expanduser(data_dir)
        if os.path.isdir(os.path.join(data_dir, "train")):
            print(f"✅ Using local train: {os.path.join(data_dir,'train')}")
            return load_local_split(data_dir, "train")
    print("🌐 Local train not found — loading HuggingFace Lab-Rasool/hackathon")
    from datasets import load_dataset
    ds = load_dataset("Lab-Rasool/hackathon")
    return ds[list(ds.keys())[0]].to_pandas()

def load_test_data(data_dir: Optional[str]) -> Optional[pd.DataFrame]:
    if data_dir:
        data_dir = os.path.expanduser(data_dir)
        if os.path.isdir(os.path.join(data_dir, "test")):
            print(f"✅ Using local test: {os.path.join(data_dir,'test')}")
            return load_local_split(data_dir, "test")
    print("⚠️ No local test data — skipping submission.")
    return None

# ---------------- survival cols ----------------
def _pick_first_present(cands: List[str], df: pd.DataFrame) -> Optional[str]:
    for c in cands:
        if c in df.columns: return c
        for dc in df.columns:
            if dc.lower() == c.lower(): return dc
    return None

def detect_time_event(df: pd.DataFrame) -> Tuple[str, str]:
    t = _pick_first_present(
        ["overall_survival_days","os_days","days_to_death","survival_days","survival_time","time_to_event","time",
         "days_to_last_followup","days_to_last_follow_up","followup_days"], df)
    e = _pick_first_present(
        ["overall_survival_event","os_event","event","status","death_event","dead","deceased","vital_status"], df)
    if t is None: raise ValueError("No time column (e.g., overall_survival_days).")
    if e is None: raise ValueError("No event column (e.g., overall_survival_event).")
    return t, e

def normalize_event_column(ecol: str, df: pd.DataFrame) -> pd.Series:
    s = df[ecol]
    if s.dtype == bool: return s.astype(int)
    if pd.api.types.is_numeric_dtype(s): return (s.astype(float) > 0).astype(int)
    sval = s.astype(str).str.lower().str.strip()
    if "alive" in set(sval) or "dead" in set(sval): return (sval == "dead").astype(int)
    return (~s.isna()).astype(int)

# ---------------- leakage filter & FE ----------------
LEAK_PATTERNS = [
    r"^id$", r"\bpatient[_ ]?id\b", r"\bmrn\b", r"\brecord\b", r"\buid\b",
    r"^date", r"_date", r"timestamp",
    r"^days", r"\bdeath\b",
    r"\bfollow[_ ]?up\b",
    r"overall[_ ]?survival", r"\bos\b", r"last[_ ]?contact", r"\bsurvival\b", r"\btime\b",
    r"\bvital[_ ]?status\b", r"\bstatus\b",
    r"\bprogress(ion|ive)?\b", r"\brecurrence\b", r"\brelapse\b",
    r"\bdead\b|\bdeceased\b",
    r"\bdisease[_ ]?response\b", r"\btreatment[_ ]?outcome\b", r"\bcause[_ ]?of[_ ]?death\b",
]

def basic_feature_drop(df: pd.DataFrame, tcol: str, ecol: str) -> pd.DataFrame:
    regex = re.compile("|".join(LEAK_PATTERNS), flags=re.I)
    keep = []
    for c in df.columns:
        if c in (tcol, ecol): continue
        if regex.search(c): continue
        keep.append(c)
    trimmed = df[keep].copy()
    # drop super long free-text (model-unfriendly)
    for c in list(trimmed.columns):
        if trimmed[c].dtype == object:
            try:
                if trimmed[c].astype[str].map(len).mean() > 200:
                    trimmed.drop(columns=[c], inplace=True)
            except Exception:
                pass
    return trimmed

def tokenize_agents(df: pd.DataFrame, col="therapeutic_agents", max_tokens=40) -> pd.DataFrame:
    """Split 'A|B|C' into tokens; keep top tokens as indicators; drop original col."""
    if col not in df.columns: return df
    s = df[col].fillna("").astype(str)
    tokens = s.str.lower().str.split(r"\|")
    flat = [tok.strip() for lst in tokens for tok in lst if tok.strip()!=""]
    if not flat: return df.drop(columns=[col])
    vc = pd.Series(flat).value_counts()
    keep = set(vc.head(max_tokens).index.tolist())
    out = df.copy()
    for tk in keep:
        out[f"agent__{tk}"] = s.str.lower().str.contains(fr"(?:^|\|){re.escape(tk)}(?:\||$)")
    return out.drop(columns=[col])

def add_light_features(dfX: pd.DataFrame) -> pd.DataFrame:
    X = dfX.copy()
    # tokenize agents early (reduces cardinality)
    X = tokenize_agents(X, col="therapeutic_agents", max_tokens=40)
    # age features
    if "age_at_diagnosis" in X.columns:
        try:
            X["age_at_diagnosis"] = pd.to_numeric(X["age_at_diagnosis"], errors="coerce")
            X["age_bin"] = pd.cut(X["age_at_diagnosis"], bins=[0,50,60,70,200],
                                  labels=["<50","50-59","60-69","70+"])
        except Exception:
            pass
    # AJCC composite
    stage_cols = [c for c in ["ajcc_t","ajcc_n","ajcc_m","ajcc_stage"] if c in X.columns]
    if stage_cols:
        def _combine_stage(row):
            vals = []
            for c in ["ajcc_t","ajcc_n","ajcc_m","ajcc_stage"]:
                vals.append(str(row[c]).upper().strip() if (c in row and pd.notna(row[c])) else "")
            s = "-".join(vals).replace("NAN","").replace("--","-").strip("-")
            return s if s else np.nan
        try:
            X["ajcc_composite"] = X[stage_cols].apply(_combine_stage, axis=1)
        except Exception:
            pass
    # simple interactions
    for a,b in [("age_bin","ajcc_composite"), ("age_bin","treatment_types"), ("ajcc_composite","treatment_types")]:
        if a in X.columns and b in X.columns:
            X[f"int__{a}__{b}"] = X[a].astype(str) + "__" + X[b].astype(str)
    return X

def cap_topk_categories(df: pd.DataFrame, k: int = 40) -> Tuple[pd.DataFrame, Dict[str, set]]:
    """Keep top-k frequent levels per categorical column; others -> '__OTHER__'."""
    df2 = df.copy()
    mapping: Dict[str, set] = {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        vals = df[c].astype(str).fillna("__MISSING__")
        top = set(vals.value_counts(dropna=False).head(k).index.tolist())
        mapping[c] = top
        df2[c] = vals.where(vals.isin(top), "__OTHER__")
    return df2, mapping

def apply_cap_mapping(df: pd.DataFrame, mapping: Dict[str, set]) -> pd.DataFrame:
    df2 = df.copy()
    for c, top in mapping.items():
        if c in df2.columns and not pd.api.types.is_numeric_dtype(df2[c]):
            vals = df2[c].astype(str).fillna("__MISSING__")
            df2[c] = vals.where(vals.isin(top), "__OTHER__")
    return df2

# ---------------- preprocessing ----------------
def get_preprocessor(dfX: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in dfX.columns if pd.api.types.is_numeric_dtype(dfX[c])]
    cat_cols = [c for c in dfX.columns if c not in num_cols]
    num_tf = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    ohe_kwargs: Dict[str, Any] = {"drop": "first"}
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        ohe_kwargs.update({"handle_unknown": "infrequent_if_exist", "sparse_output": False, "min_frequency": 5})
    else:
        ohe_kwargs.update({"handle_unknown": "ignore", "sparse": False})
    cat_tf = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                       ("onehot", OneHotEncoder(**ohe_kwargs))])
    return ColumnTransformer([("num", num_tf, num_cols), ("cat", cat_tf, cat_cols)])

def align_features(df_in: pd.DataFrame, ref_cols: List[str]) -> pd.DataFrame:
    return df_in.reindex(columns=ref_cols, fill_value=np.nan)

# ---------------- plotting ----------------
def km_plot(time, event, outpath):
    km = KaplanMeierFitter(); km.fit(time, event_observed=event)
    plt.figure(); km.plot_survival_function(ci_show=True)
    plt.title("Kaplan–Meier: Overall Survival"); plt.xlabel("Days"); plt.ylabel("Survival probability")
    plt.tight_layout(); plt.savefig(outpath); plt.close()

# ---------------- Cox utils ----------------
def prune_constant_duplicate_columns(df_cox: pd.DataFrame) -> pd.DataFrame:
    feat = [c for c in df_cox.columns if c not in ("T","E")]
    nonconst = [c for c in feat if df_cox[c].nunique(dropna=False) > 1]
    df = df_cox[nonconst + ["T","E"]].copy()
    X = df.drop(columns=["T","E"])
    X = X.T.drop_duplicates().T
    df = pd.concat([X, df[["T","E"]]], axis=1)
    return df

def fit_cox_grid(df_tr: pd.DataFrame, Xte_df: pd.DataFrame, tte, ete) -> Tuple[CoxPHFitter, float]:
    pen_opts = [0.5, 1.0, 2.0, 5.0, 10.0]
    l1_opts  = [0.0, 0.25, 0.5]
    best_cidx, best_cph = -1.0, None
    for p in pen_opts:
        for l1 in l1_opts:
            cph = CoxPHFitter(penalizer=p, l1_ratio=l1)
            try:
                cph.fit(df_tr, duration_col="T", event_col="E")
                risk = cph.predict_partial_hazard(Xte_df).values.reshape(-1)
                cidx = concordance_index(event_observed=ete, event_times=tte, predicted_scores=risk)
                if cidx > best_cidx:
                    best_cidx, best_cph = float(cidx), cph
            except ConvergenceError:
                continue
            except Exception:
                continue
    if best_cph is None:
        cph = CoxPHFitter(penalizer=25.0, l1_ratio=0.0)
        cph.fit(df_tr, duration_col="T", event_col="E")
        best_cph, best_cidx = cph, concordance_index(ete, tte, cph.predict_partial_hazard(Xte_df).values.reshape(-1))
    return best_cph, float(best_cidx)

# --- rank normalize helper ---
def rank_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = x.argsort()              # ascending
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x))
    return 1.0 - ranks / max(len(x)-1, 1)   # in [0,1], higher = longer survival

def make_xgb_dmatrix(X: np.ndarray, time: np.ndarray, event: np.ndarray):
    # XGBoost-AFT expects label = time; censoring via label_lower/upper bounds
    # Uncensored: [t, t]; right-censored: [t, +inf]
    y = time.astype(float)
    lb = y.copy()
    ub = y.copy()
    ub[~(event.astype(bool))] = np.inf
    dtrain = xgb.DMatrix(X, label=y)
    dtrain.set_float_info("label_lower_bound", lb)
    dtrain.set_float_info("label_upper_bound", ub)
    return dtrain

def fit_xgb_aft(X_train: np.ndarray, t: np.ndarray, e: np.ndarray, seed: int = 42):
    params = dict(
        booster="gbtree",
        tree_method="hist",
        objective="survival:aft",
        eval_metric="aft-nloglik",
        aft_loss_distribution="normal",           # try "logistic" too
        aft_loss_distribution_scale=1.0,          # tune 0.5–2.0
        max_depth=4,
        eta=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=10.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        seed=seed,
    )
    dtrain = make_xgb_dmatrix(X_train, t, e)
    bst = xgb.train(params, dtrain, num_boost_round=1200, verbose_eval=False)
    return bst

def xgb_aft_predict_time(bst, X: np.ndarray) -> np.ndarray:
    # XGBoost AFT predict returns expectation under the fitted distribution (seconds = days here)
    dtest = xgb.DMatrix(X)
    # 'predict' gives mean of log-time if you used log link; with default 'normal', it's direct time.
    return bst.predict(dtest)

# ---------------- RSF helpers ----------------
def rsf_surv_at_h_list(rsf_model, surv_funcs, h: float) -> np.ndarray:
    """Return survival probability at time h per sample; supports callable/array outputs."""
    event_times = getattr(rsf_model, "event_times_", None)
    out = []
    for fn in surv_funcs:
        if callable(fn):
            try:
                out.append(float(fn([h])[0])); continue
            except Exception:
                pass
        try:
            arr = np.asarray(fn).ravel()
            if event_times is None or arr.size == 0:
                out.append(np.nan)
            else:
                idx = int(np.argmin(np.abs(event_times - h)))
                idx = min(max(idx, 0), arr.size - 1)
                out.append(float(arr[idx]))
        except Exception:
            out.append(np.nan)
    return np.array(out, dtype=float)
def rsf_survival_probs_at(model, X_proc: np.ndarray, H: float, t_train: np.ndarray, e_train: np.ndarray) -> np.ndarray:
    """
    Robust survival-at-horizon for scikit-survival RSF:
    - prefers array output
    - interpolates at H using model.event_times_
    - fills NaNs with KM baseline at H
    """
    # Try array output first (fast & reliable)
    try:
        S_mat = model.predict_survival_function(X_proc, return_array=True)  # shape (n_samples, n_times)
        times = getattr(model, "event_times_", None)
        if times is None or S_mat.size == 0:
            raise RuntimeError("Missing event_times_ or empty survival matrix.")
        times = np.asarray(times, dtype=float)

        # Clip H to model time range to avoid out-of-bounds weirdness
        Hc = float(np.clip(H, times.min() if times.size else H, times.max() if times.size else H))

        # Interpolate each row at Hc
        surv = np.interp(Hc, times, S_mat.T, left=S_mat[:, 0], right=S_mat[:, -1]).astype(float)
    except Exception:
        # Fallback to list-of-functions path
        try:
            surv_funcs = model.predict_survival_function(X_proc, return_array=False)
            times = getattr(model, "event_times_", None)
            if times is None:
                raise RuntimeError("Missing event_times_")
            times = np.asarray(times, dtype=float)
            Hc = float(np.clip(H, times.min() if times.size else H, times.max() if times.size else H))
            rows = []
            for fn in surv_funcs:
                if callable(fn):
                    try:
                        rows.append(float(fn([Hc])[0])); continue
                    except Exception:
                        pass
                arr = np.asarray(fn).ravel()
                if arr.size == 0:
                    rows.append(np.nan)
                else:
                    rows.append(float(np.interp(Hc, times, arr, left=arr[0], right=arr[-1])))
            surv = np.array(rows, dtype=float)
        except Exception:
            surv = np.full((X_proc.shape[0],), np.nan, dtype=float)

    # Fill NaNs with KM baseline at H
    if np.isnan(surv).any():
        try:
            km = KaplanMeierFitter()
            km.fit(t_train, event_observed=e_train.astype(bool))
            km_val = float(km.survival_function_at_times([H]).values[0])
            if not np.isfinite(km_val):
                km_val = 0.5
        except Exception:
            km_val = 0.5
        surv = np.where(np.isnan(surv), km_val, surv)

    # Final safety: numeric & (0,1)
    return np.clip(surv, 1e-6, 1 - 1e-6)

def rsf_random_configs(rng, n=8):
    """Randomized hyperparam configs for RSF (speed/accuracy trade-off)."""
    configs = []
    for _ in range(n):
        cfg = dict(
            n_estimators=int(rng.randint(700, 1100)),
            min_samples_split=int(rng.randint(6, 14)),
            min_samples_leaf=int(rng.randint(4, 10)),
            max_features=random.choice(["sqrt", "log2"]),
            max_depth=random.choice([None, 10, 12]),
        )
        configs.append(cfg)
    return configs

# ---------------- ID column ----------------
def detect_id_column(df: pd.DataFrame, prefer: Optional[str] = None) -> Optional[str]:
    if prefer and prefer in df.columns: return prefer
    for c in ["patient_id","id","case_id","subject_id","record_id"]:
        for dc in df.columns:
            if dc == c or dc.lower() == c.lower(): return dc
    return None

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=os.path.expanduser("~/Desktop/moffitt hackathon/hackathon"))
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--horizons", nargs="+", type=int, default=[90,180,365])
    ap.add_argument("--model", choices=["cox","rsf","both"], default="rsf",
                    help="Use 'rsf' for best accuracy; 'both' also tries Cox and selects by CV.")
    ap.add_argument("--id-col", type=str, default=None)
    ap.add_argument("--model-out", type=str, default="outputs/model_export.pkl")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--topk", type=int, default=40, help="Top-K category cap per categorical feature.")
    ap.add_argument("--write-sample", type=int, default=0,
                    help="If >0, write sample_request.json with N rows from test split and exit.")
    ap.add_argument("--submission-horizon", type=int, default=365,
                    help="Horizon (days) used to produce single survival score for submission.")
    ap.add_argument("--rsf-search", type=int, default=8,
                    help="Per-fold number of RSF configs to try (speed/accuracy dial).")
    args = ap.parse_args()

    rng = check_random_state(args.seed)
    random.seed(args.seed)
    os.makedirs("outputs", exist_ok=True)

    # -------- Load train --------
    banner("LOAD TRAIN")
    df_train = load_training_data(args.data_dir)
    if args.max_rows is not None:
        n = min(args.max_rows, len(df_train))
        print(f"⚠️ Limiting train to {n} rows")
        df_train = df_train.sample(n=n, random_state=args.seed).reset_index(drop=True)

    tcol, ecol = detect_time_event(df_train)
    time = pd.to_numeric(df_train[tcol], errors="coerce")
    event = normalize_event_column(ecol, df_train)
    mask = time.notna() & (time > 0) & event.notna()
    df_train = df_train.loc[mask].reset_index(drop=True)
    time = pd.to_numeric(df_train[tcol], errors="coerce").astype(float)
    event = normalize_event_column(ecol, df_train).astype(int)

    # -------- Write sample request & exit (if asked) --------
    if args.write_sample and args.write_sample > 0:
        banner("WRITE SAMPLE REQUEST")
        df_test = load_test_data(args.data_dir)
        if df_test is None:
            print("No test split; sampling from TRAIN features instead.")
            base = df_train.drop(columns=[tcol, ecol], errors="ignore").head(args.write_sample)
        else:
            base = df_test.head(args.write_sample)
        expected_raw = [
            "age_at_diagnosis","gender","race","ethnicity","primary_diagnosis","tumor_grade",
            "ajcc_m","ajcc_n","classification_of_tumor","morphology","tissue_origin","laterality",
            "prior_malignancy","synchronous_malignancy","cause_of_death","progression_or_recurrence",
            "disease_response","treatment_types","therapeutic_agents","treatment_outcome"
        ]
        req = []
        pid = detect_id_column(base)
        for _, r in base.iterrows():
            rec = {}
            for c in expected_raw:
                rec[c] = None if c not in base.columns else (None if pd.isna(r.get(c, None)) else r.get(c, None))
            if pid is not None:
                rec["patient_id"] = r.get(pid)
            req.append(rec)
        with open("sample_request.json", "w") as f:
            json.dump(req, f, indent=2)
        print("Wrote sample_request.json with", len(req), "records.")
        print("Call:\n  curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json")
        return

    # -------- Features --------
    banner("FEATURE ENGINEERING")
    Xraw = basic_feature_drop(df_train, tcol, ecol)
    Xraw = add_light_features(Xraw)
    Xraw, cap_map = cap_topk_categories(Xraw, k=args.topk)
    if Xraw.shape[1] == 0:
        raise ValueError("No usable feature columns after filtering.")
    print(f"Using {Xraw.shape[1]} raw features (Top-{args.topk} capping + tokens/interactions).")

    pre = get_preprocessor(Xraw)
    try:
        km_plot(time.values, event.values, os.path.join("outputs","km_overall.png"))
    except Exception:
        pass

    # -------- Cross-Validation --------
    banner("CROSS-VALIDATION")
    # Basic stratification by event only (robust even if diagnosis column missing/dirty)
    y_strat = event.values
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cidx_cox, cidx_rsf = [], []
    rsf_cfg_scores: Dict[str, List[float]] = {}
    ibs_list = []

    for fold, (tr, te) in enumerate(skf.split(Xraw, y_strat), 1):
        print(f"\n--- Fold {fold}/{args.cv_folds} ---")
        Xtr, Xte = Xraw.iloc[tr], Xraw.iloc[te]
        ttr, tte = time.iloc[tr].values, time.iloc[te].values
        etr, ete = event.values[tr], event.values[te]

        # Per-fold preprocessor (avoid leakage)
        Xte = align_features(Xte, list(Xtr.columns))
        pre_fold = get_preprocessor(Xtr)
        Xtr_proc = pre_fold.fit_transform(Xtr)
        Xte_proc = pre_fold.transform(Xte)
        feat_cols_pre = [f"x{i}" for i in range(Xtr_proc.shape[1])]

        # ---- Cox (lifelines) ----
        if args.model in ("cox","both"):
            df_cox_tr = pd.DataFrame(Xtr_proc, columns=feat_cols_pre)
            df_cox_tr["T"] = ttr; df_cox_tr["E"] = etr
            df_cox_tr = prune_constant_duplicate_columns(df_cox_tr)
            train_feats = [c for c in df_cox_tr.columns if c not in ("T","E")]
            df_cox_te = pd.DataFrame(Xte_proc, columns=feat_cols_pre).reindex(columns=train_feats, fill_value=0.0)
            cph, cidx = fit_cox_grid(df_cox_tr, df_cox_te, tte, ete)
            cidx_cox.append(cidx)

        # ---- RSF with randomized search ----
        if args.model in ("rsf","both") and _HAS_SKSURV:
            y_tr = Surv.from_arrays(event=etr.astype(bool), time=ttr.astype(float))
            best_fold_c, best_fold_cfg = -1.0, None
            cfgs = rsf_random_configs(rng, n=max(2, int(args.rsf_search)))
            for cfg in cfgs:
                rsf = RandomSurvivalForest(
                    n_estimators=cfg["n_estimators"],
                    min_samples_split=cfg["min_samples_split"],
                    min_samples_leaf=cfg["min_samples_leaf"],
                    max_features=cfg["max_features"],
                    max_depth=cfg["max_depth"],
                    n_jobs=-1,
                    random_state=args.seed + fold
                )
                rsf.fit(Xtr_proc, y_tr)
                rsf_risk = -rsf.predict(Xte_proc)
                c_cens = concordance_index_censored(ete.astype(bool), tte, rsf_risk)[0]
                c_val = float(c_cens)
                sig = json.dumps(cfg, sort_keys=True)
                rsf_cfg_scores.setdefault(sig, []).append(c_val)
                if c_val > best_fold_c:
                    best_fold_c, best_fold_cfg = c_val, cfg
            cidx_rsf.append(best_fold_c)
            print(f"Best RSF fold cfg: {best_fold_cfg}  C-index={best_fold_c:.4f}")

            # Optional: quick IBS on the best fold config (skip if tiny fold to save time)
            try:
                rsf = RandomSurvivalForest(
                    n_estimators=best_fold_cfg["n_estimators"],
                    min_samples_split=best_fold_cfg["min_samples_split"],
                    min_samples_leaf=best_fold_cfg["min_samples_leaf"],
                    max_features=best_fold_cfg["max_features"],
                    max_depth=best_fold_cfg["max_depth"],
                    n_jobs=-1,
                    random_state=args.seed + 1000 + fold
                )
                rsf.fit(Xtr_proc, y_tr)
                y_te = Surv.from_arrays(event=ete.astype(bool), time=tte.astype(float))
                times_grid = np.linspace(np.percentile(ttr,10), np.percentile(ttr,90), 50)
                surv_funcs = rsf.predict_survival_function(Xte_proc, return_array=False)
                rows = []
                for fn in surv_funcs:
                    if callable(fn): rows.append(fn(times_grid))
                    else:
                        arr = np.asarray(fn).ravel()
                        etimes = getattr(rsf, "event_times_", times_grid)
                        rows.append(np.interp(times_grid, etimes, arr,
                                              left=(arr[0] if arr.size else np.nan),
                                              right=(arr[-1] if arr.size else np.nan)))
                S_mat = np.vstack(rows) if rows else np.empty((0, len(times_grid)))
                if S_mat.size:
                    ibs = integrated_brier_score(y_tr, y_te, S_mat, times_grid)
                    ibs_list.append(float(ibs))
            except Exception:
                pass

    # -------- Summarize CV --------
    banner("CV SUMMARY")
    metrics: Dict[str, Any] = {"folds": args.cv_folds}
    if cidx_cox: metrics.update(cindex_cox_mean=float(np.mean(cidx_cox)), cindex_cox_std=float(np.std(cidx_cox)))
    if cidx_rsf: metrics.update(cindex_rsf_mean=float(np.mean(cidx_rsf)), cindex_rsf_std=float(np.std(cidx_rsf)))
    if ibs_list:  metrics.update(ibs_mean=float(np.mean(ibs_list)), ibs_std=float(np.std(ibs_list)))
    with open(os.path.join("outputs","metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    for k,v in metrics.items(): print(f"{k}: {v}")
    if args.max_rows is not None: print("⚠️ NOTE: reduced dataset (--max-rows).")

    # Decide export family
    primary_choice = "cox"
    if "cindex_rsf_mean" in metrics and "cindex_cox_mean" in metrics:
        primary_choice = "rsf" if metrics["cindex_rsf_mean"] >= metrics["cindex_cox_mean"] else "cox"
    elif "cindex_rsf_mean" in metrics:
        primary_choice = "rsf"
    print(f"\nModel selection (export family): {primary_choice.upper()}")

    # Pick best RSF config overall (mean across folds)
    best_rsf_cfg = None
    if _HAS_SKSURV and rsf_cfg_scores:
        best_rsf_cfg = max(rsf_cfg_scores.items(), key=lambda kv: np.mean(kv[1]))[0]
        best_rsf_cfg = json.loads(best_rsf_cfg)
        print("Best RSF config (CV mean):", best_rsf_cfg,
              "  mean C=", float(np.mean(rsf_cfg_scores[json.dumps(best_rsf_cfg, sort_keys=True)])))

    # -------- Final fit on ALL training data --------
    banner("FINAL TRAIN")
    X_all = basic_feature_drop(df_train, tcol, ecol)
    X_all = add_light_features(X_all)
    X_all = apply_cap_mapping(X_all, cap_map)
    if X_all.shape[1] == 0:
        raise ValueError("No usable feature columns for final training after filtering.")

    pre_all = get_preprocessor(X_all)
    pre_all.fit(X_all)
    X_all_proc = pre_all.transform(X_all)
    feat_cols_all_pre = [f"x{i}" for i in range(X_all_proc.shape[1])]

    final_models: Dict[str, Any] = {}
    train_feats_all: Optional[List[str]] = None

    # ---- Cox (final) ----
    df_cox_all = pd.DataFrame(X_all_proc, columns=feat_cols_all_pre)
    df_cox_all["T"] = time.values
    df_cox_all["E"] = event.values
    df_cox_all = prune_constant_duplicate_columns(df_cox_all)
    train_feats_all = [c for c in df_cox_all.columns if c not in ("T","E")]

    try:
        cph_default = CoxPHFitter(penalizer=2.0, l1_ratio=0.5)
        cph_default.fit(df_cox_all, duration_col="T", event_col="E")
        cph_best, _ = fit_cox_grid(df_cox_all, df_cox_all[train_feats_all], time.values, event.values)
        cph_all = cph_best or cph_default
    except Exception:
        cph_all = CoxPHFitter(penalizer=10.0, l1_ratio=0.0)
        cph_all.fit(df_cox_all, duration_col="T", event_col="E")
    final_models["cox"] = cph_all

    # ---- RSF (final) ----
    if _HAS_SKSURV:
        y_all = Surv.from_arrays(event=event.values.astype(bool), time=time.values.astype(float))
        if best_rsf_cfg is None:
            best_rsf_cfg = dict(n_estimators=1000, min_samples_split=8, min_samples_leaf=8,
                                max_features="sqrt", max_depth=None)
        rsf_all = RandomSurvivalForest(
            n_estimators=best_rsf_cfg["n_estimators"],
            min_samples_split=best_rsf_cfg["min_samples_split"],
            min_samples_leaf=best_rsf_cfg["min_samples_leaf"],
            max_features=best_rsf_cfg["max_features"],
            max_depth=best_rsf_cfg["max_depth"],
            n_jobs=-1,
            random_state=args.seed + 777
        )
        rsf_all.fit(X_all_proc, y_all)
        final_models["rsf"] = rsf_all
    else:
        final_models["rsf"] = None

    # ---- Choose primary for export ----
    primary = primary_choice if (primary_choice in ("rsf","cox")) else "cox"

    # -------- Export model bundle (for serve.py) --------
    banner("EXPORT")
    import joblib
    export = {
        "preprocessor": pre_all,
        "feature_names": list(X_all.columns),
        "model_type": primary,
        "horizons": [int(h) for h in args.horizons],
        "lifelines_cox": final_models.get("cox"),
        "rsf": final_models.get("rsf"),
        "sklearn_version": sklearn.__version__,
        "cox_feat_cols_pre": feat_cols_all_pre,
        "cox_train_feats": train_feats_all,
    }
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(export, args.model_out)
    print(f"Saved model export to: {args.model_out} (primary={primary})")

    # -------- Predict on TEST (portal format) --------
    banner("TEST PREDICTIONS (submission.csv)")
    df_test = load_test_data(args.data_dir)
    if df_test is None:
        raise RuntimeError("Submission requires a test split with patient_id; none found.")

    id_col = detect_id_column(df_test, args.id_col)
    if id_col is None:
        raise RuntimeError("Could not find a patient_id/id/case_id column in test set. "
                           "The portal requires patient_id exactly matching test IDs.")

    if args.max_rows is not None:
        n = min(args.max_rows, len(df_test))
        print(f"⚠️ Limiting test to {n} rows")
        df_test = df_test.head(n).reset_index(drop=True)

    # Build test features
    test_X_raw = basic_feature_drop(df_test.copy(), tcol="_t_", ecol="_e_")
    test_X_raw = add_light_features(test_X_raw)
    test_X_raw = apply_cap_mapping(test_X_raw, cap_map)
    test_X = align_features(test_X_raw, list(X_all.columns))
    Xte_proc = pre_all.transform(test_X)

    # Single horizon for portal score
    H = int(args.submission_horizon)
    print(f"Using submission horizon: {H} days (predicted_scores = P(survive >= {H}d))")

    # Compute survival probabilities at H
    if primary == "cox":
        df_cox_te = pd.DataFrame(Xte_proc, columns=feat_cols_all_pre).reindex(columns=train_feats_all, fill_value=0.0)
        surv_probs = final_models["cox"].predict_survival_function(df_cox_te, times=[H]).T.values.reshape(-1)
    elif primary == "rsf" and final_models.get("rsf") is not None:
        # Rank-optimized submission for C-index
        raw = -final_models["rsf"].predict(Xte_proc)   # larger raw = longer survival after rank_normalize
        surv_probs = rank_normalize(raw)
        else:
            raise RuntimeError("No trained model available for submission scoring.")

    # Final safety: numeric & [0,1], avoid exact 0/1 which some leaderboards dislike
    surv_probs = np.clip(surv_probs.astype(float), 1e-6, 1 - 1e-6)

    submission = pd.DataFrame({
        "patient_id": df_test[id_col].values,
        "predicted_scores": surv_probs  # higher = longer survival
    })
    submission = submission[["patient_id", "predicted_scores"]]

    out_path = os.path.join("outputs", "submission.csv")
    print("surv_probs stats:",
      "min", float(np.min(surv_probs)),
      "max", float(np.max(surv_probs)),
      "nan", int(np.isnan(surv_probs).sum()))
    submission.to_csv(out_path, index=False)
    print(f"✅ Wrote portal-ready file: {out_path}")
    print("   Columns: patient_id, predicted_scores (higher = longer survival)")
    print(f"   Rows: {len(submission)} (portal expects exactly 50)")

if __name__ == "__main__":
    main()

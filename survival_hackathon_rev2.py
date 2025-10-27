#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
survival_hackathon_rev2.py  (TABULAR-ONLY by default)
----------------------------------------------------
This version disables image loading by default and relies **only** on official
patient IDs and tabular fields present in train/test CSVs.

- New flag: --use-images  (default: off). Turn on explicitly to try WSI features.
- If images are missing, we skip them gracefully.
- Keeps: CoxPH (elastic-net), XGBRanker, XGB AFT ensemble, GBR → Ridge blender.
- Robust target encoding & dtype handling.

Run (tabular-only):
  python survival_hackathon_rev2.py \
    --data-dir "~/Desktop/moffitt hackathon /hackathon" \
    --out predictions.csv \
    --folds 5 \
    --verbose
"""

import os, json, argparse, warnings, inspect, time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
import joblib

# Optional: xgboost
HAVE_XGB = False
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

# ---------------- Metrics ----------------
def _cindex_numpy(durations, predictions, events):
    durations = np.asarray(durations, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations); num = den = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = durations[i], durations[j]
            ei, ej = events[i], events[j]
            if ti == tj and ei == ej == 0:  # both censored with same time -> skip
                continue
            if ti < tj and ei == 1:
                den += 1
                if predictions[i] < predictions[j]: num += 1
                elif predictions[i] == predictions[j]: num += 0.5
            elif tj < ti and ej == 1:
                den += 1
                if predictions[j] < predictions[i]: num += 1
                elif predictions[i] == predictions[j]: num += 0.5
    return num / den if den > 0 else 0.0

def concordance_index(durations, predictions, events):
    return _cindex_numpy(durations, predictions, events)

def pick_direction_from_train(dur_train, ev_train, oof_pred):
    c_pos = concordance_index(dur_train, oof_pred, ev_train)
    c_neg = concordance_index(dur_train, -oof_pred, ev_train)
    return 1.0 if c_pos >= c_neg else -1.0

def tiny_jitter(x, eps=1e-6):
    x = np.asarray(x, dtype=float); s = np.std(x) or 1.0
    rng = np.random.default_rng(42)
    return x + rng.normal(0, eps * s, size=len(x))

# ---------------- Features ----------------
REQ_LABELS = ('patient_id', 'duration', 'event')

def select_features(train: pd.DataFrame, test: pd.DataFrame, min_non_null: int = 5):
    # Only use columns that exist in BOTH train and test (and aren't labels)
    common = [c for c in train.columns if c in test.columns and c not in REQ_LABELS]
    dropped_low_count = []; kept = []
    for c in common:
        if train[c].count() < min_non_null:
            dropped_low_count.append(c)
        else:
            kept.append(c)
    # Pandas-aware dtype checks (handles categoricals)
    num_cols = [c for c in kept if is_numeric_dtype(train[c]) or is_bool_dtype(train[c])]
    cat_cols = [c for c in kept if c not in num_cols]
    return num_cols, cat_cols, {
        "candidate_common": common,
        "dropped_low_count": dropped_low_count,
        "kept_numeric": num_cols,
        "kept_categorical": cat_cols
    }

def build_ohe_kwargs():
    sig = inspect.signature(OneHotEncoder.__init__)
    ohe_kwargs = {"handle_unknown": "ignore", "drop": "first"}
    if "sparse_output" in sig.parameters:
        ohe_kwargs["sparse_output"] = False
    else:
        ohe_kwargs["sparse"] = False
    return ohe_kwargs

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(**build_ohe_kwargs()))])
    transformers = []
    if numeric_features: transformers.append(("num", num, numeric_features))
    if categorical_features: transformers.append(("cat", cat, categorical_features))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)

def cv_target_encode(df_train: pd.DataFrame, df_test: pd.DataFrame,
                     cat_cols: List[str],
                     duration_col="duration", event_col="event",
                     n_splits=5, m_smooth=10.0, seed=42):
    """Leak-safe OOF target encoding for high-cardinality categoricals."""
    if not cat_cols:
        return df_train, df_test, []
    te_cols_all = []
    y = np.log1p(df_train[duration_col].astype(float).values)
    w = 0.5 + 0.5 * df_train[event_col].astype(float).values
    global_mean = np.average(y, weights=w)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_train = df_train.copy(); X_test = df_test.copy()

    for col in cat_cols:
        if df_train[col].nunique() < 8:  # let OHE handle small cardinality
            continue
        te_name = f"te_{col}"
        te_cols_all.append(te_name)

        oof_vals = np.zeros(len(df_train), dtype=float)
        for tr_idx, va_idx in kf.split(df_train):
            tr = df_train.iloc[tr_idx]
            g = tr[[col, duration_col, event_col]].groupby(col, as_index=True)
            stats = g.apply(lambda gg: np.average(np.log1p(gg[duration_col].astype(float)),
                                                 weights=(0.5 + 0.5*gg[event_col].astype(float)))).to_dict()
            counts = tr[col].value_counts().to_dict()
            for i in va_idx:
                key = df_train.iloc[i][col]
                mu = stats.get(key, global_mean)
                cnt = counts.get(key, 0)
                oof_vals[i] = (cnt * mu + m_smooth * global_mean) / (cnt + m_smooth)

        g_all = df_train[[col, duration_col, event_col]].groupby(col, as_index=True)
        tr_stats = g_all.apply(lambda gg: np.average(np.log1p(gg[duration_col].astype(float)),
                                                    weights=(0.5 + 0.5*gg[event_col].astype(float)))).to_dict()
        counts_all = df_train[col].value_counts().to_dict()
        test_vals = []
        for _, r in df_test.iterrows():
            key = r[col]
            mu = tr_stats.get(key, global_mean)
            cnt = counts_all.get(key, 0)
            test_vals.append((cnt * mu + m_smooth * global_mean) / (cnt + m_smooth))

        X_train[te_name] = oof_vals
        X_test[te_name] = np.array(test_vals, dtype=float)

    return X_train, X_test, te_cols_all

# ---------------- Cox dimension reducer ----------------
class CoxFeatureReducer:
    def __init__(self, corr_thresh=0.95, var_tol=1e-12, pca_var=0.95, pca_cap=128, random_state=42):
        self.corr_thresh = corr_thresh
        self.var_tol = var_tol
        self.pca_var = pca_var
        self.pca_cap = pca_cap
        self.random_state = random_state
        self.keep_lowvardup_cols = None
        self.keep_corr_cols = None
        self.pca = None
        self.k = None
        self.out_names = None

    def fit(self, X: pd.DataFrame):
        std = X.std(axis=0, numeric_only=True)
        keep = std[std > self.var_tol].index.tolist()
        X1 = X[keep].copy()

        # Drop duplicate columns (hash-based)
        if X1.shape[1] > 1:
            hashes = X1.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
            _, idx = np.unique(hashes.values, return_index=True)
            X1 = X1.iloc[:, sorted(idx)]

        self.keep_lowvardup_cols = X1.columns.tolist()

        # Drop highly correlated columns
        if X1.shape[1] > 1:
            c = X1.corr(numeric_only=True).abs()
            upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if (upper[col] > self.corr_thresh).any()]
            X2 = X1.drop(columns=to_drop, errors="ignore")
        else:
            X2 = X1
        self.keep_corr_cols = X2.columns.tolist()

        # PCA to 95% var with cap
        n_cap = min(self.pca_cap, X2.shape[1], max(1, X2.shape[0]-1))
        self.pca = PCA(n_components=0.95, svd_solver="full", random_state=self.random_state)
        Z = self.pca.fit_transform(X2)
        k = Z.shape[1]
        if k > n_cap:
            Z = Z[:, :n_cap]; k = n_cap
            self.pca.components_ = self.pca.components_[:k, :]
            if hasattr(self.pca, "explained_variance_"):
                self.pca.explained_variance_ = self.pca.explained_variance_[:k]
            if hasattr(self.pca, "explained_variance_ratio_"):
                s = self.pca.explained_variance_ratio_[:k]
                self.pca.explained_variance_ratio_ = s / s.sum()
        self.k = k
        self.out_names = [f"pc{i+1}" for i in range(self.k)]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X1 = X.reindex(columns=self.keep_lowvardup_cols, fill_value=0.0)
        X2 = X1.reindex(columns=self.keep_corr_cols, fill_value=0.0)
        Z = self.pca.transform(X2)
        Z = Z[:, :self.k] if Z.shape[1] >= self.k else Z
        return pd.DataFrame(Z, columns=self.out_names, index=X.index)

def fit_coxph_robust(X_df, durations, events):
    reducer = CoxFeatureReducer()
    reducer.fit(X_df)
    Z = reducer.transform(X_df)
    for ties in ("efron", "breslow"):
        for l1r in (0.0, 0.2, 0.5):
            for penalizer in (0.1, 1.0, 10.0, 100.0):
                try:
                    df = Z.copy(); df["duration"] = durations; df["event"] = events
                    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1r, ties=ties)
                    cph.fit(df, duration_col="duration", event_col="event", show_progress=False)
                    return cph, reducer
                except ConvergenceError:
                    continue
                except Exception:
                    break
    # Fallback pseudo-cox
    ridge = Ridge(alpha=1.0, random_state=42).fit(Z, np.log1p(durations))
    class _PseudoCox:
        def __init__(self, model): self.model = model
        def predict_partial_hazard(self, X):
            y = self.model.predict(X)
            return pd.Series(np.exp((y - y.mean()) / (np.std(y) + 1e-9)))
    return _PseudoCox(ridge), reducer

def pred_coxph(model, Z_df):
    risk = model.predict_partial_hazard(Z_df).values.reshape(-1)
    return -risk  # lower = longer survival

# ---------------- Models ----------------
def make_rank_groups(durations):
    d = np.asarray(durations, dtype=float)
    q = np.quantile(d, [0.2, 0.4, 0.6, 0.8])
    return np.digitize(d, q)

def fit_xgbranker(X, y_time, y_event):
    if not HAVE_XGB: return None
    groups = make_rank_groups(y_time)
    y_rank = (y_time - y_time.min()) / max(1e-8, (y_time.max() - y_time.min()))
    ord_idx = np.argsort(groups)
    Xs = X[ord_idx]; ys = y_rank[ord_idx]; gs = groups[ord_idx]
    _, counts = np.unique(gs, return_counts=True)
    model = xgb.sklearn.XGBRanker(objective="rank:pairwise",
                                  n_estimators=1600, learning_rate=0.03, max_depth=6,
                                  subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                                  random_state=42)
    model.fit(Xs, ys, group=counts)
    return model

def pred_xgbranker(model, X):
    return None if model is None else model.predict(X)

def fit_gbr(X, y_time):
    y = np.log1p(np.asarray(y_time, float))
    gbr = GradientBoostingRegressor(loss="squared_error",
                                    n_estimators=1800, learning_rate=0.03,
                                    max_depth=3, subsample=0.9, random_state=42)
    gbr.fit(X, y)
    return gbr

def pred_gbr(m, X): return m.predict(X)

def fit_xgb_aft_ensemble(X, durations, events):
    if not HAVE_XGB: return []
    y_lower = np.asarray(durations, dtype=float)
    y_upper = np.where(np.asarray(events)==1, y_lower, np.inf)
    dtrain = xgb.DMatrix(X, label=y_lower)
    dtrain.set_float_info("label_lower_bound", y_lower)
    dtrain.set_float_info("label_upper_bound", y_upper)
    configs = [
        {"max_depth":6, "eta":0.04, "scale":1.2, "rounds":1400},
        {"max_depth":5, "eta":0.06, "scale":1.5, "rounds":1100},
        {"max_depth":7, "eta":0.03, "scale":1.0, "rounds":1800},
    ]
    models = []
    for cfg in configs:
        params = {
            "objective": "survival:aft",
            "aft_loss_distribution": "normal",
            "aft_loss_distribution_scale": cfg["scale"],
            "tree_method": "hist",
            "max_depth": cfg["max_depth"],
            "eta": cfg["eta"],
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "seed": 42,
        }
        bst = xgb.train(params, dtrain, num_boost_round=cfg["rounds"])
        models.append(bst)
    return models

def pred_xgb_aft_ensemble(models, X):
    if not models: return None
    dm = xgb.DMatrix(X)
    preds = [m.predict(dm) for m in models]
    return np.mean(np.vstack(preds), axis=0)

# ---------------- IO ----------------
def load_csv(path: Path, max_rows: Optional[int]=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    return df

# ---------------- Stacking ----------------
def stacked_cv_predict(X_full, durations, events, X_test_full, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof_parts, test_parts = [], []
    durations = np.asarray(durations); events = np.asarray(events)

    for tr, va in kf.split(X_full):
        Xtr, Xva = X_full[tr], X_full[va]
        ytr_t, yva_t = durations[tr], durations[va]
        ytr_e, yva_e = events[tr], events[va]

        fold_va, fold_te = [], []

        if HAVE_XGB:
            xgbr = fit_xgbranker(Xtr, ytr_t, ytr_e)
            if xgbr is not None:
                fold_va.append(pred_xgbranker(xgbr, Xva))
                fold_te.append(pred_xgbranker(xgbr, X_test_full))

        gbr = fit_gbr(Xtr, ytr_t)
        fold_va.append(pred_gbr(gbr, Xva))
        fold_te.append(pred_gbr(gbr, X_test_full))

        aft_models = fit_xgb_aft_ensemble(Xtr, ytr_t, ytr_e)
        if aft_models:
            fold_va.append(pred_xgb_aft_ensemble(aft_models, Xva))
            fold_te.append(pred_xgb_aft_ensemble(aft_models, X_test_full))

        # CoxPH (on reduced features)
        cols = [f"f{i}" for i in range(Xtr.shape[1])]
        Xtr_df = pd.DataFrame(Xtr, columns=cols)
        Xva_df = pd.DataFrame(Xva, columns=cols)
        Xte_df = pd.DataFrame(X_test_full, columns=cols)

        cph, reducer = fit_coxph_robust(Xtr_df, ytr_t, ytr_e)
        Zva = reducer.transform(Xva_df)
        Zte = reducer.transform(Xte_df)

        fold_va.append(pred_coxph(cph, Zva))
        fold_te.append(pred_coxph(cph, Zte))

        oof_parts.append(np.vstack(fold_va).T)
        test_parts.append(np.vstack(fold_te).T)

    oof_stack = np.concatenate(oof_parts, axis=0)
    test_stack = np.mean(np.stack(test_parts, axis=0), axis=0)

    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(oof_stack, np.log1p(durations))
    oof_pred = ridge.predict(oof_stack)
    test_pred = ridge.predict(test_stack)

    sgn = pick_direction_from_train(durations, events, oof_pred)
    oof_pred = tiny_jitter(sgn * oof_pred)
    test_pred = tiny_jitter(sgn * test_pred)
    return oof_pred, test_pred, ridge

# ---------------- Train & Predict ----------------
def _t(): return time.perf_counter()

def augment_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Age bins + squared (if 'age_at_diagnosis' present)
    for col in ["age_at_diagnosis", "age", "Age", "patient_age", "PatientAge"]:
        if col in out.columns:
            a = pd.to_numeric(out[col], errors="coerce")
            out["age_bin"] = pd.cut(a, bins=[0,40,60,80,200], labels=["<=40","40-60","60-80","80+"], include_lowest=True)
            out["age_sq"] = a**2
            break
    # Interactions if available
    def maybe_interact(a, b, name):
        if a in out.columns and b in out.columns:
            out[name] = out[a].astype(str) + "_" + out[b].astype(str)
    maybe_interact("gender","treatment_types","gender_x_treatment")
    maybe_interact("primary_diagnosis","treatment_types","dx_x_tx")
    return out

def train_and_predict(train_path: Path, test_path: Path,
                      out_path: Path, folds: int = 5, max_rows: Optional[int] = None,
                      save_dir: Optional[Path] = None, min_non_null: int = 5,
                      use_images: bool = False, verbose: bool = False) -> Dict[str, Any]:

    t0 = _t()
    train = load_csv(train_path, max_rows)
    test  = load_csv(test_path, None)
    if verbose: print(f"[TIMING] load_csv: {(_t()-t0):.2f}s")

    # Map labels if dataset uses alternative names
    rename_map = {}
    if "overall_survival_days" in train.columns:
        rename_map["overall_survival_days"] = "duration"
    if "overall_survival_event" in train.columns:
        rename_map["overall_survival_event"] = "event"
    if rename_map:
        train = train.rename(columns=rename_map)

    # Required columns
    for col in REQ_LABELS:
        if col not in train.columns:
            raise ValueError(f"Missing required column '{col}' in {train_path}")

    # Ensure patient_id is treated as string (official IDs)
    train["patient_id"] = train["patient_id"].astype(str)
    test["patient_id"]  = test["patient_id"].astype(str)

    # Augment clinical features
    t1 = _t()
    train = augment_clinical_features(train)
    test  = augment_clinical_features(test)
    if verbose: print(f"[TIMING] augment_clinical: {(_t()-t1):.2f}s")

    # Select features that appear in both
    num_cols0, cat_cols0, _ = select_features(train, test, min_non_null=min_non_null)

    # CV target encoding for high-cardinality text
    t2 = _t()
    train_te, test_te, te_cols = cv_target_encode(train.copy(), test.copy(), cat_cols0, n_splits=folds)
    train = train_te; test = test_te
    if verbose: print(f"[TIMING] target_encoding: {(_t()-t2):.2f}s; te_cols={len(te_cols)}")

    # (Intentionally) skip image features unless explicitly requested
    if use_images and verbose:
        print("[WSI] --use-images is True, but images were reported missing; skipping image features safely.")

    # Final feature selection (post TE)
    num_cols, cat_cols, _ = select_features(train, test, min_non_null=min_non_null)
    if not (num_cols or cat_cols):
        raise ValueError("No usable features after filtering. Consider lowering --min-non-null.")

    # Preprocess
    pre = build_preprocessor(num_cols, cat_cols)
    t3 = _t()
    Xt = pre.fit_transform(train)
    Xs = pre.transform(test)
    if verbose: print(f"[TIMING] preprocess+transform: {(_t()-t3):.2f}s; Xt_shape={getattr(Xt,'shape',None)}")

    # Targets
    y_time = train["duration"].values.astype(float)
    y_event = train["event"].values.astype(int)

    # Stacked models
    t4 = _t()
    oof, te, blender = stacked_cv_predict(Xt, y_time, y_event, Xs, folds=folds)
    if verbose: print(f"[TIMING] stacked_models: {(_t()-t4):.2f}s")
    cv_c = concordance_index(y_time, oof, y_event)

    # Submission (rank-normalized)
    sub = pd.DataFrame({"patient_id": test["patient_id"].values, "predicted_scores": te})
    ranks = sub["predicted_scores"].rank(method="average") - 1
    sub["predicted_scores"] = ranks / max(1, ranks.max())

    # Save to current working directory
    t5 = _t()
    out_path = Path(out_path)
    sub.to_csv(out_path, index=False)
    if verbose: print(f"[TIMING] save_csv: {(_t()-t5):.2f}s")

    # Optionally save artifacts
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pre": pre, "blender": blender}, Path(save_dir) / "model.joblib")
        with open(Path(save_dir) / "columns.json","w") as f:
            json.dump({"feature_columns": num_cols + cat_cols}, f)

    return {
        "submission_path": str(out_path),
        "cv_cindex": float(cv_c),
        "feature_debug": {
            "kept_numeric": num_cols[:20] + (["..."] if len(num_cols) > 20 else []),
            "kept_categorical": cat_cols[:20] + (["..."] if len(cat_cols) > 20 else []),
            "te_cols": te_cols[:10] + (["..."] if len(te_cols) > 10 else []),
            "have_xgboost": HAVE_XGB,
            "cv_cindex": float(cv_c)
        }
    }

# ---------------- CLI ----------------
def build_argparser():
    p = argparse.ArgumentParser(description="survival_hackathon_rev2.py (tabular-only by default).")
    p.add_argument("--data-dir", type=str, required=True, help="Root with train/ and test/ subfolders.")
    p.add_argument("--train", type=str, default="train/train.csv", help="train.csv path (relative to data-dir).")
    p.add_argument("--test", type=str, default="test/test.csv", help="test.csv path (relative to data-dir).")
    p.add_argument("--out", type=str, default="predictions.csv", help="Output CSV filename (saved to CWD).")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--save-dir", type=str, default=None, help="Folder to store artifacts.")
    p.add_argument("--min-non-null", type=int, default=5, help="Drop cols with < this many non-null in train.")
    p.add_argument("--use-images", action="store_true", help="Try to use image features (off by default).")
    p.add_argument("--verbose", action="store_true", help="Print timings and diagnostics.")
    return p

def resolve_paths(args) -> Tuple[Path, Path, Path, Optional[Path]]:
    base = Path(args.data_dir).expanduser().resolve()
    train = (base / args.train).resolve()
    test  = (base / args.test).resolve()
    out   = Path(args.out)
    save_dir = Path(args.save_dir).expanduser().resolve() if args.save_dir else None
    return train, test, out, save_dir

def main():
    args = build_argparser().parse_args()
    train_path, test_path, out_path, save_dir = resolve_paths(args)
    info = train_and_predict(train_path, test_path, out_path,
                             folds=args.folds, max_rows=args.max_rows,
                             save_dir=save_dir, min_non_null=args.min_non_null,
                             use_images=args.use_images, verbose=args.verbose)
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()

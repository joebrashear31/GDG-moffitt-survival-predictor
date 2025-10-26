#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
survival_hackathon_plus_mm_autopath_v7.py
-----------------------------------------
Accuracy-focused upgrades over v6:
  • Overlapping WSI tiles by default (stride = tile_size//2)
  • Stain normalization per tile ( --stain {none,reinhard,macenko} ; default=reinhard )
  • CV Target Encoding for high-cardinality categoricals (leak-safe OOF with smoothing)
  • Deeper XGBoost AFT sweep (3-param ensemble per fold)
  • Cox (robust) kept always; Ranker + GBR kept; OOF sign logic
  • Saves CSV in current working directory

New deps: scikit-image, scipy
"""

import os, json, argparse, warnings, inspect, math, random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

# skimage & scipy
from skimage import color as skcolor
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from scipy.ndimage import binary_opening, binary_closing

# Optional readers
HAVE_OPENSLIDE = False
try:
    import openslide  # type: ignore
    HAVE_OPENSLIDE = True
except Exception:
    HAVE_OPENSLIDE = False

HAVE_TIFFFILE = False
try:
    import tifffile as tiff
    HAVE_TIFFFILE = True
except Exception:
    HAVE_TIFFFILE = False

HAVE_IMAGEIO = False
try:
    import imageio.v3 as iio
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False

# ML & survival
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
HAVE_XGB = False
try:
    import xgboost as xgb
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# ---------- C-index ----------
def _cindex_numpy(durations, predictions, events):
    durations = np.asarray(durations, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    events = np.asarray(events, dtype=int)
    n = len(durations); num = den = 0.0
    if len(predictions) != n or len(events) != n:
        raise ValueError(f"C-index size mismatch (dur={n}, pred={len(predictions)}, ev={len(events)})")
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = durations[i], durations[j]
            ei, ej = events[i], events[j]
            if ti == tj and ei == ej == 0:
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

# ---------- Feature selection & target encoding ----------
REQ_LABELS = ('patient_id', 'duration', 'event')

def select_features(train: pd.DataFrame, test: pd.DataFrame,
                    min_non_null: int = 5) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    common = [c for c in train.columns if c in test.columns and c not in REQ_LABELS]
    dropped_low_count = []; kept = []
    for c in common:
        if train[c].count() < min_non_null:
            dropped_low_count.append(c)
        else:
            kept.append(c)
    num_cols = [c for c in kept if np.issubdtype(train[c].dtype, np.number)]
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
                     n_splits=5, m_smooth=10.0, seed=42) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Leak-safe OOF target encoding for high-cardinality categoricals.
    Target = log1p(duration) with censoring weight = 0.5 + 0.5*event (uncensored weighs more).
    """
    if not cat_cols:
        return df_train, df_test, []
    rng = np.random.default_rng(seed)
    te_cols_all = []
    y = np.log1p(df_train[duration_col].astype(float).values)
    w = 0.5 + 0.5 * df_train[event_col].astype(float).values
    global_mean = np.average(y, weights=w)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_train = df_train.copy()
    X_test = df_test.copy()

    for col in cat_cols:
        if df_train[col].nunique() < 8:  # leave small-cardinality to OHE
            continue
        te_name = f"te_{col}"
        te_cols_all.append(te_name)
        oof_vals = np.zeros(len(df_train), dtype=float)
        # Precompute category stats
        for tr_idx, va_idx in kf.split(df_train):
            tr = df_train.iloc[tr_idx]
            ytr = y[tr_idx]; wtr = w[tr_idx]
            stats = tr.groupby(col).apply(lambda g: np.average(np.log1p(g[duration_col].astype(float)), weights=(0.5 + 0.5*g[event_col].astype(float)))).to_dict()
            # smoothing
            counts = tr[col].value_counts().to_dict()
            for i in va_idx:
                key = df_train.iloc[i][col]
                mu = stats.get(key, global_mean)
                cnt = counts.get(key, 0)
                oof_vals[i] = (cnt * mu + m_smooth * global_mean) / (cnt + m_smooth)
        # test transform
        tr_stats = df_train.groupby(col).apply(lambda g: np.average(np.log1p(g[duration_col].astype(float)), weights=(0.5 + 0.5*g[event_col].astype(float)))).to_dict()
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

# ---------- WSI utils: reading, staining, tiling ----------
def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(arr)
    if not np.any(mask):
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(arr[mask], [1, 99])
    if hi <= lo: hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0
    return arr.astype(np.uint8)

def read_wsi_large(path: Path, max_dim: int = 10000) -> Image.Image:
    if HAVE_OPENSLIDE:
        try:
            slide = openslide.OpenSlide(str(path))
            w, h = slide.dimensions
            scale = max(w, h) / float(max_dim)
            img = slide.get_thumbnail((int(w/scale), int(h/scale))).convert("RGB")
            slide.close()
            return img
        except Exception:
            pass
    if HAVE_TIFFFILE:
        try:
            with tiff.TiffFile(str(path)) as tf:
                arr = tf.asarray()
            if arr.ndim == 3 and arr.shape[-1] in (3,4):
                arr = arr[..., :3]
            elif arr.ndim > 2:
                arr = arr[0]
            if arr.dtype != np.uint8: arr = _rescale_to_uint8(arr)
            img = Image.fromarray(arr).convert("RGB")
            img.thumbnail((max_dim, max_dim))
            return img
        except Exception:
            pass
    with Image.open(path) as im:
        im.draft("RGB", (max_dim, max_dim))
        im = im.convert("RGB")
        im.thumbnail((max_dim, max_dim))
        return im

def hsv_tissue_mask(img: Image.Image) -> np.ndarray:
    hsv = img.convert("HSV")
    s = np.asarray(hsv.getchannel("S"), dtype=np.uint8)
    hist, _ = np.histogram(s, bins=256, range=(0,255))
    cdf = hist.cumsum()
    total = cdf[-1]
    sumB = wB = 0.0
    maximum = 0.0
    threshold = 0
    sum1 = np.dot(np.arange(256), hist)
    for t in range(256):
        wB += hist[t]; 
        if wB == 0: continue
        wF = total - wB
        if wF == 0: break
        sumB += t * hist[t]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > maximum:
            maximum = between; threshold = t
    mask = (s > threshold).astype(np.uint8)
    mask = binary_opening(mask, structure=np.ones((3,3))).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((3,3))).astype(np.uint8)
    return mask

# ---- Stain normalization ----
def reinhard_normalize(rgb: np.ndarray,
                       target_means=(60.0, 5.0, 20.0),
                       target_stds=(10.0, 3.0, 8.0),
                       mask: Optional[np.ndarray]=None) -> np.ndarray:
    """Reinhard LAB color transfer to fixed reference stats. rgb uint8 -> uint8"""
    if mask is None:
        mask = np.ones(rgb.shape[:2], dtype=bool)
    lab = skcolor.rgb2lab(rgb / 255.0)
    m = mask.astype(bool)
    L, A, B = lab[...,0][m], lab[...,1][m], lab[...,2][m]
    Lm, Am, Bm = L.mean(), A.mean(), B.mean()
    Ls, As, Bs = L.std() + 1e-6, A.std() + 1e-6, B.std() + 1e-6
    Lt, At, Bt = target_means
    Lts, Ats, Bts = target_stds
    lab[...,0] = (lab[...,0] - Lm) / Ls * Lts + Lt
    lab[...,1] = (lab[...,1] - Am) / As * Ats + At
    lab[...,2] = (lab[...,2] - Bm) / Bs * Bts + Bt
    rgbn = (np.clip(skcolor.lab2rgb(lab), 0, 1) * 255.0).astype(np.uint8)
    return rgbn

def macenko_normalize(rgb: np.ndarray, alpha=0.1, beta=0.15) -> np.ndarray:
    """
    Simplified Macenko stain normalization; returns uint8 RGB.
    """
    # Convert to OD space
    OD = -np.log((rgb.astype(np.float32)+1)/256.0)
    ODhat = OD.reshape(-1,3)
    ODhat = ODhat[(ODhat>alpha).any(axis=1)]
    if ODhat.size == 0:
        return rgb
    # SVD
    _, _, V = np.linalg.svd(ODhat, full_matrices=False)
    vec1, vec2 = V[0], V[1]
    # Project and get angle distribution
    proj = ODhat @ np.vstack([vec1, vec2]).T
    phi = np.arctan2(proj[:,1], proj[:,0])
    vmin, vmax = np.percentile(phi, [beta*100, (1-beta)*100])
    vH = vec1*np.cos(vmax) + vec2*np.sin(vmax)
    vE = vec1*np.cos(vmin) + vec2*np.sin(vmin)
    HE = np.array([vH/np.linalg.norm(vH), vE/np.linalg.norm(vE)]).T
    # Concentrations
    C = np.linalg.lstsq(HE, OD.reshape(-1,3).T, rcond=None)[0]
    # Normalize each stain to standard max
    maxC = np.percentile(C, 99, axis=1)
    C = C / (maxC[:,None] + 1e-6)
    # Reconstruct
    ODn = (HE @ (C)).T.reshape(rgb.shape)
    In = (np.exp(-ODn) * 255.0).clip(0,255).astype(np.uint8)
    return In

def apply_stain_normalization(tile: Image.Image, method: str, tissue_mask_tile: Optional[np.ndarray]) -> Image.Image:
    if method == "none":
        return tile
    rgb = np.asarray(tile.convert("RGB"), dtype=np.uint8)
    m = tissue_mask_tile.astype(bool) if tissue_mask_tile is not None else None
    try:
        if method == "reinhard":
            rgbn = reinhard_normalize(rgb, mask=m)
        elif method == "macenko":
            rgbn = macenko_normalize(rgb)
        else:
            rgbn = rgb
        return Image.fromarray(rgbn, mode="RGB")
    except Exception:
        return tile

# ---- Tiling & features ----
def tile_generator(img: Image.Image, tile_size=512, stride=256):
    W, H = img.size
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            yield x, y, img.crop((x, y, x+tile_size, y+tile_size))

def tile_features(tile: Image.Image, tissue_mask_tile: Optional[np.ndarray]) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    rgb = tile.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)
    gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.uint8)

    if tissue_mask_tile is None:
        tm = np.ones(gray.shape, dtype=bool)
    else:
        tm = tissue_mask_tile.astype(bool)
        if tm.mean() < 0.05:
            return {"skip": 1.0}

    # Color stats
    for i, cname in enumerate(["r","g","b"]):
        c = arr[..., i][tm]
        if c.size == 0: c = np.array([0], dtype=np.uint8)
        feats[f"t_{cname}_mean"] = float(np.mean(c))
        feats[f"t_{cname}_std"]  = float(np.std(c))
        for p in (5,25,50,75,95):
            feats[f"t_{cname}_p{p}"] = float(np.percentile(c, p))

    # Edges & Laplacian var
    grayf = gray.astype(np.float32)
    gx = np.zeros_like(grayf); gy = np.zeros_like(grayf)
    gx[:,1:-1] = (grayf[:,2:] - grayf[:,:-2]) * 0.5
    gy[1:-1,:] = (grayf[2:,:] - grayf[:-2,:]) * 0.5
    mag = np.hypot(gx, gy)[tm]
    feats["t_edge_density"] = float((mag > np.percentile(mag, 90)).mean()) if mag.size else 0.0
    lap = (-grayf[:-2,1:-1] - grayf[2:,1:-1] - grayf[1:-1,:-2] - grayf[1:-1,2:] + 4*grayf[1:-1,1:-1])
    feats["t_laplacian_var"] = float(np.var(lap))

    # LBP (uniform)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")[tm]
    hist, _ = np.histogram(lbp, bins=np.arange(0, 10+1), range=(0,10), density=True)
    for i, h in enumerate(hist):
        feats[f"t_lbp_u{i:02d}"] = float(h)

    # GLCM
    gl = (gray / 4).astype(np.uint8)  # 64 levels
    if tm.mean() > 0.2:
        distances = [1, 3]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        gco = graycomatrix(gl, distances=distances, angles=angles, levels=64, symmetric=True, normed=True)
        for prop in ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]:
            vals = graycoprops(gco, prop).ravel()
            feats[f"t_glcm_{prop}_mean"] = float(np.mean(vals))
            feats[f"t_glcm_{prop}_p90"] = float(np.percentile(vals, 90))
    return feats

def aggregate_tile_features(rows: List[Dict[str, float]]) -> Dict[str, float]:
    rows = [r for r in rows if "skip" not in r]
    if not rows:
        return {}
    keys = sorted({k for r in rows for k in r.keys()})
    agg: Dict[str, float] = {}
    for k in keys:
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0: continue
        agg[f"wsi_{k}_mean"] = float(np.mean(vals))
        agg[f"wsi_{k}_p90"] = float(np.percentile(vals, 90))
    agg["wsi_tiles_used"] = float(len(rows))
    return agg

def select_top_tiles_by_variance(feats_rows: List[Dict[str,float]], keep: int) -> List[Dict[str,float]]:
    if keep is None or keep <= 0 or len(feats_rows) <= keep:
        return feats_rows
    # Define a variance proxy: sum of color stds + laplacian_var + GLCM contrast
    scores = []
    for r in feats_rows:
        s = 0.0
        for k in ["t_r_std","t_g_std","t_b_std","t_laplacian_var","t_glcm_contrast_mean"]:
            s += float(r.get(k, 0.0))
        scores.append(s)
    idx = np.argsort(scores)[::-1][:keep]
    return [feats_rows[i] for i in idx]

def build_wsi_feature_row_tiled(img_path: Path, max_dim: int, tile_size: int, stride: int, max_tiles: int,
                                stain: str = "reinhard") -> Dict[str, float]:
    img = read_wsi_large(img_path, max_dim=max_dim)
    mask = hsv_tissue_mask(img)
    rows = []
    for x, y, tile in tile_generator(img, tile_size=tile_size, stride=stride):
        tm = mask[y:y+tile_size, x:x+tile_size]
        # Stain normalization per tile
        tile = apply_stain_normalization(tile, stain, tm)
        feats = tile_features(tile, tm)
        if feats.get("skip", 0.0) == 1.0:
            continue
        rows.append(feats)
    # pick top-variance tiles if too many
    if max_tiles is not None and max_tiles > 0 and len(rows) > max_tiles:
        rows = select_top_tiles_by_variance(rows, keep=max_tiles)
    slide_feats = aggregate_tile_features(rows)
    return slide_feats

def image_features_dataframe(df_ids: pd.DataFrame, id_col: str, image_dir: Path,
                             image_ext: str = "tif", max_images: Optional[int] = None,
                             max_dim: int = 10000, tile_size: int = 512, stride: int = 256,
                             max_tiles: int = 400, stain: str = "reinhard") -> pd.DataFrame:
    rows = []; processed = 0
    for _, row in df_ids.iterrows():
        pid = str(row[id_col])
        fname = f"{pid}.{image_ext}"
        img_path = image_dir / fname
        if not img_path.exists():
            cand = None
            for alt in [fname, fname.upper(), fname.lower()]:
                p = image_dir / alt
                if p.exists(): cand = p; break
            if cand is None:
                rows.append({"patient_id": pid})
                continue
            img_path = cand
        try:
            feats = build_wsi_feature_row_tiled(img_path, max_dim=max_dim, tile_size=tile_size, stride=stride,
                                                max_tiles=max_tiles, stain=stain)
        except UnidentifiedImageError:
            feats = {}
        feats["patient_id"] = pid
        rows.append(feats)
        processed += 1
        if max_images and processed >= max_images: break
    if not rows:
        return pd.DataFrame({"patient_id": df_ids[id_col].astype(str)})
    return pd.DataFrame(rows)

# ---------- Clinical feature augmentation ----------
def augment_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["age", "Age", "patient_age", "PatientAge"]:
        if col in out.columns:
            a = pd.to_numeric(out[col], errors="coerce")
            out["age_bin"] = pd.cut(a, bins=[0,40,60,80,200], labels=["<=40","40-60","60-80","80+"], include_lowest=True)
            out["age_sq"] = a**2
            break
    def maybe_interact(a, b, name):
        if a in out.columns and b in out.columns:
            out[name] = out[a].astype(str) + "_" + out[b].astype(str)
    maybe_interact("sex","treatment", "sex_x_treatment")
    maybe_interact("gender","treatment","gender_x_treatment")
    return out

# ---------- XGBoost models ----------
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
                                  n_estimators=1400, learning_rate=0.035, max_depth=6,
                                  subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                                  random_state=42)
    model.fit(Xs, ys, group=counts)
    return model

def pred_xgbranker(model, X):
    return None if model is None else model.predict(X)

def fit_gbr(X, y_time):
    y = np.log1p(np.asarray(y_time, float))
    gbr = GradientBoostingRegressor(loss="squared_error",
                                    n_estimators=1600, learning_rate=0.03,
                                    max_depth=3, subsample=0.9, random_state=42)
    gbr.fit(X, y)
    return gbr

def pred_gbr(m, X): return m.predict(X)

def fit_xgb_aft_ensemble(X, durations, events):
    if not HAVE_XGB:
        return []
    y_lower = np.asarray(durations, dtype=float)
    y_upper = np.where(np.asarray(events)==1, y_lower, np.inf)
    dtrain = xgb.DMatrix(X, label=y_lower)
    dtrain.set_float_info("label_lower_bound", y_lower)
    dtrain.set_float_info("label_upper_bound", y_upper)
    configs = [
        {"max_depth":5, "eta":0.05, "scale":1.2, "rounds":1200},
        {"max_depth":6, "eta":0.035, "scale":1.0, "rounds":1600},
        {"max_depth":4, "eta":0.07, "scale":1.5, "rounds":900},
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

# ---------- Robust Cox with deterministic reducer ----------
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
        if X1.shape[1] > 1:
            hashes = X1.apply(lambda s: pd.util.hash_pandas_object(s, index=False).sum())
            _, idx = np.unique(hashes.values, return_index=True)
            X1 = X1.iloc[:, sorted(idx)]
        self.keep_lowvardup_cols = X1.columns.tolist()
        if X1.shape[1] > 1:
            c = X1.corr(numeric_only=True).abs()
            upper = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))
            to_drop = [col for col in upper.columns if (upper[col] > self.corr_thresh).any()]
            X2 = X1.drop(columns=to_drop, errors="ignore")
        else:
            X2 = X1
        self.keep_corr_cols = X2.columns.tolist()
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
        for penalizer in (1.0, 10.0, 100.0, 0.1):
            try:
                df = Z.copy()
                df["duration"] = durations
                df["event"] = events
                cph = CoxPHFitter(penalizer=penalizer, l1_ratio=0.0, ties=ties)
                cph.fit(df, duration_col="duration", event_col="event", show_progress=False)
                return cph, reducer
            except ConvergenceError:
                continue
            except Exception:
                break
    ridge = Ridge(alpha=1.0, random_state=42).fit(Z, np.log1p(durations))
    class _PseudoCox:
        def __init__(self, model): self.model = model
        def predict_partial_hazard(self, X):
            y = self.model.predict(X)
            return pd.Series(np.exp((y - y.mean()) / (np.std(y) + 1e-9)))
    return _PseudoCox(ridge), reducer

def pred_coxph(model, Z_df):
    risk = model.predict_partial_hazard(Z_df).values.reshape(-1)
    return -risk

# ---------- IO + Merge ----------
def load_csv(path: Path, max_rows: Optional[int]=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)
    return df

def add_image_features(train: pd.DataFrame, test: pd.DataFrame,
                       id_col: str,
                       train_csv_path: Path,
                       test_csv_path: Path,
                       image_ext: str = "tif",
                       max_images: Optional[int] = None,
                       max_dim: int = 10000,
                       tile_size: int = 512,
                       stride: int = 256,
                       max_tiles: int = 400,
                       stain: str = "reinhard") -> Tuple[pd.DataFrame, pd.DataFrame]:
    img_dir_train = train_csv_path.parent / "images"
    img_dir_test  = test_csv_path.parent  / "images"
    if not img_dir_train.exists() or not img_dir_test.exists():
        warnings.warn(f"Expected image directories at {img_dir_train} and {img_dir_test}; continuing without images.")
        return train, test
    tr_img_df = image_features_dataframe(train[[id_col]].copy(), id_col, img_dir_train, image_ext, max_images,
                                         max_dim, tile_size, stride, max_tiles, stain)
    te_img_df = image_features_dataframe(test[[id_col]].copy(),  id_col, img_dir_test,  image_ext, max_images,
                                         max_dim, tile_size, stride, max_tiles, stain)
    train2 = train.merge(tr_img_df, on=id_col, how="left")
    test2  = test.merge(te_img_df, on=id_col, how="left")
    return train2, test2

# ---------- Stacking ----------
def stacked_cv_predict(X_full, durations, events, X_test_full, folds=5):
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    oof_parts, test_parts = [], []
    durations = np.asarray(durations); events = np.asarray(events)

    for tr, va in kf.split(X_full):
        Xtr, Xva = X_full[tr], X_full[va]
        ytr_t, yva_t = durations[tr], durations[va]
        ytr_e, yva_e = events[tr], events[va]

        fold_va, fold_te = [], []

        # Ranker
        if HAVE_XGB:
            xgbr = fit_xgbranker(Xtr, ytr_t, ytr_e)
            if xgbr is not None:
                fold_va.append(pred_xgbranker(xgbr, Xva))
                fold_te.append(pred_xgbranker(xgbr, X_test_full))

        # GBR
        gbr = fit_gbr(Xtr, ytr_t)
        fold_va.append(pred_gbr(gbr, Xva))
        fold_te.append(pred_gbr(gbr, X_test_full))

        # AFT ensemble
        aft_models = fit_xgb_aft_ensemble(Xtr, ytr_t, ytr_e)
        if aft_models:
            fold_va.append(pred_xgb_aft_ensemble(aft_models, Xva))
            fold_te.append(pred_xgb_aft_ensemble(aft_models, X_test_full))

        # Cox (robust)
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

# ---------- Train + Predict ----------
def train_and_predict(train_path: Path, test_path: Path,
                      out_path: Path, folds: int = 5, max_rows: Optional[int] = None,
                      model_kind: str = "stack", save_dir: Optional[Path] = None,
                      image_ext: str = "tif", max_images: Optional[int] = None,
                      max_dim: int = 10000, tile_size: int = 512, stride: int = 256, max_tiles: int = 400,
                      min_non_null: int = 5, stain: str = "reinhard") -> Dict[str, Any]:

    train = load_csv(train_path, max_rows)
    test  = load_csv(test_path, None)

    # Standard column rename if provided
    rename_map = {}
    if "overall_survival_days" in train.columns:
        rename_map["overall_survival_days"] = "duration"
    if "overall_survival_event" in train.columns:
        rename_map["overall_survival_event"] = "event"
    if rename_map: train = train.rename(columns=rename_map)

    for col in REQ_LABELS:
        if col not in train.columns:
            raise ValueError(f"Missing required column '{col}' in {train_path}")

    # Augment clinical features
    train = augment_clinical_features(train)
    test  = augment_clinical_features(test)

    # Target encoding (high-cardinality cats) before OHE
    num_cols0, cat_cols0, _ = select_features(train, test, min_non_null=min_non_null)
    train_te, test_te, te_cols = cv_target_encode(train.copy(), test.copy(), cat_cols0, n_splits=folds)
    train = train_te
    test = test_te

    # Image features (tiled + stain norm)
    train, test = add_image_features(train, test, id_col="patient_id",
                                     train_csv_path=train_path, test_csv_path=test_path,
                                     image_ext=image_ext, max_images=max_images,
                                     max_dim=max_dim, tile_size=tile_size, stride=stride, max_tiles=max_tiles,
                                     stain=stain)

    # Feature selection after TE & images
    num_cols, cat_cols, dbg = select_features(train, test, min_non_null=min_non_null)
    if not (num_cols or cat_cols):
        raise ValueError("No usable features after filtering. Consider lowering --min-non-null.")

    # Preprocess
    pre = build_preprocessor(num_cols, cat_cols)
    Xt = pre.fit_transform(train)
    Xs = pre.transform(test)

    y_time = train['duration'].values.astype(float)
    y_event = train['event'].values.astype(int)

    # Stack
    oof, te, blender = stacked_cv_predict(Xt, y_time, y_event, Xs, folds=folds)
    cv_c = concordance_index(y_time, oof, y_event)

    # Submission
    sub = pd.DataFrame({"patient_id": test["patient_id"].values, "predicted_scores": te})
    ranks = sub["predicted_scores"].rank(method="average") - 1
    sub["predicted_scores"] = ranks / max(1, ranks.max())

    out_path = Path(out_path)
    sub.to_csv(out_path, index=False)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pre": pre, "blender": blender}, Path(save_dir) / "model.joblib")
        with open(Path(save_dir) / "columns.json","w") as f:
            json.dump({"feature_columns": num_cols + cat_cols}, f)

    debug_summary = {
        "kept_numeric": num_cols[:20] + (["..."] if len(num_cols) > 20 else []),
        "kept_categorical": cat_cols[:20] + (["..."] if len(cat_cols) > 20 else []),
        "have_xgboost": HAVE_XGB,
        "stain": stain,
        "tile_params": {"max_dim": max_dim, "tile_size": tile_size, "stride": stride, "max_tiles": max_tiles},
        "te_cols": te_cols[:10] + (["..."] if len(te_cols) > 10 else []),
        "cv_cindex": float(cv_c)
    }

    return {"submission_path": str(out_path),
            "cv_cindex": float(cv_c),
            "feature_debug": debug_summary}

# ---------- CLI ----------
def build_argparser():
    p = argparse.ArgumentParser(description="Multimodal survival pipeline v7 (Overlap-tiling, stain norm, TE, AFT-ens).")
    p.add_argument("--data-dir", type=str, required=True, help="Root with train/ and test/ subfolders.")
    p.add_argument("--train", type=str, default="train/train.csv", help="train.csv path (relative to data-dir).")
    p.add_argument("--test", type=str, default="test/test.csv", help="test.csv path (relative to data-dir).")
    p.add_argument("--out", type=str, default="predictions.csv", help="Output CSV filename (saved to CWD).")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--model", type=str, default="stack", choices=["stack","gbr"])
    p.add_argument("--save-dir", type=str, default=None, help="Folder to store artifacts.")
    p.add_argument("--image-ext", type=str, default="tif", help="Image extension (default: tif).")
    p.add_argument("--max-images", type=int, default=None, help="Cap images processed (speed debug).")
    p.add_argument("--max-dim", type=int, default=10000, help="Max WSI dimension for downscale.")
    p.add_argument("--tile-size", type=int, default=512, help="Tile size for WSI tiling.")
    p.add_argument("--tile-stride", type=int, default=256, help="Stride for WSI tiling (defaults to 50% overlap).")
    p.add_argument("--max-tiles", type=int, default=400, help="Max tiles per image (post-filter); top-variance kept.")
    p.add_argument("--min-non-null", type=int, default=5, help="Drop cols with < this many non-null in train.")
    p.add_argument("--stain", type=str, default="reinhard", choices=["none","reinhard","macenko"], help="Tile stain normalization.")
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
                             model_kind=args.model, save_dir=save_dir,
                             image_ext=args.image_ext, max_images=args.max_images,
                             max_dim=args.max_dim, tile_size=args.tile_size, stride=args.tile_stride, max_tiles=args.max_tiles,
                             min_non_null=args.min_non_null, stain=args.stain)
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    main()

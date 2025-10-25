#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI service for Moffitt Survival Predictor (research-only).

- Loads joblib export produced by survival_hackathon.py
- Predicts survival probabilities at configured horizons
- Handles Cox (lifelines) and RSF (scikit-survival) models
"""

import os
import json
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --------------------------
# Config
# --------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/model_export.pkl")

# --------------------------
# Dependency pre-checks
# --------------------------
MISSING_DEPS: List[str] = []

try:
    import sklearn  # noqa: F401
except Exception:
    MISSING_DEPS.append("scikit-learn")

# Try lifelines and scikit-survival up front so unpickling won't fail silently
try:
    import lifelines  # noqa: F401
except Exception:
    # Lifelines is required if the export contains a CoxPHFitter.
    MISSING_DEPS.append("lifelines")

# scikit-survival is optional (only needed if model_type == "rsf")
try:
    import sksurv  # noqa: F401
    _HAS_SKSURV = True
except Exception:
    _HAS_SKSURV = False


# --------------------------
# App
# --------------------------
app = FastAPI(
    title="Moffitt Survival Predictor",
    version="1.1.0",
    description="Research-only API that returns survival probabilities at clinical horizons.",
)

# --------------------------
# Model Load
# --------------------------
_export: Dict[str, Any] = {}
_load_error: Optional[str] = None

def _load_model() -> None:
    """Load the joblib export with helpful error handling."""
    global _export, _load_error

    if MISSING_DEPS:
        _load_error = (
            "Missing dependencies: "
            + ", ".join(MISSING_DEPS)
            + ". Please install them (e.g., `pip install lifelines scikit-learn`)."
        )
        return

    if not os.path.exists(MODEL_PATH):
        _load_error = f"Model file not found at {MODEL_PATH}. Train/export first."
        return

    try:
        _export = joblib.load(MODEL_PATH)
    except Exception as e:
        # Often caused by missing packages during unpickle.
        _load_error = (
            f"Failed to load model at {MODEL_PATH}: {e}\n"
            f"Traceback:\n{traceback.format_exc()}\n"
            "Common cause: the environment doesn't have the same libraries as training.\n"
            "Make sure `lifelines` (for Cox) and `scikit-survival` (if using RSF) are installed."
        )
        return

    # Minimal sanity checks
    required_keys = ["preprocessor", "feature_names", "model_type", "horizons"]
    missing = [k for k in required_keys if k not in _export]
    if missing:
        _load_error = f"Export is missing keys: {missing}. Re-train/export with latest script."
        _export = {}
        return

    # If Cox, ensure Cox objects exist
    if _export["model_type"] == "cox" and _export.get("lifelines_cox") is None:
        _load_error = "Exported model_type is 'cox' but 'lifelines_cox' object is missing."
        _export = {}
        return

    # If RSF, check availability
    if _export["model_type"] == "rsf":
        if _export.get("rsf") is None:
            _load_error = "Exported model_type is 'rsf' but 'rsf' object is missing."
            _export = {}
            return
        if not _HAS_SKSURV:
            _load_error = "scikit-survival not installed but model_type is 'rsf'. Install `scikit-survival`."
            _export = {}
            return

    _load_error = None


_load_model()

# --------------------------
# Schemas
# --------------------------
class PatientPayload(BaseModel):
    records: List[Dict[str, Any]]


# --------------------------
# Utilities
# --------------------------
def _align_raw_features(df: pd.DataFrame, expected: List[str]) -> pd.DataFrame:
    """
    Align input dataframe to the raw feature names the preprocessor expects.
    Missing columns -> NaN, extras dropped.
    """
    df = df.copy()
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected]

def _predict_internal(payload: PatientPayload) -> Dict[str, Any]:
    if _load_error:
        raise HTTPException(status_code=503, detail=_load_error)

    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    feature_names = _export["feature_names"]
    preprocessor = _export["preprocessor"]
    model_type = _export["model_type"]
    horizons = _export["horizons"]

    df_raw = pd.DataFrame(payload.records)
    # Align to raw feature names (pre-preprocess)
    df_raw = _align_raw_features(df_raw, feature_names)

    # Transform with preprocessor
    X = preprocessor.transform(df_raw)

    # ---- Cox path ----
    if model_type == "cox":
        cox_model = _export["lifelines_cox"]

        # Use column names as they were *before* pruning during train
        cox_feat_cols_pre = _export.get("cox_feat_cols_pre")
        cox_train_feats = _export.get("cox_train_feats")

        # Backward compatibility: if not present, infer simple names
        if cox_feat_cols_pre is None:
            cox_feat_cols_pre = [f"x{i}" for i in range(X.shape[1])]
        if cox_train_feats is None:
            cox_train_feats = list(cox_feat_cols_pre)

        # 1) build DF with the "pre-prune" column list (must match X width)
        df_cox = pd.DataFrame(X, columns=cox_feat_cols_pre)
        # 2) reindex to the pruned training feature list
        df_cox = df_cox.reindex(columns=cox_train_feats, fill_value=0.0)

        # Predict survival per horizon and risk score
        out: List[Dict[str, float]] = []
        surv_by_h: Dict[int, np.ndarray] = {}
        for h in horizons:
            surv_by_h[int(h)] = cox_model.predict_survival_function(df_cox, times=[h]).T.values.reshape(-1)

        risk = cox_model.predict_partial_hazard(df_cox).values.reshape(-1)

        for i in range(df_cox.shape[0]):
            rec = {"risk_score": float(risk[i])}
            for h in horizons:
                rec[f"surv_prob_{int(h)}d"] = float(surv_by_h[int(h)][i])
            out.append(rec)

        return {"model_type": "cox", "horizons": horizons, "predictions": out}

    # ---- RSF path ----
    if model_type == "rsf":
        rsf_model = _export["rsf"]
        surv_funcs = rsf_model.predict_survival_function(X, return_array=False)
        out: List[Dict[str, float]] = []
        for i, fn in enumerate(surv_funcs):
            rec = {}
            for h in horizons:
                rec[f"surv_prob_{int(h)}d"] = float(fn([h])[0])
            # Larger -> worse risk; RSF outputs time -> use negative for ranking consistency
            rec["risk_score"] = float(-rsf_model.predict(X[i:i+1])[0])
            out.append(rec)
        return {"model_type": "rsf", "horizons": horizons, "predictions": out}

    # Unknown model_type
    raise HTTPException(status_code=500, detail=f"Unsupported model_type: {model_type}")


# --------------------------
# Routes
# --------------------------
@app.get("/health")
def health():
    info = {
        "status": "ok" if not _load_error else "error",
        "load_error": _load_error,
    }
    if _export:
        info.update({
            "model_path": MODEL_PATH,
            "model_type": _export.get("model_type"),
            "horizons": _export.get("horizons"),
            "sklearn_version_saved": _export.get("sklearn_version"),
            "raw_feature_count": len(_export.get("feature_names", [])),
        })
    else:
        info["model_path"] = MODEL_PATH
    return info


@app.get("/features")
def features():
    if _load_error:
        raise HTTPException(status_code=503, detail=_load_error)
    return {
        "expected_raw_features": _export.get("feature_names", []),
        "note": "These are the raw feature names expected in /predict payload; missing values will be imputed.",
    }


@app.post("/predict")
def predict(payload: PatientPayload):
    try:
        return _predict_internal(payload)
    except HTTPException:
        raise
    except Exception as e:
        err = f"Prediction failed: {e}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=err)

# serve.py
# FastAPI service that loads the exported model (joblib pkl) and predicts survival probs.

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

MODEL_PATH = os.environ.get("MODEL_PATH", "outputs/model_export.pkl")

app = FastAPI(title="Moffitt Survival Predictor", version="1.0", description="Research-only survival probabilities")

# ---- Load model at startup ----
export = joblib.load(MODEL_PATH)
preprocessor = export["preprocessor"]
feature_names = export["feature_names"]
model_type = export["model_type"]
horizons = export["horizons"]
cox_model = export.get("lifelines_cox")
rsf_model = export.get("rsf")

class PatientPayload(BaseModel):
    records: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok", "model_type": model_type, "horizons": horizons}

@app.post("/predict")
def predict(payload: PatientPayload):
    if not payload.records:
        return {"error": "Empty records"}

    # Create DataFrame with trained feature names; missing keys -> NaN
    df = pd.DataFrame(payload.records)
    # Restrict/align to training features (unknown columns dropped, missing added)
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
    df = df[feature_names].copy()

    X = preprocessor.transform(df)

    if model_type == "cox" and cox_model is not None:
        Xdf = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
        out = []
        surv_matrix = {}
        for h in horizons:
            surv_matrix[h] = cox_model.predict_survival_function(Xdf, times=[h]).T.values.reshape(-1)
        risk = cox_model.predict_partial_hazard(Xdf).values.reshape(-1)
        for i in range(X.shape[0]):
            rec = {"risk_score": float(risk[i])}
            for h in horizons:
                rec[f"surv_prob_{int(h)}d"] = float(surv_matrix[h][i])
            out.append(rec)
        return {"model_type": "cox", "predictions": out}

    if model_type == "rsf" and rsf_model is not None:
        surv_funcs = rsf_model.predict_survival_function(X, return_array=False)
        out = []
        for i, fn in enumerate(surv_funcs):
            rec = {}
            for h in horizons:
                rec[f"surv_prob_{int(h)}d"] = float(fn([h])[0])
            rec["risk_score"] = float(-rsf_model.predict(X[i:i+1])[0])
            out.append(rec)
        return {"model_type": "rsf", "predictions": out}

    return {"error": "Model not available"}

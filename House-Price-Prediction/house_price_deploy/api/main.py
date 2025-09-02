# api/main.py
import os
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------
# 1) API setup
# -------------------------
app = FastAPI(title="House Price API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# 2) Input schema (12 raw features)
# -------------------------
FEATURE_ORDER = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning",
    "parking", "prefarea", "furnishingstatus"
]

class HouseFeatures(BaseModel):
    area: float = Field(..., gt=0)
    bedrooms: int = Field(..., ge=0)
    bathrooms: int = Field(..., ge=0)
    stories: int = Field(..., ge=1)
    mainroad: Literal["yes", "no"]
    guestroom: Literal["yes", "no"]
    basement: Literal["yes", "no"]
    hotwaterheating: Literal["yes", "no"]
    airconditioning: Literal["yes", "no"]
    parking: int = Field(..., ge=0)
    prefarea: Literal["yes", "no"]
    furnishingstatus: Literal["furnished", "semi-furnished", "unfurnished"]

# -------------------------
# 3) Model loading
#    - Prefer env var MODEL_PATH
#    - Fallbacks: ./models/..., ./model/...
# -------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "").strip() or ""
_DEFAULT_CANDIDATES = [
    MODEL_PATH,
    "models/lasso_house_price_model.pkl",   # common layout
    "model/lasso_house_price_model.pkl",    # alt layout
    "lasso_house_price_model.pkl",          # same dir
]

_model = None

def _load_model():
    """Load the saved scikit-learn Pipeline (joblib or pickle)."""
    global _model

    # pick first existing candidate
    path = None
    for p in _DEFAULT_CANDIDATES:
        if p and os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(
            "Could not find model file. Checked: "
            + ", ".join([p for p in _DEFAULT_CANDIDATES if p])
            + ". Set MODEL_PATH env var if needed."
        )

    # try joblib then pickle
    last_err = None
    try:
        from joblib import load as joblib_load
        _model = joblib_load(path)
        return
    except Exception as e:
        last_err = e

    try:
        import pickle
        with open(path, "rb") as f:
            _model = pickle.load(f)
        return
    except Exception as e2:
        raise RuntimeError(
            f"Failed to load model at '{path}'. "
            f"joblib error: {last_err}; pickle error: {e2}"
        )

def _normalize_payload(d: dict) -> dict:
    """Lower/trim categorical strings to the expected forms."""
    out = dict(d)
    for k in ["mainroad","guestroom","basement",
              "hotwaterheating","airconditioning","prefarea",
              "furnishingstatus"]:
        out[k] = str(out[k]).strip().lower()
    return out

@app.on_event("startup")
def _startup():
    _load_model()

# -------------------------
# 4) Utility endpoints
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/schema")
def schema():
    return {
        "feature_order": FEATURE_ORDER,
        "model_loaded": _model is not None,
        "model_type": type(_model).__name__ if _model is not None else None,
        "expects_raw_columns": True,  # pipeline handles preprocessing internally
    }

# -------------------------
# 5) Prediction endpoint
# -------------------------
@app.post("/predict")
def predict(features: HouseFeatures):
    """
    Accepts a single example matching HouseFeatures.
    Returns a single numeric prediction.
    """
    try:
        payload = _normalize_payload(features.model_dump())
        X = pd.DataFrame([payload], columns=FEATURE_ORDER)

        # DEBUG (optional): uncomment if you need server-side insight
        print("=== Incoming DataFrame ===")
        print(X.head())
        print(X.dtypes)

        y_pred = _model.predict(X)
        return {"prediction": float(y_pred[0])}
    except Exception as e:
        # If anything goes wrong inside the model/pipeline
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

# -------------------------
# 6) Local dev entrypoint
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# To run: python api/main.py
# Then open http://localhost:8000/docs for interactive API docs
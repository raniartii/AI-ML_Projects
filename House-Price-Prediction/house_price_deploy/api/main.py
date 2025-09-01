import os
import pickle
from typing import Any, Dict, List, Union, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

class HouseFeatures(BaseModel):
    area: float
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: str
    guestroom: str
    basement: str
    hotwaterheating: str
    airconditioning: str
    parking: int
    prefarea: str
    furnishingstatus: str

# Path to the serialized model. Override with env var if needed.
MODEL_PATH = os.environ.get("MODEL_PATH", "model/lasso_house_price_model.pkl")

app = FastAPI(title="House Price Model API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_expected_columns: Optional[List[str]] = None

def _infer_expected_columns(model) -> Optional[List[str]]:
    # 1) Most reliable for sklearn >=1.0
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    # 2) Search for ColumnTransformer inside a Pipeline
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(model, Pipeline):
            # direct
            for name, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    cols = []
                    for _, trans, c in step.transformers:
                        if c == "drop" or trans == "drop":
                            continue
                        if isinstance(c, (list, tuple)):
                            cols.extend(list(c))
                    return list(dict.fromkeys(cols)) if cols else None
            # nested
            for name, step in model.named_steps.items():
                if isinstance(step, Pipeline):
                    for n2, st2 in step.named_steps.items():
                        if isinstance(st2, ColumnTransformer):
                            cols = []
                            for _, trans, c in st2.transformers:
                                if c == "drop" or trans == "drop":
                                    continue
                                if isinstance(c, (list, tuple)):
                                    cols.extend(list(c))
                            return list(dict.fromkeys(cols)) if cols else None
    except Exception:
        pass
    return None

def _load_model():
    """
    Robust model loader that handles joblib, pickle, Py2 pickles, and optionally cloudpickle.
    """
    import pickle, os

    global _model, _expected_columns
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

    # Pre-import common globals used inside sklearn objects (helps resolution)
    try:
        import numpy as _np   # noqa: F401
        import sklearn as _sk # noqa: F401
    except Exception:
        pass

    # 1) Try joblib (most common for sklearn)
    joblib_err = None
    try:
        from joblib import load as joblib_load
        _model = joblib_load(MODEL_PATH)
    except Exception as e:
        joblib_err = e
        _model = None

    # 2) Try plain pickle
    if _model is None:
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
        except Exception as e_plain:
            # 3) Try Py2-compatible pickle (bytes→str)
            try:
                with open(MODEL_PATH, "rb") as f:
                    _model = pickle.load(f, fix_imports=True, encoding="latin1")
            except Exception as e_latin1:
                # 4) Optional: cloudpickle fallback
                cloud_err = None
                try:
                    import cloudpickle
                    with open(MODEL_PATH, "rb") as f:
                        _model = cloudpickle.load(f)
                except Exception as e_cloud:
                    cloud_err = e_cloud
                    raise RuntimeError(
                        "Could not load model via joblib/pickle/cloudpickle. "
                        f"joblib: {joblib_err}\n"
                        f"pickle: {e_plain}\n"
                        f"pickle(latin1): {e_latin1}\n"
                        f"cloudpickle: {cloud_err}\n"
                        "Fixes:\n"
                        "  • Pin numpy & scikit-learn to your training versions in requirements.txt\n"
                        "  • Load with the same library you used to save (joblib/pickle/cloudpickle)\n"
                        "  • Or re-export the model from training with joblib.dump(...)\n"
                    ) from e_cloud

    _expected_columns = _infer_expected_columns(_model)


@app.on_event("startup")
def _startup():
    _load_model()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/schema")
def schema():
    return {
        "expected_columns": _expected_columns,
        "has_feature_names_in_": hasattr(_model, "feature_names_in_")
    }

@app.post("/predict")
def predict(payload: Union[HouseFeatures, List[HouseFeatures]]):
    try:
        # Pydantic models → dicts
        if isinstance(payload, list):
            payload = [item.dict() for item in payload]
        else:
            payload = payload.dict()

        df = _to_dataframe(payload)

        print("DEBUG - DataFrame:", df)

        y_pred = _model.predict(df)
        preds = [float(p) for p in y_pred]
        return {"predictions": preds, "count": len(preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/predict")
def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]] ):
    try:
        df = _to_dataframe(payload)

        # DEBUG: print to server logs
        print("=== Incoming DataFrame ===")
        print(df.head())
        print(df.dtypes)
        print("Any NaNs?", df.isna().sum())

        y_pred = _model.predict(df)
        preds = [float(p) for p in y_pred]
        return {"predictions": preds, "count": len(preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

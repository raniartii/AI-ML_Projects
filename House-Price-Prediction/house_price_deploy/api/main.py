import os
import pickle
from typing import Any, Dict, List, Union, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
    global _model, _expected_columns
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
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

def _to_dataframe(payload: Union[Dict[str, Any], List[Dict[str, Any]]] ) -> pd.DataFrame:
    if isinstance(payload, dict):
        df = pd.DataFrame([payload])
    elif isinstance(payload, list):
        if not payload:
            raise HTTPException(status_code=400, detail="Empty list given.")
        df = pd.DataFrame(payload)
    else:
        raise HTTPException(status_code=400, detail="Payload must be an object or a list of objects.")
    # Shape to expected columns if we know them
    if _expected_columns is not None:
        # Reindex to expected columns: fill missing with NaN, drop extras.
        df = df.reindex(columns=_expected_columns)
    return df

@app.post("/predict")
def predict(payload: Union[Dict[str, Any], List[Dict[str, Any]]] ):
    try:
        df = _to_dataframe(payload)
        y_pred = _model.predict(df)
        preds = [float(p) for p in y_pred]  # ensure JSON-serializable
        return {"predictions": preds, "count": len(preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

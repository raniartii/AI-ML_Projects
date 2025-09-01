import os
import json
import pickle
from typing import Any, Dict, List, Optional
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("🏠 House Price Predictor")
st.caption("Use a trained model to predict house prices. Choose to call a running API or load the model locally.")

API_BASE = st.sidebar.text_input("API base URL (leave empty to use local model)", value="http://localhost:8000")
use_local = st.sidebar.checkbox("Use local model instead of API", value=(API_BASE.strip() == ""))

MODEL_PATH = st.sidebar.text_input("Local model path", value="model/lasso_house_price_model.pkl")

# Helper to fetch expected columns
def get_expected_columns_from_api() -> Optional[List[str]]:
    try:
        r = requests.get(API_BASE.rstrip("/") + "/schema", timeout=10)
        if r.ok:
            return r.json().get("expected_columns")
    except Exception:
        pass
    return None

def get_expected_columns_from_local(model) -> Optional[List[str]]:
    if hasattr(model, "feature_names_in_"):
        return list(getattr(model, "feature_names_in_"))
    # Best effort: look for ColumnTransformer columns
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        if isinstance(model, Pipeline):
            for name, step in model.named_steps.items():
                if isinstance(step, ColumnTransformer):
                    cols = []
                    for _, trans, c in step.transformers:
                        if c == "drop" or trans == "drop":
                            continue
                        if isinstance(c, (list, tuple)):
                            cols.extend(list(c))
                    return list(dict.fromkeys(cols)) if cols else None
    except Exception:
        pass
    return None

@st.cache_resource
def load_local_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

if use_local:
    try:
        model = load_local_model(MODEL_PATH)
        exp_cols = get_expected_columns_from_local(model)
        st.success(f"Local model loaded. Expected columns: {exp_cols if exp_cols else 'Unknown (accepts DataFrame)'}")
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        st.stop()
else:
    # verify API connectivity
    try:
        h = requests.get(API_BASE.rstrip("/") + "/health", timeout=5)
        if not h.ok:
            st.error(f"API healthcheck failed with status {h.status_code}")
            st.stop()
    except Exception as e:
        st.error(f"Could not reach API at {API_BASE}: {e}")
        st.stop()
    exp_cols = get_expected_columns_from_api()
    st.info(f"Using API. Expected columns: {exp_cols if exp_cols else 'Unknown'}")

st.subheader("Single Prediction")
st.caption("Fill the inputs below and click Predict. Leave fields blank to send as missing values.")

# Provide a sensible default set of fields for common 'Housing' datasets if we don't know the schema
default_fields = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]
fields = exp_cols if exp_cols else default_fields

# Simple per-field widgets
user_inputs: Dict[str, Any] = {}
binary_yes_no = {"mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"}
for col in fields:
    key = f"fld_{col}"
    if col.lower() in binary_yes_no:
        user_inputs[col] = st.selectbox(col, ["", "yes", "no"], index=0, key=key)
    elif any(tok in col.lower() for tok in ["bed", "bath", "story", "parking"]):
        user_inputs[col] = st.number_input(col, min_value=0, step=1, value=None, placeholder="e.g. 3", key=key)
    elif any(tok in col.lower() for tok in ["area", "sqft", "lot", "size"]):
        user_inputs[col] = st.number_input(col, min_value=0.0, step=1.0, value=None, placeholder="e.g. 3000", key=key)
    elif "furnish" in col.lower():
        user_inputs[col] = st.selectbox(col, ["", "furnished", "semi-furnished", "unfurnished"], index=0, key=key)
    else:
        user_inputs[col] = st.text_input(col, value="", key=key)

if st.button("Predict"):
    # Build payload
    payload = {k: (None if (v == "" or v is None) else v) for k, v in user_inputs.items()}
    if not use_local:
        try:
            r = requests.post(API_BASE.rstrip("/") + "/predict", json=payload, timeout=20)
            st.json(r.json())
        except Exception as e:
            st.error(f"Prediction failed via API: {e}")
    else:
        try:
            import pandas as pd
            df = pd.DataFrame([payload])
            # attempt to reindex to expected columns
            if exp_cols:
                df = df.reindex(columns=exp_cols)
            pred = model.predict(df)
            st.success(f"Predicted price: {float(pred[0]):,.2f}")
        except Exception as e:
            st.error(f"Local prediction failed: {e}")

st.divider()
st.subheader("Batch Prediction (CSV)")
st.caption("Upload a CSV with the same columns as training. We'll return a downloadable CSV with predictions.")
csv = st.file_uploader("Upload CSV", type=["csv"])
if csv is not None:
    try:
        df = pd.read_csv(csv)
        st.write("Preview:", df.head())
        if st.button("Run batch prediction"):
            if not use_local:
                # Send list of records
                records = df.to_dict(orient="records")
                r = requests.post(API_BASE.rstrip("/") + "/predict", json=records, timeout=60)
                if r.ok:
                    preds = r.json().get("predictions", [])
                    out = df.copy()
                    out["prediction"] = preds
                    out_path = "batch_predictions.csv"
                    out.to_csv(out_path, index=False)
                    st.success("Done.")
                    st.download_button("Download predictions", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
                    st.dataframe(out.head())
                else:
                    st.error(f"API error: {r.status_code} {r.text}")
            else:
                if exp_cols:
                    df = df.reindex(columns=exp_cols)
                preds = model.predict(df)
                out = df.copy()
                out["prediction"] = preds
                st.success("Done.")
                st.download_button("Download predictions", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
                st.dataframe(out.head())
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")

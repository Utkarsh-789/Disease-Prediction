"""
streamlit_app.py (robust)

Features:
- Tries to load model from MODEL_PATH.
- If not found or loading fails, shows a file uploader so you can upload a saved model (.pkl).
- Displays helpful errors and lets you run predictions (single or batch) after loading a model from disk or from upload.

Run:
    streamlit run streamlit_app.py
"""
import streamlit as st
import pandas as pd
import pickle
import io
import os
from typing import Optional

# Change this to your preferred default path if you have one
MODEL_PATH = "best_model.pkl"

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

@st.cache_resource
def load_model_from_bytes(data_bytes: bytes):
    """Load a model from raw bytes (used for uploaded pickle)."""
    try:
        obj = pickle.loads(data_bytes)
        # support either model directly or a dict with 'model_pipeline' key
        if isinstance(obj, dict) and 'model_pipeline' in obj:
            return obj
        # if user saved only the pipeline, normalize into a dictionary
        return {"model_pipeline": obj, "numeric_columns": obj.get("numeric_columns", []), "categorical_columns": obj.get("categorical_columns", [])}
    except Exception as e:
        raise RuntimeError(f"Failed to load pickle: {e}")

@st.cache_resource
def load_model_from_path(path: str) -> Optional[dict]:
    """Attempt to load model from path. Returns model dict or None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and 'model_pipeline' in obj:
            return obj
        return {"model_pipeline": obj, "numeric_columns": obj.get("numeric_columns", []), "categorical_columns": obj.get("categorical_columns", [])}
    except Exception as e:
        st.error(f"Error loading model from path: {e}")
        return None

st.title("Diabetes Prediction — Inference UI")

st.markdown(
    """
    App will try to load a model from **`{}`**.  
    If that fails you can upload a `best_model.pkl` file below (the same file produced by the training script).
    """.format(MODEL_PATH)
)

model_pack = load_model_from_path(MODEL_PATH)

if model_pack is None:
    st.warning(f"Model not found at `{MODEL_PATH}` or failed to load. Please upload `best_model.pkl` (the pickle created by your training script).")
    uploaded = st.file_uploader("Upload model `.pkl` file", type=["pkl", "pickle"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            data = uploaded.read()
            model_pack = load_model_from_bytes(data)
            st.success("Model uploaded and loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
            st.stop()

if model_pack is None:
    st.stop()  # nothing to do

model = model_pack["model_pipeline"]
numeric_cols = model_pack.get("numeric_columns", [])
categorical_cols = model_pack.get("categorical_columns", [])

st.write("Model ready. You can do single prediction or batch (CSV upload).")

choice = st.radio("Input type:", ["Single row (form)", "Upload CSV (batch)"])

if choice == "Single row (form)":
    st.subheader("Single row input")
    # build a simple form based on known columns; if not known, let user enter manually
    form_values = {}
    if numeric_cols or categorical_cols:
        st.write("Detected feature columns — fill values below (missing fields will be imputed by the pipeline).")
        for c in numeric_cols:
            # a better default for age or similar could be implemented heuristically
            default = 0.0
            if "age" in c.lower():
                default = 30.0
            form_values[c] = st.number_input(c, value=float(default))
        for c in categorical_cols:
            form_values[c] = st.text_input(c, value="")
    else:
        st.info("Model didn't provide column names. Enter JSON-like input for one row.")
        raw = st.text_area("Enter one sample as JSON, e.g. {\"age\":45, \"gender\":\"Female\", \"bmi\":27.5}", height=120)
        if raw:
            try:
                import json
                form_values = json.loads(raw)
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                form_values = {}

    if st.button("Predict single row"):
        try:
            df = pd.DataFrame([form_values])
            pred = model.predict(df)[0]
            st.success(f"Predicted label: {pred}")
            if hasattr(model, "predict_proba"):
                st.write("Probabilities:", model.predict_proba(df)[0].tolist())
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.subheader("Batch prediction from CSV")
    uploaded_csv = st.file_uploader("Upload CSV with same features (no target column expected)", type=["csv"])
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.write("Uploaded data preview:", df.head())
            if st.button("Run batch prediction"):
                preds = model.predict(df)
                st.write("Predictions (first 50):", preds[:50])
                if hasattr(model, "predict_proba"):
                    st.write("Probabilities (first 5 rows):")
                    st.write(model.predict_proba(df)[:5])
        except Exception as e:
            st.error(f"Failed to read CSV or run predictions: {e}")

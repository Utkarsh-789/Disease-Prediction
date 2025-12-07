# streamlit_app.py
"""
Streamlit app that builds a form from model metadata (or uploaded sample) and runs inference.

Usage:
  export MODEL_PATH="./best_model.pkl"   # optional
  streamlit run streamlit_app.py

Requirements:
  pip install streamlit pandas scikit-learn joblib
"""

import os
import io
import pickle
from typing import Optional, Dict, Any, List
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Form-based Inference", layout="centered")

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl")

# ----------------- Utilities -----------------
def load_model_from_path(path: str) -> Optional[Dict[str, Any]]:
    """Try to load a model file from disk (pickle or joblib). Return normalized dict or None."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e_pickle:
        # fallback to joblib
        try:
            import joblib
            obj = joblib.load(path)
        except Exception as e_joblib:
            st.error(f"Failed to load model from path. pickle error: {e_pickle}; joblib error: {e_joblib}")
            return None
    return normalize_model_object(obj)

def load_model_from_bytes(b: bytes) -> Optional[Dict[str, Any]]:
    """Load a model from uploaded bytes."""
    try:
        obj = pickle.loads(b)
    except Exception as e_pickle:
        try:
            import joblib
            obj = joblib.loads(b)
        except Exception as e_joblib:
            st.error(f"Failed to load uploaded model. pickle error: {e_pickle}; joblib error: {e_joblib}")
            return None
    return normalize_model_object(obj)

def normalize_model_object(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize loaded object into a dict:
      {"model_pipeline": pipeline_obj, "numeric_columns": [...], "categorical_columns": [...]}
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        if "model_pipeline" in obj:
            return {
                "model_pipeline": obj["model_pipeline"],
                "numeric_columns": obj.get("numeric_columns", []) or [],
                "categorical_columns": obj.get("categorical_columns", []) or []
            }
        if "pipeline" in obj:
            return {
                "model_pipeline": obj["pipeline"],
                "numeric_columns": obj.get("numeric_columns", []) or [],
                "categorical_columns": obj.get("categorical_columns", []) or []
            }
    # assume it's a pipeline object
    model_pipeline = obj
    numeric_cols = getattr(obj, "numeric_columns", []) or []
    categorical_cols = getattr(obj, "categorical_columns", []) or []
    return {
        "model_pipeline": model_pipeline,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols
    }

def try_predict(model, df: pd.DataFrame):
    """Run prediction and return (pred, probs) or raise."""
    pred = model.predict(df)
    probs = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(df)
        except Exception:
            probs = None
    return pred, probs

# ----------------- App UI -----------------
st.title("Form-based Inference App")
st.markdown(
    "This app builds a form dynamically from your model metadata or a sample CSV and runs prediction for a single row."
)
st.write("Model path:", f"`{MODEL_PATH}`")

# Try to load model automatically from MODEL_PATH
model_pack = load_model_from_path(MODEL_PATH)
if model_pack is None:
    st.warning(f"No model loaded from `{MODEL_PATH}`. You can upload a model (.pkl/.joblib) below or provide sample columns.")
    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_model = st.file_uploader("Upload model (.pkl or .joblib)", type=["pkl", "pickle", "joblib"], key="upload_model")
    with col2:
        st.markdown("**Or** if you already have a model file in the working dir, set `MODEL_PATH` env var and restart.")
    if uploaded_model is not None:
        model_pack = load_model_from_bytes(uploaded_model.read())
        if model_pack is not None:
            st.success("Model uploaded and loaded successfully.")

if model_pack is None:
    st.info("If you don't have a model file, upload a small sample CSV (header only) below to build the form (no model required).")
    sample_csv = st.file_uploader("Upload sample CSV (for inferring form fields)", type=["csv"], key="upload_sample")
    inferred_numeric_cols: List[str] = []
    inferred_categorical_cols: List[str] = []
    if sample_csv is not None:
        try:
            sample_df = pd.read_csv(sample_csv, nrows=5)
            st.write("Sample preview (first 5 rows):")
            st.dataframe(sample_df.head())
            # infer numeric vs categorical by dtype
            inferred_numeric_cols = sample_df.select_dtypes(include=["number"]).columns.tolist()
            inferred_categorical_cols = sample_df.select_dtypes(exclude=["number"]).columns.tolist()
            st.success("Inferred columns from uploaded CSV.")
        except Exception as e:
            st.error(f"Failed to read uploaded CSV: {e}")
    else:
        # allow manual paste of column names
        st.markdown("Alternatively, paste comma-separated column names below (the last column will be assumed the target unless you specify otherwise).")
        col_input = st.text_area("Comma-separated feature column names (exclude target column)", value="", help="e.g. gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level")
        if col_input.strip():
            names = [c.strip() for c in col_input.split(",") if c.strip()]
            # ask which are numeric
            if names:
                st.markdown("Mark which of these are numeric (others will be treated as categorical).")
                numeric_selected = []
                for name in names:
                    is_num = st.checkbox(f"{name} (numeric)", key=f"num_{name}")
                    if is_num:
                        numeric_selected.append(name)
                inferred_numeric_cols = numeric_selected
                inferred_categorical_cols = [n for n in names if n not in numeric_selected]

    # If we got columns either from CSV or manual, build the form
    if inferred_numeric_cols or inferred_categorical_cols:
        st.markdown("### Generated form (from sample columns)")
        form_values = {}
        with st.form("generated_form"):
            for c in inferred_numeric_cols:
                default = 30.0 if "age" in c.lower() else 0.0
                form_values[c] = st.number_input(label=c, value=float(default))
            for c in inferred_categorical_cols:
                form_values[c] = st.text_input(label=c, value="")
            submit = st.form_submit_button("Predict from generated form")
        if submit:
            if 'model_pipeline' in (model_pack or {}):
                model = model_pack["model_pipeline"]
                try:
                    df_in = pd.DataFrame([form_values])
                    pred, probs = try_predict(model, df_in)
                    st.success(f"Predicted label: {pred[0]}")
                    if probs is not None:
                        st.write("Probabilities:", probs[0].tolist())
                except Exception as e:
                    st.error(f"Prediction failed (model present but error): {e}")
            else:
                st.error("No model loaded for prediction. Upload a model to enable prediction.")
    else:
        st.info("Upload a CSV or paste column names to generate the input form.")
    st.stop()  # done in case no model_pack loaded

# If we get here, a model is loaded
model = model_pack["model_pipeline"]
numeric_cols = model_pack.get("numeric_columns", []) or []
categorical_cols = model_pack.get("categorical_columns", []) or []

st.success("Model loaded successfully.")
st.subheader("Model metadata")
st.write("Numeric columns:", numeric_cols if numeric_cols else "—")
st.write("Categorical columns:", categorical_cols if categorical_cols else "—")

# If metadata is missing, let user upload a CSV or paste columns to create the form
if not numeric_cols and not categorical_cols:
    st.info("Model does not provide column metadata. Upload a sample CSV or paste column names to build the form.")
    sample_csv2 = st.file_uploader("Upload sample CSV (to infer columns)", type=["csv"], key="sample2")
    inferred_numeric_cols = []
    inferred_categorical_cols = []
    if sample_csv2 is not None:
        try:
            sample_df = pd.read_csv(sample_csv2, nrows=5)
            inferred_numeric_cols = sample_df.select_dtypes(include=["number"]).columns.tolist()
            inferred_categorical_cols = sample_df.select_dtypes(exclude=["number"]).columns.tolist()
            st.success("Inferred columns from sample CSV.")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
    col_input2 = st.text_area("Or paste comma-separated column names:", value="")
    if col_input2.strip():
        names = [c.strip() for c in col_input2.split(",") if c.strip()]
        # ask user which names are numeric
        numeric_selected = []
        if names:
            st.markdown("Tick which of these are numeric:")
            for name in names:
                if st.checkbox(name + " (numeric)", key=f"meta_num_{name}"):
                    numeric_selected.append(name)
            inferred_numeric_cols = numeric_selected
            inferred_categorical_cols = [n for n in names if n not in numeric_selected]
    # update lists
    if not inferred_numeric_cols and not inferred_categorical_cols:
        st.stop()
    numeric_cols = inferred_numeric_cols
    categorical_cols = inferred_categorical_cols

# Build the form from numeric_cols and categorical_cols
st.markdown("### Input form — enter feature values")
form_values = {}
with st.form("input_form"):
    # numeric inputs
    for c in numeric_cols:
        default = 30.0 if "age" in c.lower() else 0.0
        form_values[c] = st.number_input(label=c, value=float(default), format="%.6f", key=f"n_{c}")
    # categorical inputs
    for c in categorical_cols:
        form_values[c] = st.text_input(label=c, value="", key=f"c_{c}")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build dataframe and predict
    try:
        input_df = pd.DataFrame([form_values])
        pred, probs = try_predict(model, input_df)
        st.success(f"Predicted label: {pred[0]}")
        if probs is not None:
            st.write("Probabilities:", probs[0].tolist())
        st.write("Input used for prediction:")
        st.dataframe(input_df.T.rename(columns={0: "value"}))
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

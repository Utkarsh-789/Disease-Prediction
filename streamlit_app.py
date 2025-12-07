"""
streamlit_app.py

Usage:
    # Set MODEL_PATH environment variable if your model is somewhere else:
    export MODEL_PATH="/path/to/best_model.pkl"
    streamlit run streamlit_app.py

Features:
 - Load model from disk or uploaded file (.pkl or .joblib)
 - Accept single-row input via form or JSON
 - Accept batch CSV for predictions
 - Displays prediction and probabilities (if model supports it)
"""

import os
import io
import pickle
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd

# Config
DEFAULT_MODEL_FILENAME = "best_model.pkl"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL_FILENAME)

st.set_page_config(page_title="Disease Prediction — Inference", layout="centered")

# ---- Helpers ----
def try_pickle_load_bytes(data_bytes: bytes) -> Dict[str, Any]:
    """
    Load a pickle/joblib object from raw bytes and normalize into a dict:
      {"model_pipeline": pipeline, "numeric_columns": [...], "categorical_columns": [...]}
    Accepts:
      - pickled pipeline object (sklearn Pipeline)
      - pickled dict with keys 'model_pipeline' etc.
    Raises RuntimeError on failure.
    """
    # try pickle first
    try:
        obj = pickle.loads(data_bytes)
    except Exception as e_pickle:
        # try joblib if available
        try:
            import joblib
            obj = joblib.loads(data_bytes)
        except Exception as e_joblib:
            raise RuntimeError(f"Unable to load model bytes via pickle or joblib. Errors:\npickle: {e_pickle}\njoblib: {e_joblib}")

    if isinstance(obj, dict):
        # assume user saved a dict with metadata
        if "model_pipeline" in obj:
            return obj
        # also accept {'pipeline': ...}
        if "pipeline" in obj:
            return {"model_pipeline": obj["pipeline"],
                    "numeric_columns": obj.get("numeric_columns", []),
                    "categorical_columns": obj.get("categorical_columns", [])}
    else:
        # assume it's the pipeline itself
        model_pipeline = obj
        # try to extract column names if they were attached as attributes
        numeric_cols = getattr(obj, "numeric_columns", []) or []
        categorical_cols = getattr(obj, "categorical_columns", []) or []
        # normalize
        return {"model_pipeline": model_pipeline,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols}

def try_load_model_from_path(path: str) -> Optional[Dict[str, Any]]:
    """Try reading a model file from disk and normalize into model_pack dict; return None on failure."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
        return try_pickle_load_bytes(data)
    except Exception as e:
        st.error(f"Failed to load model from path `{path}`: {e}")
        return None

# cache model load to avoid repeated heavy work
@st.cache_resource
def load_model_pack_from_disk_or_bytes(path: Optional[str] = None, bytes_data: Optional[bytes] = None):
    if bytes_data is not None:
        return try_pickle_load_bytes(bytes_data)
    if path is not None:
        return try_load_model_from_path(path)
    return None

# ---- UI ----
st.title("Disease / Diabetes Prediction — Inference")
st.markdown(
    "This app expects a saved model pipeline (a pickle or joblib file). "
    "If the model isn't found at the configured path, upload it below."
)
st.write("Model path:", f"`{MODEL_PATH}`")

# try to load from default path first
model_pack = load_model_pack_from_disk_or_bytes(path=MODEL_PATH)

if model_pack is None:
    st.warning(f"No model loaded from `{MODEL_PATH}`. Please upload a model file (pickle/.pkl or joblib/.joblib).")
    uploaded_model = st.file_uploader("Upload model file (.pkl or .joblib)", type=["pkl", "pickle", "joblib"], accept_multiple_files=False)
    if uploaded_model is not None:
        try:
            data_bytes = uploaded_model.read()
            model_pack = load_model_pack_from_disk_or_bytes(bytes_data=data_bytes)
            st.success("Model uploaded and loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
            st.stop()

if model_pack is None:
    st.info("No model available yet. Upload one or set MODEL_PATH to a valid model file.")
    st.stop()

# Normalize model and metadata
model = model_pack.get("model_pipeline")
numeric_cols = model_pack.get("numeric_columns", []) or []
categorical_cols = model_pack.get("categorical_columns", []) or []

st.subheader("Model information")
st.write("Model object type:", type(model))
if numeric_cols or categorical_cols:
    st.write("Detected numeric columns:", numeric_cols)
    st.write("Detected categorical columns:", categorical_cols)
else:
    st.info("Model metadata doesn't include column lists. You can still provide input manually (JSON or CSV).")

# Prediction UI
mode = st.radio("Choose input mode", ["Single row (form)", "Single row (JSON)", "Batch CSV upload"])

if mode == "Single row (form)":
    st.markdown("Fill the form for one sample. Missing fields will be imputed by the pipeline if needed.")
    sample = {}
    # If model provided column names, show friendly form; otherwise let user add key/value pairs
    if numeric_cols or categorical_cols:
        for c in numeric_cols:
            # provide a sensible numeric default for 'age'
            default = 30.0 if "age" in c.lower() else 0.0
            sample[c] = st.number_input(label=c, value=float(default), format="%.6f")
        for c in categorical_cols:
            sample[c] = st.text_input(label=c, value="")
    else:
        st.info("No column metadata: use the JSON input mode for free-form entry or upload a CSV.")
        st.stop()

    if st.button("Predict single sample"):
        try:
            df = pd.DataFrame([sample])
            pred = model.predict(df)
            st.success(f"Predicted label: {pred[0]}")
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(df)[0]
                st.write("Prediction probabilities:", probs.tolist())
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif mode == "Single row (JSON)":
    st.markdown("Paste a JSON object for one sample. Example: `{\"age\":45, \"gender\":\"Female\", \"bmi\":28.5}`")
    raw = st.text_area("JSON input", height=150)
    if st.button("Predict from JSON"):
        if not raw:
            st.error("Please provide JSON input.")
        else:
            try:
                sample_dict = pd.read_json(io.StringIO("[" + raw.strip().lstrip("{").rstrip("}") + "]"), orient="records")
            except Exception:
                # fallback to Python json parsing for more forgiving behavior
                import json
                try:
                    parsed = json.loads(raw)
                    sample_dict = pd.DataFrame([parsed])
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")
                    st.stop()

            try:
                pred = model.predict(sample_dict)
                st.success(f"Predicted label: {pred[0]}")
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(sample_dict)[0]
                    st.write("Prediction probabilities:", probs.tolist())
            except Exception as e:
                st.error(f"Prediction failed: {e}")

else:  # Batch CSV
    st.markdown("Upload a CSV file for batch predictions. Do not include the target column.")
    uploaded_csv = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_csv is not None:
        try:
            df = pd.read_csv(uploaded_csv)
            st.write("Preview:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        if st.button("Run batch prediction"):
            try:
                preds = model.predict(df)
                st.write("Predictions (first 100):")
                st.write(pd.Series(preds).head(100).to_list())
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df)
                    # show first 5 rows of probabilities
                    st.write("Probabilities (first 5 rows):")
                    st.dataframe(pd.DataFrame(probs).head(5))
                # Offer download of results
                out_df = df.copy()
                out_df["_prediction"] = preds
                csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")

# Footer / tips
st.markdown("---")
st.markdown(
    "Tips:\n"
    "- If loading fails due to pickle version mismatch, re-create the model pickle in the same Python environment as your deployment.\n"
    "- To avoid uploading each time, set the environment variable `MODEL_PATH` to the absolute path of your model file before launching Streamlit."
)

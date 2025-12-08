# streamlit_app.py
"""
Streamlit app that uses artifacts produced by preprocessing/training:
  ./artifacts/model_best.pkl
  ./artifacts/preprocessor.pkl   (optional)
  ./artifacts/metadata.json
  ./artifacts/feature_importances.csv  (optional)
  ./artifacts/plots/*  (optional)

Run:
  streamlit run streamlit_app.py
"""
import os
import json
import joblib
import pickle
from typing import Any, Dict, List

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

st.set_page_config(page_title="Diabetes Predictor (artifacts)", layout="centered")
st.title("Diabetes Predictor (using artifacts from preprocessing/training)")

ARTIFACTS_DIR = st.sidebar.text_input("Artifacts directory", value="artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model_best.pkl")
PREPROC_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.pkl")
METADATA_PATH = os.path.join(ARTIFACTS_DIR, "metadata.json")
FEATURE_IMP_CSV = os.path.join(ARTIFACTS_DIR, "feature_importances.csv")
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
TRAINING_REPORT = os.path.join(ARTIFACTS_DIR, "training_report.txt")

# ---------------- Utilities ----------------
@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        # fallback to pickle
        with open(path, "rb") as f:
            return pickle.load(f)

@st.cache_data
def load_metadata(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

def human_label(pred_int: int) -> str:
    return "Diabetes" if int(pred_int) == 1 else "No diabetes"

def safe_float2(x):
    try:
        return float(round(float(x), 2))
    except Exception:
        return 0.0

# ---------------- Load artifacts ----------------
st.sidebar.header("Load / override artifacts")
st.sidebar.write("Default metadata path:", METADATA_PATH)

model = load_model(MODEL_PATH)
if model is None:
    st.sidebar.warning(f"No model found at `{MODEL_PATH}`. You can upload a model file (.pkl/.joblib).")
    uploaded_model = st.sidebar.file_uploader("Upload model (joblib/pickle)", type=["pkl","joblib","pickle"])
    if uploaded_model is not None:
        try:
            uploaded_model.seek(0)
            model = joblib.load(uploaded_model)
            st.sidebar.success("Uploaded model loaded.")
        except Exception:
            try:
                uploaded_model.seek(0)
                model = pickle.load(uploaded_model)
                st.sidebar.success("Uploaded model loaded (pickle).")
            except Exception as e:
                st.sidebar.error(f"Failed to load model: {e}")
                model = None
else:
    st.sidebar.success(f"Loaded model from {MODEL_PATH}")

metadata = load_metadata(METADATA_PATH)
if metadata is None:
    st.sidebar.warning(f"No metadata found at `{METADATA_PATH}`.")
    uploaded_meta = st.sidebar.file_uploader("Upload metadata.json (optional)", type=["json"])
    if uploaded_meta is not None:
        try:
            metadata = json.load(uploaded_meta)
            st.sidebar.success("metadata.json loaded.")
        except Exception as e:
            st.sidebar.error(f"Bad metadata.json: {e}")
            metadata = None

# If metadata present, show summary metrics (human readable)
if metadata:
    st.sidebar.markdown("### Model metadata (from metadata.json)")
    st.sidebar.write(f"Model: {metadata.get('model_name', '—')}")
    st.sidebar.write(f"Threshold: {metadata.get('threshold', 0.5):.4f}")
    # show short metrics
    if "metrics_at_threshold" in metadata:
        st.sidebar.markdown("**Metrics (at chosen threshold)**")
        for k, v in metadata["metrics_at_threshold"].items():
            if k != "confusion_matrix":
                st.sidebar.write(f"- {k}: {v}")
    st.sidebar.markdown("**Columns**")
    st.sidebar.write("Numeric:", metadata.get("numeric_columns", []))
    st.sidebar.write("Categorical:", metadata.get("categorical_columns", []))
    # cite metadata file
    st.sidebar.caption("Metadata loaded from training step. :contentReference[oaicite:3]{index=3}")

# Show training report if available
if os.path.exists(TRAINING_REPORT):
    try:
        with open(TRAINING_REPORT, "r") as f:
            rep = f.read()
        st.sidebar.markdown("### Training report")
        st.sidebar.text(rep)
        st.sidebar.caption("Saved by training script. :contentReference[oaicite:4]{index=4}")
    except Exception:
        pass

# ---------------- Show EDA plots and feature importances ----------------
st.header("EDA & Model artifacts")

col1, col2 = st.columns([1, 2])
with col1:
    if os.path.exists(FEATURE_IMP_CSV):
        try:
            fi = pd.read_csv(FEATURE_IMP_CSV)
            st.subheader("Top feature importances")
            st.dataframe(fi.head(20))
            st.bar_chart(fi.set_index("feature").importance.head(20))
        except Exception as e:
            st.write("Could not load feature_importances.csv:", e)
with col2:
    # show available plot images
    if os.path.isdir(PLOTS_DIR):
        imgs = sorted([p for p in os.listdir(PLOTS_DIR) if p.lower().endswith((".png",".jpg",".jpeg"))])
        if imgs:
            st.subheader("EDA plots")
            for img_name in imgs:
                img_path = os.path.join(PLOTS_DIR, img_name)
                try:
                    st.image(img_path, caption=img_name, use_column_width=True)
                except Exception:
                    st.write("Couldn't load image", img_name)
        else:
            st.info("No plots found in artifacts/plots/")

# ---------------- Build input form from metadata ----------------
st.header("Single-sample prediction")

# If metadata exists extract columns; else fallback to asking user
if metadata:
    numeric_cols: List[str] = metadata.get("numeric_columns", []) or []
    categorical_cols: List[str] = metadata.get("categorical_columns", []) or []
    threshold: float = float(metadata.get("threshold", 0.5))
else:
    # fallback
    numeric_cols = []
    categorical_cols = []
    threshold = 0.5

# If neither set present, ask user to paste columns
if not numeric_cols and not categorical_cols:
    st.info("No column metadata available — please paste comma-separated feature names (exclude target).")
    cols_text = st.text_area("Feature column names (comma-separated)", value="")
    if cols_text.strip():
        names = [c.strip() for c in cols_text.split(",") if c.strip()]
        # quick numeric choice
        st.write("Mark numeric columns (others will be categorical):")
        numeric_cols = []
        categorical_cols = []
        for n in names:
            if st.checkbox(f"{n} numeric?", key=f"num_{n}"):
                numeric_cols.append(n)
            else:
                categorical_cols.append(n)

# Build the simple, user-friendly form
form_values: Dict[str, Any] = {}
with st.form("pred_form"):
    st.markdown("Enter patient values")
    # numeric widgets first (2 decimal precision except age integer)
    for col in numeric_cols:
        lname = col.lower()
        if "age" == lname or lname == "age":
            val = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1, format="%d", key=f"age_{col}")
            form_values[col] = int(val)
        else:
            default = 0.0
            if "bmi" in lname:
                default = 25.0
            if "hba" in lname or "hba1c" in lname:
                default = 5.5
            if "glucose" in lname or "blood" in lname:
                default = 100.0
            val = st.number_input(col, value=float(default), step=0.01, format="%.2f", key=f"num_{col}")
            form_values[col] = safe_float2(val)

    # categorical widgets
    for col in categorical_cols:
        lname = col.lower()
        if "gender" in lname:
            form_values[col] = st.radio("Gender", ["Female", "Male", "Other"], index=0, horizontal=True, key=f"gender_{col}")
        elif "hypert" in lname or "hyper" in lname:
            form_values[col] = st.radio("Hypertension", ["No", "Yes"], index=0, horizontal=True, key=f"htn_{col}")
        elif "heart" in lname and "disease" in lname:
            form_values[col] = st.radio("Heart disease", ["No", "Yes"], index=0, horizontal=True, key=f"hd_{col}")
        elif "smok" in lname:
            form_values[col] = st.selectbox("Smoking history", ["never", "former", "current", "not current", "unknown"], index=0, key=f"smk_{col}")
        else:
            form_values[col] = st.text_input(col, value="", key=f"cat_{col}")

    submitted = st.form_submit_button("Predict")

# ---------------- Predict ----------------
if submitted:
    if model is None:
        st.error("No model loaded. Place model_best.pkl in artifacts/ or upload one in the sidebar.")
    else:
        # build DataFrame
        X_input = pd.DataFrame([form_values])

        # normalize Yes/No -> 1/0 for columns where that pattern exists
        for c in X_input.columns:
            vals = X_input[c].astype(str).str.lower().tolist()
            if all(v in ("yes", "no") or v in ("1","0") for v in vals):
                X_input[c] = X_input[c].map(lambda x: 1 if str(x).lower() == "yes" or str(x) == "1" else 0)

        # ensure numeric types
        for c in numeric_cols:
            if c in X_input.columns:
                try:
                    if c.lower() == "age":
                        X_input[c] = X_input[c].astype(int)
                    else:
                        X_input[c] = X_input[c].astype(float)
                except Exception:
                    pass

        st.subheader("Input used for prediction")
        st.dataframe(X_input.T.rename(columns={0: "value"}))

        # get pipeline (if model was saved as dict)
        pipeline = model.get("model_pipeline") if isinstance(model, dict) and "model_pipeline" in model else model

        try:
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_input)[0]
                # binary classes -> positive index assumed 1
                if len(probs) == 2:
                    pos_prob = float(probs[1])
                else:
                    pos_prob = float(max(probs))
            else:
                pos_prob = None
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            pos_prob = None

        # determine label using metadata threshold (if present)
        thresh = float(metadata.get("threshold", 0.5)) if metadata else 0.5
        if pos_prob is not None:
            pred_int = 1 if pos_prob >= thresh else 0
            pred_text = human_label(pred_int)
            if pred_int == 1:
                st.error(f"{pred_text} — Confidence {pos_prob*100:.2f}% (threshold {thresh:.3f})")
            else:
                st.success(f"{pred_text} — Confidence {pos_prob*100:.2f}% (threshold {thresh:.3f})")
            st.write("Raw probabilities:", probs)
        else:
            # fallback to predict()
            try:
                pred = pipeline.predict(X_input)[0]
                st.write("Prediction:", human_label(int(pred)))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------- Footer / tips ----------------
st.markdown("---")
st.markdown(
    "Notes:\n"
    "- This app expects artifacts produced by the preprocessing/training script (model, metadata, feature importances, plots). "
    "If you used the provided training script it creates these files. :contentReference[oaicite:5]{index=5}\n"
    "- The metadata.json contains the chosen threshold and column lists used here. :contentReference[oaicite:6]{index=6}\n"
    "- The training report is available in artifacts/training_report.txt. :contentReference[oaicite:7]{index=7}"
)

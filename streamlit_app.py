# streamlit_app.py
"""
Simple Streamlit UI for single-row diabetes prediction.
- Age: integer
- Numeric values: 2 decimal places
- Gender: radio buttons
- Human-readable output with probability percent
- Loads model from ./best_model.pkl or from MODEL_PATH env var
- If model not found, allows uploading a .pkl file
"""

import os
import pickle
from typing import Optional, Dict, Any
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simple Diabetes Predictor", layout="centered")

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model.pkl")

# ---- Helpers ----
def load_model_from_path(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model from {path}: {e}")
        return None
    # normalize: support either dict with 'model_pipeline' or raw pipeline
    if isinstance(obj, dict) and "model_pipeline" in obj:
        return obj
    # if pipeline saved directly, try to attach sensible metadata
    return {"model_pipeline": obj, "numeric_columns": ["age","bmi","HbA1c_level","blood_glucose_level"], "categorical_columns": ["gender","hypertension","heart_disease","smoking_history"]}

@st.cache_resource
def cached_load_model(path: str):
    return load_model_from_path(path)

def human_label(pred):
    # adjust based on model's label encoding; common assumption: 1 -> diabetic, 0 -> not
    if pred == 1 or str(pred) == "1":
        return "Diabetes"
    return "No diabetes"

# ---- Load model ----
st.title("Simple Diabetes Prediction")
st.caption("Fill the form below and click Predict")

model_pack = cached_load_model(MODEL_PATH)
if model_pack is None:
    st.warning(f"Model not found at `{MODEL_PATH}`. Upload a model (.pkl) or place the file next to this app.")
    uploaded = st.file_uploader("Upload model file (.pkl)", type=["pkl","pickle"], accept_multiple_files=False)
    if uploaded is not None:
        try:
            model_bytes = uploaded.read()
            model_obj = pickle.loads(model_bytes)
            if isinstance(model_obj, dict) and "model_pipeline" in model_obj:
                model_pack = model_obj
            else:
                model_pack = {"model_pipeline": model_obj,
                              "numeric_columns": ["age","bmi","HbA1c_level","blood_glucose_level"],
                              "categorical_columns": ["gender","hypertension","heart_disease","smoking_history"]}
            st.success("Model uploaded and loaded.")
        except Exception as e:
            st.error(f"Failed to load uploaded model: {e}")
            st.stop()

# stop here if still no model
if model_pack is None:
    st.stop()

model = model_pack["model_pipeline"]

# ---- Simple form ----
with st.form("predict_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.radio("Gender", options=["Female", "Male", "Other"], horizontal=True)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1, format="%d")
        hypertension = st.radio("Hypertension", options=["No", "Yes"], index=0, horizontal=True)
        heart_disease = st.radio("Heart disease", options=["No", "Yes"], index=0, horizontal=True)
    with col2:
        smoking_history = st.selectbox("Smoking history", options=["never", "former", "current", "not current", "unknown"], index=0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.00, step=0.01, format="%.2f")
        HbA1c_level = st.number_input("HbA1c level (%)", min_value=0.0, max_value=30.0, value=5.50, step=0.01, format="%.2f")
        blood_glucose_level = st.number_input("Blood glucose level (mg/dL)", min_value=0.0, max_value=1000.0, value=100.00, step=0.01, format="%.2f")

    submitted = st.form_submit_button("Predict")

# ---- Prediction and display ----
if submitted:
    # Normalize yes/no to 0/1 if model expects integers
    try:
        input_dict = {
            "gender": gender,
            "age": int(age),
            "hypertension": 1 if hypertension == "Yes" else 0,
            "heart_disease": 1 if heart_disease == "Yes" else 0,
            "smoking_history": smoking_history,
            "bmi": float(round(bmi, 2)),
            "HbA1c_level": float(round(HbA1c_level, 2)),
            "blood_glucose_level": float(round(blood_glucose_level, 2))
        }

        df = pd.DataFrame([input_dict])

        # Predict
        pred = model.predict(df)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob_arr = model.predict_proba(df)[0]
                # choose probability of positive class; assume class order [0,1]
                if len(prob_arr) == 2:
                    prob = prob_arr[1]
                else:
                    # fallback: show max prob
                    prob = max(prob_arr)
            except Exception:
                prob = None

        # Human readable output
        label = human_label(pred)
        st.markdown("### Result")
        st.write(f"**Prediction:** {label}")
        if prob is not None:
            st.write(f"**Confidence:** {prob*100:.2f}%")
        else:
            st.write("**Confidence:** not available")

        st.markdown("**Input used:**")
        st.table(pd.DataFrame(input_dict, index=["value"]).T)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)

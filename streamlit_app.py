# streamlit_app.py
"""
Simple Diabetes Predictor UI (unchanged)
Loads:
  - model_best.pkl
  - metadata.json
from the SAME DIRECTORY as this app.
"""

import os
import pickle
import json
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simple Diabetes Predictor", layout="centered")

# ------------------------------
# Load model + metadata (same folder)
# ------------------------------
MODEL_PATH = "model_best.pkl"
METADATA_PATH = "metadata.json"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

model = load_model()
metadata = load_metadata()

if model is None:
    st.error("❌ model_best.pkl not found. Place model_best.pkl in the same folder as streamlit_app.py.")
    st.stop()

if metadata is None:
    st.error("❌ metadata.json not found. Place metadata.json in the same folder as streamlit_app.py.")
    st.stop()

numeric_cols = metadata.get("numeric_columns", [])
categorical_cols = metadata.get("categorical_columns", [])
threshold = metadata.get("threshold", 0.5)

# ------------------------------
# UI (exact layout preserved)
# ------------------------------
st.title("Simple Diabetes Prediction")
st.caption("Fill the form below and click Predict")

with st.form("predict_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    # LEFT SIDE INPUTS
    with col1:
        gender = st.radio("Gender", ["Female", "Male", "Other"], horizontal=True)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1, format="%d")
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart disease", ["No", "Yes"], horizontal=True)

    # RIGHT SIDE INPUTS
    with col2:
        smoking_history = st.selectbox("Smoking history",
                                       ["never", "former", "current", "not current", "unknown"])
        bmi = st.number_input("BMI", value=25.00, min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
        HbA1c_level = st.number_input("HbA1c level (%)", value=5.50, min_value=0.0, max_value=30.0,
                                      step=0.01, format="%.2f")
        blood_glucose_level = st.number_input("Blood glucose level (mg/dL)",
                                              value=100.00,
                                              min_value=0.0, max_value=1000.0,
                                              step=0.01, format="%.2f")

    submitted = st.form_submit_button("Predict")

# ------------------------------
# Prediction
# ------------------------------
if submitted:
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

    try:
        pipeline = model

        # Predict probability
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(df)[0][1]
        else:
            prob = None

        # Use threshold from metadata.json
        if prob is not None:
            pred = 1 if prob >= threshold else 0
        else:
            pred = pipeline.predict(df)[0]

        result = "Diabetes" if pred == 1 else "No diabetes"

        # Display result
        st.markdown("### Result")
        if pred == 1:
            st.error(f"**{result}** — Probability: {prob*100:.2f}% (threshold {threshold:.2f})")
        else:
            st.success(f"**{result}** — Probability: {prob*100:.2f}% (threshold {threshold:.2f})")

        st.markdown("### Input used")
        st.table(df.T.rename(columns={0: "value"}))

        if prob is not None:
            st.write("Raw predicted probability:", prob)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

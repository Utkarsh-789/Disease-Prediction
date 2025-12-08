# streamlit_app.py
"""
Simple Diabetes Predictor — simplified result display.

- Loads model_best.pkl and metadata.json from same folder.
- UI layout preserved.
- Output: "Diabetes: Yes/No" and "Model accuracy: XX.XX%"
"""

import os
import pickle
import json
import joblib
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Simple Diabetes Predictor", layout="centered")

MODEL_PATH = "model_best.pkl"
METADATA_PATH = "metadata.json"

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

def load_metadata():
    if not os.path.exists(METADATA_PATH):
        return None
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

def pretty_accuracy_from_metadata(meta: dict):
    """
    Try to extract a sensible accuracy number from metadata.
    Priority:
      1) metrics_at_threshold.accuracy
      2) metrics_at_0.5.accuracy
      3) metrics.accuracy
      4) metrics_at_threshold.roc_auc * 100 (fallback)
      If nothing, return None.
    """
    if not meta:
        return None
    for key in ("metrics_at_threshold", "metrics_at_0.5", "metrics"):
        d = meta.get(key)
        if isinstance(d, dict):
            acc = d.get("accuracy")
            if acc is not None:
                try:
                    return float(acc) * 100.0
                except Exception:
                    try:
                        return float(acc)
                    except Exception:
                        pass
    # fallback: use roc_auc if present
    for key in ("metrics_at_threshold", "metrics_at_0.5", "metrics"):
        d = meta.get(key)
        if isinstance(d, dict):
            roc = d.get("roc_auc")
            if roc is not None:
                try:
                    return float(roc) * 100.0
                except Exception:
                    pass
    return None

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
threshold = float(metadata.get("threshold", 0.5))

st.title("Simple Diabetes Prediction")
st.caption("Fill the form below and click Predict")

with st.form("predict_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.radio("Gender", ["Female", "Male", "Other"], horizontal=True)
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=30, step=1, format="%d")
        hypertension = st.radio("Hypertension", ["No", "Yes"], horizontal=True)
        heart_disease = st.radio("Heart disease", ["No", "Yes"], horizontal=True)

    with col2:
        smoking_history = st.selectbox("Smoking history",
                                       ["never", "former", "current", "not current", "unknown"])
        bmi = st.number_input("BMI", value=25.00, min_value=0.0, max_value=100.0, step=0.01, format="%.2f")
        HbA1c_level = st.number_input("HbA1c level (%)", value=5.50, min_value=0.0, max_value=30.0,
                                      step=0.01, format="%.2f")
        blood_glucose_level = st.number_input("Blood glucose level (mg/dL)", value=100.00,
                                              min_value=0.0, max_value=1000.0, step=0.01, format="%.2f")

    submitted = st.form_submit_button("Predict")

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
        pipeline = model.get("model_pipeline") if isinstance(model, dict) and "model_pipeline" in model else model

        pos_prob = None
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(df)[0]
            # assume binary classification; positive class index 1
            pos_prob = float(proba[1]) if len(proba) >= 2 else float(max(proba))

        # Determine predicted label using threshold from metadata
        if pos_prob is not None:
            pred_label = 1 if pos_prob >= threshold else 0
        else:
            pred_label = int(pipeline.predict(df)[0])

        # Simple Yes / No output
        yes_no = "Yes" if pred_label == 1 else "No"

        # Model accuracy from metadata (percent)
        acc_pct = pretty_accuracy_from_metadata(metadata)

        # Display clean result
        st.markdown("### Diabetes")
        if pred_label == 1:
            st.markdown(f"<h2 style='color:red'>Yes</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color:green'>No</h2>", unsafe_allow_html=True)

        if acc_pct is not None:
            st.write(f"**Model accuracy:** {acc_pct:.2f}%")
        else:
            st.write("**Model accuracy:** not available in metadata")

        # Optionally show the input used (small)
        st.markdown("**Input used:**")
        st.table(df.T.rename(columns={0: "value"}))

        # If you still want to show probability (commented out by default)
        # if pos_prob is not None:
        #     st.write(f"Model confidence (positive class): {pos_prob*100:.2f}% (threshold {threshold:.2f})")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

"""
streamlit_app.py

Run:
  streamlit run streamlit_app.py

This simple app:
 - Lets you upload a CSV for batch prediction OR
 - Fill a form for a single prediction
 - Shows predicted label and probability (if available)
"""
import streamlit as st
import pandas as pd
import pickle

MODEL_PATH = "best_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

st.title("Diabetes Prediction - Demo")

model_pack = load_model()

choice = st.radio("Choose input type:", ["Single input (form)", "Upload CSV for batch"])

if choice == "Single input (form)":
    st.write("Fill features below (use appropriate types). Missing fields will be imputed by pipeline.")
    # attempt to adapt form dynamically if feature names are known:
    numeric_cols = model_pack.get("numeric_columns", [])
    categorical_cols = model_pack.get("categorical_columns", [])

    form = {}
    for c in numeric_cols:
        form[c] = st.number_input(c, value=float(0) if "age" not in c.lower() else 30.0)
    for c in categorical_cols:
        form[c] = st.text_input(c, value="unknown")

    if st.button("Predict"):
        df = pd.DataFrame([form])
        model = model_pack['model_pipeline']
        pred = model.predict(df)[0]
        st.write("Predicted label:", pred)
        if hasattr(model, "predict_proba"):
            st.write("Prediction probabilities:", model.predict_proba(df)[0])

else:
    uploaded = st.file_uploader("Upload CSV (no target column expected)", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview of uploaded data:", df.head())
        if st.button("Run batch prediction"):
            model = model_pack['model_pipeline']
            preds = model.predict(df)
            st.write("Predictions (first 50):", preds[:50])
            if hasattr(model, "predict_proba"):
                st.write("Probabilities (first 5 rows):")
                st.write(model.predict_proba(df)[:5])

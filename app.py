import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.title("Heart Disease Risk Predictor (XGBoost)")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "final_xgb_pipeline.joblib"
model = joblib.load(MODEL_PATH)

FEATURES = [
    'Age',
    'Sex',
    'Chest pain type',
    'BP',
    'Cholesterol',
    'FBS over 120',
    'EKG results',
    'Max HR',
    'Exercise angina',
    'ST depression',
    'Slope of ST',
    'Number of vessels fluro',
    'Thallium'
]

st.write("Fill the inputs below and click Predict to get probability.")

# --- 4-column layout for inputs ---
cols = st.columns(4)   # create 4 columns
input_data = {}

for i, col in enumerate(FEATURES):
    with cols[i % 4]:
        input_data[col] = st.text_input(col, "")

if st.button("Predict"):
    df = pd.DataFrame([input_data])

    # Convert numeric columns safely (text stays for categoricals)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    prob = model.predict_proba(df)[:, 1][0]
    st.success(f"Predicted Heart Disease Probability: {prob:.4f}")

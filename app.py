import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("Heart Disease Risk Predictor")

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

# -------- Helpers ----------
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try numeric conversion where possible. Keep categoricals if they are strings."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out

def risk_label(prob: float) -> str:
    if prob < 0.10: return "Very-Low"
    if prob < 0.30: return "Low"
    if prob < 0.60: return "Medium"
    if prob < 0.85: return "High"
    return "Very-High"

# -------- UI mode ----------
mode = st.radio("Choose input method", ["Manual input", "Upload CSV"], horizontal=True)

# -------- Manual input ----------
if mode == "Manual input":
    st.write("Fill inputs and click Predict.")

    cols = st.columns(4)
    input_data = {}

    # optional defaults
    DEFAULTS = {
        "Age": "45", "Sex": "1", "BP": "120", "Cholesterol": "200",
        "FBS over 120": "0", "Max HR": "150", "Exercise angina": "0",
        "ST depression": "0.0", "Number of vessels fluro": "0"
    }

    for i, col in enumerate(FEATURES):
        with cols[i % 4]:
            input_data[col] = st.text_input(col, value=DEFAULTS.get(col, ""))

    if st.button("Predict"):
        df = pd.DataFrame([input_data])
        df = coerce_types(df)

        prob = model.predict_proba(df)[:, 1][0]
        label = risk_label(prob)

        st.success(f"Risk: {label} | Probability: {prob:.4f}")
        st.dataframe(df, use_container_width=True)

# -------- CSV upload ----------
else:
    st.write("Upload a CSV with these columns (exact names):")
    st.code(", ".join(FEATURES))

    # Download template CSV
    template = pd.DataFrame([{c: "" for c in FEATURES}])
    st.download_button(
        "Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="heart_input_template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded is not None:
        df_all = pd.read_csv(uploaded)

        # Check required columns
        missing = [c for c in FEATURES if c not in df_all.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Keep only feature columns (ignore extras)
        X = df_all[FEATURES].copy()
        X = coerce_types(X)

        st.subheader("Preview")
        st.dataframe(X.head(20), use_container_width=True)

        # Select row(s)
        if len(X) == 1:
            idxs = [0]
        else:
            idxs = st.multiselect(
                "Select row index(es) to predict",
                options=list(range(len(X))),
                default=[0]
            )

        if st.button("Predict selected"):
            if not idxs:
                st.warning("Select at least one row.")
                st.stop()

            X_sel = X.iloc[idxs].copy()
            probs = model.predict_proba(X_sel)[:, 1]

            result = X_sel.copy()
            result["probability"] = probs
            result["risk"] = [risk_label(p) for p in probs]

            st.subheader("Predictions")
            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download predictions CSV",
                data=result.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )

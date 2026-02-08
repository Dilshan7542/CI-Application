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
    "Age",
    "Sex",
    "Chest pain type",
    "BP",
    "Cholesterol",
    "FBS over 120",
    "EKG results",
    "Max HR",
    "Exercise angina",
    "ST depression",
    "Slope of ST",
    "Number of vessels fluro",
    "Thallium",
]

MAX_SELECT = 10  # row limit


# -------- Helpers ----------
def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Try numeric conversion where possible. Keep categoricals if they are strings."""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def risk_label(prob: float) -> str:
    if prob < 0.10:
        return "Very-Low"
    if prob < 0.30:
        return "Low"
    if prob < 0.60:
        return "Medium"
    if prob < 0.85:
        return "High"
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
        "Age": "45",
        "Sex": "1",
        "BP": "120",
        "Cholesterol": "200",
        "FBS over 120": "0",
        "Max HR": "150",
        "Exercise angina": "0",
        "ST depression": "0.0",
        "Number of vessels fluro": "0",
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
        mime="text/csv",
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

        st.subheader("Preview (first 20 rows)")
        st.dataframe(X.head(20), use_container_width=True)

        #  Row selection (limit to 10)
        st.subheader(f"Select rows (max {MAX_SELECT})")

        if len(X) == 1:
            idxs = [0]
            st.info("Only 1 row found in CSV. It will be predicted automatically.")
        else:
            select_mode = st.radio(
                "Row selection method",
                ["Pick up to 10 (recommended)", "Range (auto limits to 10)"],
                horizontal=True,
            )

            if select_mode == "Pick up to 10 (recommended)":
            
                cand = X.copy()
                options = cand.index.tolist()
                st.caption(f"Matching rows: {len(options)}")

                default = options[: min(3, len(options))]  # small default
                idxs = st.multiselect(
                    f"Pick up to {MAX_SELECT} row index(es)",
                    options=options,
                    default=default,
                    max_selections=MAX_SELECT, 
                )

            else:
                max_idx = len(X) - 1
                col1, col2 = st.columns(2)
                with col1:
                    start_i = st.number_input(
                        "Start index", min_value=0, max_value=max_idx, value=0, step=1
                    )
                with col2:
                    end_i = st.number_input(
                        "End index (inclusive)",
                        min_value=0,
                        max_value=max_idx,
                        value=min(MAX_SELECT - 1, max_idx),
                        step=1,
                    )

                start_i = int(start_i)
                end_i = int(end_i)
                if start_i > end_i:
                    st.error("Start index must be <= End index")
                    st.stop()

                idxs = list(range(start_i, end_i + 1))

                # auto row limit to 10
                if len(idxs) > MAX_SELECT:
                    st.warning(
                        f"Range selected {len(idxs)} rows. Limiting to first {MAX_SELECT}."
                    )
                    idxs = idxs[:MAX_SELECT]

        # show selected preview
        if idxs:
            st.caption(f"Selected rows: {idxs}")
            st.dataframe(X.iloc[idxs], use_container_width=True)
        else:
            st.warning("No rows selected yet.")

        # -------- Predict ----------
        if st.button("Predict selected"):
            if not idxs:
                st.warning("Select at least one row.")
                st.stop()

            # final safety limit (even if UI changes)
            if len(idxs) > MAX_SELECT:
                st.warning(
                    f"Too many rows selected ({len(idxs)}). Limiting to first {MAX_SELECT}."
                )
                idxs = idxs[:MAX_SELECT]

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
                mime="text/csv",
            )

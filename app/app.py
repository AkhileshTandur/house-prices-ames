import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

@st.cache_resource
def load_artifacts():
    model = load(MODELS_DIR / "house_price_pipeline.joblib")
    with open(MODELS_DIR / "feature_schema.json", "r") as f:
        schema = json.load(f)
    return model, schema

model, schema = load_artifacts()
expected_cols = schema["feature_columns"]

st.set_page_config(page_title="Ames House Price Predictor", layout="centered")
st.title("🏠 Ames House Price Predictor")
st.write("Enter a few key features or upload a CSV with full schema.")

# ----- Manual form -----
with st.form("manual_input"):
    col1, col2 = st.columns(2)
    with col1:
        GrLivArea = st.number_input("GrLivArea", min_value=100, max_value=10000, value=1500)
        TotalBsmtSF = st.number_input("TotalBsmtSF", min_value=0, max_value=4000, value=800)
        GarageCars = st.number_input("GarageCars", min_value=0, max_value=5, value=2)
        OverallQual = st.slider("OverallQual", 1, 10, 6)
    with col2:
        YearBuilt = st.number_input("YearBuilt", min_value=1872, max_value=2025, value=1995)
        FullBath = st.number_input("FullBath", min_value=0, max_value=4, value=2)
        HalfBath = st.number_input("HalfBath", min_value=0, max_value=3, value=1)

        # Dropdown of valid values if available; fallback to text input
        nb_options = (schema.get("categorical_values", {}) or {}).get("Neighborhood", None)
        if nb_options:
            Neighborhood = st.selectbox("Neighborhood", nb_options, index=0)
        else:
            Neighborhood = st.text_input("Neighborhood", value="NAmes")

    submitted = st.form_submit_button("Predict from form")

if submitted:
    base = {c: np.nan for c in expected_cols}
    base.update({
        "GrLivArea": GrLivArea,
        "TotalBsmtSF": TotalBsmtSF,
        "GarageCars": GarageCars,
        "OverallQual": OverallQual,
        "YearBuilt": YearBuilt,
        "FullBath": FullBath,
        "HalfBath": HalfBath,
        "Neighborhood": Neighborhood,
    })
    X = pd.DataFrame([base])
    pred = float(model.predict(X)[0])
    st.success(f"Estimated Sale Price: ${pred:,.0f}")

# ----- CSV uploader (robust) -----
st.divider()
st.subheader("📤 Or upload CSV for batch predictions")
uploaded = st.file_uploader("Upload CSV (any subset of columns is fine).", type=["csv"])
if uploaded:
    raw = pd.read_csv(uploaded)

    if raw.shape[0] == 0:
        st.error("Uploaded CSV has no rows. Please upload a non-empty CSV.")
    else:
        n = len(raw)
        safe = {}
        for c in expected_cols:
            if c in raw.columns:
                # ensure arrays (avoid pandas scalar issue)
                safe[c] = raw[c].to_numpy()
            else:
                # broadcast NaNs to correct length to avoid 'all scalars' error
                safe[c] = np.full(n, np.nan, dtype=float)

        X_safe = pd.DataFrame(safe)
        preds = model.predict(X_safe)

        out = raw.copy()
        out["Predicted_SalePrice"] = preds
        st.dataframe(out.head(20))
        st.download_button(
            "Download predictions",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv"
        )

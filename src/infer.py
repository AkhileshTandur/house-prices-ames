import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def load_schema():
    with open(MODELS_DIR / "feature_schema.json", "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Batch inference for Ames house prices")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV of houses")
    parser.add_argument("--output_csv", default=str(ROOT / "predictions.csv"), help="Where to save predictions")
    args = parser.parse_args()

    model = load(MODELS_DIR / "house_price_pipeline.joblib")
    schema = load_schema()
    expected_cols = schema["feature_columns"]

    X = pd.read_csv(args.input_csv)

    # Align columns to schema to avoid KeyErrors
    safe = {c: (X[c] if c in X.columns else np.nan) for c in expected_cols}
    X_safe = pd.DataFrame(safe)

    preds = model.predict(X_safe)
    out = X.copy()
    out["Predicted_SalePrice"] = preds
    out.to_csv(args.output_csv, index=False)
    print(f"[OK] Saved predictions → {args.output_csv}")

if __name__ == "__main__":
    main()

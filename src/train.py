import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW = DATA_DIR / "train.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def _make_dummy_train(path: Path, n: int = 800, seed: int = 42) -> None:
    """Create a synthetic Ames-like dataset so training works without Kaggle."""
    rng = np.random.default_rng(seed)
    neighborhoods = ["NAmes","CollgCr","OldTown","Edwards","Somerst","Gilbert","Sawyer"]
    df = pd.DataFrame({
        "GrLivArea": rng.integers(500, 3500, size=n),
        "TotalBsmtSF": rng.integers(0, 1800, size=n),
        "GarageCars": rng.integers(0, 4, size=n),
        "OverallQual": rng.integers(1, 10, size=n),
        "YearBuilt": rng.integers(1900, 2010, size=n),
        "FullBath": rng.integers(0, 3, size=n),
        "HalfBath": rng.integers(0, 2, size=n),
        "BsmtFullBath": rng.integers(0, 2, size=n),
        "BsmtHalfBath": rng.integers(0, 2, size=n),
        "Neighborhood": rng.choice(neighborhoods, size=n),
    })
    base_price = (
        50_000
        + df["GrLivArea"] * 110
        + df["TotalBsmtSF"] * 40
        + df["GarageCars"] * 8_000
        + df["OverallQual"] * 5_000
        + (df["YearBuilt"] - 1900) * 400
        + df["FullBath"] * 4_000
        + df["HalfBath"] * 2_000
    )
    noise = rng.normal(0, 20_000, size=n)
    df["SalePrice"] = (base_price + noise).clip(20_000, 800_000).round(0).astype(int)
    path.write_text(df.to_csv(index=False, encoding="utf-8"))
    print(f"[INFO] No dataset found. Created synthetic dataset at: {path}")

def main():
    # Ensure data exists; auto-create if missing
    if not DATA_RAW.exists():
        _make_dummy_train(DATA_RAW)

    df = pd.read_csv(DATA_RAW)
    if "SalePrice" not in df.columns:
        raise RuntimeError("train.csv is missing 'SalePrice' column.")

    # Light, safe features
    if {"FullBath","HalfBath","BsmtFullBath","BsmtHalfBath"}.issubset(df.columns):
        df["TotalBathrooms"] = (
            df["FullBath"].fillna(0)
            + 0.5 * df["HalfBath"].fillna(0)
            + df["BsmtFullBath"].fillna(0)
            + 0.5 * df["BsmtHalfBath"].fillna(0)
        )
    if {"GrLivArea","TotalBsmtSF"}.issubset(df.columns):
        df["TotalLivArea"] = df["GrLivArea"].fillna(0) + df["TotalBsmtSF"].fillna(0)

    y = df["SalePrice"].astype(float)
    X = df.drop(columns=["SalePrice"])

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # Dense output so HGBR works across sklearn versions
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    reg = HistGradientBoostingRegressor(random_state=RANDOM_STATE, learning_rate=0.08, max_leaf_nodes=31)

    model = TransformedTargetRegressor(
        regressor=Pipeline([("prep", preprocessor), ("reg", reg)]),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    # CV
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rmse = -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=kf, n_jobs=-1)
    print(f"[INFO] CV RMSE: mean={rmse.mean():.2f} std={rmse.std():.2f}")

    # Fit on full data
    model.fit(X, y)

    # Save model
    model_path = MODELS_DIR / "house_price_pipeline.joblib"
    dump(model, model_path)

    # Save schema + categorical values for UI
    cat_values = {}
    if "Neighborhood" in X.columns:
        cat_values["Neighborhood"] = (
            pd.Series(X["Neighborhood"].dropna().astype(str).unique())
            .sort_values().tolist()
        )
    schema = {
        "feature_columns": X.columns.tolist(),
        "categorical_columns": cat_cols,
        "numeric_columns": num_cols,
        "categorical_values": cat_values,
    }
    (MODELS_DIR / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    # Permutation importance (estimator-level, version-safe)
    rng = np.random.default_rng(RANDOM_STATE)
    val_idx = rng.choice(len(X), size=max(1, int(0.2 * len(X))), replace=False)
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    imp = permutation_importance(
        model, X_val, y_val, n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1, scoring=None
    )
    pd.DataFrame({
        "feature": X.columns,
        "importance_mean": imp.importances_mean,
        "importance_std": imp.importances_std,
    }).sort_values("importance_mean", ascending=False)\
     .to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    # Save metrics for README/reporting
    metrics = {
        "cv_rmse_mean": float(rmse.mean()),
        "cv_rmse_std": float(rmse.std()),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"[OK] Saved model → {model_path}")
    print(f"[OK] Saved schema → {MODELS_DIR / 'feature_schema.json'}")
    print(f"[OK] Saved feature importances → {REPORTS_DIR / 'feature_importance.csv'}")
    print(f"[OK] Saved metrics → {MODELS_DIR / 'metrics.json'}")

if __name__ == "__main__":
    main()

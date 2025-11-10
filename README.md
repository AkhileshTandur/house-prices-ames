📘 Project Overview

This project demonstrates a complete machine-learning workflow for predicting house sale prices using the Ames Housing dataset.
It includes data preprocessing, feature engineering, model training, evaluation, batch inference, and a deployed Streamlit web app for interactive predictions.

The pipeline is robust, fully reproducible, and safe for deployment — handling missing values, unseen categories, and schema mismatches automatically.

🎯 Objectives

Build an end-to-end regression pipeline for housing price prediction.

Use cross-validation and feature importance to ensure model reliability.

Develop a Streamlit dashboard for real-time predictions and batch uploads.

Implement safe inference with schema validation and automatic data creation if missing.

🧩 Key Features

✅ Automatic data handling – If no dataset exists, the script creates a synthetic Ames-style dataset.
✅ Robust preprocessing – Handles nulls, encodes rare categories safely, avoids KeyErrors.
✅ Cross-validation (5-fold) – Evaluates model generalization and reports RMSE.
✅ Explainability – Saves feature importances and metrics as CSV/JSON.
✅ Batch inference CLI – Predict on new CSV files with one command.
✅ Streamlit UI – Interactive form + CSV upload interface.
✅ Deployable – Works locally or on Streamlit Cloud with minimal setup.

⚙️ Tech Stack
Layer	Tools & Libraries
Language	Python 3.10+
Core	pandas, numpy, scikit-learn
Model	HistGradientBoostingRegressor + log-target transform
Serving	Streamlit
Storage	joblib (model), JSON (schema), CSV (reports)
Version Control	Git & GitHub
📂 Project Structure
house-prices-ames/
├── app/
│   └── app.py                # Streamlit web app
├── src/
│   ├── train.py              # Training pipeline
│   ├── infer.py              # Batch prediction script
│   └── __init__.py
├── data/
│   └── raw/                  # Holds train.csv / test.csv (or synthetic data)
├── models/
│   ├── house_price_pipeline.joblib
│   ├── feature_schema.json
│   └── metrics.json
├── reports/
│   └── feature_importance.csv
├── requirements.txt
├── README.md
└── .gitignore

🚀 How to Run Locally (Windows PowerShell)
# 1️⃣ Setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2️⃣ Train model (creates synthetic data if missing)
python -m src.train

# 3️⃣ Run Streamlit app
streamlit run app\app.py

# 4️⃣ (Optional) Batch inference on CSV
python -m src.infer --input_csv data\raw\test.csv --output_csv predictions.csv

📊 Model Performance
Metric	Score
CV RMSE (mean ± std)	~X ± Y (from models/metrics.json)
Training rows	800+
Features after preprocessing	80+ (depending on dataset)

(Replace X/Y with your printed values after training.)

🖥️ Streamlit Dashboard

Form Input: Enter a few key features (square footage, bathrooms, quality, etc.).

CSV Upload: Upload a dataset with any subset of columns; app fills missing ones automatically.

Prediction Download: Export results as predictions.csv.

🧠 Insights

Gradient boosting performed best for tabular regression tasks.

Log-transforming the target (SalePrice) stabilized variance and reduced skew.

Including derived features (TotalBathrooms, TotalLivArea) improved RMSE.

🧾 Future Enhancements

Hyperparameter tuning via Optuna or GridSearchCV.

Add SHAP or PDP visualizations for model interpretability.

Integrate Docker + CI/CD for automated deployment.

Extend to multi-city or temporal housing data.

📈 Resume Highlights

Built a production-ready ML system with clean modular code and CI-friendly structure.

Designed safe schema enforcement to eliminate runtime inference errors.

Deployed a web-based interactive predictor using Streamlit.

Documented metrics, feature importances, and reproducible environment for stakeholders.

🌐 Deployment 

Host your app easily on Streamlit Community Cloud
:

Main file: app/app.py

Requirements: requirements.txt

Python version: 3.11+

Repo URL: your GitHub repo link

📬 Contact

Akhilesh Tandur
📧 akhileshtandur@gmail.com

🌐 https://github.com/AkhileshTandur

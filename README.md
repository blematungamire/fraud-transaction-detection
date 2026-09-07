# Fraud Transaction Detection — Streamlit Application

An AI-powered tool for detecting fraudulent financial transactions using an ensemble of XGBoost, Random Forest, and Isolation Forest models.

## Problem

Financial fraud costs institutions billions annually. This application provides a screening tool that scores individual transactions for fraud probability, enabling analysts to prioritize investigations.

## Features

- **Upload CSV** — Score thousands of transactions in bulk
- **Manual Entry** — Score a single transaction interactively
- **Risk Classification** — CRITICAL / HIGH / MEDIUM / LOW / MINIMAL
- **Visual Dashboard** — Histograms, box plots, pie charts of results
- **Adjustable Threshold** — Tune sensitivity via sidebar slider
- **Export Results** — Download flagged transactions as CSV
- **Model Analytics** — View feature importance, confusion matrix, ROC-AUC

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| ML Models | XGBoost, scikit-learn, imbalanced-learn |
| Visualization | Plotly |
| Data | Pandas, NumPy |

## Project Structure

```
fraud-detection-app/
├── app.py                 # Main Streamlit application
├── model.py               # ML model training, prediction, feature engineering
├── generate_data.py       # Synthetic transaction data generator
├── test_app.py            # Pytest test suite (normal + edge cases)
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── saved_models/          # Saved model artifacts (auto-generated)
    └── fraud_model.joblib
```

## Setup & Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/fraud-detection-app.git
cd fraud-detection-app
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open http://localhost:8501 in your browser.

### Running Tests

```bash
pytest test_app.py -v          # 30 unit tests: data, model, prediction, edge cases
python test_app_native.py      # 3 functional AppTest checks: render, upload+score, manual entry
```

## How It Works

1. **Data Input**: User uploads a CSV or enters a single transaction manually.
2. **Feature Engineering**: 12+ derived features are computed (amount ratios, time flags, balance changes, etc.).
3. **Ensemble Scoring**: Three models produce independent fraud scores:
   - XGBoost (50% weight) — supervised gradient boosting
   - Random Forest (30% weight) — supervised ensemble bagging
   - Isolation Forest (20% weight) — unsupervised anomaly detection
4. **Threshold Application**: Final ensemble score compared against user-set threshold (default 0.5).
5. **Risk Classification**: Score mapped to CRITICAL (≥0.8), HIGH (≥0.6), MEDIUM (≥0.4), LOW (≥0.2), MINIMAL (<0.2).

## Assumptions

- Fraud rate is ~3% of all transactions (industry average)
- Fraudulent transactions tend to have higher amounts, occur at night, and involve international/online channels
- Training data is synthetically generated and mirrors realistic patterns
- No real PII or financial data is used anywhere in the application

## Data Sources

All data is **synthetically generated** via `generate_data.py`. No real financial records, API keys, or personal data are used.

## Limitations

- Synthetic data may not capture all real-world fraud patterns
- Model requires periodic retraining on fresh, real-world data
- No account-level behavioral baselines (each transaction is scored independently)
- No real-time streaming capability in this version

## Deployment

### Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy — the app will be live at `https://<your-app>.streamlit.app`

## License

This project is for educational purposes (HBF2212 — Artificial Intelligence in Finance).

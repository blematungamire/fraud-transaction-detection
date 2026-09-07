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
- **Automated Audit Trail** — every decision is logged with risk score, rules triggered, AI reasoning, data used, action taken, and space for an investigator's decision and final outcome

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
├── audit_trail.py         # Automated audit trail (append-only decision log)
├── generate_data.py       # Synthetic transaction data generator
├── test_app.py            # Pytest test suite (normal + edge cases + audit)
├── test_app_native.py     # Streamlit AppTest functional checks
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── saved_models/          # Saved model artifacts (auto-generated)
│   └── fraud_model.joblib
└── audit_log.csv          # Runtime audit log (auto-created, git-ignored)
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
pytest test_app.py -v          # 37 tests: data, model, prediction, edge cases, audit trail
python test_app_native.py      # 4 functional AppTest checks: render, upload+score, manual entry, audit trail
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
6. **Audit Trail**: Every decision — from both CSV batch and manual entry — is automatically recorded to `audit_log.csv` with the risk score, business rules triggered, AI reasoning, data used, and the action taken. Investigators review flagged transactions and can record their decision and the final outcome.

### Audit Trail Columns

| Column | Description |
|--------|-------------|
| `transaction_id` | Unique identifier of the scored transaction |
| `date_time` | UTC timestamp of the decision |
| `risk_score` | Model fraud probability (0–1) |
| `rules_triggered` | Human-readable business rules that fired |
| `ai_reasoning` | Plain-English explanation of the model's score |
| `data_used` | JSON: input source, channel, type, amount |
| `action_taken` | System action (flagged for review / cleared) |
| `investigator_decision` | Human verdict (filled via the Audit Trail tab) |
| `final_outcome` | Closed-loop outcome of the case |

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

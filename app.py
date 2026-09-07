"""
Fraud Transaction Detection - Streamlit Application
====================================================
A machine learning-powered tool for detecting fraudulent financial transactions.

This app accepts user-uploaded CSV files or manual transaction input, scores each
transaction for fraud probability, and returns flagged transactions with risk levels.

Data Sources: Synthetic data generated via generate_data.py (no real PII used).
Model: Ensemble of XGBoost, Random Forest, and Isolation Forest.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from model import (
    train_model, save_model, load_model, predict,
    get_feature_importance, MODEL_DIR, engineer_features, validate_csv
)
from audit_trail import (
    record_decision, record_batch, update_investigator_decision,
    load_audit_log_dataframe, AUDIT_COLUMNS
)
from currency import (
    SUPPORTED_CURRENCIES, RATE_PER_USD, BASE_CURRENCY,
    to_usd, from_usd, format_amount, currency_options, code_from_option
)
from generate_data import generate_transactions

# ---------------------------------------------------------------------------
# Multi-currency helpers
# ---------------------------------------------------------------------------
AMOUNT_COLS = ("amount", "balance_before", "balance_after")


def df_to_usd(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Return a copy of df with monetary columns converted to USD."""
    out = df.copy()
    for col in AMOUNT_COLS:
        if col in out.columns:
            out[col] = out[col].astype(float) / rate
    return out


def df_to_local(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Return a copy of df with monetary columns converted to local currency."""
    out = df.copy()
    for col in AMOUNT_COLS:
        if col in out.columns:
            out[col] = (out[col].astype(float) * rate).round(2)
    return out

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Transaction Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 14px; opacity: 0.8; }
    .metric-card p { margin: 5px 0 0; font-size: 28px; font-weight: bold; }
    .risk-critical { color: #ff1744; font-weight: bold; }
    .risk-high { color: #ff9100; font-weight: bold; }
    .risk-medium { color: #ffea00; font-weight: bold; }
    .risk-low { color: #00e676; font-weight: bold; }
    .risk-minimal { color: #448aff; font-weight: bold; }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# State Management
# ---------------------------------------------------------------------------
if "model_artifacts" not in st.session_state:
    st.session_state.model_artifacts = None
if "results_df" not in st.session_state:
    st.session_state.results_df = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def auto_train_model():
    """Train model on synthetic data and cache in session state."""
    with st.spinner("Generating synthetic training data (10,000 transactions)..."):
        train_df = generate_transactions(10000)
    with st.spinner("Training ensemble model (XGBoost + RandomForest + IsolationForest)..."):
        artifacts = train_model(train_df)
    save_model(artifacts)
    st.session_state.model_artifacts = artifacts
    return artifacts


def display_metrics(metrics: dict):
    """Display model evaluation metrics."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    with col2:
        st.metric("Avg Precision", f"{metrics['average_precision']:.4f}")
    with col3:
        cm = metrics["confusion_matrix"]
        st.metric("False Positives", f"{cm[0][1]}")
    with col4:
        st.metric("False Negatives", f"{cm[1][0]}")

    st.subheader("Confusion Matrix")
    cm = np.array(metrics["confusion_matrix"])
    labels = ["Legitimate", "Fraud"]
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale="Blues", text=cm, texttemplate="%{text}",
        textfont={"size": 18}
    ))
    fig.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=350, width=500
    )
    st.plotly_chart(fig, width="content")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🛡️ Fraud Detector")
    st.markdown("---")

    st.subheader("⚙️ Settings")
    threshold = st.slider(
        "Fraud Detection Threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Lower = more sensitive (more flags), Higher = more selective"
    )

    st.markdown("---")
    st.subheader("💱 Currency")
    currency_option = st.selectbox(
        "Working currency",
        currency_options(),
        index=0,
        help="Amounts you enter or upload are interpreted in this currency. "
             "They are converted to USD for model scoring and shown back to "
             "you in this currency."
    )
    currency = code_from_option(currency_option)
    rate = float(RATE_PER_USD.get(currency, 1.0))
    c_sym = SUPPORTED_CURRENCIES[currency]["symbol"]

    with st.expander("Exchange rates (per 1 USD)"):
        rate_rows = [f"{k:<5} {v:.4f}" for k, v in RATE_PER_USD.items()]
        st.text("\n".join(rate_rows))
        st.caption("Static reference rates. Replace with a live FX API via "
                   "Streamlit secrets for production.")
    st.caption(f"Inputs & outputs are shown in **{currency} ({c_sym})**.")

    st.markdown("---")
    st.subheader("📊 Model Status")
    if st.session_state.model_artifacts:
        st.success("Model loaded & ready")
    else:
        st.warning("No model loaded")

    if st.button("🔄 Retrain Model", width="stretch"):
        auto_train_model()
        st.success("Model retrained!")
        st.rerun()

    st.markdown("---")
    st.caption("Built with Streamlit + XGBoost + Isolation Forest")
    st.caption("Synthetic data only — no real PII used")

# ---------------------------------------------------------------------------
# Main Content
# ---------------------------------------------------------------------------
st.title("🛡️ Fraud Transaction Detection")
st.markdown(
    "Upload a CSV of transactions or enter one manually. The ensemble ML model "
    "scores each transaction and flags suspicious activity."
)

# Load or train model
if st.session_state.model_artifacts is None:
    if os.path.exists(os.path.join(MODEL_DIR, "fraud_model.joblib")):
        st.session_state.model_artifacts = load_model()
    else:
        auto_train_model()

artifacts = st.session_state.model_artifacts

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_upload, tab_manual, tab_analytics, tab_audit, tab_about = st.tabs(
    ["📁 Upload CSV", "✏️ Manual Entry", "📈 Analytics Dashboard", "📋 Audit Trail", "ℹ️ About"]
)

# ===================== TAB 1: FILE UPLOAD =====================
with tab_upload:
    st.subheader("Upload Transaction Data")
    st.markdown(
        "Upload a CSV file with columns: `amount`, `tx_type`, `channel`, "
        "`hour_of_day`, `day_of_week`, `merchant_category`, `is_international`, "
        "`is_online`, `balance_before`, `balance_after`."
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    col_a, col_b = st.columns(2)
    with col_a:
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(df_upload):,} transactions")
                st.dataframe(df_upload.head(10), width="stretch")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                df_upload = None
        else:
            df_upload = None

    with col_b:
        if st.button("📥 Download Sample CSV", width="stretch"):
            sample = generate_transactions(200, seed=99)
            for col in AMOUNT_COLS:
                sample[col] = (sample[col].astype(float) * rate).round(2)
            st.download_button(
                f"⬇️ Click to download sample ({currency})",
                data=sample.to_csv(index=False),
                file_name="sample_transactions.csv",
                mime="text/csv"
            )

    if df_upload is not None:
        missing = validate_csv(df_upload)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
        else:
            if st.button("🔍 Run Fraud Detection", type="primary", width="stretch"):
                with st.spinner(f"Scoring transactions in {currency}..."):
                    scored_input = df_to_usd(df_upload, rate)
                    results = predict(scored_input, artifacts, threshold=threshold,
                                      currency=currency, symbol=c_sym, usd_per_unit=rate)
                # Persist every decision to the automated audit trail (USD base)
                n_logged = record_batch(results, data_source=f"CSV upload ({uploaded_file.name})",
                                        currency=currency, rate=rate)
                # Convert results back to the selected currency for display
                results = df_to_local(results, rate)
                st.session_state.results_df = results
                st.success(f"📋 {n_logged} decisions were recorded to the Audit Trail (see Audit Trail tab).")

    if st.session_state.results_df is not None:
        results = st.session_state.results_df
        st.markdown("---")
        st.subheader(f"Results ({currency})")

        n_flagged = int(results["is_flagged"].sum())
        total = len(results)
        fraud_pct = results["fraud_score"].mean()
        total_flagged_amount = results.loc[results["is_flagged"] == 1, "amount"].sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Transactions", f"{total:,}")
        m2.metric("Flagged as Fraud", f"{n_flagged:,}", f"{n_flagged/total*100:.1f}%")
        m3.metric("Avg Fraud Score", f"{fraud_pct:.4f}")
        m4.metric("Flagged Amount", f"{format_amount(total_flagged_amount, currency)}")

        st.markdown("---")

        filter_opt = st.radio(
            "Show:", ["All", "Flagged Only", "Clear Only"],
            horizontal=True
        )
        if filter_opt == "Flagged Only":
            display_df = results[results["is_flagged"] == 1].sort_values("fraud_score", ascending=False)
        elif filter_opt == "Clear Only":
            display_df = results[results["is_flagged"] == 0].sort_values("fraud_score")
        else:
            display_df = results.sort_values("fraud_score", ascending=False)

        st.dataframe(
            display_df.style.map(
                lambda v: "background-color: #ffcccc" if isinstance(v, (int, float)) and v >= 0.6 else "",
                subset=["fraud_score"]
            ),
            width="stretch",
            height=400
        )

        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_data = results.to_csv(index=False)
            st.download_button(
                "⬇️ Download Full Results",
                data=csv_data,
                file_name="fraud_detection_results.csv",
                mime="text/csv",
                width="stretch"
            )
        with col_dl2:
            flagged_data = results[results["is_flagged"] == 1].to_csv(index=False)
            st.download_button(
                "⬇️ Download Flagged Transactions Only",
                data=flagged_data,
                file_name="flagged_transactions.csv",
                mime="text/csv",
                width="stretch"
            )

        # --- Charts ---
        st.markdown("---")
        st.subheader("Visualizations")

        vc1, vc2 = st.columns(2)
        with vc1:
            fig_dist = px.histogram(
                results, x="fraud_score", color="is_flagged",
                nbins=50, barmode="overlay",
                color_discrete_map={0: "#448aff", 1: "#ff1744"},
                title="Fraud Score Distribution",
                labels={"is_flagged": "Flagged"}
            )
            fig_dist.update_layout(height=350)
            st.plotly_chart(fig_dist, width="stretch")

        with vc2:
            risk_counts = results["risk_level"].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map={
                    "CRITICAL": "#ff1744", "HIGH": "#ff9100",
                    "MEDIUM": "#ffea00", "LOW": "#00e676", "MINIMAL": "#448aff"
                }
            )
            fig_risk.update_layout(height=350)
            st.plotly_chart(fig_risk, width="stretch")

        vc3, vc4 = st.columns(2)
        with vc3:
            fig_box = px.box(
                results, x="is_flagged", y="amount", color="is_flagged",
                color_discrete_map={0: "#448aff", 1: "#ff1744"},
                title=f"Transaction Amount by Flag Status ({currency})",
                labels={"is_flagged": "Is Flagged", "amount": f"Amount ({currency})"}
            )
            fig_box.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_box, width="stretch")

        with vc4:
            fig_hour = px.histogram(
                results, x="hour_of_day", color="is_flagged",
                nbins=24, barmode="group",
                color_discrete_map={0: "#448aff", 1: "#ff1744"},
                title="Transactions by Hour of Day",
                labels={"is_flagged": "Flagged", "hour_of_day": "Hour"}
            )
            fig_hour.update_layout(height=350)
            st.plotly_chart(fig_hour, width="stretch")


# ===================== TAB 2: MANUAL ENTRY =====================
with tab_manual:
    st.subheader(f"Enter a Single Transaction ({currency})")

    with st.form("manual_entry_form"):
        fc1, fc2, fc3 = st.columns(3)

        with fc1:
            amount = st.number_input(f"Transaction Amount ({c_sym})", min_value=0.01, value=150.0, step=10.0)
            tx_type = st.selectbox("Transaction Type", ["purchase", "withdrawal", "transfer", "payment"])
            channel = st.selectbox("Channel", ["ATM", "Online", "POS", "Mobile", "Branch"])

        with fc2:
            hour_of_day = st.slider("Hour of Day", 0, 23, 12)
            day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            merchant_category = st.selectbox(
                "Merchant Category",
                ["grocery", "electronics", "restaurant", "travel", "gas",
                 "jewelry", "atm_withdrawal", "transfer", "subscription", "other"]
            )

        with fc3:
            is_international = st.checkbox("International Transaction")
            is_online = st.checkbox("Online Transaction")
            balance_before = st.number_input(f"Balance Before ({c_sym})", min_value=0.0, value=5000.0, step=100.0)
            balance_after = st.number_input(f"Balance After ({c_sym})", min_value=-500.0, value=4850.0, step=100.0)

        st.caption(f"Amounts are entered in {currency} and converted to USD ({BASE_CURRENCY}) for model scoring.")
        submitted = st.form_submit_button("🔍 Score Transaction", type="primary", width="stretch")

    if submitted:
        single_df = pd.DataFrame([{
            "transaction_id": "MANUAL_001",
            "amount": amount,
            "tx_type": tx_type,
            "channel": channel,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "merchant_category": merchant_category,
            "is_international": int(is_international),
            "is_online": int(is_online),
            "balance_before": balance_before,
            "balance_after": balance_after,
        }])

        result = predict(df_to_usd(single_df, rate), artifacts, threshold=threshold,
                         currency=currency, symbol=c_sym, usd_per_unit=rate)
        score = result["fraud_score"].iloc[0]
        flagged = result["is_flagged"].iloc[0]
        risk = result["risk_level"].iloc[0]

        # Persist this decision to the automated audit trail
        action = "Flagged for review" if flagged else "No action (low risk cleared)"
        record_decision(
            transaction_id="MANUAL_001",
            risk_score=float(score),
            rules_triggered=str(result["rules_triggered"].iloc[0]),
            ai_reasoning=str(result["ai_reasoning"].iloc[0]),
            data_used={
                "source": "Manual entry",
                "channel": channel,
                "tx_type": tx_type,
                "currency": currency,
                "amount": round(float(amount), 2),
                "amount_in_usd": round(to_usd(float(amount), currency), 2),
            },
            action_taken=action,
        )
        st.caption("📋 This decision was recorded to the Audit Trail.")

        st.markdown("---")
        st.subheader(f"Scoring Result ({currency})")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Fraud Score", f"{score:.4f}")
        with r2:
            color_map = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "MINIMAL": "🔵"}
            st.metric("Risk Level", f"{color_map.get(risk, '⚪')} {risk}")
        with r3:
            status = "🚨 FLAGGED" if flagged else "✅ CLEAR"
            st.metric("Status", status, delta=f"{currency}")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Fraud Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1a1a2e"},
                "steps": [
                    {"range": [0, 20], "color": "#448aff"},
                    {"range": [20, 40], "color": "#00e676"},
                    {"range": [40, 60], "color": "#ffea00"},
                    {"range": [60, 80], "color": "#ff9100"},
                    {"range": [80, 100], "color": "#ff1744"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width="stretch")

        st.markdown("---")
        st.subheader("Feature Explanation")
        single_engineered = engineer_features(single_df)
        feature_vals = {
            "Amount": format_amount(amount, currency),
            "Hour": f"{hour_of_day}:00",
            f"Amount/Balance Ratio ({currency})": f"{amount / (balance_before + 1):.4f}",
            "Balance Change": format_amount(balance_before - balance_after, currency),
            "International": "Yes" if is_international else "No",
            "Online": "Yes" if is_online else "No",
            "Night Transaction": "Yes" if hour_of_day < 6 or hour_of_day > 22 else "No",
            "Weekend": "Yes" if day_of_week in ["Sat", "Sun"] else "No",
        }
        for k, v in feature_vals.items():
            st.text(f"  {k}: {v}")


# ===================== TAB 3: ANALYTICS =====================
with tab_analytics:
    st.subheader("Model Analytics & Performance")

    metrics = artifacts["metrics"]

    st.markdown("### Evaluation Metrics")
    display_metrics(metrics)

    st.markdown("---")
    st.subheader("Feature Importance (XGBoost)")
    feat_imp = get_feature_importance(artifacts)
    fig_imp = px.bar(
        feat_imp.head(15), x="importance", y="feature",
        orientation="h", title="Top 15 Feature Importances",
        color="importance", color_continuous_scale="Blues"
    )
    fig_imp.update_layout(height=450, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_imp, width="stretch")

    st.markdown("---")
    st.subheader("Test on Generated Data")
    n_gen = st.slider("Number of test transactions", 100, 5000, 1000, step=100)
    if st.button("Generate & Score Test Data", width="stretch"):
        test_df = generate_transactions(n_gen, seed=123)
        with st.spinner("Scoring..."):
            test_results = predict(test_df, artifacts, threshold=threshold,
                                   currency=currency, symbol=c_sym, usd_per_unit=rate)
            # Generated data is USD-denominated; localise for display
            test_results = df_to_local(test_results, rate)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total", f"{len(test_results):,}")
        t2.metric("Flagged", f"{test_results['is_flagged'].sum():,}")
        t3.metric("Avg Score", f"{test_results['fraud_score'].mean():.4f}")
        t4.metric("Actual Fraud", f"{test_df['is_fraud'].sum():,}")

        fig_scatter = px.scatter(
            test_results, x="amount", y="fraud_score",
            color="is_flagged", color_discrete_map={0: "#448aff", 1: "#ff1744"},
            title=f"Amount vs Fraud Score ({currency})",
            labels={"amount": f"Amount ({currency})"},
            opacity=0.6
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, width="stretch")


# ===================== TAB 4: AUDIT TRAIL =====================
with tab_audit:
    st.subheader("📋 Automated Audit Trail")
    st.markdown(
        "Every decision made by the fraud-detection system is recorded here — "
        "append-only, timestamped, and fully traceable."
    )

    audit_df = load_audit_log_dataframe()

    col_stats = st.columns(4)
    with col_stats[0]:
        st.metric("Total Decisions Logged", f"{len(audit_df):,}")
    with col_stats[1]:
        flagged_count = len(audit_df[audit_df["action_taken"].str.startswith("Flagged", na=False)]) if len(audit_df) else 0
        st.metric("Flagged Actions", f"{flagged_count:,}")
    with col_stats[2]:
        pending = len(audit_df[audit_df["investigator_decision"].eq("Pending review")]) if len(audit_df) else 0
        st.metric("Awaiting Review", f"{pending:,}")
    with col_stats[3]:
        critical = len(audit_df[audit_df["risk_score"].astype(float) >= 0.8]) if len(audit_df) else 0
        st.metric("Critical (score ≥ 0.8)", f"{critical:,}")

    st.markdown("---")

    if audit_df.empty:
        st.info("No decisions recorded yet. Run a prediction from the Upload or Manual Entry tabs.")
    else:
        st.markdown("### Full Audit Trail")
        st.dataframe(
            audit_df[AUDIT_COLUMNS],
            width="stretch", height=350, hide_index=True
        )

        st.download_button(
            "⬇️ Download Audit Trail (CSV)",
            data=audit_df[AUDIT_COLUMNS].to_csv(index=False),
            file_name="audit_trail.csv",
            mime="text/csv",
            width="stretch"
        )

        st.markdown("---")
        st.markdown("### Investigator Review")
        st.markdown("Update the human decision and final outcome for a logged transaction.")

        tx_options = [""] + sorted(audit_df["transaction_id"].unique().tolist())
        selected_tx = st.selectbox("Select Transaction ID", tx_options, key="audit_tx_select")

        if selected_tx:
            row = audit_df[audit_df["transaction_id"] == selected_tx].iloc[-1]
            st.markdown(f"""
            **Details for {selected_tx}**
            - Risk score: `{row['risk_score']}`
            - Risk level: {row['action_taken']}
            - AI reasoning: {row['ai_reasoning']}
            - Rules triggered: {row['rules_triggered']}
            """)

            with st.form("investigator_form"):
                c1, c2 = st.columns(2)
                with c1:
                    decision = st.selectbox(
                        "Investigator Decision",
                        ["Confirmed fraudulent", "Confirmed legitimate",
                         "Escalated to senior review", "Requires more data", "Pending review"],
                        index=4
                    )
                with c2:
                    outcome = st.selectbox(
                        "Final Outcome",
                        ["Awaiting investigator", "Fraud — account suspended",
                         "Fraud — refund issued", "False positive — cleared",
                         "Monitoring account"],
                        index=0
                    )
                submitted_review = st.form_submit_button("✔️ Update Decision", type="primary", width="stretch")

            if submitted_review:
                ok = update_investigator_decision(selected_tx, decision, outcome)
                if ok:
                    st.success(f"Decision updated for {selected_tx}.")
                    st.rerun()
                else:
                    st.error(f"Transaction {selected_tx} not found in audit log.")

    st.markdown("---")
    st.caption("Audit log stored in `audit_log.csv` in the project directory.")


# ===================== TAB 5: ABOUT =====================
with tab_about:
    st.subheader("About This Application")

    st.markdown("""
    ### Problem Statement
    Financial fraud costs institutions billions annually. This application provides
    an AI-powered screening tool that scores individual transactions for fraud
    probability, enabling analysts to prioritize investigations.

    ### How It Works
    1. **Data Input**: Upload a CSV or enter a transaction manually
    2. **Currency Normalisation**: Inputs are converted from the sidebar-selected
       currency to USD (the model's training currency) before scoring
    3. **Feature Engineering**: 12+ derived features from raw transaction attributes
    4. **Ensemble Scoring**: Three models vote — XGBoost (50%), Random Forest (30%), Isolation Forest (20%)
    5. **Risk Classification**: Score mapped to CRITICAL / HIGH / MEDIUM / LOW / MINIMAL
    6. **Audit Trail**: Every decision is logged with rules, reasoning, and data used

    ### Model Details
    | Component | Purpose |
    |-----------|---------|
    | XGBoost | Supervised classification with gradient boosting |
    | Random Forest | Ensemble bagging for robust predictions |
    | Isolation Forest | Unsupervised anomaly detection |
    | SMOTE | Handles class imbalance (fraud is rare) |

    ### Assumptions
    - Fraud rate is ~3% (industry average)
    - Fraudulent transactions tend to have higher amounts, occur at night, and are international/online
    - The model is trained on synthetic data — real-world performance may vary
    - Threshold of 0.5 balances precision and recall (adjustable)

    ### Data Sources
    - All data is **synthetically generated** — no real PII or financial data is used
    - Training data mirrors realistic transaction patterns (amounts, times, categories)

    ### Limitations
    - Synthetic data may not capture all real-world fraud patterns
    - Model requires periodic retraining on fresh data
    - Does not account for account-level behavioral baselines
    - No real-time streaming capability in this version

    ### Tech Stack
    - **Frontend**: Streamlit
    - **ML**: XGBoost, scikit-learn, imbalanced-learn
    - **Visualization**: Plotly
    - **Data**: Pandas, NumPy
    """)

    st.markdown("---")
    st.caption("HBF2212 — Artificial Intelligence in Finance | Project 1")

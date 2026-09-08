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
from collections import Counter

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


def explain(title: str, body: str):
    """Collapsible 'how to read this' explanation for charts & tables.

    Keeps the UI clean while giving new analysts a plain-English guide.
    """
    with st.expander(f"💡 How to read this: {title}"):
        st.markdown(body)


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

        explain(
            "the results table",
            """
            Each row is one transaction with the model's verdict added:

            - **fraud_score** — probability (0–1) that this transaction is fraud. Rows with
              a score ≥ 0.6 are shaded red so high-risk cases jump out.
            - **is_flagged** — 1 if the score is at/above the sidebar threshold, else 0.
            - **risk_level** — a human label graded from the score: CRITICAL (≥ 0.8),
              HIGH (≥ 0.6), MEDIUM (≥ 0.4), LOW (≥ 0.2), MINIMAL (< 0.2).
            - **currency** — the working currency; every amount column is shown in it.
            - **rules_triggered / ai_reasoning** — which business rules fired and a
              plain-English explanation of *why* the model scored the way it did.

            **What to do:** sort by `fraud_score` (descending), open the flagged rows'
            reasoning, and start your investigation from the top. The `Flagged Only`
            radio button hides the noise so you focus purely on alerts.
            """
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
        explain(
            "these four charts",
            """
            All four charts are drawn from **your uploaded data** and give a 10-second
            health check of the batch you just scored:

            - **Fraud Score Distribution** — how many transactions sit at each score.
              A good system shows most mass on the left (safe) and a small red tail on
              the right (risky). A wide red tail = your threshold is too generous, or
              the batch genuinely contains many risky cases.
            - **Risk Level Distribution** — the same scores sliced into buckets.
              The red/orange slice is your *to-do list*; the blue/green is already cleared.
            - **Transaction Amount by Flag Status** — box plots. If the flagged box sits
              *above* the cleared box, high amounts are driving the alerts (typical).
            - **Transactions by Hour** — red bars clustered around late night / early
              morning are the classic fraud pattern and worth reporting.

            **What to do:** read them left-to-right. Distribution → risk mix → amount →
            time. Then drill into any pattern with the table above.
            """
        )

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

        explain(
            "the fraud score gauge",
            """
            The needle shows this single transaction's fraud score (0–100). The coloured
            zones are the risk bands the system uses internally:

            - **Blue** (0–20) → MINIMAL, **Green** (20–40) → LOW,
            - **Yellow** (40–60) → MEDIUM, **Orange** (60–80) → HIGH,
            - **Red** (80–100) → CRITICAL.

            The red tick mark is your **sidebar threshold**. Anything scoring at/above it
            is flagged for review, even if it sits in a "medium" colour band. Moving the
            threshold left makes the system more sensitive; right makes it more selective.

            **What to do:** for any score ≥ 60, open the Feature Explanation below to see
            *which* inputs pushed the score up before making a decision.
            """
        )

        st.markdown("---")
        st.subheader("Feature Explanation")
        explain(
            "the features behind the score",
            """
            This table shows the raw inputs the model considered when scoring. They map
            to the model's engineered features — the same ones used to train it:

            - **Amount/Balance Ratio** — how much of the available balance this transaction
              consumes. A ratio ≥ 1.5 triggers the *"Amount exceeds 150% of available
              balance"* rule.
            - **Balance Change** — how much the balance dropped. Negative balances trigger
              the *"transactions drove balance negative"* rule.
            - **International / Online** — both together fire the *"online + international
              (typical fraud vector)"* rule.
            - **Night / Weekend** — unusual-hour flags; late-night online purchases are a
              classic fraud signal.

            **What to do:** compare the "No / Yes" flags with the `rules_triggered` text in
            the audit trail. If the system flagged a case you disagree with, the feature
            table tells you exactly which signal it reacted to — so you can learn the
            system's behaviour and challenge it with evidence.
            """,
        )
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
    cm = np.array(metrics["confusion_matrix"])

    st.markdown("### Evaluation Metrics")
    display_metrics(metrics)

    explain(
        "the evaluation metrics & confusion matrix",
        """
        These numbers describe *how well the model performed on data it had never
        seen* (a held-out test set) during training. New analysts should know:

        - **ROC-AUC** — the probability a random fraud gets a higher score than a random
          legitimate transaction. 1.0 = perfect, 0.5 = pure guessing. Higher is better.
        - **Average Precision** — precision averaged over all thresholds; concentrates on
          *how precise the model is on fraud*, not just on rank order.
        - **False Positives (FP)** — legitimate transactions wrongly flagged. Each one
          costs an analyst's time but is recoverable.
        - **False Negatives (FN)** — fraud that slipped through unflagged. This is the
          expensive error a fraud team exists to avoid.

        The **Confusion Matrix** shows the four outcomes as a grid: rows = actual
        (Legitimate/Fraud), columns = predicted. Top-left = correctly cleared (TN),
        top-right = false alarm (FP), bottom-right = correctly caught (TP),
        bottom-left = missed fraud (FN).

        **What to do:** check that **FN is close to zero** — that matters more than a
        low FP. If FN grows after retraining, dial the threshold down (sidebar) and add
        the missed cases to your training data.
        """
    )

    st.markdown("### Per-Class Classification Report")
    report = metrics["classification_report"]
    rep_rows = []
    for cls in ["Legitimate", "Fraud"]:
        r = report.get(cls, {})
        rep_rows.append({
            "Class": cls,
            "Precision": round(r.get("precision", 0), 4),
            "Recall": round(r.get("recall", 0), 4),
            "F1-Score": round(r.get("f1-score", 0), 4),
            "Support": int(r.get("support", 0)),
        })
    rep_df = pd.DataFrame(rep_rows)
    st.dataframe(rep_df, width="stretch", hide_index=True)

    explain(
        "the per-class report",
        """
        One row per outcome class — how the model behaves *on each*:

        - **Precision** — of everything the model called fraud, how many really were.
          High = few wasted reviews. Low = many false alarms.
        - **Recall** — of all real fraud, how much the model caught. High = few missed
          frauds. Low = dangerous blind spots.
        - **F1-Score** — a single balance of the two (harmonic mean). Compare F1 across
          classes: a large gap means the model treats the two classes very differently.
        - **Support** — number of test transactions in that class; put the percentages
          in context (a recall of 100% on 5 frauds is less impressive than on 500).

        **The trade-off:** you cannot usually maximise precision and recall together.
        Raising the sidebar threshold raises precision but lowers recall (missed fraud);
        lowering it does the reverse. This project deliberately weights **fraud recall**
        higher, because a missed fraud is costlier than a reviewable alert.

        **What to do:** aim for fraud recall ≥ 0.8 and check that precision stays above
        roughly 0.5 so analysts aren't overwhelmed with false alarms.
        """
    )

    tn, fp, fn, tp = cm.ravel()
    fraud_prec = report.get("Fraud", {}).get("precision", 0)
    fraud_rec = report.get("Fraud", {}).get("recall", 0)
    st.info(
        f"The ensemble separates fraud from legitimate transactions with a "
        f"**ROC-AUC of {metrics['roc_auc']:.3f}** and an average precision of "
        f"**{metrics['average_precision']:.3f}**. Across {int(metrics['test_size']):,} "
        f"held-out transactions, the model caught **{tp} of {int(metrics['fraud_count_test']):,}** "
        f"actual frauds (recall {fraud_rec:.0%}) while raising **{fp:,}** false alarms "
        f"(precision {fraud_prec:.0%}). Fraud recall is deliberately prioritised — a single "
        f"missed fraud costs far more than a reviewable false positive."
    )

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

    top_feat = str(feat_imp.iloc[0]["feature"])
    top_contrib = ", ".join(str(f) for f in feat_imp["feature"].head(3))
    st.caption(
        f"The strongest predictor is **{top_feat}**, followed by **{top_contrib}**. "
        f"These are the signals the model leans on most when deciding risk — "
        f"review flagged transactions with unusual values on these features first."
    )
    explain(
        "feature importance",
        """
        Feature importance tells you **what the model is actually looking at**. The bars
        show how often / how strongly each engineered feature was used to split the data
        in the XGBoost trees. Longer bar = bigger influence on the score.

        - Amount-related and balance-ratio features dominating is expected and healthy.
        - If a feature dwarfs everything else, the model may be over-reliant on one
          signal — investigate before trusting it in production.
        - Features near zero barely matter: changing them will barely move scores.

        **What to do:** use this chart to *sanity-check* decisions and to explain the
        system to stakeholders — "the model flagged this because high-value + unusual
        balance ratios drive its decisions, matching the chart above."
        """
    )

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
        explain(
            "the batch summary numbers",
            """
            Quick read on the batch you just generated:

            - **Total / Flagged** — how many transactions and how many were alerted.
            - **Avg Score** — the mean fraud probability across the whole batch; a low
              average with a handful of very high scores is the normal fraud pattern.
            - **Actual Fraud** — the *ground truth* injected into this synthetic batch.
              Compare it to **Flagged**: flagged ≈ actual means the threshold is
              well-calibrated (not under- or over-alerting).

            **What to do:** if Flagged is far above Actual Fraud, raise the sidebar
            threshold; if it is far below, lower it — then re-run to see the effect
            immediately.
            """
        )

        scatter_c1, scatter_c2 = st.columns(2)
        with scatter_c1:
            fig_scatter = px.scatter(
                test_results, x="amount", y="fraud_score",
                color="is_flagged", color_discrete_map={0: "#448aff", 1: "#ff1744"},
                title=f"Amount vs Fraud Score ({currency})",
                labels={"amount": f"Amount ({currency})"},
                opacity=0.6
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, width="stretch")
        with scatter_c2:
            fig_hist = px.histogram(
                test_results, x="fraud_score", nbins=40,
                color="is_flagged", color_discrete_map={0: "#448aff", 1: "#ff1744"},
                title="Distribution of Fraud Scores",
                labels={"fraud_score": "Fraud Score", "count": "Transactions"}
            )
            fig_hist.update_layout(height=400, bargap=0.05)
            st.plotly_chart(fig_hist, width="stretch")
        explain(
            "the scatter & score-distribution charts",
            """
            The **scatter plot** maps amount (x) against fraud score (y); red dots are
            flagged. If red dots rise as amounts rise, the model is saying *"big money moves
            more risk"* — normal. Watch for a cluster of high-score red dots at *low*
            amounts: that suggests rules like night-time or balance-ratio are what fired,
            not size.

            The **distribution chart** is the "how much is in each zone" view: blue mass on
            the left is safely scored, red on the right is flagged. A clean right-hand tail
            (rather than a blob near the threshold) means the model is confident, not
            borderline — borderline-heavy outputs usually need a threshold review.

            **What to do:** hover over individual dots in the scatter to inspect
            transaction details; use the distribution to judge whether the threshold is
            separating the two groups cleanly.
            """
        )

        st.markdown("### Where Does Fraud Concentrate?")
        flagged = test_results[test_results["is_flagged"] == 1]
        conc_c1, conc_c2 = st.columns(2)
        with conc_c1:
            top_cats = flagged["merchant_category"].value_counts().head(8).reset_index()
            top_cats.columns = ["Merchant Category", "Flagged Count"]
            fig_cat = px.bar(
                top_cats, x="Flagged Count", y="Merchant Category",
                orientation="h", title="Top Merchant Categories Among Flagged",
                color="Flagged Count", color_continuous_scale="Reds"
            )
            fig_cat.update_layout(height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_cat, width="stretch")
        with conc_c2:
            chan_rate = (test_results.groupby("channel")["is_flagged"]
                         .mean().sort_values(ascending=False).reset_index())
            chan_rate.columns = ["Channel", "Flag Rate"]
            fig_chan = px.bar(
                chan_rate, x="Channel", y="Flag Rate",
                title="Flag Rate by Channel", color="Flag Rate",
                color_continuous_scale="Reds", text_auto=".0%"
            )
            fig_chan.update_layout(height=380, showlegend=False)
            st.plotly_chart(fig_chan, width="stretch")
        explain(
            "where fraud concentrates",
            """
            Two lenses on *where* the fraud is hiding:

            - **Top Merchant Categories Among Flagged** — the categories where flagged
              cases pile up. Electronics, jewellery and travel topping the list is the
              expected high-ticket pattern; an unexpected category (e.g. "grocery")
              dominating isn't inherently wrong, just worth explaining.
            - **Flag Rate by Channel** — the *proportion* of each channel that got flagged,
              not raw counts. A small channel with a huge flag rate (e.g. Online 25%) is
              a red flag worth escalating; a big channel with a low rate is healthy.

            **The difference between the two charts matters:** the left counts *how many*,
            the right measures *how risky each channel is*. Analysts often over-react to raw
            counts — always check the rate chart too.
            """
        )

        rule_counter = Counter()
        for rules in test_results["rules_triggered"].astype(str):
            if rules and rules != "No high-risk rules triggered":
                for r in rules.split("; "):
                    rule_counter[r] += 1
        if rule_counter:
            st.markdown("### Which Business Rules Fire Most?")
            rules_df = pd.DataFrame(
                rule_counter.most_common(8), columns=["Rule", "Occurrences"]
            )
            fig_rules = px.bar(
                rules_df, x="Occurrences", y="Rule", orientation="h",
                title="Most Frequently Triggered Rules",
                color="Occurrences", color_continuous_scale="Blues"
            )
            fig_rules.update_layout(height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_rules, width="stretch")
            explain(
                "the rules-fired chart",
                """
                Every decision in this system is rule-transparent: the model also reports
                *which business rules fired* (these appear in the `rules_triggered` column
                and the audit trail). This chart counts how often each rule fired across the
                batch.

                - Use it to see **which warnings drive your flag volume** — the top rule is
                  effectively your team's most common trigger and the first to tune.
                - A rule firing on a huge share of *cleared* transactions is a candidate for
                  weakening; one that never fires may be dead weight or simply rare.
                - Rules explain *part* of a decision — scores also come from the ML models —
                  so pair this chart with the correlation chart below.

                **What to do:** share the top-3 rules in your investigation notes. It gives
                reviewers a shared vocabulary for why things got flagged.
                """
            )

        st.markdown("### Feature Correlation with Fraud Score")
        numeric_feats = [
            c for c in test_results.columns
            if c not in ("transaction_id", "rules_triggered", "ai_reasoning",
                         "risk_level", "is_fraud")
            and pd.api.types.is_numeric_dtype(test_results[c])
        ]
        corr_series = (
            test_results[numeric_feats].corr()["fraud_score"]
            .drop("fraud_score").dropna()
        )
        corr_df = (
            corr_series.to_frame("Correlation").reset_index()
            .rename(columns={"index": "Feature"})
        )
        corr_df["Abs"] = corr_df["Correlation"].abs()
        corr_df = corr_df.sort_values("Abs", ascending=False).head(10).drop(columns="Abs")
        fig_corr = px.bar(
            corr_df, x="Correlation", y="Feature", orientation="h",
            title="Top Correlations with Fraud Score",
            color="Correlation", color_continuous_scale="RdBu",
            range_color=[-1, 1]
        )
        fig_corr.update_layout(height=400, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_corr, width="stretch")
        explain(
            "the correlation chart",
            """
            Correlation measures how strongly each feature *moves together* with the fraud
            score (-1 → +1). It is **not** proof of causation, but it points at which inputs
            are most responsible for high scores.

            - **Positive values** (red, right of centre) — as the feature increases, fraud
              score rises. E.g. `amount` positively correlated = bigger transactions score
              riskier.
            - **Negative values** (blue, left of centre) — as the feature increases, score
              falls. E.g. `balance_after` negatively correlated = lower remaining balances
              score riskier.
            - Values near zero mean the feature plays almost no role for this batch.

            **What to do:** compare this chart with Feature Importance above. If a feature
            has high importance but near-zero correlation (or vice-versa), that's a
            mismatch worth investigating — the model may be combining signals in ways the
            simple correlation can't see.
            """
        )

        flagged_rate = len(flagged) / len(test_results)
        true_rate = test_df["is_fraud"].mean()
        avg_flag = flagged["fraud_score"].mean() if len(flagged) else 0
        avg_clear = test_results.loc[test_results["is_flagged"] == 0, "fraud_score"].mean()
        top_rule = rules_df.iloc[0]["Rule"] if rule_counter else "n/a"
        top_rule_pct = (
            rules_df.iloc[0]["Occurrences"] / len(test_results) * 100
            if rule_counter else 0
        )
        strong_feat = corr_df.iloc[0]["Feature"]
        strong_val = corr_df.iloc[0]["Correlation"]
        st.markdown("---")
        st.markdown("#### 📖 How to Read These Results")
        st.markdown(
            f"Of **{len(test_results):,}** generated transactions, the model flagged "
            f"**{len(flagged):,} ({flagged_rate:.1%})** at the current threshold of "
            f"**{threshold:.2f}** — close to the **{true_rate:.1%}** fraud rate injected "
            f"into the synthetic data, indicating the threshold is well-calibrated. "
            f"Flagged transactions carry an average fraud score of **{avg_flag:.3f}** versus "
            f"**{avg_clear:.3f}** for cleared ones, a clear separation between the two groups. "
            f"The single most common warning was **\"{top_rule}\"**, fired on "
            f"**{top_rule_pct:.1f}%** of all transactions, so it is rule #1 to investigate "
            f"first. The feature most strongly linked to fraud scores is **{strong_feat}** "
            f"(correlation **{strong_val:+.2f}**{', so unusually high values here deserve special scrutiny' if strong_val > 0 else ', so unusually low values here deserve special scrutiny'}). "
            f"Use the Upload tab to score your own data and compare against these baselines."
        )


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
        explain(
            "the audit trail table",
            """
            One immutable row per decision the system has made — the paper trail for every
            alert. Reading the columns:

            - **transaction_id** — which transaction it refers to (matches your CSV or
              `MANUAL_001` for manual entries).
            - **date_time** — when the decision was made (UTC).
            - **risk_score** — the model's fraud probability, 0–1.
            - **rules_triggered** — business rules that fired for this transaction.
            - **ai_reasoning** — plain-English explanation of *why* the model scored it.
            - **data_used** — JSON: input source, channel, type, currency, amount and its
              USD equivalent (so you can audit which inputs produced the score).
            - **action_taken** — what the system did ("Flagged for review" / "No action...").
            - **investigator_decision / final_outcome** — human fields you fill in below to
              close the loop (they default to "Pending review").

            **Why this matters:** `audit_log.csv` is append-only — rows are never rewritten or
            silently deleted — which is what makes this demonstrable for compliance and
            regulatory-grade reporting.
            """
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
        explain(
            "the investigator review",
            """
            This is where a human closes the loop on a machine-generated alert — required in
            this system because the model is a *screener*, not an authority.

            **Investigator Decision** — what you conclude after reviewing the case:
            - *Confirmed fraudulent* / *Confirmed legitimate* — the machine got it right.
            - *Escalated to senior review* — you need a second opinion or a policy call.
            - *Requires more data* — insufficient information to judge (state what's missing).

            **Final Outcome** — the business result of the case (account suspended, refund
            issued, false positive cleared, monitoring). This becomes the record your team's
            performance metrics are judged on.

            **What to do:** work top-down through the audit table: open one transaction,
            read its `ai_reasoning` + `rules_triggered`, form your verdict, and submit. Every
            update is appended to `audit_log.csv` with the transaction's risk score — giving
            you a clean dataset of *model verdicts vs human outcomes* you can later use to
            retrain and improve the model.
            """,
        )

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
    st.subheader("🎓 Quick Guide for New Analysts")
    st.markdown("""
    **What this system does, in one sentence:** it reads a batch of transactions, scores
    each one's probability of being fraud (0 = safe, 1 = certain fraud), flags the risky
    ones, and logs every decision with a reason so a human can review and close the loop.

    **The 4-step workflow**
    1. **Load data** (Upload tab) or **create one case** (Manual Entry tab) — amounts can
       be in any of the 16 sidebar currencies; they are normalised to USD for scoring and
       shown back to you in your chosen currency.
    2. **Model scores** — an ensemble of three models votes (XGBoost 50% + Random Forest
       30% + Isolation Forest 20%); the result is one score per transaction.
    3. **Threshold applies** — the sidebar threshold (default 0.5) decides flagged vs
       cleared. Lower = more sensitive, higher = more selective.
    4. **Human closes the loop** — review flagged cases in the Audit Trail tab and record
       your verdict. The audit log even tracks *model answer vs human answer* for later
       retraining.

    **Plain-English definitions**
    - **Fraud score** — model probability (0–1) the transaction is fraud.
    - **Encoded / engineered feature** — a raw input (e.g. amount, hour) transformed into
      something the model can learn from (e.g. *amount / balance ratio*, *is_night*).
    - **Ensemble** — several models combined; each covers the others' blind spots.
    - **SMOTE** — a technique that creates extra synthetic fraud examples during training
      because real fraud is rare (~3%), which would otherwise bias the model toward
      "everything is legitimate".
    - **ROC-AUC** — how well the model *ranks* fraud above legitimate (1.0 = perfect).
    - **Precision** — of what the model flags, how much is really fraud (false-alarm rate).
    - **Recall** — of the actual fraud, how much the model catches (missed-fraud rate).
    - **Confusion matrix** — the 2×2 grid of caught/missed vs correct/false for each class.
    - **Threshold** — the cut-off score for flagging; change it in the sidebar and re-run.
    - **Audit trail** — an append-only log of every decision the system has made, with the
      reasoning, so results are reproducible and auditable.
    - **Currency normalisation** — converting inputs from your chosen currency to USD (the
      model's training currency) before scoring, then converting back for display.

    **How to form a verdict on a flagged case**
    1. Read the row's `ai_reasoning` and `rules_triggered` in the audit trail.
    2. Check the Manual-Entry Feature Explanation for what each input contributed.
    3. Compare with the Analytics charts (concentration, rules, correlation) to see if the
      pattern is normal for your book of business.
    4. Record your decision + outcome. If you disagree with the model, keep the case in
      your training set — your human verdicts are how the model improves.
    """)

    st.markdown("---")
    st.caption("HBF2212 — Artificial Intelligence in Finance | Project 1")

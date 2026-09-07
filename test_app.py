"""
Test Suite for Fraud Transaction Detection Application
=====================================================
Covers normal, edge-case, and boundary-condition testing.
Run with: pytest test_app.py -v
"""
import sys
import os
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from generate_data import generate_transactions
from audit_trail import (
    record_decision, record_batch, update_investigator_decision,
    load_audit_log, load_audit_log_dataframe, reset_audit_log, AUDIT_COLUMNS
)
from model import (
    train_model, predict, engineer_features,
    prepare_data, save_model, load_model, get_feature_importance,
    validate_csv
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trained_artifacts():
    """Train model once for all tests in this module."""
    df = generate_transactions(2000, seed=42)
    return train_model(df, test_size=0.2, random_state=42)


@pytest.fixture
def sample_single_transaction():
    return pd.DataFrame([{
        "transaction_id": "TEST_001",
        "amount": 150.00,
        "tx_type": "purchase",
        "channel": "POS",
        "hour_of_day": 14,
        "day_of_week": "Wed",
        "merchant_category": "grocery",
        "is_international": 0,
        "is_online": 0,
        "balance_before": 5000.00,
        "balance_after": 4850.00,
    }])


@pytest.fixture
def sample_batch():
    return generate_transactions(100, seed=99).drop(columns=["is_fraud"])


# ---------------------------------------------------------------------------
# Test: Data Generation
# ---------------------------------------------------------------------------
class TestDataGeneration:
    def test_generates_correct_columns(self):
        df = generate_transactions(100)
        expected_cols = {
            "transaction_id", "amount", "tx_type", "channel",
            "hour_of_day", "day_of_week", "merchant_category",
            "is_international", "is_online", "balance_before",
            "balance_after", "is_fraud"
        }
        assert expected_cols == set(df.columns)

    def test_generates_correct_count(self):
        df = generate_transactions(500)
        assert len(df) == 500

    def test_fraud_ratio_approximate(self):
        df = generate_transactions(10000, fraud_ratio=0.03)
        actual_ratio = df["is_fraud"].mean()
        assert 0.01 < actual_ratio < 0.06, f"Fraud ratio {actual_ratio} out of expected range"

    def test_amounts_are_positive(self):
        df = generate_transactions(500)
        assert (df["amount"] > 0).all()

    def test_reproducibility_with_seed(self):
        df1 = generate_transactions(100, seed=123)
        df2 = generate_transactions(100, seed=123)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_transactions(100, seed=1)
        df2 = generate_transactions(100, seed=2)
        assert not df1.equals(df2)

    def test_balance_after_less_than_before_for_purchases(self):
        df = generate_transactions(1000)
        purchases = df[df["tx_type"] == "purchase"]
        # Most purchases should reduce balance
        reduced = (purchases["balance_after"] < purchases["balance_before"]).mean()
        assert reduced > 0.8


# ---------------------------------------------------------------------------
# Test: Feature Engineering
# ---------------------------------------------------------------------------
class TestFeatureEngineering:
    def test_adds_expected_columns(self):
        df = generate_transactions(100)
        result = engineer_features(df)
        new_cols = [
            "amount_to_balance_ratio", "balance_change",
            "balance_change_ratio", "is_night", "is_weekend",
            "log_amount", "amount_digit_length", "is_round_amount"
        ]
        for col in new_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_is_night_flag(self):
        df = pd.DataFrame({"hour_of_day": [0, 3, 8, 14, 23], "amount": [10]*5,
                           "balance_before": [1000]*5, "balance_after": [990]*5,
                           "day_of_week": ["Mon"]*5})
        result = engineer_features(df)
        assert result["is_night"].tolist() == [1, 1, 0, 0, 1]

    def test_log_amount_positive(self):
        df = generate_transactions(50)
        result = engineer_features(df)
        assert (result["log_amount"] >= 0).all()


# ---------------------------------------------------------------------------
# Test: Model Training
# ---------------------------------------------------------------------------
class TestModelTraining:
    def test_returns_expected_keys(self, trained_artifacts):
        expected_keys = {"xgb_model", "rf_model", "iso_model", "scaler", "encoders", "metrics"}
        assert expected_keys == set(trained_artifacts.keys())

    def test_metrics_are_valid(self, trained_artifacts):
        m = trained_artifacts["metrics"]
        assert 0 <= m["roc_auc"] <= 1
        assert 0 <= m["average_precision"] <= 1
        assert m["confusion_matrix"][0][0] >= 0

    def test_model_has_positive_auc(self, trained_artifacts):
        assert trained_artifacts["metrics"]["roc_auc"] > 0.5

    def test_save_and_load(self, trained_artifacts, tmp_path):
        path = str(tmp_path / "test_model.joblib")
        save_model(trained_artifacts, path)
        loaded = load_model(path)
        assert set(loaded.keys()) == set(trained_artifacts.keys())


# ---------------------------------------------------------------------------
# Test: Prediction - Normal Cases
# ---------------------------------------------------------------------------
class TestPredictionNormal:
    def test_single_transaction_returns_score(self, trained_artifacts, sample_single_transaction):
        result = predict(sample_single_transaction, trained_artifacts)
        assert "fraud_score" in result.columns
        assert "is_flagged" in result.columns
        assert "risk_level" in result.columns
        assert 0 <= result["fraud_score"].iloc[0] <= 1

    def test_batch_prediction(self, trained_artifacts, sample_batch):
        result = predict(sample_batch, trained_artifacts)
        assert len(result) == len(sample_batch)
        assert result["fraud_score"].between(0, 1).all()

    def test_legitimate_transaction_low_score(self, trained_artifacts):
        """A typical grocery purchase should have a low fraud score."""
        tx = pd.DataFrame([{
            "transaction_id": "LEGIT_001",
            "amount": 25.50,
            "tx_type": "purchase",
            "channel": "POS",
            "hour_of_day": 12,
            "day_of_week": "Tue",
            "merchant_category": "grocery",
            "is_international": 0,
            "is_online": 0,
            "balance_before": 3000.00,
            "balance_after": 2974.50,
        }])
        result = predict(tx, trained_artifacts, threshold=0.5)
        assert result["fraud_score"].iloc[0] < 0.5, "Legitimate tx scored too high"

    def test_suspicious_transaction_high_score(self, trained_artifacts):
        """A high-amount international online transaction at 3 AM should score high."""
        tx = pd.DataFrame([{
            "transaction_id": "SUSP_001",
            "amount": 9500.00,
            "tx_type": "transfer",
            "channel": "Online",
            "hour_of_day": 3,
            "day_of_week": "Sun",
            "merchant_category": "jewelry",
            "is_international": 1,
            "is_online": 1,
            "balance_before": 500.00,
            "balance_after": -9000.00,
        }])
        result = predict(tx, trained_artifacts, threshold=0.5)
        assert result["fraud_score"].iloc[0] > 0.3, "Suspicious tx scored too low"


# ---------------------------------------------------------------------------
# Test: Prediction - Edge Cases
# ---------------------------------------------------------------------------
class TestPredictionEdgeCases:
    def test_zero_amount(self, trained_artifacts):
        """Zero amount transaction."""
        tx = pd.DataFrame([{
            "transaction_id": "EDGE_001",
            "amount": 0.01,
            "tx_type": "purchase",
            "channel": "POS",
            "hour_of_day": 12,
            "day_of_week": "Mon",
            "merchant_category": "grocery",
            "is_international": 0,
            "is_online": 0,
            "balance_before": 100.00,
            "balance_after": 99.99,
        }])
        result = predict(tx, trained_artifacts)
        assert len(result) == 1
        assert 0 <= result["fraud_score"].iloc[0] <= 1

    def test_very_large_amount(self, trained_artifacts):
        tx = pd.DataFrame([{
            "transaction_id": "EDGE_002",
            "amount": 999999.99,
            "tx_type": "transfer",
            "channel": "Online",
            "hour_of_day": 0,
            "day_of_week": "Sat",
            "merchant_category": "transfer",
            "is_international": 1,
            "is_online": 1,
            "balance_before": 1000000.00,
            "balance_after": 1.00,
        }])
        result = predict(tx, trained_artifacts)
        assert len(result) == 1

    def test_midnight_hour(self, trained_artifacts):
        tx = pd.DataFrame([{
            "transaction_id": "EDGE_003",
            "amount": 100.00,
            "tx_type": "withdrawal",
            "channel": "ATM",
            "hour_of_day": 0,
            "day_of_week": "Sat",
            "merchant_category": "atm_withdrawal",
            "is_international": 0,
            "is_online": 0,
            "balance_before": 500.00,
            "balance_after": 400.00,
        }])
        result = predict(tx, trained_artifacts)
        assert 0 <= result["fraud_score"].iloc[0] <= 1

    def test_end_of_day_hour(self, trained_artifacts):
        tx = pd.DataFrame([{
            "transaction_id": "EDGE_004",
            "amount": 100.00,
            "tx_type": "purchase",
            "channel": "Online",
            "hour_of_day": 23,
            "day_of_week": "Fri",
            "merchant_category": "subscription",
            "is_international": 0,
            "is_online": 1,
            "balance_before": 200.00,
            "balance_after": 100.00,
        }])
        result = predict(tx, trained_artifacts)
        assert 0 <= result["fraud_score"].iloc[0] <= 1

    def test_all_amounts_same(self, trained_artifacts):
        """Batch where every transaction has the same amount."""
        base = {
            "tx_type": "purchase", "channel": "POS", "hour_of_day": 12,
            "day_of_week": "Wed", "merchant_category": "grocery",
            "is_international": 0, "is_online": 0,
            "balance_before": 1000.0, "balance_after": 900.0
        }
        rows = [{**base, "transaction_id": f"BATCH_{i}", "amount": 100.0} for i in range(10)]
        df = pd.DataFrame(rows)
        result = predict(df, trained_artifacts)
        assert len(result) == 10
        assert result["fraud_score"].between(0, 1).all()

    def test_threshold_affects_flagging(self, trained_artifacts, sample_batch):
        """Lower threshold should flag more transactions."""
        r_high = predict(sample_batch, trained_artifacts, threshold=0.9)
        r_low = predict(sample_batch, trained_artifacts, threshold=0.1)
        assert r_low["is_flagged"].sum() >= r_high["is_flagged"].sum()

    def test_negative_balance(self, trained_artifacts):
        tx = pd.DataFrame([{
            "transaction_id": "EDGE_005",
            "amount": 500.00,
            "tx_type": "withdrawal",
            "channel": "ATM",
            "hour_of_day": 2,
            "day_of_week": "Sun",
            "merchant_category": "atm_withdrawal",
            "is_international": 0,
            "is_online": 0,
            "balance_before": 100.00,
            "balance_after": -400.00,
        }])
        result = predict(tx, trained_artifacts)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: Feature Importance
# ---------------------------------------------------------------------------
class TestFeatureImportance:
    def test_returns_dataframe(self, trained_artifacts):
        fi = get_feature_importance(trained_artifacts)
        assert isinstance(fi, pd.DataFrame)
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) > 0

    def test_importances_non_negative(self, trained_artifacts):
        fi = get_feature_importance(trained_artifacts)
        assert (fi["importance"] >= 0).all()

    def test_importances_sum_positive(self, trained_artifacts):
        fi = get_feature_importance(trained_artifacts)
        assert fi["importance"].sum() > 0


# ---------------------------------------------------------------------------
# Test: CSV Validation
# ---------------------------------------------------------------------------
class TestCSVValidation:
    def test_valid_csv_returns_empty(self):
        df = pd.DataFrame({
            "amount": [100], "tx_type": ["purchase"], "channel": ["POS"],
            "hour_of_day": [12], "day_of_week": ["Mon"],
            "merchant_category": ["grocery"], "is_international": [0],
            "is_online": [0], "balance_before": [1000],
            "balance_after": [900]
        })
        missing = validate_csv(df)
        assert len(missing) == 0

    def test_missing_columns_detected(self):
        df = pd.DataFrame({"amount": [100], "tx_type": ["purchase"]})
        missing = validate_csv(df)
        assert len(missing) > 0
        assert "channel" in missing


# ---------------------------------------------------------------------------
# Test: Automated Audit Trail
# ---------------------------------------------------------------------------
class TestAuditTrail:

    def _write_defaults(self):
        # ensure the log starts clean for isolated tests
        reset_audit_log()

    def test_records_required_columns(self):
        reset_audit_log()
        rec = record_decision(
            transaction_id="AUD_001", risk_score=0.87,
            rules_triggered="High-value transaction (> $5,000)",
            ai_reasoning="flagged due to large amount",
            data_used={"source": "test", "amount": 9000},
            action_taken="Flagged for review",
        )
        for col in AUDIT_COLUMNS:
            assert col in rec, f"missing column {col}"
        assert rec["transaction_id"] == "AUD_001"
        assert rec["investigator_decision"] == "Pending review"
        assert rec["final_outcome"] == "Awaiting investigator"

    def test_appends_new_records_not_overwrites(self):
        reset_audit_log()
        record_decision(transaction_id="R1", risk_score=0.1,
                        rules_triggered="", ai_reasoning="",
                        data_used={}, action_taken="No action")
        record_decision(transaction_id="R2", risk_score=0.9,
                        rules_triggered="", ai_reasoning="",
                        data_used={}, action_taken="Flagged for review")
        log = load_audit_log()
        assert len(log) == 2

    def test_record_batch_creates_one_row_per_transaction(self):
        reset_audit_log()
        df = pd.DataFrame([
            {"transaction_id": "T1", "amount": 100, "channel": "POS",
             "tx_type": "purchase", "fraud_score": 0.1, "is_flagged": 0,
             "rules_triggered": "none", "ai_reasoning": "low risk"},
            {"transaction_id": "T2", "amount": 9000, "channel": "Online",
             "tx_type": "transfer", "fraud_score": 0.92, "is_flagged": 1,
             "rules_triggered": "High-value", "ai_reasoning": "large amount"},
        ])
        count = record_batch(df, data_source="CSV upload")
        assert count == 2
        log = load_audit_log()
        assert len(log) == 2
        assert log[1]["transaction_id"] == "T2"
        assert log[1]["action_taken"].startswith("Flagged")
        assert log[0]["action_taken"].startswith("No action")

    def test_update_investigator_decision(self):
        reset_audit_log()
        record_decision(transaction_id="AUD_003", risk_score=0.85,
                        rules_triggered="x", ai_reasoning="y",
                        data_used={}, action_taken="Flagged for review")
        ok = update_investigator_decision(
            "AUD_003", "Confirmed fraudulent", "Fraud — account suspended")
        assert ok is True
        log = load_audit_log()
        assert log[0]["investigator_decision"] == "Confirmed fraudulent"
        assert log[0]["final_outcome"] == "Fraud — account suspended"

    def test_update_missing_transaction_returns_false(self):
        reset_audit_log()
        ok = update_investigator_decision("NOPE", "Confirmed legitimate")
        assert ok is False

    def test_predict_output_feeds_audit_trail(self, trained_artifacts):
        """predict() must expose rules_triggered and ai_reasoning for auditing."""
        tx = pd.DataFrame([{
            "transaction_id": "AUD_TX_1",
            "amount": 9999.99, "tx_type": "transfer", "channel": "Online",
            "hour_of_day": 3, "day_of_week": "Sun",
            "merchant_category": "jewelry", "is_international": 1,
            "is_online": 1, "balance_before": 500.0, "balance_after": -9000.0,
        }])
        result = predict(tx, trained_artifacts)
        assert "rules_triggered" in result.columns
        assert "ai_reasoning" in result.columns
        assert result["rules_triggered"].iloc[0] != ""
        assert result["ai_reasoning"].iloc[0] != ""

    def test_round_amount_and_night_rules(self):
        from model import _rules_triggered
        eng = engineer_features(pd.DataFrame([{
            "amount": 5000, "tx_type": "purchase", "channel": "Online",
            "hour_of_day": 2, "day_of_week": "Sat",
            "merchant_category": "electronics", "is_international": 1,
            "is_online": 1, "balance_before": 1000, "balance_after": -4000,
        }]))
        row = eng.iloc[0]
        rules = _rules_triggered(row)
        assert "High-value" in rules
        assert "night" in rules.lower() or "Night" in rules
        assert "Online + international" in rules


# ---------------------------------------------------------------------------
# Test: Multi-currency System
# ---------------------------------------------------------------------------
from currency import RATE_PER_USD as RATE_TO_USD


class TestCurrency:
    def test_roundtrip_usd(self):
        from currency import to_usd, from_usd
        for code, r in RATE_TO_USD.items():
            amt = 100.0
            assert abs(from_usd(to_usd(amt, code), code) - amt) < 1e-6

    def test_convert_between_currencies(self):
        from currency import convert
        # 100 EUR -> GBP should be positive and reasonable
        gbp = convert(100, "EUR", "GBP")
        assert 50 < gbp < 150

    def test_high_value_threshold_localised(self):
        from currency import high_value_threshold
        from currency import RATE_PER_USD
        assert high_value_threshold("INR") == 5000.0 * RATE_PER_USD["INR"]
        assert high_value_threshold("USD") == 5000.0

    def test_predict_currency_labelling(self, trained_artifacts):
        """predict() with a non-USD currency records currency + localised text."""
        tx = pd.DataFrame([{
            "transaction_id": "CCY_001",
            "amount": 9000.0, "tx_type": "transfer", "channel": "Online",
            "hour_of_day": 3, "day_of_week": "Sat",
            "merchant_category": "jewelry", "is_international": 1,
            "is_online": 1, "balance_before": 1000.0, "balance_after": -8000.0,
        }])
        rate = RATE_TO_USD["INR"]
        result = predict(tx, trained_artifacts, currency="INR", symbol="₹", usd_per_unit=rate)
        assert result["currency"].iloc[0] == "INR"
        reasoning = result["ai_reasoning"].iloc[0]
        rules = result["rules_triggered"].iloc[0]
        assert "High-value transaction (₹" in rules
        assert "₹" in reasoning

    def test_format_amount_decimals(self):
        from currency import format_amount
        assert format_amount(149.5, "USD") == "$149.50"
        assert format_amount(15000, "JPY") == "¥15,000"

    def test_batch_currency_metadata_in_audit(self):
        reset_audit_log()
        df = pd.DataFrame([
            {"transaction_id": "CCY_A", "amount": 120.0, "channel": "POS",
             "tx_type": "purchase", "fraud_score": 0.1, "is_flagged": 0,
             "rules_triggered": "none", "ai_reasoning": "low risk"},
        ])
        record_batch(df, data_source="CSV upload", currency="GBP", rate=0.79)
        rec = load_audit_log()[0]
        import json
        used = json.loads(rec["data_used"])
        assert used["currency"] == "GBP"
        assert abs(used["amount"] - 94.8) < 0.01  # 120 USD * 0.79 GBP/USD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Fraud Detection Model Module
Handles feature engineering, model training, evaluation, and prediction.
Uses an ensemble of XGBoost and Isolation Forest for robust fraud detection.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_recall_curve, roc_auc_score,
                             average_precision_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
CATEGORICAL_COLS = ["tx_type", "channel", "day_of_week", "merchant_category"]
NUMERICAL_COLS = ["amount", "hour_of_day", "balance_before", "balance_after"]
BOOLEAN_COLS = ["is_international", "is_online"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from raw transaction data.

    Assumptions:
    - Amount-based ratios capture spending anomalies
    - Time-of-day patterns differ between fraud and legitimate transactions
    - Balance changes relative to amount indicate unusual behavior
    """
    df = df.copy()

    df["amount_to_balance_ratio"] = df["amount"] / (df["balance_before"] + 1)
    df["balance_change"] = df["balance_before"] - df["balance_after"]
    df["balance_change_ratio"] = df["balance_change"] / (df["amount"] + 1)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if h < 6 or h > 22 else 0)
    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d in ["Sat", "Sun"] else 0)
    df["log_amount"] = np.log1p(df["amount"])
    df["amount_digit_length"] = df["amount"].apply(lambda x: len(str(int(x))))
    df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

    return df


def prepare_data(df: pd.DataFrame, fit_encoders: bool = True,
                 encoders: Dict = None) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Encode categorical variables, scale numericals, and split features/target.
    """
    df = engineer_features(df)

    all_cats = CATEGORICAL_COLS
    all_nums = NUMERICAL_COLS + [
        "amount_to_balance_ratio", "balance_change", "balance_change_ratio",
        "is_night", "is_weekend", "log_amount", "amount_digit_length", "is_round_amount"
    ]
    all_bools = BOOLEAN_COLS

    if fit_encoders:
        encoders = {}
        for col in all_cats:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in all_cats:
            le = encoders[col]
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

    feature_cols = all_nums + all_bools + all_cats
    X = df[feature_cols].values.astype(np.float32)
    y = df["is_fraud"].values.astype(np.int32) if "is_fraud" in df.columns else None

    return X, y, encoders


def train_model(df: pd.DataFrame, test_size: float = 0.2,
                random_state: int = 42) -> Dict[str, Any]:
    """
    Train the fraud detection ensemble model.

    Steps:
    1. Feature engineering and encoding
    2. Handle class imbalance with SMOTE
    3. Train XGBoost classifier
    4. Train Isolation Forest for anomaly detection
    5. Combine predictions via weighted ensemble

    Returns dict with model artifacts and evaluation metrics.
    """
    X, y, encoders = prepare_data(df, fit_encoders=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=random_state, sampling_strategy=0.5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=random_state,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_res, y_train_res)

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=random_state,
        class_weight="balanced"
    )
    rf_model.fit(X_train_res, y_train_res)

    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.03,
        random_state=random_state
    )
    iso_model.fit(X_train_scaled)

    xgb_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    rf_prob = rf_model.predict_proba(X_test_scaled)[:, 1]
    iso_scores = -iso_model.score_samples(X_test_scaled)
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)

    ensemble_prob = 0.5 * xgb_prob + 0.3 * rf_prob + 0.2 * iso_norm

    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    metrics = {
        "classification_report": classification_report(
            y_test, ensemble_pred, target_names=["Legitimate", "Fraud"], output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, ensemble_pred).tolist(),
        "roc_auc": float(roc_auc_score(y_test, ensemble_prob)),
        "average_precision": float(average_precision_score(y_test, ensemble_prob)),
        "test_size": len(y_test),
        "fraud_count_test": int(y_test.sum()),
    }

    artifacts = {
        "xgb_model": xgb_model,
        "rf_model": rf_model,
        "iso_model": iso_model,
        "scaler": scaler,
        "encoders": encoders,
        "metrics": metrics,
    }

    return artifacts


def save_model(artifacts: Dict[str, Any], path: str = None):
    """Save trained model artifacts to disk."""
    if path is None:
        path = os.path.join(MODEL_DIR, "fraud_model.joblib")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(artifacts, path)
    return path


def load_model(path: str = None) -> Dict[str, Any]:
    """Load trained model artifacts from disk."""
    if path is None:
        path = os.path.join(MODEL_DIR, "fraud_model.joblib")
    return joblib.load(path)


def predict(df: pd.DataFrame, artifacts: Dict[str, Any],
            threshold: float = 0.5) -> pd.DataFrame:
    """
    Score transactions for fraud probability and flag suspicious ones.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data (without 'is_fraud' column).
    artifacts : dict
        Trained model artifacts from train_model().
    threshold : float
        Probability threshold for flagging fraud (0.0 - 1.0).

    Returns
    -------
    pd.DataFrame with original data plus fraud_score and is_flagged columns.
    """
    df = df.copy()
    X, _, _ = prepare_data(df, fit_encoders=False, encoders=artifacts["encoders"])

    X_scaled = artifacts["scaler"].transform(X)

    xgb_prob = artifacts["xgb_model"].predict_proba(X_scaled)[:, 1]
    rf_prob = artifacts["rf_model"].predict_proba(X_scaled)[:, 1]
    iso_scores = -artifacts["iso_model"].score_samples(X_scaled)
    iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)

    ensemble_prob = 0.5 * xgb_prob + 0.3 * rf_prob + 0.2 * iso_norm

    df["fraud_score"] = np.round(ensemble_prob, 4)
    df["is_flagged"] = (ensemble_prob >= threshold).astype(int)
    df["risk_level"] = df["fraud_score"].apply(_risk_level)

    return df


def _risk_level(score: float) -> str:
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    return "MINIMAL"


def validate_csv(df: pd.DataFrame):
    """Validate that an uploaded DataFrame has all required columns."""
    required = {"amount", "tx_type", "channel", "hour_of_day", "day_of_week",
                "merchant_category", "is_international", "is_online",
                "balance_before", "balance_after"}
    missing = required - set(df.columns)
    return missing


def get_feature_importance(artifacts: Dict[str, Any]) -> pd.DataFrame:
    """Extract feature importance from the XGBoost model."""
    feature_names = (
        NUMERICAL_COLS +
        ["amount_to_balance_ratio", "balance_change", "balance_change_ratio",
         "is_night", "is_weekend", "log_amount", "amount_digit_length", "is_round_amount"] +
        BOOLEAN_COLS + CATEGORICAL_COLS
    )
    importance = artifacts["xgb_model"].feature_importances_
    df = pd.DataFrame({
        "feature": feature_names[:len(importance)],
        "importance": importance
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df

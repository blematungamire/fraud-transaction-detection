"""
Synthetic Financial Transaction Data Generator
Generates realistic transaction data with fraud labels for training and testing.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_transactions(n_samples: int = 10000, fraud_ratio: float = 0.03,
                          seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic financial transactions with fraud labels.

    Assumptions:
    - Fraud accounts for ~3% of transactions (industry average)
    - Fraudulent transactions tend to have higher amounts and unusual patterns
    - Features mimic real-world banking transaction attributes

    Parameters
    ----------
    n_samples : int
        Number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions (0.0 to 1.0).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns matching the expected schema.
    """
    rng = np.random.RandomState(seed)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    records = []

    # --- Legitimate transactions ---
    for _ in range(n_legit):
        amt = rng.lognormal(mean=3.5, sigma=1.2)
        amt = round(min(max(amt, 0.5), 5000), 2)
        hour = int(rng.choice(range(24), p=_hour_weights_legit()))
        records.append(_make_record(amt, hour, is_fraud=False, rng=rng))

    # --- Fraudulent transactions ---
    for _ in range(n_fraud):
        amt = rng.lognormal(mean=5.5, sigma=1.5)
        amt = round(min(max(amt, 10), 25000), 2)
        hour = int(rng.choice(range(24), p=_hour_weights_fraud()))
        records.append(_make_record(amt, hour, is_fraud=True, rng=rng))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def _hour_weights_legit():
    w = np.array([
        1, 0.5, 0.3, 0.2, 0.2, 0.5, 1, 2,   # 0-7
        4, 6, 8, 9, 10, 9, 8, 7,              # 8-15
        6, 5, 4, 3, 2, 2, 1.5, 1              # 16-23
    ])
    return w / w.sum()


def _hour_weights_fraud():
    w = np.array([
        5, 6, 7, 6, 4, 2, 1, 1,              # 0-7
        1, 1, 1, 1, 1, 1, 1, 1,              # 8-15
        1, 1, 2, 3, 4, 5, 6, 6               # 16-23
    ])
    return w / w.sum()


def _make_record(amount: float, hour: int, is_fraud: bool, rng: np.random.RandomState):
    tx_type = rng.choice(
        ["purchase", "withdrawal", "transfer", "payment"],
        p=[0.45, 0.20, 0.20, 0.15]
    )

    if is_fraud:
        balance_before = round(rng.uniform(100, 15000), 2)
        balance_after = round(max(balance_before - amount * rng.uniform(0.5, 1.5), -500), 2)
    else:
        balance_before = round(rng.uniform(500, 50000), 2)
        balance_after = round(balance_before - amount, 2)

    is_international = bool(rng.choice([0, 1], p=[0.85, 0.15]) if not is_fraud
                            else rng.choice([0, 1], p=[0.40, 0.60]))

    is_online = bool(rng.choice([0, 1], p=[0.55, 0.45] if not is_fraud else [0.30, 0.70]))

    if is_fraud:
        txn_hour = hour
    else:
        txn_hour = hour

    day_of_week = rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

    channel = rng.choice(
        ["ATM", "Online", "POS", "Mobile", "Branch"],
        p=[0.15, 0.30, 0.25, 0.20, 0.10] if not is_fraud
        else [0.10, 0.40, 0.15, 0.30, 0.05]
    )

    merchant_category = rng.choice(
        ["grocery", "electronics", "restaurant", "travel", "gas",
         "jewelry", "atm_withdrawal", "transfer", "subscription", "other"],
        p=[0.20, 0.10, 0.15, 0.08, 0.10,
           0.02, 0.10, 0.10, 0.08, 0.07] if not is_fraud
        else [0.05, 0.15, 0.05, 0.20, 0.03,
              0.15, 0.10, 0.15, 0.02, 0.10]
    )

    return {
        "transaction_id": f"TXN_{rng.randint(100000, 999999)}",
        "amount": amount,
        "tx_type": tx_type,
        "channel": channel,
        "hour_of_day": txn_hour,
        "day_of_week": day_of_week,
        "merchant_category": merchant_category,
        "is_international": is_international,
        "is_online": is_online,
        "balance_before": balance_before,
        "balance_after": balance_after,
        "is_fraud": int(is_fraud),
    }


if __name__ == "__main__":
    df = generate_transactions(10000)
    df.to_csv("data/transactions.csv", index=False)
    print(f"Generated {len(df)} transactions, fraud rate: {df['is_fraud'].mean():.2%}")

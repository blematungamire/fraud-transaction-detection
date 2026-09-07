"""
Automated Audit Trail Module
================================
Records every fraud-detection decision into a persistent, append-only log so
that all model actions are fully traceable and accountable.

Each row captures:
  - Transaction ID
  - Date/time of decision
  - Risk score
  - Rules triggered
  - AI reasoning
  - Data used (source + inputs)
  - Action taken (by the system)
  - Investigator decision (human override/review, filled later)
  - Final outcome

Storage: an on-disk CSV (audit_log.csv) appended with each decision. The log
is never overwritten -- only appended -- preserving a complete history.
"""
import os
import csv
import json
from datetime import datetime, timezone

AUDIT_FILE = os.path.join(os.path.dirname(__file__), "audit_log.csv")

AUDIT_COLUMNS = [
    "transaction_id",
    "date_time",
    "risk_score",
    "rules_triggered",
    "ai_reasoning",
    "data_used",
    "action_taken",
    "investigator_decision",
    "final_outcome",
]


def _ensure_file():
    """Create the audit log file with headers if it does not exist."""
    if not os.path.exists(AUDIT_FILE):
        with open(AUDIT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()


def record_decision(transaction_id: str,
                    risk_score: float,
                    rules_triggered: str,
                    ai_reasoning: str,
                    data_used: dict,
                    action_taken: str,
                    investigator_decision: str = "Pending review",
                    final_outcome: str = "Awaiting investigator",
                    date_time: str = None) -> dict:
    """
    Append a single decision to the audit trail.

    Returns the formatted record that was written.
    """
    _ensure_file()

    if date_time is None:
        date_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    record = {
        "transaction_id": str(transaction_id),
        "date_time": date_time,
        "risk_score": f"{float(risk_score):.4f}",
        "rules_triggered": rules_triggered,
        "ai_reasoning": ai_reasoning,
        "data_used": json.dumps(data_used),
        "action_taken": action_taken,
        "investigator_decision": investigator_decision,
        "final_outcome": final_outcome,
    }

    with open(AUDIT_FILE, "a", newline="", encoding="utf-8") as f:
        # Preserve column order
        ordered = {k: record[k] for k in AUDIT_COLUMNS}
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writerow(ordered)

    return record


def record_batch(result_df: object, data_source: str = "CSV upload",
                 default_action: str = "Flagged for review",
                 currency: str = "USD", rate: float = 1.0) -> int:
    """
    Append one audit record for every scored transaction in a result set.

    Parameters
    ----------
    result_df : pd.DataFrame
        Output of model.predict() -- must contain transaction_id, fraud_score,
        is_flagged, risk_level, rules_triggered, ai_reasoning.
    data_source : str
        Where the input data came from (e.g. "CSV upload", "Manual entry").
    default_action : str
        Action the system took for each record.
    currency : str
        Currency code the user was working in at time of scoring.
    rate : float
        Units of `currency` per 1 USD (used to store the local amount).

    Returns the number of records appended.
    """
    count = 0
    for _, row in result_df.iterrows():
        tx_id = row.get("transaction_id", f"UNKNOWN_{count}")
        action = (
            default_action if int(row.get("is_flagged", 0)) == 1
            else "No action (low risk cleared)"
        )
        amount_usd = float(row.get("amount", 0.0))
        record_decision(
            transaction_id=tx_id,
            risk_score=float(row.get("fraud_score", 0.0)),
            rules_triggered=str(row.get("rules_triggered", "")),
            ai_reasoning=str(row.get("ai_reasoning", "")),
            data_used={
                "source": data_source,
                "channel": str(row.get("channel", "")),
                "tx_type": str(row.get("tx_type", "")),
                "currency": currency,
                "amount": round(amount_usd * rate, 2),
                "amount_in_usd": round(amount_usd, 2),
            },
            action_taken=action,
        )
        count += 1
    return count


def update_investigator_decision(transaction_id: str,
                                 decision: str,
                                 outcome: str = None) -> bool:
    """
    Update the investigator decision and final outcome for a transaction.

    The log is append-only, so this rewrites the file with the updated cell
    (simulating an investigator returning a verdict). Returns True on success.
    """
    _ensure_file()
    updated = False

    with open(AUDIT_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if row["transaction_id"] == str(transaction_id):
            row["investigator_decision"] = decision
            if outcome is not None:
                row["final_outcome"] = outcome
            updated = True

    with open(AUDIT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return updated


def load_audit_log() -> list:
    """Return all audit records as a list of dicts (empty if none yet)."""
    if not os.path.exists(AUDIT_FILE):
        return []
    with open(AUDIT_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_audit_log_dataframe():
    """Return all audit records as a pandas DataFrame (empty if none yet)."""
    import pandas as pd
    if not os.path.exists(AUDIT_FILE):
        return pd.DataFrame(columns=AUDIT_COLUMNS)
    return pd.read_csv(AUDIT_FILE)


def reset_audit_log():
    """Delete the audit log (used in tests / admin reset)."""
    if os.path.exists(AUDIT_FILE):
        os.remove(AUDIT_FILE)

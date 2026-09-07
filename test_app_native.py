"""Functional tests: verify the Streamlit app end-to-end via AppTest.

Simulates a user uploading a CSV, running fraud detection, and checking
results. Also exercises the manual entry path.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from streamlit.testing.v1 import AppTest

from audit_trail import reset_audit_log, load_audit_log

APP_PATH = os.path.join(os.path.dirname(__file__), "app.py")
SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "data", "sample_transactions.csv")


def test_app_renders():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception, f"App raised: {at.exception}"
    assert any("Fraud Transaction Detection" in str(t.value) for t in at.title)
    assert len(at.slider) >= 1
    # Audit Trail tab must exist
    assert any("Audit Trail" in str(t.label) for t in at.tabs)


def test_currency_selection_flow():
    reset_audit_log()

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # Select INR in the sidebar currency selector
    assert at.sidebar.selectbox, "expected at least one sidebar selectbox"
    currency_box = None
    for sb in at.sidebar.selectbox:
        if "Working currency" in str(sb.label):
            currency_box = sb
            break
    assert currency_box is not None, "currency selectbox not found"
    currency_box.select("INR — Indian Rupee")
    at.run()
    assert not at.exception, f"App raised after currency change: {at.exception}"

    with open(SAMPLE_CSV, "rb") as f:
        content = f.read()
    at.file_uploader[0].set_value(
        [(os.path.basename(SAMPLE_CSV), content, "text/csv")]
    )
    at.run()
    for btn in at.button:
        if "Run Fraud Detection" in str(btn.label):
            btn.click()
            break
    at.run()
    assert not at.exception, f"App raised after INR scoring: {at.exception}"

    # Audit records should carry INR metadata
    log = load_audit_log()
    assert len(log) > 0
    import json
    used = json.loads(log[0]["data_used"])
    assert used["currency"] == "INR"
    print(f"Currency flow wrote {len(log)} INR decisions OK")


def test_audit_trail_writes_on_scoring():
    reset_audit_log()
    assert len(load_audit_log()) == 0

    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    with open(SAMPLE_CSV, "rb") as f:
        content = f.read()
    at.file_uploader[0].set_value(
        [(os.path.basename(SAMPLE_CSV), content, "text/csv")]
    )
    at.run()
    for btn in at.button:
        if "Run Fraud Detection" in str(btn.label):
            btn.click()
            break
    at.run()
    assert not at.exception, f"App raised after scoring: {at.exception}"

    log = load_audit_log()
    assert len(log) > 0, "Audit trail should contain decisions after scoring"
    # Verify expected audit columns are populated
    first = log[0]
    for col in ["transaction_id", "date_time", "risk_score", "rules_triggered",
                "ai_reasoning", "data_used", "action_taken",
                "investigator_decision", "final_outcome"]:
        assert col in first, f"audit record missing {col}"
    print(f"Audit trail wrote {len(log)} decisions OK")


def test_upload_and_score_flow():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # Upload sample CSV
    with open(SAMPLE_CSV, "rb") as f:
        content = f.read()
    at.file_uploader[0].set_value(
        [(os.path.basename(SAMPLE_CSV), content, "text/csv")]
    )
    at.run()
    assert not at.exception, f"App raised after upload: {at.exception}"
    assert any("Loaded 1,000 transactions" in str(e.value) for e in at.success)

    # Click "Run Fraud Detection"
    def click(pred_btn):
        pred_btn.click()
    at.button(key=None).__class__  # noqa
    # Find the primary run button by label
    for btn in at.button:
        if "Run Fraud Detection" in str(btn.label):
            btn.click()
            break
    at.run()
    assert not at.exception, f"App raised after scoring: {at.exception}"

    # Metrics should now show results
    metric_labels = [str(m.label) for m in at.metric]
    # results dataframe rendered
    dataframes = at.dataframe
    assert len(dataframes) >= 1

    # Verify a download button for results appears
    dls = [d.label for d in at.download_button]
    assert any("Download Full Results" in l for l in dls)
    print("Upload-and-score flow OK")


def test_manual_entry_flow():
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    # All tab elements render in AppTest; submit the manual entry form.
    # st.form_submit_button is exposed as a Button in AppTest.
    for sb in at.button:
        if "Score Transaction" in str(sb.label):
            sb.click()
            break
    at.run()
    assert not at.exception, f"App raised after manual scoring: {at.exception}"

    # Result gauges/metadata should render
    metric_labels = [str(m.label) for m in at.metric]
    assert any("Fraud Score" in l for l in metric_labels)
    assert any("Risk Level" in l for l in metric_labels)
    print("Manual-entry flow OK")


if __name__ == "__main__":
    test_app_renders()
    test_audit_trail_writes_on_scoring()
    test_upload_and_score_flow()
    test_manual_entry_flow()
    print("ALL FUNCTIONAL TESTS PASSED")
"""End-to-end tests for PDF report generation, CSV upload parsing, and Streamlit UI."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.app import adapters
from src.app.report import build_monthly_pdf_report
from src.recommend.rules import generate_recommendations


def test_user_csv_upload_parsing() -> None:
    csv_text = (
        "timestamp,generation_kwh,consumption_kwh\n"
        "2025-01-01 00:00,0.0,0.25\n"
        "2025-01-01 00:30,0.0,0.20\n"
        "2025-01-01 12:00,1.1,0.35\n"
        "2025-01-01 12:30,1.2,0.40\n"
    )
    df = adapters.load_user_csv(StringIO(csv_text), pv_kw=4.0)
    assert not df.empty
    assert (df["home_id"] == 0).all()
    assert {"ts", "gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_kwh"}.issubset(df.columns)


def test_monthly_pdf_report_generation() -> None:
    df = adapters.load_home(1)
    assert not df.empty
    kpis = adapters.compute_kpis(df)
    monthly_df = adapters.monthly_summary(df)
    anomalies = adapters.detect_anomalies(df)
    recs = generate_recommendations(df, kpis, anomalies)
    month = str(monthly_df["month"].iloc[-1])
    ai_text = adapters.monthly_report_text(1, month)

    pdf_bytes = build_monthly_pdf_report(
        home_label="Home 1",
        month=month,
        kpis=kpis,
        monthly_df=monthly_df,
        anomalies=anomalies,
        recommendations=recs,
        ai_summary=ai_text,
        import_tariff=8.0,
        export_tariff=3.0,
    )
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 3000


def test_streamlit_app_smoke_and_interactions() -> None:
    app_path = Path(__file__).resolve().parents[1] / "src" / "app" / "streamlit_app.py"
    at = AppTest.from_file(app_path, default_timeout=30)
    at.run()
    assert not at.exception, f"Streamlit raised exception on initial load: {at.exception}"
    assert len(at.tabs) == 7

    # Trigger PDF generation button in Tab 7
    gen_btn = [b for b in at.button if "Generate Monthly PDF Report" in b.label]
    assert gen_btn, "Generate Monthly PDF Report button missing"
    gen_btn[0].click().run()
    assert not at.exception, f"Streamlit raised exception on PDF generation: {at.exception}"
    assert "last_pdf_bytes" in at.session_state
    assert at.session_state["last_pdf_bytes"].startswith(b"%PDF-")

    # Trigger Ask SolarSense chat input
    assert len(at.chat_input) == 1
    at.chat_input[0].set_value("Is a 5 kWh battery worth it for me?").run()
    assert not at.exception, f"Streamlit raised exception on chat: {at.exception}"
    assert 1 in at.session_state["chat_histories"]
    assert len(at.session_state["chat_histories"][1]) == 2

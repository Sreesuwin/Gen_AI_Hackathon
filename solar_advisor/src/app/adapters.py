"""Integration adapters connecting M3/UI to Shashikiran (M1/M2) and Sreesuwin (M4).

Imports the real team modules when available in `src/pipeline`, `src/analytics`,
and `src/llm`, and cleanly falls back to `src/app/temporary_mocks.py` during
parallel development or when a team function still raises NotImplementedError.
"""

from __future__ import annotations

import functools

from src.app import temporary_mocks


def _with_not_implemented_fallback(real_fn, mock_fn):
    @functools.wraps(real_fn)
    def _wrapper(*args, **kwargs):
        try:
            return real_fn(*args, **kwargs)
        except NotImplementedError:
            return mock_fn(*args, **kwargs)

    return _wrapper


# --- Pipeline (Shashikiran) ---
try:
    from src.pipeline import list_homes, load_home, load_user_csv as _real_load_user_csv  # type: ignore
    load_user_csv = _with_not_implemented_fallback(_real_load_user_csv, temporary_mocks.mock_load_user_csv)
    USING_MOCK_PIPELINE = False
except ImportError:
    try:
        from src.pipeline.load import list_homes, load_home, load_user_csv as _real_load_user_csv  # type: ignore
        load_user_csv = _with_not_implemented_fallback(_real_load_user_csv, temporary_mocks.mock_load_user_csv)
        USING_MOCK_PIPELINE = False
    except ImportError:
        list_homes = temporary_mocks.mock_list_homes
        load_home = temporary_mocks.mock_load_home
        load_user_csv = temporary_mocks.mock_load_user_csv
        USING_MOCK_PIPELINE = True

# --- Analytics (Shashikiran) ---
try:
    from src.analytics import (  # type: ignore
        compute_kpis,
        daily_profile,
        detect_anomalies,
        forecast,
        monthly_summary,
    )
    USING_MOCK_ANALYTICS = False
except ImportError:
    try:
        from src.analytics.anomalies import detect_anomalies  # type: ignore
        from src.analytics.forecast import forecast  # type: ignore
        from src.analytics.kpis import compute_kpis, daily_profile, monthly_summary  # type: ignore
        USING_MOCK_ANALYTICS = False
    except ImportError:
        compute_kpis = temporary_mocks.mock_compute_kpis
        daily_profile = temporary_mocks.mock_daily_profile
        monthly_summary = temporary_mocks.mock_monthly_summary
        detect_anomalies = temporary_mocks.mock_detect_anomalies
        forecast = temporary_mocks.mock_forecast
        USING_MOCK_ANALYTICS = True

# --- GenAI Advisor (Sreesuwin) ---
try:
    from src.llm.advisor import chat, monthly_report_text as _real_monthly_report_text  # type: ignore
    monthly_report_text = _with_not_implemented_fallback(
        _real_monthly_report_text, temporary_mocks.mock_monthly_report_text
    )
    USING_MOCK_LLM = False
except ImportError:
    try:
        from src.llm import chat, monthly_report_text as _real_monthly_report_text  # type: ignore
        monthly_report_text = _with_not_implemented_fallback(
            _real_monthly_report_text, temporary_mocks.mock_monthly_report_text
        )
        USING_MOCK_LLM = False
    except ImportError:
        chat = temporary_mocks.mock_chat
        monthly_report_text = temporary_mocks.mock_monthly_report_text
        USING_MOCK_LLM = True

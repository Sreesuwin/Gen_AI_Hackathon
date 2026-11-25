"""Contract tests: every module must return the agreed shapes.   Run: pytest -q

These pass on the stubs. They must still pass after you replace a stub with real code.
"""
import json

import pandas as pd
import pytest

from src import contracts as C
from src.analytics.anomalies import detect_anomalies
from src.analytics.forecast import forecast
from src.analytics.kpis import compute_kpis, daily_profile, monthly_summary
from src.llm.guardrail import verify_numbers
from src.llm.tools import DISPATCH, TOOLS
from src.pipeline.load import list_homes, load_home
from src.recommend.rules import generate_recommendations
from src.recommend.simulator import simulate_battery, simulate_ev, simulate_extra_pv, simulate_load_shift


@pytest.fixture(scope="module")
def df():
    return load_home(int(list_homes()["home_id"].iloc[0]))


def _plain(obj):
    """Must be JSON-serialisable with plain types and <= 2 decimals."""
    json.dumps(obj)
    if isinstance(obj, float):
        assert round(obj, 2) == obj, f"{obj} has more than 2 decimals"
    elif isinstance(obj, dict):
        for v in obj.values():
            _plain(v)
    elif isinstance(obj, list):
        for v in obj:
            _plain(v)


def _keys(d, keys):
    missing = [k for k in keys if k not in d]
    assert not missing, f"missing keys: {missing}"
    _plain(d)


def test_load_home_schema(df):
    assert list(df.columns) == C.HALFHOURLY_COLUMNS
    assert len(df) > 0 and df["valid"].all()
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])


def test_list_homes():
    assert set(C.HOMES_COLUMNS) <= set(list_homes().columns)


def test_kpis(df):
    _keys(compute_kpis(df), C.KPI_KEYS)


def test_profiles(df):
    assert list(daily_profile(df).columns) == ["hour", "gen_kw", "cons_kw"]
    assert "self_consumption_pct" in monthly_summary(df).columns


def test_anomalies(df):
    out = detect_anomalies(df)
    assert isinstance(out, list)
    for a in out:
        _keys(a, C.ANOMALY_KEYS)
        assert a["type"] in C.ANOMALY_TYPES


def test_forecast(df):
    f = forecast(df, 7, "cons")
    _keys(f, C.FORECAST_KEYS)
    assert len(f["p50"]) == 7 and f["method"] in C.FORECAST_METHODS


def test_simulators(df):
    _keys(simulate_battery(df, 5), C.BATTERY_KEYS)
    _keys(simulate_extra_pv(df, 2), C.EXTRA_PV_KEYS)
    _keys(simulate_load_shift(df, 1.5), C.LOAD_SHIFT_KEYS)
    _keys(simulate_ev(df, 40), C.EV_KEYS)


def test_recommendations(df):
    recs = generate_recommendations(df, compute_kpis(df), detect_anomalies(df))
    assert isinstance(recs, list) and recs
    for r in recs:
        _keys(r, C.RECOMMENDATION_KEYS)
        assert r["category"] in C.RECOMMENDATION_CATEGORIES
    saved = [r["inr_saved_month"] for r in recs]
    assert saved == sorted(saved, reverse=True), "recommendations must be sorted by inr_saved_month"


def test_tools_dispatch(df):
    names = {t["function"]["name"] for t in TOOLS}
    assert names == set(DISPATCH)
    args = {"forecast": {"target": "cons"}, "simulate_battery": {"capacity_kwh": 5},
            "simulate_extra_pv": {"extra_kw": 2}, "simulate_ev": {"km_per_day": 40}}
    for n in names:
        _plain(DISPATCH[n](df, **args.get(n, {})))


def test_guardrail_shape():
    _keys(verify_numbers("You save ₹6052.0 a year", [{"output": {"savings_inr_year": 6052.0}}]), C.GUARDRAIL_KEYS)

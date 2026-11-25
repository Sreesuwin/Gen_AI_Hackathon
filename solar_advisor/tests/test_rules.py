"""Tests for M3 recommendation rules (R1–R6)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.recommend.rules import generate_recommendations
from tests.test_simulators import load_test_home_df

REQUIRED_KEYS = {
    "id",
    "title",
    "category",
    "action",
    "kwh_saved_month",
    "inr_saved_month",
    "effort",
    "evidence",
}


def _make_synthetic_df(
    days: int = 30,
    night_cons_per_half_hr: float = 0.30,
    day_gen_per_half_hr: float = 1.2,
    eve_cons_per_half_hr: float = 0.6,
    cl_night_per_half_hr: float = 0.5,
    spike_cons: float = 2.5,
) -> pd.DataFrame:
    ts = pd.date_range("2025-01-01", periods=days * 48, freq="30min")
    hours = ts.hour
    gen = np.where((hours >= 9) & (hours < 16), day_gen_per_half_hr, 0.0)
    gc = np.full(len(ts), 0.15)
    gc[(hours >= 1) & (hours < 5)] = night_cons_per_half_hr
    gc[(hours >= 18) & (hours < 22)] = eve_cons_per_half_hr
    gc[(hours == 19) & (ts.minute == 0)] = spike_cons
    cl = np.where((hours >= 1) & (hours < 5), cl_night_per_half_hr, 0.0)
    cons = gc + cl
    return pd.DataFrame({
        "home_id": 999,
        "ts": ts,
        "gen_kwh": gen,
        "gc_kwh": gc,
        "cl_kwh": cl,
        "cons_kwh": cons,
        "self_kwh": np.minimum(gen, cons),
        "import_kwh": np.maximum(cons - gen, 0.0),
        "export_kwh": np.maximum(gen - cons, 0.0),
        "valid": True,
    })


def _assert_rec_schema(recs: list[dict]) -> None:
    json.dumps(recs)
    inrs = [r["inr_saved_month"] for r in recs]
    assert inrs == sorted(inrs, reverse=True), "Recommendations must be sorted by inr_saved_month descending"
    for r in recs:
        assert REQUIRED_KEYS.issubset(r.keys())
        assert type(r["kwh_saved_month"]) is float
        assert type(r["inr_saved_month"]) is float
        assert round(r["kwh_saved_month"], 2) == r["kwh_saved_month"]
        assert round(r["inr_saved_month"], 2) == r["inr_saved_month"]
        assert isinstance(r["evidence"], dict) and len(r["evidence"]) > 0


def test_rules_on_real_home() -> None:
    df = load_test_home_df(1)
    recs = generate_recommendations(df, kpis=None, anomalies=[])
    assert len(recs) >= 2
    _assert_rec_schema(recs)
    ids = {r["id"] for r in recs}
    assert "R1" in ids  # Home 1 has ~24.6% self-consumption & evening import
    assert "R4" in ids  # Home 1 has CL & large solar exports


def test_all_six_rules_trigger_and_rank() -> None:
    df = _make_synthetic_df(
        days=90,
        night_cons_per_half_hr=0.35,   # baseload = 0.70 kW > 0.40 kW -> R2
        day_gen_per_half_hr=1.5,       # large daytime surplus -> R1, R4, R5
        eve_cons_per_half_hr=0.6,      # evening import > 2 kWh/day -> R1, R5
        cl_night_per_half_hr=0.5,      # controlled load at night -> R4
        spike_cons=3.5,                # peak_kw > 2.5 * avg_evening_kw -> R6
    )
    anomalies = [
        {
            "date": "2025-03-20",
            "type": "low_generation",
            "value": 0.55,
            "expected": 1.0,
            "unit": "ratio",
            "score": 0.85,
            "reason": "possible dust, shading or fault",
        }
    ]
    # With higher import spread (12 INR import / 1 INR export), battery payback < 12 years triggers R5
    recs = generate_recommendations(
        df,
        kpis=None,
        anomalies=anomalies,
        import_tariff=12.0,
        export_tariff=1.0,
    )
    _assert_rec_schema(recs)
    ids = {r["id"] for r in recs}
    assert ids == {"R1", "R2", "R3", "R4", "R5", "R6"}, f"Expected all 6 rules, got {ids}"

    # At default tariffs (8/3), 1 cycle/day has >= 17.6 yr payback, so R5 should NOT trigger
    recs_default = generate_recommendations(df, kpis=None, anomalies=anomalies)
    assert "R5" not in {r["id"] for r in recs_default}


def test_r3_ignores_old_low_gen_anomalies() -> None:
    df = _make_synthetic_df(days=90)
    old_anomalies = [
        {
            "date": "2024-01-01",  # > 60 days before 2025-03-31
            "type": "low_generation",
            "value": 0.5,
            "expected": 1.0,
            "unit": "ratio",
            "score": 0.8,
            "reason": "old dust event",
        }
    ]
    recs = generate_recommendations(df, kpis=None, anomalies=old_anomalies)
    assert "R3" not in {r["id"] for r in recs}


def test_empty_df_returns_empty() -> None:
    assert generate_recommendations(pd.DataFrame(), {}, []) == []

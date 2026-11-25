"""Tests for M3 simulators (simulate_battery, simulate_extra_pv, simulate_load_shift, simulate_ev)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.recommend.simulator import (
    simulate_battery,
    simulate_ev,
    simulate_extra_pv,
    simulate_load_shift,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PARQUET = ROOT / "data" / "sample" / "sample_home.parquet"
ZIP_PATH = ROOT / "Dataset" / "Ausgrid_solar_home_data.zip"


def load_test_home_df(home_id: int = 1) -> pd.DataFrame:
    """Load sample_home.parquet if present, else stream one home from the Ausgrid zip in memory."""
    if SAMPLE_PARQUET.exists():
        df = pd.read_parquet(SAMPLE_PARQUET)
        if "valid" in df.columns:
            df = df[df["valid"]].copy()
        return df

    assert ZIP_PATH.exists(), f"Neither {SAMPLE_PARQUET} nor {ZIP_PATH} exists."
    with zipfile.ZipFile(ZIP_PATH) as zf:
        raw = pd.read_csv(zf.open("Solar home 2012-2013.csv"), skiprows=1, nrows=3500)
    raw = raw[raw["Customer"] == home_id].copy()
    if "Row Quality" in raw.columns:
        raw = raw[raw["Row Quality"] != "NA"].copy()
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y")
    slots = list(raw.columns[5:53])
    slot_map = {s: i for i, s in enumerate(slots)}
    long = raw.melt(
        id_vars=["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"],
        value_vars=slots,
        var_name="slot",
        value_name="kwh",
    )
    long["ts"] = long["date"] + pd.to_timedelta(long["slot"].map(slot_map) * 30, unit="min")
    piv = long.pivot_table(
        index=["Customer", "Generator Capacity", "ts"],
        columns="Consumption Category",
        values="kwh",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for cat in ("GG", "GC", "CL"):
        if cat not in piv.columns:
            piv[cat] = 0.0
    out = pd.DataFrame({
        "home_id": piv["Customer"].astype(int),
        "ts": piv["ts"],
        "gen_kwh": piv["GG"].astype(float),
        "gc_kwh": piv["GC"].astype(float),
        "cl_kwh": piv["CL"].astype(float),
    })
    out["cons_kwh"] = out["gc_kwh"] + out["cl_kwh"]
    out["self_kwh"] = out[["gen_kwh", "cons_kwh"]].min(axis=1)
    out["import_kwh"] = (out["cons_kwh"] - out["gen_kwh"]).clip(lower=0.0)
    out["export_kwh"] = (out["gen_kwh"] - out["cons_kwh"]).clip(lower=0.0)
    pv_kw = float(piv["Generator Capacity"].iloc[0])
    out["valid"] = out["gen_kwh"] <= pv_kw * 0.5 * 1.05
    return out[out["valid"]].sort_values("ts").reset_index(drop=True)


@pytest.fixture(scope="module")
def sample_home() -> pd.DataFrame:
    return load_test_home_df(1)


def _assert_plain_python_and_rounded(result: dict) -> None:
    json.dumps(result)
    for key, val in result.items():
        assert type(val) in (int, float, str, bool), f"{key} has non-plain type {type(val)}"
        if isinstance(val, float):
            assert round(val, 2) == val, f"{key}={val} has >2 decimal places"


def test_simulate_battery_contract(sample_home: pd.DataFrame) -> None:
    res = simulate_battery(sample_home, capacity_kwh=5.0)
    _assert_plain_python_and_rounded(res)
    expected_keys = {
        "capacity_kwh",
        "power_kw",
        "import_avoided_kwh",
        "savings_inr_year",
        "self_consumption_pct_before",
        "self_consumption_pct_after",
        "cost_inr",
        "payback_years",
        "import_before_kwh",
        "import_after_kwh",
        "export_before_kwh",
        "export_after_kwh",
    }
    assert expected_keys.issubset(res.keys())
    assert res["power_kw"] == 2.5
    assert res["cost_inr"] == 150000.0
    assert res["import_after_kwh"] <= res["import_before_kwh"]
    assert res["export_after_kwh"] <= res["export_before_kwh"]
    assert res["self_consumption_pct_after"] >= res["self_consumption_pct_before"]
    assert res["import_avoided_kwh"] > 0


def test_simulate_battery_zero_and_empty(sample_home: pd.DataFrame) -> None:
    zero = simulate_battery(sample_home, capacity_kwh=0.0)
    assert zero["import_avoided_kwh"] == 0.0
    assert zero["savings_inr_year"] == 0.0
    assert zero["payback_years"] == 0.0

    empty = simulate_battery(pd.DataFrame(columns=["ts", "gen_kwh", "cons_kwh"]), capacity_kwh=5.0)
    _assert_plain_python_and_rounded(empty)
    assert empty["import_avoided_kwh"] == 0.0


def test_simulate_extra_pv(sample_home: pd.DataFrame) -> None:
    res = simulate_extra_pv(sample_home, extra_kw=2.0)
    _assert_plain_python_and_rounded(res)
    assert res["extra_gen_kwh"] > 0
    assert res["import_after_kwh"] <= res["import_before_kwh"]
    assert res["export_after_kwh"] >= res["export_before_kwh"]
    assert res["savings_inr_year"] > 0
    assert res["cost_inr"] == 100000.0


def test_simulate_load_shift(sample_home: pd.DataFrame) -> None:
    res = simulate_load_shift(sample_home, kwh_per_day=1.5)
    _assert_plain_python_and_rounded(res)
    assert res["import_avoided_kwh"] >= 0
    assert res["import_after_kwh"] <= res["import_before_kwh"]
    assert res["self_consumption_pct_after"] >= res["self_consumption_pct_before"]
    assert res["savings_inr_year"] >= 0


def test_simulate_ev(sample_home: pd.DataFrame) -> None:
    res = simulate_ev(sample_home, km_per_day=40.0)
    _assert_plain_python_and_rounded(res)
    assert res["ev_demand_day_kwh"] == 6.0
    assert res["ev_demand_kwh"] > 0
    assert 0.0 <= res["solar_coverage_pct"] <= 100.0
    assert abs((res["ev_from_solar_kwh"] + res["ev_from_grid_kwh"]) - res["ev_demand_kwh"]) <= 0.5
    assert res["additional_cost_inr_year"] >= res["grid_cost_inr_year"]

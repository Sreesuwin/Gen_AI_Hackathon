"""What-if simulators (battery, extra PV, load shift, EV).   OWNER: Divya

STUB: fixed example outputs. Real versions step through the half-hours (README 6.1).
Results are scaled to one year (365 days) whatever the data length.
"""
import pandas as pd


def simulate_battery(df: pd.DataFrame, capacity_kwh: float, power_kw: float | None = None) -> dict:
    power_kw = power_kw or capacity_kwh / 2
    # STUB
    return {"capacity_kwh": capacity_kwh, "power_kw": power_kw, "import_avoided_kwh": 1210.4,
            "savings_inr_year": 6052.0, "self_consumption_pct_before": 52.6, "self_consumption_pct_after": 81.3,
            "cost_inr": round(capacity_kwh * 30000, 2), "payback_years": 24.8}


def simulate_extra_pv(df: pd.DataFrame, extra_kw: float) -> dict:
    # STUB
    return {"extra_kw": extra_kw, "extra_gen_kwh_year": 1350.0, "import_avoided_kwh": 410.2,
            "extra_export_kwh": 939.8, "savings_inr_year": 6101.0, "cost_inr": round(extra_kw * 50000, 2),
            "payback_years": 16.4, "self_sufficiency_pct_after": 24.3}


def simulate_load_shift(df: pd.DataFrame, kwh_per_day: float, from_hours=(18, 22), to_hours=(10, 14)) -> dict:
    # STUB
    return {"kwh_per_day": kwh_per_day, "shifted_kwh_year": 438.0, "savings_inr_year": 2190.0,
            "self_consumption_pct_after": 72.7}


def simulate_ev(df: pd.DataFrame, km_per_day: float, charge_hours=(10, 15)) -> dict:
    # STUB
    return {"km_per_day": km_per_day, "ev_kwh_year": round(km_per_day * 0.15 * 365, 2), "solar_share_pct": 38.5,
            "extra_import_kwh_year": 1347.2, "extra_cost_inr_year": 10777.6}

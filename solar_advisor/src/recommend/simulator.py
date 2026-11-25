"""Energy optimization simulators for SolarSense (M3).

Implements battery storage, extra solar PV, load-shifting, and EV charging
simulations over half-hourly household generation/consumption time series.
All numeric outputs are annualized to 365 days, rounded to 2 decimals, and
returned as plain Python types for JSON/LLM guardrail compatibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BATTERY_COST_PER_KWH,
    BATTERY_EFFICIENCY,
    EXPORT_TARIFF,
    IMPORT_TARIFF,
    PV_COST_PER_KW,
)

EV_KWH_PER_KM = 0.15


def _prepare_df(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Validate, sort, and compute annualization scale factor (365 / days)."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "gen_kwh", "cons_kwh"]), 1.0

    work = df.copy()
    if "ts" in work.columns:
        work["ts"] = pd.to_datetime(work["ts"])
        work = work.sort_values("ts").reset_index(drop=True)

    work["gen_kwh"] = pd.to_numeric(work["gen_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)
    work["cons_kwh"] = pd.to_numeric(work["cons_kwh"], errors="coerce").fillna(0.0).clip(lower=0.0)

    days = max(len(work) / 48.0, 1.0 / 48.0)
    scale = 365.0 / days
    return work, float(scale)


def _r2(val: float) -> float:
    """Convert to plain Python float rounded to 2 decimals."""
    return round(float(val), 2)


def _self_consumption_pct(gen_total: float, export_total: float) -> float:
    if gen_total <= 0:
        return 0.0
    used = max(gen_total - export_total, 0.0)
    return min(max((used / gen_total) * 100.0, 0.0), 100.0)


def simulate_battery(
    df: pd.DataFrame,
    capacity_kwh: float,
    power_kw: float | None = None,
    import_tariff: float | None = None,
    export_tariff: float | None = None,
) -> dict:
    """Simulate a residential battery over half-hourly generation and load.

    Parameters
    ----------
    df : pd.DataFrame
        Half-hourly household data with ``gen_kwh`` and ``cons_kwh`` columns.
    capacity_kwh : float
        Usable battery capacity in kWh.
    power_kw : float | None
        Max charge/discharge power in kW. Defaults to ``capacity_kwh / 2``.
    import_tariff : float | None
        Grid import price (INR/kWh). Defaults to ``config.IMPORT_TARIFF``.
    export_tariff : float | None
        Solar export credit (INR/kWh). Defaults to ``config.EXPORT_TARIFF``.
    """
    cap = max(float(capacity_kwh), 0.0)
    pwr = float(power_kw) if power_kw is not None else cap / 2.0
    pwr = max(pwr, 0.0)
    imp_rate = IMPORT_TARIFF if import_tariff is None else float(import_tariff)
    exp_rate = EXPORT_TARIFF if export_tariff is None else float(export_tariff)
    eff = float(BATTERY_EFFICIENCY)

    work, scale = _prepare_df(df)
    if work.empty:
        return {
            "capacity_kwh": _r2(cap),
            "power_kw": _r2(pwr),
            "import_before_kwh": 0.0,
            "import_after_kwh": 0.0,
            "import_avoided_kwh": 0.0,
            "export_before_kwh": 0.0,
            "export_after_kwh": 0.0,
            "savings_inr_year": 0.0,
            "self_consumption_pct_before": 0.0,
            "self_consumption_pct_after": 0.0,
            "cost_inr": _r2(cap * BATTERY_COST_PER_KWH),
            "payback_years": 0.0 if cap == 0 else 999.0,
        }

    gen_arr = work["gen_kwh"].to_numpy(dtype=float)
    cons_arr = work["cons_kwh"].to_numpy(dtype=float)

    surplus_arr = gen_arr - cons_arr
    gen_total = float(gen_arr.sum())
    imp_old = float(np.maximum(-surplus_arr, 0.0).sum())
    exp_old = float(np.maximum(surplus_arr, 0.0).sum())

    if cap <= 0.0 or pwr <= 0.0:
        imp_new = imp_old
        exp_new = exp_old
    else:
        soc = 0.0
        step_kwh = pwr * 0.5
        imp_new = 0.0
        exp_new = 0.0
        for surplus in surplus_arr:
            if surplus > 0.0:
                charge = min(surplus, step_kwh, (cap - soc) / eff)
                soc += charge * eff
                exp_new += surplus - charge
            else:
                need = -surplus
                dis = min(need, step_kwh, soc)
                soc -= dis
                imp_new += need - dis

    imp_before_yr = imp_old * scale
    imp_after_yr = imp_new * scale
    exp_before_yr = exp_old * scale
    exp_after_yr = exp_new * scale
    import_avoided_yr = max(imp_before_yr - imp_after_yr, 0.0)

    savings_yr = (imp_before_yr - imp_after_yr) * imp_rate - (exp_before_yr - exp_after_yr) * exp_rate
    cost_inr = cap * BATTERY_COST_PER_KWH
    if cap == 0.0:
        payback_years = 0.0
    elif savings_yr <= 0.0:
        payback_years = 999.0
    else:
        payback_years = cost_inr / savings_yr

    sc_before = _self_consumption_pct(gen_total, exp_old)
    sc_after = _self_consumption_pct(gen_total, exp_new)

    return {
        "capacity_kwh": _r2(cap),
        "power_kw": _r2(pwr),
        "import_before_kwh": _r2(imp_before_yr),
        "import_after_kwh": _r2(imp_after_yr),
        "import_avoided_kwh": _r2(import_avoided_yr),
        "export_before_kwh": _r2(exp_before_yr),
        "export_after_kwh": _r2(exp_after_yr),
        "savings_inr_year": _r2(savings_yr),
        "self_consumption_pct_before": _r2(sc_before),
        "self_consumption_pct_after": _r2(sc_after),
        "cost_inr": _r2(cost_inr),
        "payback_years": _r2(payback_years),
    }


def simulate_extra_pv(
    df: pd.DataFrame,
    extra_kw: float,
    pv_kw: float | None = None,
    import_tariff: float | None = None,
    export_tariff: float | None = None,
) -> dict:
    """Simulate adding extra rooftop solar PV capacity (kW).

    Scales historical generation by ``(base_pv_kw + extra_kw) / base_pv_kw``
    when existing generation is present, then recomputes grid import, solar
    export, savings, and self-consumption percentage.
    """
    extra = max(float(extra_kw), 0.0)
    imp_rate = IMPORT_TARIFF if import_tariff is None else float(import_tariff)
    exp_rate = EXPORT_TARIFF if export_tariff is None else float(export_tariff)

    work, scale = _prepare_df(df)
    if work.empty:
        return {
            "extra_kw": _r2(extra),
            "base_pv_kw": _r2(pv_kw or 0.0),
            "gen_before_kwh": 0.0,
            "gen_after_kwh": 0.0,
            "extra_gen_kwh": 0.0,
            "extra_gen_kwh_year": 0.0,
            "import_before_kwh": 0.0,
            "import_after_kwh": 0.0,
            "import_avoided_kwh": 0.0,
            "export_before_kwh": 0.0,
            "export_after_kwh": 0.0,
            "extra_export_kwh": 0.0,
            "savings_inr_year": 0.0,
            "self_consumption_pct_before": 0.0,
            "self_consumption_pct_after": 0.0,
            "self_consumption_change_pct": 0.0,
            "self_sufficiency_pct_after": 0.0,
            "cost_inr": _r2(extra * PV_COST_PER_KW),
            "payback_years": 0.0 if extra == 0 else 999.0,
        }

    gen_arr = work["gen_kwh"].to_numpy(dtype=float)
    cons_arr = work["cons_kwh"].to_numpy(dtype=float)

    if pv_kw is not None and float(pv_kw) > 0.0:
        base_pv = float(pv_kw)
    elif "pv_kw" in work.columns and pd.to_numeric(work["pv_kw"], errors="coerce").dropna().gt(0).any():
        base_pv = float(pd.to_numeric(work["pv_kw"], errors="coerce").dropna().iloc[0])
    else:
        peak_gen_kw = float(gen_arr.max()) * 2.0
        base_pv = max(peak_gen_kw, 0.5) if gen_arr.sum() > 0 else 0.0

    if extra <= 0.0:
        gen_new_arr = gen_arr
    elif base_pv > 0.0 and gen_arr.sum() > 0.0:
        factor = (base_pv + extra) / base_pv
        gen_new_arr = gen_arr * factor
    else:
        # Fallback if a home has zero existing solar generation: ~4 kWh/kW/day over 08:00-16:00
        if "ts" in work.columns:
            hours = work["ts"].dt.hour
            daylight = ((hours >= 8) & (hours < 16)).to_numpy(dtype=float)
            slots_per_day = 16.0
            gen_new_arr = daylight * (extra * 4.0 / slots_per_day)
        else:
            gen_new_arr = np.full(len(work), extra * 4.0 / 48.0)

    imp_old = float(np.maximum(cons_arr - gen_arr, 0.0).sum())
    exp_old = float(np.maximum(gen_arr - cons_arr, 0.0).sum())
    imp_new = float(np.maximum(cons_arr - gen_new_arr, 0.0).sum())
    exp_new = float(np.maximum(gen_new_arr - cons_arr, 0.0).sum())

    gen_before_yr = float(gen_arr.sum()) * scale
    gen_after_yr = float(gen_new_arr.sum()) * scale
    imp_before_yr = imp_old * scale
    imp_after_yr = imp_new * scale
    exp_before_yr = exp_old * scale
    exp_after_yr = exp_new * scale

    savings_yr = (imp_before_yr - imp_after_yr) * imp_rate + (exp_after_yr - exp_before_yr) * exp_rate
    cost_inr = extra * PV_COST_PER_KW
    if extra == 0.0:
        payback_years = 0.0
    elif savings_yr <= 0.0:
        payback_years = 999.0
    else:
        payback_years = cost_inr / savings_yr

    sc_before = _self_consumption_pct(gen_before_yr, exp_before_yr)
    sc_after = _self_consumption_pct(gen_after_yr, exp_after_yr)
    cons_total_yr = float(cons_arr.sum()) * scale
    self_after_yr = max(gen_after_yr - exp_after_yr, 0.0)
    ss_after = min(max((self_after_yr / cons_total_yr * 100.0) if cons_total_yr > 0.0 else 0.0, 0.0), 100.0)
    extra_gen_yr = max(gen_after_yr - gen_before_yr, 0.0)

    return {
        "extra_kw": _r2(extra),
        "base_pv_kw": _r2(base_pv),
        "gen_before_kwh": _r2(gen_before_yr),
        "gen_after_kwh": _r2(gen_after_yr),
        "extra_gen_kwh": _r2(extra_gen_yr),
        "extra_gen_kwh_year": _r2(extra_gen_yr),
        "import_before_kwh": _r2(imp_before_yr),
        "import_after_kwh": _r2(imp_after_yr),
        "import_avoided_kwh": _r2(max(imp_before_yr - imp_after_yr, 0.0)),
        "export_before_kwh": _r2(exp_before_yr),
        "export_after_kwh": _r2(exp_after_yr),
        "extra_export_kwh": _r2(max(exp_after_yr - exp_before_yr, 0.0)),
        "savings_inr_year": _r2(savings_yr),
        "self_consumption_pct_before": _r2(sc_before),
        "self_consumption_pct_after": _r2(sc_after),
        "self_consumption_change_pct": _r2(sc_after - sc_before),
        "self_sufficiency_pct_after": _r2(ss_after),
        "cost_inr": _r2(cost_inr),
        "payback_years": _r2(payback_years),
    }


def simulate_load_shift(
    df: pd.DataFrame,
    kwh_per_day: float,
    from_hours: tuple[int, int] = (18, 22),
    to_hours: tuple[int, int] = (10, 14),
    import_tariff: float | None = None,
    export_tariff: float | None = None,
) -> dict:
    """Simulate shifting flexible appliance load from evening to midday solar hours.

    Each day moves ``min(kwh_per_day, evening_import, midday_surplus)`` from
    ``from_hours`` to ``to_hours``.
    """
    req_kwh = max(float(kwh_per_day), 0.0)
    imp_rate = IMPORT_TARIFF if import_tariff is None else float(import_tariff)
    exp_rate = EXPORT_TARIFF if export_tariff is None else float(export_tariff)

    work, scale = _prepare_df(df)
    if work.empty or "ts" not in work.columns:
        return {
            "kwh_per_day": _r2(req_kwh),
            "avg_daily_shifted_kwh": 0.0,
            "import_before_kwh": 0.0,
            "import_after_kwh": 0.0,
            "import_avoided_kwh": 0.0,
            "shifted_kwh_year": 0.0,
            "export_before_kwh": 0.0,
            "export_after_kwh": 0.0,
            "savings_inr_year": 0.0,
            "savings_inr_month": 0.0,
            "self_consumption_pct_before": 0.0,
            "self_consumption_pct_after": 0.0,
        }

    work["date"] = work["ts"].dt.date
    work["hour"] = work["ts"].dt.hour
    work["imp"] = np.maximum(work["cons_kwh"] - work["gen_kwh"], 0.0)
    work["exp"] = np.maximum(work["gen_kwh"] - work["cons_kwh"], 0.0)

    from_mask = (work["hour"] >= from_hours[0]) & (work["hour"] < from_hours[1])
    to_mask = (work["hour"] >= to_hours[0]) & (work["hour"] < to_hours[1])

    eve_imp_by_day = work.loc[from_mask].groupby("date")["imp"].sum()
    mid_exp_by_day = work.loc[to_mask].groupby("date")["exp"].sum()
    all_dates = work["date"].unique()

    shifted_total = 0.0
    if req_kwh > 0.0:
        for d in all_dates:
            eve_imp = float(eve_imp_by_day.get(d, 0.0))
            mid_exp = float(mid_exp_by_day.get(d, 0.0))
            shifted_total += min(req_kwh, eve_imp, mid_exp)

    gen_total = float(work["gen_kwh"].sum())
    imp_old = float(work["imp"].sum())
    exp_old = float(work["exp"].sum())
    imp_new = max(imp_old - shifted_total, 0.0)
    exp_new = max(exp_old - shifted_total, 0.0)

    n_days = max(len(all_dates), 1)
    avg_daily_shifted = shifted_total / n_days

    imp_before_yr = imp_old * scale
    imp_after_yr = imp_new * scale
    exp_before_yr = exp_old * scale
    exp_after_yr = exp_new * scale
    shifted_yr = shifted_total * scale

    savings_yr = shifted_yr * (imp_rate - exp_rate)
    sc_before = _self_consumption_pct(gen_total, exp_old)
    sc_after = _self_consumption_pct(gen_total, exp_new)

    return {
        "kwh_per_day": _r2(req_kwh),
        "avg_daily_shifted_kwh": _r2(avg_daily_shifted),
        "import_before_kwh": _r2(imp_before_yr),
        "import_after_kwh": _r2(imp_after_yr),
        "import_avoided_kwh": _r2(shifted_yr),
        "shifted_kwh_year": _r2(shifted_yr),
        "export_before_kwh": _r2(exp_before_yr),
        "export_after_kwh": _r2(exp_after_yr),
        "savings_inr_year": _r2(savings_yr),
        "savings_inr_month": _r2(savings_yr / 12.0),
        "self_consumption_pct_before": _r2(sc_before),
        "self_consumption_pct_after": _r2(sc_after),
    }


def simulate_ev(
    df: pd.DataFrame,
    km_per_day: float,
    charge_hours: tuple[int, int] = (10, 15),
    import_tariff: float | None = None,
    export_tariff: float | None = None,
) -> dict:
    """Simulate adding an electric vehicle charged during ``charge_hours``.

    Assumes 0.15 kWh/km distributed evenly across ``charge_hours`` each day.
    """
    km = max(float(km_per_day), 0.0)
    daily_ev_kwh = km * EV_KWH_PER_KM
    imp_rate = IMPORT_TARIFF if import_tariff is None else float(import_tariff)
    exp_rate = EXPORT_TARIFF if export_tariff is None else float(export_tariff)

    work, scale = _prepare_df(df)
    if work.empty or "ts" not in work.columns:
        return {
            "km_per_day": _r2(km),
            "kwh_per_km": EV_KWH_PER_KM,
            "ev_demand_day_kwh": _r2(daily_ev_kwh),
            "ev_demand_kwh": 0.0,
            "ev_kwh_year": 0.0,
            "solar_hours_demand_kwh": 0.0,
            "ev_from_solar_kwh": 0.0,
            "ev_from_grid_kwh": 0.0,
            "extra_import_kwh_year": 0.0,
            "solar_coverage_pct": 0.0,
            "solar_share_pct": 0.0,
            "grid_cost_inr_year": 0.0,
            "additional_cost_inr_year": 0.0,
            "extra_cost_inr_year": 0.0,
            "additional_cost_inr_month": 0.0,
        }

    work["date"] = work["ts"].dt.date
    work["hour"] = work["ts"].dt.hour

    charge_mask = (work["hour"] >= charge_hours[0]) & (work["hour"] < charge_hours[1])
    slots_per_day = work.loc[charge_mask].groupby("date")["ts"].transform("count")

    ev_added = np.zeros(len(work), dtype=float)
    if daily_ev_kwh > 0.0 and charge_mask.any():
        ev_added[charge_mask.to_numpy()] = daily_ev_kwh / slots_per_day.to_numpy(dtype=float)

    gen_arr = work["gen_kwh"].to_numpy(dtype=float)
    cons_arr = work["cons_kwh"].to_numpy(dtype=float)
    cons_new = cons_arr + ev_added

    imp_old = float(np.maximum(cons_arr - gen_arr, 0.0).sum())
    exp_old = float(np.maximum(gen_arr - cons_arr, 0.0).sum())
    imp_new = float(np.maximum(cons_new - gen_arr, 0.0).sum())
    exp_new = float(np.maximum(gen_arr - cons_new, 0.0).sum())

    ev_demand_yr = float(ev_added.sum()) * scale
    solar_hours_demand_yr = float(ev_added[gen_arr > 0.0].sum()) * scale
    ev_from_solar_yr = max((exp_old - exp_new) * scale, 0.0)
    ev_from_grid_yr = max((imp_new - imp_old) * scale, 0.0)

    solar_coverage_pct = (ev_from_solar_yr / ev_demand_yr * 100.0) if ev_demand_yr > 0.0 else 0.0
    grid_cost_yr = ev_from_grid_yr * imp_rate
    # Net bill impact includes both extra grid imports and foregone solar export credit
    additional_cost_yr = ev_from_grid_yr * imp_rate + ev_from_solar_yr * exp_rate

    return {
        "km_per_day": _r2(km),
        "kwh_per_km": EV_KWH_PER_KM,
        "ev_demand_day_kwh": _r2(daily_ev_kwh),
        "ev_demand_kwh": _r2(ev_demand_yr),
        "ev_kwh_year": _r2(ev_demand_yr),
        "solar_hours_demand_kwh": _r2(solar_hours_demand_yr),
        "ev_from_solar_kwh": _r2(ev_from_solar_yr),
        "ev_from_grid_kwh": _r2(ev_from_grid_yr),
        "extra_import_kwh_year": _r2(ev_from_grid_yr),
        "solar_coverage_pct": _r2(solar_coverage_pct),
        "solar_share_pct": _r2(solar_coverage_pct),
        "grid_cost_inr_year": _r2(grid_cost_yr),
        "additional_cost_inr_year": _r2(additional_cost_yr),
        "extra_cost_inr_year": _r2(additional_cost_yr),
        "additional_cost_inr_month": _r2(additional_cost_yr / 12.0),
    }

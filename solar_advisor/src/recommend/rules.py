"""Rule-based recommendation engine for SolarSense (M3).

Implements rules R1–R6 from Team_Build_Plan.md §4.2 and returns ranked
recommendations sorted by estimated monthly INR savings (highest first).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.config import EXPORT_TARIFF, IMPORT_TARIFF
from src.recommend.simulator import simulate_battery, simulate_load_shift

MEDIAN_HOME_BASELOAD_KW = 0.26
BATTERY_CANDIDATES_KWH = (3.0, 5.0, 10.0)
BATTERY_MAX_PAYBACK_YEARS = 12.0


def _r2(val: float) -> float:
    return round(float(val), 2)


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ts", "gen_kwh", "cons_kwh", "cl_kwh"])
    work = df.copy()
    if "ts" in work.columns:
        work["ts"] = pd.to_datetime(work["ts"])
        work = work.sort_values("ts").reset_index(drop=True)
    for col in ("gen_kwh", "cons_kwh", "cl_kwh"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0).clip(lower=0.0)
        else:
            work[col] = 0.0
    return work


def _compute_fallback_metrics(work: pd.DataFrame, kpis: dict | None) -> dict:
    """Derive missing KPI metrics from half-hourly data if not already in kpis."""
    k = dict(kpis or {})
    if work.empty or "ts" not in work.columns:
        return k

    work_hours = work["ts"].dt.hour
    work_dates = work["ts"].dt.date
    n_days = max(int(work_dates.nunique()), 1)

    gen_total = float(work["gen_kwh"].sum())
    imp_arr = np.maximum(work["cons_kwh"] - work["gen_kwh"], 0.0)
    exp_arr = np.maximum(work["gen_kwh"] - work["cons_kwh"], 0.0)

    if "self_consumption_pct" not in k:
        used = max(gen_total - float(exp_arr.sum()), 0.0)
        k["self_consumption_pct"] = (used / gen_total * 100.0) if gen_total > 0 else 0.0

    if "baseload_kw" not in k:
        night = work[(work_hours >= 1) & (work_hours < 5)]
        if not night.empty:
            daily_night_mean = night.groupby(night["ts"].dt.date)["cons_kwh"].mean()
            k["baseload_kw"] = float(daily_night_mean.median()) * 2.0
        else:
            k["baseload_kw"] = 0.0

    if "peak_kw" not in k:
        k["peak_kw"] = float(np.percentile(work["cons_kwh"] * 2.0, 99))

    if "avg_daily_gen_kwh" not in k:
        k["avg_daily_gen_kwh"] = gen_total / n_days

    if "export_kwh" not in k:
        k["export_kwh"] = float(exp_arr.sum())
    if "days" not in k:
        k["days"] = n_days

    eve_mask = (work_hours >= 18) & (work_hours < 22)
    eve_imp_total = float(imp_arr[eve_mask].sum()) if eve_mask.any() else 0.0
    k["_evening_import_kwh_day"] = eve_imp_total / n_days
    k["_avg_evening_kw"] = float(work.loc[eve_mask, "cons_kwh"].mean() * 2.0) if eve_mask.any() else 0.0
    k["_avg_daily_export_kwh"] = float(exp_arr.sum()) / n_days
    k["_has_cl"] = float(work["cl_kwh"].sum()) > 0.0

    non_solar_mask = (work_hours < 8) | (work_hours >= 17)
    k["_night_cl_kwh_day"] = (
        float(work.loc[non_solar_mask, "cl_kwh"].sum()) / n_days if non_solar_mask.any() else 0.0
    )
    return k


def generate_recommendations(
    df: pd.DataFrame,
    kpis: dict | None,
    anomalies: list[dict] | None,
    import_tariff: float | None = None,
    export_tariff: float | None = None,
) -> list[dict]:
    """Evaluate recommendation rules R1–R6 and return sorted list (highest monthly INR first)."""
    imp_rate = IMPORT_TARIFF if import_tariff is None else float(import_tariff)
    exp_rate = EXPORT_TARIFF if export_tariff is None else float(export_tariff)

    work = _ensure_df(df)
    if work.empty:
        return []

    metrics = _compute_fallback_metrics(work, kpis)
    anoms = list(anomalies or [])
    recs: list[dict] = []

    # R1 – Load shifting
    sc_pct = float(metrics.get("self_consumption_pct", 0.0))
    eve_imp_day = float(metrics.get("_evening_import_kwh_day", 0.0))
    if sc_pct < 60.0 and eve_imp_day > 2.0:
        shift_res = simulate_load_shift(
            work,
            kwh_per_day=1.5,
            from_hours=(18, 22),
            to_hours=(10, 14),
            import_tariff=imp_rate,
            export_tariff=exp_rate,
        )
        if shift_res["savings_inr_month"] > 0.0:
            recs.append({
                "id": "R1",
                "title": "Shift flexible loads to midday solar",
                "category": "utilization",
                "action": "Move washing machine, dishwasher, pumps, or water heating to 10:00–14:00 when solar surplus is high.",
                "kwh_saved_month": _r2(shift_res["import_avoided_kwh"] / 12.0),
                "inr_saved_month": _r2(shift_res["savings_inr_month"]),
                "effort": "low",
                "evidence": {
                    "self_consumption_pct": _r2(sc_pct),
                    "evening_import_kwh_day": _r2(eve_imp_day),
                    "avg_daily_shifted_kwh": _r2(shift_res["avg_daily_shifted_kwh"]),
                },
            })

    # R2 – Night load
    baseload_kw = float(metrics.get("baseload_kw", 0.0))
    baseload_threshold = max(0.4, 1.5 * MEDIAN_HOME_BASELOAD_KW)
    if baseload_kw > 0.4 or baseload_kw > 1.5 * MEDIAN_HOME_BASELOAD_KW:
        excess_kw = max(baseload_kw - MEDIAN_HOME_BASELOAD_KW, 0.0)
        kwh_month = excess_kw * 24.0 * 30.0
        inr_month = kwh_month * imp_rate
        recs.append({
            "id": "R2",
            "title": "Cut always-on night load",
            "category": "inefficiency",
            "action": "Inspect devices left running overnight (older refrigerators, pool/water pumps, standby chargers, set-top boxes).",
            "kwh_saved_month": _r2(kwh_month),
            "inr_saved_month": _r2(inr_month),
            "effort": "low",
            "evidence": {
                "baseload_kw": _r2(baseload_kw),
                "median_home_baseload_kw": _r2(MEDIAN_HOME_BASELOAD_KW),
                "threshold_kw": _r2(baseload_threshold),
            },
        })

    # R3 – Solar panel check (low_generation anomaly in the last 60 days of dataset)
    latest_date = work["ts"].max().date() if "ts" in work.columns else None
    cutoff_date = latest_date - timedelta(days=60) if latest_date is not None else None
    low_gen_recent: list[dict] = []
    for a in anoms:
        if a.get("type") != "low_generation":
            continue
        raw_date = a.get("date")
        if cutoff_date is not None and raw_date:
            try:
                d = pd.to_datetime(raw_date).date()
                if d < cutoff_date:
                    continue
            except (ValueError, TypeError):
                pass
        low_gen_recent.append(a)

    if low_gen_recent:
        avg_daily_gen = float(metrics.get("avg_daily_gen_kwh", 0.0))
        daily_losses: list[float] = []
        for a in low_gen_recent:
            expected = float(a.get("expected", 0.0) or 0.0)
            actual = float(a.get("value", 0.0) or 0.0)
            unit = str(a.get("unit", "")).lower()
            gap = max(expected - actual, 0.0)
            if unit == "kwh" or expected > 2.0:
                daily_losses.append(gap)
            else:
                # yield_ratio-based anomaly (e.g. expected ~1.0, actual ~0.6)
                daily_losses.append(gap * max(avg_daily_gen, 0.0))

        avg_daily_loss = float(np.mean(daily_losses)) if daily_losses else 0.0
        kwh_month = avg_daily_loss * 30.0
        sc_frac = min(max(sc_pct / 100.0, 0.0), 1.0)
        blended_rate = sc_frac * imp_rate + (1.0 - sc_frac) * exp_rate
        inr_month = kwh_month * blended_rate
        latest_anom_date = max(str(a.get("date", "")) for a in low_gen_recent)

        recs.append({
            "id": "R3",
            "title": "Clean or inspect solar panels",
            "category": "inefficiency",
            "action": "Inspect panels for dust, shading, bird droppings, or inverter faults causing sustained low solar yield.",
            "kwh_saved_month": _r2(kwh_month),
            "inr_saved_month": _r2(inr_month),
            "effort": "medium",
            "evidence": {
                "low_gen_anomalies_60d": int(len(low_gen_recent)),
                "latest_anomaly_date": latest_anom_date,
                "estimated_daily_loss_kwh": _r2(avg_daily_loss),
            },
        })

    # R4 – Hot water (controlled load + exports > 3 kWh/day)
    has_cl = bool(metrics.get("_has_cl", False))
    avg_daily_export = float(metrics.get("_avg_daily_export_kwh", 0.0))
    night_cl_day = float(metrics.get("_night_cl_kwh_day", 0.0))
    if has_cl and avg_daily_export > 3.0 and night_cl_day > 0.0:
        shiftable_day = min(night_cl_day, avg_daily_export)
        kwh_month = shiftable_day * 30.0
        inr_month = kwh_month * max(imp_rate - exp_rate, 0.0)
        recs.append({
            "id": "R4",
            "title": "Shift controlled-load hot water to midday",
            "category": "utilization",
            "action": "Re-time electric hot-water heating from overnight controlled load to midday solar surplus hours.",
            "kwh_saved_month": _r2(kwh_month),
            "inr_saved_month": _r2(inr_month),
            "effort": "low",
            "evidence": {
                "avg_daily_export_kwh": _r2(avg_daily_export),
                "night_cl_kwh_day": _r2(night_cl_day),
                "shiftable_kwh_day": _r2(shiftable_day),
            },
        })

    # R5 – Battery (evaluate 3, 5, 10 kWh; recommend best if payback < 12 years)
    evaluated_batteries = [
        simulate_battery(
            work,
            capacity_kwh=cap,
            import_tariff=imp_rate,
            export_tariff=exp_rate,
        )
        for cap in BATTERY_CANDIDATES_KWH
    ]
    viable = [
        b for b in evaluated_batteries
        if 0.0 < b["payback_years"] < BATTERY_MAX_PAYBACK_YEARS and b["savings_inr_year"] > 0.0
    ]
    if viable:
        best = min(viable, key=lambda b: (b["payback_years"], -b["savings_inr_year"]))
        recs.append({
            "id": "R5",
            "title": f"Install a {int(best['capacity_kwh'])} kWh home battery",
            "category": "planning",
            "action": (
                f"Consider adding a {best['capacity_kwh']} kWh ({best['power_kw']} kW) battery to store excess "
                f"daytime solar for evening use (estimated payback {best['payback_years']} years)."
            ),
            "kwh_saved_month": _r2(best["import_avoided_kwh"] / 12.0),
            "inr_saved_month": _r2(best["savings_inr_year"] / 12.0),
            "effort": "high",
            "evidence": {
                "recommended_capacity_kwh": _r2(best["capacity_kwh"]),
                "payback_years": _r2(best["payback_years"]),
                "cost_inr": _r2(best["cost_inr"]),
                "self_consumption_pct_after": _r2(best["self_consumption_pct_after"]),
                "options_evaluated": [
                    {
                        "capacity_kwh": b["capacity_kwh"],
                        "payback_years": b["payback_years"],
                        "savings_inr_year": b["savings_inr_year"],
                    }
                    for b in evaluated_batteries
                ],
            },
        })

    # R6 – Peak spikes
    peak_kw = float(metrics.get("peak_kw", 0.0))
    avg_evening_kw = float(metrics.get("_avg_evening_kw", 0.0))
    if avg_evening_kw > 0.0 and peak_kw > 2.5 * avg_evening_kw:
        est_peak_reduction = max(peak_kw - 2.0 * avg_evening_kw, 0.0)
        recs.append({
            "id": "R6",
            "title": "Stagger heavy evening appliances",
            "category": "inefficiency",
            "action": (
                f"Avoid running multiple high-draw appliances together during evening hours; staggering loads "
                f"can reduce peak demand by roughly {_r2(est_peak_reduction)} kW."
            ),
            "kwh_saved_month": 0.0,
            "inr_saved_month": 0.0,
            "effort": "low",
            "evidence": {
                "peak_kw": _r2(peak_kw),
                "avg_evening_kw": _r2(avg_evening_kw),
                "peak_to_evening_ratio": _r2(peak_kw / avg_evening_kw),
                "estimated_peak_reduction_kw": _r2(est_peak_reduction),
            },
        })

    recs.sort(key=lambda r: (-r["inr_saved_month"], -r["kwh_saved_month"], r["id"]))
    return recs

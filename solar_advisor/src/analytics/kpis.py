"""KPI engine, daily load/generation profiles, and monthly energy summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import EXPORT_TARIFF, IMPORT_TARIFF

DAILY_PROFILE_COLUMNS = ["hour", "gen_kw", "cons_kw"]
MONTHLY_SUMMARY_COLUMNS = [
    "month",
    "gen_kwh",
    "cons_kwh",
    "import_kwh",
    "export_kwh",
    "self_consumption_pct",
]


def _empty_kpis() -> dict:
    """Return a zeroed, JSON-compatible KPI dictionary matching Section 1.4."""
    return {
        "period": "",
        "days": 0,
        "gen_kwh": 0.0,
        "cons_kwh": 0.0,
        "self_kwh": 0.0,
        "import_kwh": 0.0,
        "export_kwh": 0.0,
        "self_consumption_pct": 0.0,
        "self_sufficiency_pct": 0.0,
        "baseload_kw": 0.0,
        "peak_kw": 0.0,
        "peak_hour": 0,
        "avg_daily_gen_kwh": 0.0,
        "avg_daily_cons_kwh": 0.0,
        "grid_cost_inr": 0.0,
        "export_credit_inr": 0.0,
        "net_bill_inr": 0.0,
    }


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute household solar and consumption KPIs conforming to Section 1.4."""
    if df is None or df.empty:
        return _empty_kpis()

    ts = pd.to_datetime(df["ts"])
    ts_min = ts.min()
    ts_max = ts.max()
    days = int(ts.dt.normalize().nunique())

    gen_kwh = float(df["gen_kwh"].sum())
    cons_kwh = float(df["cons_kwh"].sum())
    self_kwh = float(df["self_kwh"].sum())
    import_kwh = float(df["import_kwh"].sum())
    export_kwh = float(df["export_kwh"].sum())

    self_consumption_pct = (self_kwh / gen_kwh * 100.0) if gen_kwh > 0 else 0.0
    self_sufficiency_pct = (self_kwh / cons_kwh * 100.0) if cons_kwh > 0 else 0.0

    # Baseload: median over days of mean general consumption (gc_kwh) in [01:00, 05:00),
    # * 2 for kWh/30m -> kW. Controlled load (off-peak hot water) is scheduled, not
    # always-on, so it is left out; this matches DATASET_MEDIAN_BASELOAD_KW (0.26 kW).
    hours = ts.dt.hour
    night_mask = (hours >= 1) & (hours < 5)
    base_col = "gc_kwh" if "gc_kwh" in df.columns else "cons_kwh"
    if night_mask.any():
        night_daily_mean = (
            df.loc[night_mask, base_col]
            .groupby(ts.loc[night_mask].dt.normalize())
            .mean()
        )
        baseload_kw = float(night_daily_mean.median() * 2.0)
    else:
        baseload_kw = 0.0

    # Peak power: 99th percentile of half-hourly kW (cons_kwh * 2)
    peak_kw = float(np.percentile(df["cons_kwh"].to_numpy(dtype=float) * 2.0, 99))

    # Peak hour: hour of day with highest average consumption
    hourly_cons = df["cons_kwh"].groupby(hours).mean()
    peak_hour = int(hourly_cons.idxmax())

    avg_daily_gen_kwh = (gen_kwh / days) if days > 0 else 0.0
    avg_daily_cons_kwh = (cons_kwh / days) if days > 0 else 0.0

    grid_cost_inr = round(import_kwh * float(IMPORT_TARIFF), 2)
    export_credit_inr = round(export_kwh * float(EXPORT_TARIFF), 2)
    net_bill_inr = round(grid_cost_inr - export_credit_inr, 2)

    return {
        "period": f"{ts_min.strftime('%Y-%m-%d')} to {ts_max.strftime('%Y-%m-%d')}",
        "days": int(days),
        "gen_kwh": round(gen_kwh, 2),
        "cons_kwh": round(cons_kwh, 2),
        "self_kwh": round(self_kwh, 2),
        "import_kwh": round(import_kwh, 2),
        "export_kwh": round(export_kwh, 2),
        "self_consumption_pct": round(float(self_consumption_pct), 2),
        "self_sufficiency_pct": round(float(self_sufficiency_pct), 2),
        "baseload_kw": round(baseload_kw, 2),
        "peak_kw": round(peak_kw, 2),
        "peak_hour": int(peak_hour),
        "avg_daily_gen_kwh": round(float(avg_daily_gen_kwh), 2),
        "avg_daily_cons_kwh": round(float(avg_daily_cons_kwh), 2),
        "grid_cost_inr": float(grid_cost_inr),
        "export_credit_inr": float(export_credit_inr),
        "net_bill_inr": float(net_bill_inr),
    }


def daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return average hourly generation and consumption in kW (``hour, gen_kw, cons_kw``)."""
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "hour": pd.Series(dtype="int64"),
                "gen_kw": pd.Series(dtype="float64"),
                "cons_kw": pd.Series(dtype="float64"),
            }
        )

    ts = pd.to_datetime(df["ts"])
    grouped = (
        pd.DataFrame(
            {
                "hour": ts.dt.hour.astype("int64"),
                "gen_kw": df["gen_kwh"].astype("float64") * 2.0,
                "cons_kw": df["cons_kwh"].astype("float64") * 2.0,
            }
        )
        .groupby("hour", as_index=False)[["gen_kw", "cons_kw"]]
        .mean()
        .sort_values("hour")
        .reset_index(drop=True)
    )
    grouped["gen_kw"] = grouped["gen_kw"].round(2)
    grouped["cons_kw"] = grouped["cons_kw"].round(2)
    return grouped[DAILY_PROFILE_COLUMNS]


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return monthly totals (``month, gen_kwh, cons_kwh, import_kwh, export_kwh, self_consumption_pct``)."""
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "month": pd.Series(dtype="str"),
                "gen_kwh": pd.Series(dtype="float64"),
                "cons_kwh": pd.Series(dtype="float64"),
                "import_kwh": pd.Series(dtype="float64"),
                "export_kwh": pd.Series(dtype="float64"),
                "self_consumption_pct": pd.Series(dtype="float64"),
            }
        )

    ts = pd.to_datetime(df["ts"])
    work = pd.DataFrame(
        {
            "month": ts.dt.to_period("M").astype(str),
            "gen_kwh": df["gen_kwh"].astype("float64"),
            "cons_kwh": df["cons_kwh"].astype("float64"),
            "self_kwh": df["self_kwh"].astype("float64"),
            "import_kwh": df["import_kwh"].astype("float64"),
            "export_kwh": df["export_kwh"].astype("float64"),
        }
    )
    monthly = (
        work.groupby("month", as_index=False)[
            ["gen_kwh", "cons_kwh", "self_kwh", "import_kwh", "export_kwh"]
        ]
        .sum()
        .sort_values("month")
        .reset_index(drop=True)
    )
    monthly["self_consumption_pct"] = np.where(
        monthly["gen_kwh"] > 0,
        monthly["self_kwh"] / monthly["gen_kwh"] * 100.0,
        0.0,
    )
    for col in ("gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_consumption_pct"):
        monthly[col] = monthly[col].round(2)

    return monthly[MONTHLY_SUMMARY_COLUMNS]


if __name__ == "__main__":
    from src.pipeline.load import load_home

    home1 = load_home(1)
    kpis = compute_kpis(home1)
    profile = daily_profile(home1)
    monthly = monthly_summary(home1)
    print("Home 1 KPIs:", kpis)
    print("\nDaily profile (head):\n", profile.head().to_string(index=False))
    print("\nMonthly summary (tail):\n", monthly.tail().to_string(index=False))

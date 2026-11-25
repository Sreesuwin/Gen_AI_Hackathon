"""KPIs and profiles.   OWNER: Shashikiran

STUB: returns the agreed example shapes so others can build now. Replace the bodies;
keep the signatures and keys (src/contracts.py, checked by tests/test_contracts.py).
"""
import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict:
    # STUB - fixed example values
    return {"period": "2012-07-01 to 2013-06-30", "days": 365,
            "gen_kwh": 2180.4, "cons_kwh": 6391.2, "self_kwh": 1146.3, "import_kwh": 5244.9, "export_kwh": 1034.1,
            "self_consumption_pct": 52.6, "self_sufficiency_pct": 17.9,
            "baseload_kw": 0.26, "peak_kw": 3.8, "peak_hour": 18,
            "avg_daily_gen_kwh": 5.97, "avg_daily_cons_kwh": 17.51,
            "grid_cost_inr": 41959.2, "export_credit_inr": 3102.3, "net_bill_inr": 38856.9}


def daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Columns: hour (0-23), gen_kw, cons_kw - the average day."""
    # this one is simple enough to be real already
    d = df.assign(hour=df["ts"].dt.hour)
    out = d.groupby("hour")[["gen_kwh", "cons_kwh"]].mean() * 2      # kWh per half hour -> kW
    return out.rename(columns={"gen_kwh": "gen_kw", "cons_kwh": "cons_kw"}).reset_index().round(3)


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Columns: month (YYYY-MM), gen_kwh, cons_kwh, import_kwh, export_kwh, self_consumption_pct."""
    m = df.set_index("ts")[["gen_kwh", "cons_kwh", "self_kwh", "import_kwh", "export_kwh"]].resample("MS").sum()
    m["self_consumption_pct"] = (100 * m["self_kwh"] / m["gen_kwh"]).round(1)
    m = m.drop(columns="self_kwh").round(1).reset_index()
    m["month"] = m["ts"].dt.strftime("%Y-%m")
    return m.drop(columns="ts")[["month", "gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_consumption_pct"]]

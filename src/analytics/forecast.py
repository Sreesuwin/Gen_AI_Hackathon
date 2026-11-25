"""Forecasting.   OWNER: Shashikiran

Review 2: seasonal baseline (mean of same weekday over the last 4 weeks).
Review 3: Chronos-2 (or chronos-bolt-small) on daily totals, CPU is fine. Always set "method".
"""
import pandas as pd


def forecast(df: pd.DataFrame, horizon_days: int = 14, target: str = "cons") -> dict:
    # STUB - flat example values
    dates = pd.date_range("2013-07-01", periods=horizon_days, freq="D").strftime("%Y-%m-%d").tolist()
    p50 = [17.25] * horizon_days
    return {"target": target, "horizon_days": horizon_days, "dates": dates,
            "p10": [14.1] * horizon_days, "p50": p50, "p90": [20.8] * horizon_days,
            "total_p50_kwh": round(sum(p50), 2), "method": "seasonal_baseline"}

"""
src/analytics/forecast.py
=========================
M2 Task 3 – Energy Forecasting for SolarSense (PS15).

Public API
----------
forecast(df, horizon_days=14, target="cons") -> dict
    Forecast daily energy totals for the next `horizon_days` days.

    Parameters
    ----------
    df : pd.DataFrame
        Half-hourly data in the standard schema (columns: ts, gen_kwh, cons_kwh, …).
        Must contain at least one day of history.
    horizon_days : int
        Number of future calendar days to forecast (default 14).
    target : str
        "cons" for electricity consumption, "gen" for solar generation.

    Returns
    -------
    dict with keys:
        target          : str    – "cons" or "gen"
        horizon_days    : int    – as supplied
        dates           : list[str] – ISO-8601 date strings (YYYY-MM-DD)
        p10             : list[float] – 10th-percentile forecast (rounded to 2 dp)
        p50             : list[float] – median / best-estimate forecast
        p90             : list[float] – 90th-percentile forecast
        total_p50_kwh   : float  – sum of p50 (rounded to 2 dp)
        method          : str    – "chronos-2" | "seasonal_baseline"

evaluate_forecast(daily_actuals, p50_preds) -> dict
    Compute MAE, RMSE, MAPE for evaluation and reporting only.
    Not included in the forecast() return dict.

Notes
-----
* Chronos-2 is attempted first; if torch is blocked by WDAC (ImportError / OSError)
  the function falls back silently to the seasonal-baseline implementation.
* All returned numbers are native Python floats/ints/strings (JSON-serialisable).
* Values are rounded to at most 2 decimal places per the data-contract in §1.3.

Anomaly limitation note (Option A, approved 2026-11-25)
---------------------------------------------------------
Anomaly detection flags 510 low-generation events = 47.09 % of eligible days.
The 0.75 yield-ratio threshold lies near the global median (0.7515).
Flags indicate possible underperformance but do not independently confirm
equipment faults — seasonal dimming is the most common cause.
"""

from __future__ import annotations

import functools
import math
import warnings
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _aggregate_daily(df: pd.DataFrame, target: str) -> pd.Series:
    """Return a date-indexed Series of daily totals for *target*.

    NaN half-hourly values are treated as 0 before summing.
    """
    if df.empty:
        return pd.Series(dtype=float, name=target)

    col = "cons_kwh" if target == "cons" else "gen_kwh"
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    work = df[["ts", col]].copy()
    work[col] = work[col].fillna(0.0)
    work["date"] = pd.to_datetime(work["ts"]).dt.normalize()
    daily = work.groupby("date")[col].sum()
    daily.index = pd.DatetimeIndex(daily.index).normalize()
    daily.name = target
    return daily


def _seasonal_baseline(
    daily: pd.Series,
    horizon_days: int,
) -> tuple[List[str], List[float], List[float], List[float]]:
    """Seasonal-baseline forecast.

    For each future day d (1 … horizon_days):
        - Identify target weekday = (last_date + d).weekday()
        - Collect the 4 most recent historical days with the same weekday,
          looking back up to 112 days (4 × 28).
        - p50 = mean of collected values
        - p10 = min  of collected values
        - p90 = max  of collected values
        - If fewer than 4 days exist use all available.
        - If 0 days exist fall back to global mean.

    Returns
    -------
    dates  : list[str]   – ISO date strings
    p10    : list[float]
    p50    : list[float]
    p90    : list[float]
    """
    daily = daily.sort_index()
    last_date: date = daily.index[-1].date()
    global_mean = float(daily.mean()) if len(daily) > 0 else 0.0

    dates_out: List[str] = []
    p10_out:   List[float] = []
    p50_out:   List[float] = []
    p90_out:   List[float] = []

    for d in range(1, horizon_days + 1):
        future_date = last_date + timedelta(days=d)
        target_wd = future_date.weekday()          # 0=Monday … 6=Sunday
        dates_out.append(future_date.isoformat())

        # Restrict look-back to 112 days (4 × 28) for efficiency
        lookback_start = pd.Timestamp(last_date - timedelta(days=112))
        window = daily[daily.index >= lookback_start]

        # Filter to matching weekday
        same_wd = window[window.index.day_of_week == target_wd]

        # Keep the 4 most recent
        obs = same_wd.sort_index().iloc[-4:].values

        if len(obs) == 0:
            p50 = global_mean
            p10 = global_mean
            p90 = global_mean
        elif len(obs) == 1:
            p50 = float(obs[0])
            p10 = float(obs[0])
            p90 = float(obs[0])
        else:
            p50 = float(np.mean(obs))
            p10 = float(np.min(obs))
            p90 = float(np.max(obs))

        p10_out.append(round(max(p10, 0.0), 2))
        p50_out.append(round(max(p50, 0.0), 2))
        p90_out.append(round(max(p90, 0.0), 2))

    return dates_out, p10_out, p50_out, p90_out


# ---------------------------------------------------------------------------
# Chronos-2 (optional — requires torch, may be blocked by WDAC)
# ---------------------------------------------------------------------------

# Module-level cache: set to the loaded pipeline object once imported, or
# to the sentinel _CHRONOS_UNAVAILABLE if the import fails.
_CHRONOS_UNAVAILABLE = object()
_chronos_pipeline = None          # None = not yet attempted


def _load_chronos():
    """Attempt to load Chronos-2 / chronos-bolt-small. Returns pipeline or None."""
    global _chronos_pipeline
    if _chronos_pipeline is _CHRONOS_UNAVAILABLE:
        return None
    if _chronos_pipeline is not None:
        return _chronos_pipeline

    try:
        # torch must be importable before chronos
        import torch  # noqa: F401
        from chronos import BaseChronosPipeline  # type: ignore

        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cpu",
        )
        _chronos_pipeline = pipeline
        return pipeline
    except (ImportError, OSError, Exception):  # WDAC or missing package
        _chronos_pipeline = _CHRONOS_UNAVAILABLE
        return None


def _chronos_forecast(
    daily: pd.Series,
    horizon_days: int,
) -> Optional[tuple[List[str], List[float], List[float], List[float]]]:
    """Try Chronos-2 forecast. Returns None if unavailable."""
    pipeline = _load_chronos()
    if pipeline is None:
        return None

    try:
        import torch

        daily_sorted = daily.sort_index()
        last_date: date = daily_sorted.index[-1].date()

        # Use up to 365 days of context
        context_series = daily_sorted.iloc[-365:].values.astype(float)
        context_tensor = torch.tensor(context_series, dtype=torch.float32).unsqueeze(0)

        quantile_levels = [0.1, 0.5, 0.9]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            quantiles, _ = pipeline.predict_quantiles(
                context=context_tensor,
                prediction_length=horizon_days,
                quantile_levels=quantile_levels,
            )

        # quantiles shape: (1, horizon_days, 3)
        q = quantiles.squeeze(0).numpy()   # (horizon_days, 3)

        dates_out = [
            (last_date + timedelta(days=d)).isoformat()
            for d in range(1, horizon_days + 1)
        ]
        p10_out = [round(max(float(v), 0.0), 2) for v in q[:, 0]]
        p50_out = [round(max(float(v), 0.0), 2) for v in q[:, 1]]
        p90_out = [round(max(float(v), 0.0), 2) for v in q[:, 2]]

        return dates_out, p10_out, p50_out, p90_out

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def forecast(
    df: pd.DataFrame,
    horizon_days: int = 14,
    target: str = "cons",
) -> dict:
    """Forecast daily energy totals for the next *horizon_days* days.

    Parameters
    ----------
    df : pd.DataFrame
        Half-hourly data with at minimum columns ``ts`` and either
        ``cons_kwh`` (target="cons") or ``gen_kwh`` (target="gen").
    horizon_days : int
        Number of future calendar days to forecast.  Default 14.
    target : str
        ``"cons"`` for electricity consumption, ``"gen"`` for solar generation.

    Returns
    -------
    dict
        Keys: target, horizon_days, dates, p10, p50, p90,
              total_p50_kwh, method.
        All numeric values are native Python floats rounded to ≤ 2 dp.

    Raises
    ------
    ValueError
        If *target* is not "cons" or "gen", or if *df* is empty.
    """
    # --- Validate inputs ---------------------------------------------------
    if target not in ("cons", "gen"):
        raise ValueError(
            f"target must be 'cons' or 'gen', got {target!r}"
        )
    if horizon_days < 1:
        raise ValueError(f"horizon_days must be ≥ 1, got {horizon_days}")

    if df is None or (hasattr(df, "empty") and df.empty):
        raise ValueError(
            "df is empty — at least one day of history is required to forecast."
        )

    # --- Aggregate to daily totals ----------------------------------------
    daily = _aggregate_daily(df, target)

    if daily.empty:
        raise ValueError(
            "No valid rows in df after aggregation — cannot produce a forecast."
        )

    # --- Attempt Chronos-2 first ------------------------------------------
    chronos_result = _chronos_forecast(daily, horizon_days)

    if chronos_result is not None:
        dates_out, p10_out, p50_out, p90_out = chronos_result
        method = "chronos-2"
    else:
        # Seasonal baseline (always available)
        dates_out, p10_out, p50_out, p90_out = _seasonal_baseline(
            daily, horizon_days
        )
        method = "seasonal_baseline"

    total_p50 = round(sum(p50_out), 2)

    return {
        "target":        target,
        "horizon_days":  int(horizon_days),
        "dates":         dates_out,
        "p10":           p10_out,
        "p50":           p50_out,
        "p90":           p90_out,
        "total_p50_kwh": total_p50,
        "method":        method,
    }


# ---------------------------------------------------------------------------
# Evaluation helper (not part of the forecast dict)
# ---------------------------------------------------------------------------

def evaluate_forecast(
    daily_actuals: List[float],
    p50_preds: List[float],
) -> dict:
    """Compute MAE, RMSE and MAPE between actuals and p50 predictions.

    Parameters
    ----------
    daily_actuals : list[float]
        Ground-truth daily totals (kWh).
    p50_preds : list[float]
        Model p50 predictions for the same days.

    Returns
    -------
    dict
        ``mae``, ``rmse``, ``mape`` (all floats, rounded to 4 dp).
        MAPE uses zero-safe division: days where actual == 0 are excluded
        from the percentage calculation.

    Raises
    ------
    ValueError
        If the two lists have different lengths or are empty.
    """
    if len(daily_actuals) != len(p50_preds):
        raise ValueError(
            f"daily_actuals (len={len(daily_actuals)}) and "
            f"p50_preds (len={len(p50_preds)}) must have the same length."
        )
    if len(daily_actuals) == 0:
        raise ValueError("Cannot evaluate an empty forecast.")

    actuals = np.array(daily_actuals, dtype=float)
    preds   = np.array(p50_preds,     dtype=float)
    errors  = actuals - preds

    mae  = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: exclude days where actual == 0 to avoid division by zero
    nonzero_mask = actuals > 0
    if nonzero_mask.sum() == 0:
        mape = float("nan")
    else:
        mape = float(
            np.mean(np.abs(errors[nonzero_mask] / actuals[nonzero_mask])) * 100
        )

    return {
        "mae":  round(mae,  4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if not math.isnan(mape) else None,
        "n":    len(daily_actuals),
    }

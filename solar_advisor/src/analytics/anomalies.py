"""Anomaly detection for household electricity consumption and solar generation.

Follows Section 1.4 and Section 2.4 of Team_Build_Plan.md:
- Daily feature engineering (`daily_cons_kwh`, `night_load_kw`, `peak_kw`,
  `evening_share`, `yield_ratio`).
- Primary detector: `sklearn.ensemble.IsolationForest(contamination=0.03, random_state=42)`.
- Explicit opt-in fallback (`allow_fallback=True`): `PurePythonIsolationForest` when
  scikit-learn compiled extensions are blocked by OS policy.
- Median-based z-score explanations (`high_night_load`, `consumption_spike`).
- Independent 14-day rolling median `yield_ratio < 0.75` check (`low_generation`).

Limitation (documented per Option A decision, 2026-11-25)
-----------------------------------------------------------
For Home 1 (all 3 years), 510 low_generation flags = 47.09 % of eligible days.
The 0.75 yield-ratio threshold lies near the global median yield_ratio (0.7515).
Flags indicate possible underperformance but do NOT independently confirm
equipment faults — seasonal dimming is the most common cause.  Low-generation
alerts should be treated as a prompt to investigate rather than a confirmed fault.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONSUMPTION_FEATURES = [
    "daily_cons_kwh",
    "night_load_kw",
    "peak_kw",
    "evening_share",
]
MIN_DAYS_FOR_ISOLATION_FOREST = 7
ROLLING_YIELD_WINDOW = 14
LOW_GENERATION_THRESHOLD = 0.75

BACKEND_SKLEARN = "sklearn_isolation_forest"
BACKEND_FALLBACK = "pure_python_isolation_forest_fallback"


def _create_isolation_forest(allow_fallback: bool = False):
    """Return (detector, backend_name), preferring scikit-learn IsolationForest.

    Never silently substitutes the fallback: if scikit-learn cannot be imported
    and ``allow_fallback`` is False, raises RuntimeError.
    """
    try:
        from sklearn.ensemble import IsolationForest

        return IsolationForest(contamination=0.03, random_state=42), BACKEND_SKLEARN
    except (ImportError, OSError) as exc:
        if not allow_fallback:
            raise RuntimeError(
                "scikit-learn IsolationForest is unavailable "
                f"({type(exc).__name__}: {exc}). "
                "Fallback is disabled by default; pass allow_fallback=True explicitly "
                "to run PurePythonIsolationForest."
            ) from exc

        from src.analytics.isolation_forest_fallback import PurePythonIsolationForest

        return (
            PurePythonIsolationForest(
                n_estimators=100,
                max_samples=256,
                contamination=0.03,
                random_state=42,
            ),
            BACKEND_FALLBACK,
        )


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate half-hourly readings into one row per day with Section 2.4 features."""
    empty_cols = [
        "daily_cons_kwh",
        "night_load_kw",
        "peak_kw",
        "evening_share",
        "yield_ratio",
        "daily_gen_kwh",
        "p90_month_gen_kwh",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=empty_cols)

    work = df.copy()
    if "valid" in work.columns:
        work = work[work["valid"].fillna(False).astype(bool)].copy()

    if work.empty:
        return pd.DataFrame(columns=empty_cols)

    work["ts"] = pd.to_datetime(work["ts"], errors="coerce")
    work = work.dropna(subset=["ts"]).copy()
    if work.empty:
        return pd.DataFrame(columns=empty_cols)

    work["cons_kwh"] = pd.to_numeric(work["cons_kwh"], errors="coerce").fillna(0.0)
    work["gen_kwh"] = pd.to_numeric(work["gen_kwh"], errors="coerce").fillna(0.0)
    work["date"] = work["ts"].dt.normalize()
    work["hour"] = work["ts"].dt.hour

    daily_cons = work.groupby("date")["cons_kwh"].sum()
    daily_gen = work.groupby("date")["gen_kwh"].sum()

    night_mask = (work["hour"] >= 1) & (work["hour"] < 5)
    night_load_kw = (
        work.loc[night_mask].groupby("date")["cons_kwh"].mean() * 2.0
    ).reindex(daily_cons.index, fill_value=0.0)

    peak_kw = work.groupby("date")["cons_kwh"].max() * 2.0

    eve_mask = (work["hour"] >= 18) & (work["hour"] < 22)
    eve_kwh = (
        work.loc[eve_mask].groupby("date")["cons_kwh"].sum()
    ).reindex(daily_cons.index, fill_value=0.0)

    evening_share = np.where(daily_cons > 0.0, eve_kwh / daily_cons, 0.0)

    daily = pd.DataFrame(
        {
            "daily_cons_kwh": daily_cons.astype("float64"),
            "night_load_kw": night_load_kw.astype("float64"),
            "peak_kw": peak_kw.astype("float64"),
            "evening_share": pd.Series(evening_share, index=daily_cons.index, dtype="float64"),
            "daily_gen_kwh": daily_gen.astype("float64"),
        }
    ).sort_index()

    months = daily.index.month
    p90_month_gen = daily.groupby(months)["daily_gen_kwh"].transform(
        lambda s: float(s.quantile(0.90))
    )
    daily["p90_month_gen_kwh"] = p90_month_gen.astype("float64")
    daily["yield_ratio"] = np.where(
        daily["p90_month_gen_kwh"] > 0.0,
        daily["daily_gen_kwh"] / daily["p90_month_gen_kwh"],
        1.0,
    ).astype("float64")

    return daily


def _compute_median_z_scores(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Compute signed z-scores for each feature against the household median."""
    medians: dict[str, float] = {}
    z_scores = pd.DataFrame(index=daily.index)

    for col in CONSUMPTION_FEATURES:
        series = daily[col].astype("float64")
        med = float(series.median())
        medians[col] = med

        std = float(series.std(ddof=0))
        if std > 1e-9:
            scale = std
        else:
            mad = float((series - med).abs().median()) * 1.4826
            scale = mad if mad > 1e-9 else max(abs(med) * 0.05, 1e-6)

        z_scores[col] = (series - med) / scale

    return z_scores, medians


def detect_anomalies_with_backend(
    df: pd.DataFrame,
    *,
    allow_fallback: bool = False,
) -> tuple[list[dict], str]:
    """Detect anomalies and return ``(anomalies, backend_used)``."""
    detector, backend_used = _create_isolation_forest(allow_fallback=allow_fallback)

    daily = build_daily_features(df)
    if daily.empty or len(daily) < MIN_DAYS_FOR_ISOLATION_FOREST:
        return [], backend_used

    anomalies: list[dict] = []

    # 1. Consumption anomalies via Isolation Forest
    X = daily[CONSUMPTION_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    preds = detector.fit_predict(X)
    raw_scores = -detector.score_samples(X)
    z_scores, medians = _compute_median_z_scores(daily)

    for idx, (day_ts, is_anom) in enumerate(zip(daily.index, preds == -1)):
        if not is_anom:
            continue

        day_z = z_scores.loc[day_ts]
        if float(day_z.max()) <= 0.0:
            continue

        top_feature = str(day_z.idxmax())
        score = round(float(np.clip(raw_scores[idx], 0.0, 1.0)), 2)
        date_str = day_ts.strftime("%Y-%m-%d")

        if top_feature == "night_load_kw":
            val = round(float(daily.at[day_ts, "night_load_kw"]), 2)
            exp = round(float(medians["night_load_kw"]), 2)
            if exp > 0.0:
                ratio = round(val / exp, 1)
                reason = f"Night load {ratio}x higher than this home's usual level"
            else:
                reason = f"Night load ({val} kW) unusually high vs usual {exp} kW"
            anomalies.append(
                {
                    "date": str(date_str),
                    "type": "high_night_load",
                    "value": float(val),
                    "expected": float(exp),
                    "unit": "kW",
                    "score": float(score),
                    "reason": str(reason),
                }
            )
        elif top_feature == "peak_kw":
            val = round(float(daily.at[day_ts, "peak_kw"]), 2)
            exp = round(float(medians["peak_kw"]), 2)
            anomalies.append(
                {
                    "date": str(date_str),
                    "type": "consumption_spike",
                    "value": float(val),
                    "expected": float(exp),
                    "unit": "kW",
                    "score": float(score),
                    "reason": f"Peak demand reached {val} kW vs usual {exp} kW",
                }
            )
        elif top_feature == "daily_cons_kwh":
            val = round(float(daily.at[day_ts, "daily_cons_kwh"]), 2)
            exp = round(float(medians["daily_cons_kwh"]), 2)
            anomalies.append(
                {
                    "date": str(date_str),
                    "type": "consumption_spike",
                    "value": float(val),
                    "expected": float(exp),
                    "unit": "kWh",
                    "score": float(score),
                    "reason": f"Daily consumption reached {val} kWh vs usual {exp} kWh",
                }
            )
        else:
            val = round(float(daily.at[day_ts, "daily_cons_kwh"]), 2)
            exp = round(float(medians["daily_cons_kwh"]), 2)
            eve_pct = round(float(daily.at[day_ts, "evening_share"]) * 100.0, 1)
            exp_eve_pct = round(float(medians["evening_share"]) * 100.0, 1)
            anomalies.append(
                {
                    "date": str(date_str),
                    "type": "consumption_spike",
                    "value": float(val),
                    "expected": float(exp),
                    "unit": "kWh",
                    "score": float(score),
                    "reason": (
                        f"Evening consumption share surged to {eve_pct}% "
                        f"(usual {exp_eve_pct}%), total {val} kWh"
                    ),
                }
            )

    # 2. Independent solar panel check: 14-day rolling median of yield_ratio < 0.75
    if len(daily) >= ROLLING_YIELD_WINDOW:
        roll_median = (
            daily["yield_ratio"]
            .rolling(window=ROLLING_YIELD_WINDOW, min_periods=ROLLING_YIELD_WINDOW)
            .median()
        )
        low_gen_mask = (roll_median < LOW_GENERATION_THRESHOLD) & (
            daily["p90_month_gen_kwh"] > 0.0
        )
        for day_ts in daily.index[low_gen_mask]:
            val = round(float(daily.at[day_ts, "daily_gen_kwh"]), 2)
            exp = round(float(daily.at[day_ts, "p90_month_gen_kwh"]), 2)
            roll_val = float(roll_median.at[day_ts])
            score = round(float(np.clip(1.0 - roll_val, 0.0, 1.0)), 2)
            anomalies.append(
                {
                    "date": str(day_ts.strftime("%Y-%m-%d")),
                    "type": "low_generation",
                    "value": float(val),
                    "expected": float(exp),
                    "unit": "kWh",
                    "score": float(score),
                    "reason": "Possible dust, shading or fault",
                }
            )

    anomalies.sort(key=lambda item: (item["date"], item["type"]))
    return anomalies, backend_used


def detect_anomalies(df: pd.DataFrame, *, allow_fallback: bool = False) -> list[dict]:
    """Detect consumption and solar generation anomalies for a household.

    Follows the Section 1.3 signature ``detect_anomalies(df) -> list[dict]``.
    """
    anomalies, _ = detect_anomalies_with_backend(df, allow_fallback=allow_fallback)
    return anomalies


if __name__ == "__main__":
    import argparse
    from collections import Counter
    from src.pipeline.load import load_home

    parser = argparse.ArgumentParser(description="Run SolarSense anomaly detection on Home 1.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Explicitly allow PurePythonIsolationForest if scikit-learn is blocked.",
    )
    args = parser.parse_args()

    home1 = load_home(1)
    results, backend = detect_anomalies_with_backend(home1, allow_fallback=args.allow_fallback)
    counts = Counter(item["type"] for item in results)
    print(f"Backend used: {backend}")
    print(f"Total anomalies detected for Home 1: {len(results)}")
    print(f"Breakdown by type: {dict(counts)}")
    for sample in results[:5]:
        print(sample)

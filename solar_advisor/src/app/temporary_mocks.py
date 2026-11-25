"""TEMPORARY MOCKS for team modules not yet present in the workspace.

This file provides temporary stand-in implementations for:
- Shashikiran's pipeline (`src/pipeline/`)
- Shashikiran's analytics (`src/analytics/`)
- Sreesuwin's LLM advisor (`src/llm/advisor.py`)

They are used ONLY when the corresponding team modules raise ImportError, and
are backed by real Ausgrid half-hourly records (never fabricated numbers).
Replace/remove once all team modules are merged into `src/`.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config
from src.config import EXPORT_TARIFF, IMPORT_TARIFF, LLM_MODEL
from src.recommend.rules import generate_recommendations
from src.recommend.simulator import simulate_battery, simulate_ev, simulate_extra_pv

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = Path(getattr(config, "PROCESSED_DIR", ROOT / getattr(config, "PROCESSED", "data/processed")))
SAMPLE_PARQUET = Path(getattr(config, "SAMPLE_FILE", ROOT / "data" / "sample" / "sample_home.parquet"))
ZIP_PATH = ROOT / "Dataset" / "Ausgrid_solar_home_data.zip"
DEMO_HOME_IDS = (1, 2, 3, 4, 5)


def _reshape_ausgrid_rows(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if "Row Quality" in raw.columns:
        raw = raw[raw["Row Quality"] != "NA"].copy()
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce")
    raw = raw.dropna(subset=["date"])
    slots = list(raw.columns[5:53])
    slot_map = {s: i for i, s in enumerate(slots)}
    long = raw.melt(
        id_vars=["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date"],
        value_vars=slots,
        var_name="slot",
        value_name="kwh",
    )
    long["kwh"] = pd.to_numeric(long["kwh"], errors="coerce").fillna(0.0)
    long["ts"] = long["date"] + pd.to_timedelta(long["slot"].map(slot_map) * 30, unit="min")
    piv = long.pivot_table(
        index=["Customer", "Generator Capacity", "Postcode", "ts"],
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
        "gen_kwh": piv["GG"].astype(float).clip(lower=0.0),
        "gc_kwh": piv["GC"].astype(float).clip(lower=0.0),
        "cl_kwh": piv["CL"].astype(float).clip(lower=0.0),
        "pv_kw": piv["Generator Capacity"].astype(float),
        "postcode": piv["Postcode"].astype(int),
    })
    out["cons_kwh"] = out["gc_kwh"] + out["cl_kwh"]
    out["self_kwh"] = np.minimum(out["gen_kwh"], out["cons_kwh"])
    out["import_kwh"] = np.maximum(out["cons_kwh"] - out["gen_kwh"], 0.0)
    out["export_kwh"] = np.maximum(out["gen_kwh"] - out["cons_kwh"], 0.0)
    out["valid"] = out["gen_kwh"] <= out["pv_kw"] * 0.5 * 1.05
    return out.sort_values(["home_id", "ts"]).reset_index(drop=True)


def _read_demo_homes_from_zip(home_ids: tuple[int, ...] = DEMO_HOME_IDS) -> pd.DataFrame:
    if not ZIP_PATH.exists():
        return pd.DataFrame()
    chunks = []
    target_set = set(home_ids)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        with zf.open("Solar home 2012-2013.csv") as fh:
            for chunk in pd.read_csv(fh, skiprows=1, chunksize=5000):
                sub = chunk[chunk["Customer"].isin(target_set)]
                if not sub.empty:
                    chunks.append(sub)
                if not chunk.empty and int(chunk["Customer"].max()) > max(target_set):
                    break
    if not chunks:
        return pd.DataFrame()
    return _reshape_ausgrid_rows(pd.concat(chunks, ignore_index=True))


# ---------------------------------------------------------------------------
# TEMPORARY MOCKS FOR SHASHIKIRAN'S PIPELINE (src/pipeline)
# ---------------------------------------------------------------------------

def mock_list_homes() -> pd.DataFrame:
    homes_pq = PROCESSED_DIR / "homes.parquet"
    if homes_pq.exists():
        return pd.read_parquet(homes_pq)
    if SAMPLE_PARQUET.exists():
        df = pd.read_parquet(SAMPLE_PARQUET)
        hid = int(df["home_id"].iloc[0]) if "home_id" in df.columns and not df.empty else 1
        pv = float(df["pv_kw"].iloc[0]) if "pv_kw" in df.columns and not df.empty else 3.78
        return pd.DataFrame([{
            "home_id": hid,
            "pv_kw": pv,
            "postcode": 2076,
            "has_cl": bool((df["cl_kwh"].sum() > 0) if "cl_kwh" in df.columns else False),
            "n_days": int(pd.to_datetime(df["ts"]).dt.date.nunique()) if not df.empty else 0,
        }])

    demo_df = _read_demo_homes_from_zip()
    if demo_df.empty:
        return pd.DataFrame(columns=["home_id", "pv_kw", "postcode", "has_cl", "n_days"])
    rows = []
    for hid, grp in demo_df.groupby("home_id"):
        rows.append({
            "home_id": int(hid),
            "pv_kw": round(float(grp["pv_kw"].iloc[0]), 2),
            "postcode": int(grp["postcode"].iloc[0]),
            "has_cl": bool(grp["cl_kwh"].sum() > 0),
            "n_days": int(grp["ts"].dt.date.nunique()),
        })
    return pd.DataFrame(rows).sort_values("home_id").reset_index(drop=True)


def mock_load_home(home_id: int, start=None, end=None) -> pd.DataFrame:
    hh_pq = PROCESSED_DIR / "halfhourly.parquet"
    if hh_pq.exists():
        df = pd.read_parquet(hh_pq, filters=[("home_id", "==", int(home_id))])
    elif SAMPLE_PARQUET.exists() and int(home_id) == 1:
        df = pd.read_parquet(SAMPLE_PARQUET)
    else:
        df = _read_demo_homes_from_zip((int(home_id),))

    if df.empty:
        return df

    df = df.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    if "valid" in df.columns:
        df = df[df["valid"]].copy()
    if start is not None:
        df = df[df["ts"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["ts"] <= pd.to_datetime(end)]
    return df.sort_values("ts").reset_index(drop=True)


def mock_load_user_csv(file, pv_kw: float) -> pd.DataFrame:
    raw = pd.read_csv(file)
    col_map = {c.strip().lower(): c for c in raw.columns}

    ts_col = next((col_map[k] for k in ("timestamp", "ts", "datetime", "date_time") if k in col_map), raw.columns[0])
    gen_col = next((col_map[k] for k in ("generation_kwh", "gen_kwh", "gg", "generation") if k in col_map), None)
    cons_col = next((col_map[k] for k in ("consumption_kwh", "cons_kwh", "gc", "consumption") if k in col_map), None)
    cl_col = next((col_map[k] for k in ("cl_kwh", "cl", "controlled_load", "controlled_load_kwh") if k in col_map), None)

    if gen_col is None or cons_col is None:
        raise ValueError(
            "CSV must include timestamp, generation_kwh (or GG), and consumption_kwh (or GC) columns."
        )

    parsed = pd.DataFrame({
        "ts": pd.to_datetime(raw[ts_col], errors="coerce"),
        "gen_kwh": pd.to_numeric(raw[gen_col], errors="coerce").fillna(0.0).clip(lower=0.0),
        "gc_kwh": pd.to_numeric(raw[cons_col], errors="coerce").fillna(0.0).clip(lower=0.0),
        "cl_kwh": pd.to_numeric(raw[cl_col], errors="coerce").fillna(0.0).clip(lower=0.0) if cl_col else 0.0,
    }).dropna(subset=["ts"])

    if parsed.empty:
        raise ValueError("No valid rows found in uploaded CSV after parsing timestamps.")

    parsed = parsed.set_index("ts").resample("30min").sum().reset_index()
    parsed["home_id"] = 0
    parsed["cons_kwh"] = parsed["gc_kwh"] + parsed["cl_kwh"]
    parsed["self_kwh"] = np.minimum(parsed["gen_kwh"], parsed["cons_kwh"])
    parsed["import_kwh"] = np.maximum(parsed["cons_kwh"] - parsed["gen_kwh"], 0.0)
    parsed["export_kwh"] = np.maximum(parsed["gen_kwh"] - parsed["cons_kwh"], 0.0)
    cap_pv = max(float(pv_kw), 0.1)
    parsed["valid"] = parsed["gen_kwh"] <= cap_pv * 0.5 * 1.05
    parsed["pv_kw"] = float(pv_kw)
    cols = ["home_id", "ts", "gen_kwh", "gc_kwh", "cl_kwh", "cons_kwh", "self_kwh", "import_kwh", "export_kwh", "valid", "pv_kw"]
    return parsed[parsed["valid"]][cols].sort_values("ts").reset_index(drop=True)


# ---------------------------------------------------------------------------
# TEMPORARY MOCKS FOR SHASHIKIRAN'S ANALYTICS (src/analytics)
# ---------------------------------------------------------------------------

def mock_compute_kpis(df: pd.DataFrame) -> dict:
    if df is None or df.empty:
        return {
            "period": "N/A",
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

    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    gen = float(work["gen_kwh"].sum())
    cons = float(work["cons_kwh"].sum())
    self_kwh = float(np.minimum(work["gen_kwh"], work["cons_kwh"]).sum())
    imp = float(np.maximum(work["cons_kwh"] - work["gen_kwh"], 0.0).sum())
    exp = float(np.maximum(work["gen_kwh"] - work["cons_kwh"], 0.0).sum())
    days = max(int(work["ts"].dt.date.nunique()), 1)

    hours = work["ts"].dt.hour
    night = work[(hours >= 1) & (hours < 5)]
    if not night.empty:
        baseload_kw = float(night.groupby(night["ts"].dt.date)["cons_kwh"].mean().median()) * 2.0
    else:
        baseload_kw = 0.0

    peak_kw = float(np.percentile(work["cons_kwh"] * 2.0, 99))
    hourly_cons = work.groupby(hours)["cons_kwh"].mean()
    peak_hour = int(hourly_cons.idxmax()) if not hourly_cons.empty else 0

    grid_cost = imp * IMPORT_TARIFF
    export_credit = exp * EXPORT_TARIFF
    net_bill = grid_cost - export_credit

    return {
        "period": f"{work['ts'].min().date()} to {work['ts'].max().date()}",
        "days": int(days),
        "gen_kwh": round(gen, 2),
        "cons_kwh": round(cons, 2),
        "self_kwh": round(self_kwh, 2),
        "import_kwh": round(imp, 2),
        "export_kwh": round(exp, 2),
        "self_consumption_pct": round((self_kwh / gen * 100.0) if gen > 0 else 0.0, 2),
        "self_sufficiency_pct": round((self_kwh / cons * 100.0) if cons > 0 else 0.0, 2),
        "baseload_kw": round(baseload_kw, 2),
        "peak_kw": round(peak_kw, 2),
        "peak_hour": peak_hour,
        "avg_daily_gen_kwh": round(gen / days, 2),
        "avg_daily_cons_kwh": round(cons / days, 2),
        "grid_cost_inr": round(grid_cost, 2),
        "export_credit_inr": round(export_credit, 2),
        "net_bill_inr": round(net_bill, 2),
    }


def mock_daily_profile(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["hour", "gen_kw", "cons_kw"])
    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    work["hour"] = work["ts"].dt.hour
    prof = work.groupby("hour", as_index=False)[["gen_kwh", "cons_kwh"]].mean()
    prof["gen_kw"] = (prof["gen_kwh"] * 2.0).round(2)
    prof["cons_kw"] = (prof["cons_kwh"] * 2.0).round(2)
    return prof[["hour", "gen_kw", "cons_kw"]]


def mock_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["month", "gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_consumption_pct"])
    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    work["month"] = work["ts"].dt.to_period("M").astype(str)
    work["import_kwh"] = np.maximum(work["cons_kwh"] - work["gen_kwh"], 0.0)
    work["export_kwh"] = np.maximum(work["gen_kwh"] - work["cons_kwh"], 0.0)
    work["self_kwh"] = np.minimum(work["gen_kwh"], work["cons_kwh"])
    m = work.groupby("month", as_index=False)[["gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_kwh"]].sum()
    m["self_consumption_pct"] = np.where(m["gen_kwh"] > 0, (m["self_kwh"] / m["gen_kwh"]) * 100.0, 0.0).round(2)
    for col in ("gen_kwh", "cons_kwh", "import_kwh", "export_kwh"):
        m[col] = m[col].round(2)
    return m[["month", "gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_consumption_pct"]]


def mock_detect_anomalies(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    work["date"] = work["ts"].dt.date
    work["hour"] = work["ts"].dt.hour
    work["month"] = work["ts"].dt.to_period("M").astype(str)

    daily = work.groupby("date", as_index=False).agg(
        daily_cons_kwh=("cons_kwh", "sum"),
        daily_gen_kwh=("gen_kwh", "sum"),
        peak_kw=("cons_kwh", lambda s: float(s.max()) * 2.0),
        month=("month", "first"),
    )
    if len(daily) < 15:
        return []

    night = work[(work["hour"] >= 1) & (work["hour"] < 5)].groupby("date")["cons_kwh"].mean() * 2.0
    eve = work[(work["hour"] >= 18) & (work["hour"] < 22)].groupby("date")["cons_kwh"].sum()
    daily["night_load_kw"] = daily["date"].map(night).fillna(0.0)
    daily["evening_share"] = np.where(
        daily["daily_cons_kwh"] > 0,
        daily["date"].map(eve).fillna(0.0) / daily["daily_cons_kwh"],
        0.0,
    )

    p90_month = daily.groupby("month")["daily_gen_kwh"].transform(lambda s: np.percentile(s, 90))
    daily["yield_ratio"] = np.where(p90_month > 0, daily["daily_gen_kwh"] / p90_month, 1.0)

    feat_cols = ["daily_cons_kwh", "night_load_kw", "peak_kw", "evening_share"]
    iso = IsolationForest(contamination=0.03, random_state=42)
    preds = iso.fit_predict(daily[feat_cols])
    raw_scores = -iso.decision_function(daily[feat_cols])
    min_s, max_s = float(raw_scores.min()), float(raw_scores.max())
    norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-9)

    medians = daily[["daily_cons_kwh", "night_load_kw", "peak_kw"]].median()
    stds = daily[["daily_cons_kwh", "night_load_kw", "peak_kw"]].std().replace(0.0, 1.0)

    anomalies: list[dict] = []
    for idx, row in daily[preds == -1].iterrows():
        z_night = abs(row["night_load_kw"] - medians["night_load_kw"]) / stds["night_load_kw"]
        z_cons = abs(row["daily_cons_kwh"] - medians["daily_cons_kwh"]) / stds["daily_cons_kwh"]
        z_peak = abs(row["peak_kw"] - medians["peak_kw"]) / stds["peak_kw"]

        if z_night >= z_cons and z_night >= z_peak and row["night_load_kw"] > medians["night_load_kw"]:
            anomalies.append({
                "date": str(row["date"]),
                "type": "high_night_load",
                "value": round(float(row["night_load_kw"]), 2),
                "expected": round(float(medians["night_load_kw"]), 2),
                "unit": "kW",
                "score": round(float(norm_scores[idx]), 2),
                "reason": "Night load substantially higher than this home's usual level",
            })
        else:
            is_peak = z_peak > z_cons
            val = float(row["peak_kw"] if is_peak else row["daily_cons_kwh"])
            exp = float(medians["peak_kw"] if is_peak else medians["daily_cons_kwh"])
            unit = "kW" if is_peak else "kWh"
            anomalies.append({
                "date": str(row["date"]),
                "type": "consumption_spike",
                "value": round(val, 2),
                "expected": round(exp, 2),
                "unit": unit,
                "score": round(float(norm_scores[idx]), 2),
                "reason": "Unusually high daily or peak household consumption",
            })

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["rolling_yield"] = daily["yield_ratio"].rolling(14, min_periods=14).median()
    low_gen = daily[daily["rolling_yield"] < 0.75]
    for _, row in low_gen.iterrows():
        anomalies.append({
            "date": str(row["date"]),
            "type": "low_generation",
            "value": round(float(row["rolling_yield"]), 2),
            "expected": 1.0,
            "unit": "ratio",
            "score": round(float(min(max(1.0 - row["rolling_yield"], 0.0), 1.0)), 2),
            "reason": "14-day rolling yield below 0.75 (possible dust, shading or fault)",
        })

    anomalies.sort(key=lambda x: x["date"], reverse=True)
    return anomalies


def mock_forecast(df: pd.DataFrame, horizon_days: int = 14, target: str = "cons") -> dict:
    if df is None or df.empty:
        return {
            "target": target,
            "horizon_days": int(horizon_days),
            "dates": [],
            "p10": [],
            "p50": [],
            "p90": [],
            "total_p50_kwh": 0.0,
            "method": "seasonal_baseline",
        }

    col = "gen_kwh" if target == "gen" else "cons_kwh"
    work = df.copy()
    work["ts"] = pd.to_datetime(work["ts"])
    daily = work.groupby(work["ts"].dt.date)[col].sum()
    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    last_date = daily.index.max()
    recent = daily.tail(28)
    dates, p10_list, p50_list, p90_list = [], [], [], []
    for i in range(1, int(horizon_days) + 1):
        d = last_date + pd.Timedelta(days=i)
        same_wd = recent[recent.index.dayofweek == d.dayofweek]
        if same_wd.empty:
            same_wd = recent
        vals = same_wd.to_numpy(dtype=float)
        p10_list.append(round(float(np.percentile(vals, 10)), 2))
        p50_list.append(round(float(np.mean(vals)), 2))
        p90_list.append(round(float(np.percentile(vals, 90)), 2))
        dates.append(str(d.date()))

    return {
        "target": target,
        "horizon_days": int(horizon_days),
        "dates": dates,
        "p10": p10_list,
        "p50": p50_list,
        "p90": p90_list,
        "total_p50_kwh": round(float(sum(p50_list)), 2),
        "method": "seasonal_baseline",
    }


# ---------------------------------------------------------------------------
# TEMPORARY MOCKS FOR SREESUWIN'S LLM MODULE (src/llm/advisor.py)
# ---------------------------------------------------------------------------

def mock_chat(home_id: int, history: list, question: str) -> dict:
    """Temporary grounded mock used only when src/llm/advisor.py is not present."""
    t0 = time.time()
    df = mock_load_home(home_id)
    kpis = mock_compute_kpis(df)
    anomalies = mock_detect_anomalies(df)
    recs = generate_recommendations(df, kpis, anomalies)

    q_lower = question.lower()
    tool_calls = []

    if "battery" in q_lower:
        sim = simulate_battery(df, 5.0)
        tool_calls.append({"name": "simulate_battery", "args": {"capacity_kwh": 5.0}, "output": sim})
        reply = (
            f"For Home {home_id}, a {sim['capacity_kwh']} kWh battery ({sim['power_kw']} kW) avoids "
            f"{sim['import_avoided_kwh']} kWh/year of grid imports, saving ₹{sim['savings_inr_year']}/year. "
            f"Self-consumption improves from {sim['self_consumption_pct_before']}% to {sim['self_consumption_pct_after']}%, "
            f"with an installation cost of ₹{sim['cost_inr']} and an estimated payback of {sim['payback_years']} years. "
            f"(Temporary mock response — waiting for `src/llm/advisor.py` with model `{LLM_MODEL}`.)"
        )
    elif "ev" in q_lower or "car" in q_lower or "vehicle" in q_lower:
        sim = simulate_ev(df, 40.0)
        tool_calls.append({"name": "simulate_ev", "args": {"km_per_day": 40.0}, "output": sim})
        reply = (
            f"Driving {sim['km_per_day']} km/day adds {sim['ev_demand_kwh']} kWh/year of demand. "
            f"Solar covers {sim['solar_coverage_pct']}% ({sim['ev_from_solar_kwh']} kWh/year) during daytime charging, "
            f"while {sim['ev_from_grid_kwh']} kWh/year comes from the grid, adding ₹{sim['additional_cost_inr_year']}/year. "
            f"(Temporary mock response — waiting for `src/llm/advisor.py`.)"
        )
    elif "solar" in q_lower and ("extra" in q_lower or "add" in q_lower or "panel" in q_lower):
        sim = simulate_extra_pv(df, 2.0)
        tool_calls.append({"name": "simulate_extra_pv", "args": {"extra_kw": 2.0}, "output": sim})
        reply = (
            f"Adding {sim['extra_kw']} kW of solar generates {sim['extra_gen_kwh']} kWh/year extra and saves "
            f"₹{sim['savings_inr_year']}/year (estimated payback {sim['payback_years']} years). "
            f"(Temporary mock response — waiting for `src/llm/advisor.py`.)"
        )
    elif "recommend" in q_lower or "save" in q_lower or "first" in q_lower or "change" in q_lower:
        tool_calls.append({"name": "get_recommendations", "args": {}, "output": recs})
        if recs:
            top = recs[0]
            reply = (
                f"Top recommendation ({top['id']}): {top['title']}. {top['action']} "
                f"Estimated savings: {top['kwh_saved_month']} kWh/month (₹{top['inr_saved_month']}/month). "
                f"(Temporary mock response — waiting for `src/llm/advisor.py`.)"
            )
        else:
            reply = "No actionable recommendations triggered for the current dataset. (Temporary mock response.)"
    elif "anomal" in q_lower or "night" in q_lower or "spike" in q_lower:
        tool_calls.append({"name": "get_anomalies", "args": {}, "output": anomalies[:5]})
        reply = (
            f"Detected {len(anomalies)} anomalous days for Home {home_id}. "
            + (
                f"Most recent: {anomalies[0]['date']} ({anomalies[0]['type']}: {anomalies[0]['value']} {anomalies[0]['unit']} vs expected {anomalies[0]['expected']}). "
                if anomalies
                else "No anomalies were flagged. "
            )
            + "(Temporary mock response — waiting for `src/llm/advisor.py`.)"
        )
    else:
        tool_calls.append({"name": "get_kpis", "args": {}, "output": kpis})
        reply = (
            f"Home {home_id} ({kpis['period']}) generated {kpis['gen_kwh']} kWh and consumed {kpis['cons_kwh']} kWh. "
            f"Self-consumption is {kpis['self_consumption_pct']}%, self-sufficiency is {kpis['self_sufficiency_pct']}%, "
            f"solar export is {kpis['export_kwh']} kWh, and estimated net bill is ₹{kpis['net_bill_inr']}. "
            f"(Temporary mock response — waiting for `src/llm/advisor.py` with `{LLM_MODEL}`.)"
        )

    return {
        "reply": reply,
        "tool_calls": tool_calls,
        "verified": True,
        "unverified_numbers": [],
        "latency_s": round(time.time() - t0, 2),
    }


def mock_monthly_report_text(home_id: int, month: str) -> str:
    """Temporary grounded monthly summary used only when src/llm/advisor.py is absent."""
    df = mock_load_home(home_id)
    if df.empty:
        return "No household data available to generate a monthly summary."
    df_m = df[pd.to_datetime(df["ts"]).dt.to_period("M").astype(str) == str(month)]
    if df_m.empty:
        df_m = df
    kpis = mock_compute_kpis(df_m)
    anomalies = mock_detect_anomalies(df_m)
    recs = generate_recommendations(df_m, kpis, anomalies)

    top_actions = "; ".join(
        f"{r['title']} (est. INR {r['inr_saved_month']}/month)" for r in recs[:3]
    ) or "Continue monitoring daytime solar usage"

    return (
        f"During {month}, Home {home_id} generated {kpis['gen_kwh']} kWh of solar electricity and "
        f"consumed {kpis['cons_kwh']} kWh, achieving {kpis['self_consumption_pct']}% solar self-consumption "
        f"and {kpis['self_sufficiency_pct']}% energy self-sufficiency. Grid imports totaled {kpis['import_kwh']} kWh "
        f"while {kpis['export_kwh']} kWh was exported, resulting in an estimated net bill of INR {kpis['net_bill_inr']}. "
        f"Analytics flagged {len(anomalies)} unusual day(s) in this period. "
        f"Top recommended actions based on historical patterns (estimates, not guarantees): {top_actions}."
    )

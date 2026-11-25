"""Tools the LLM can call.   OWNER: Sreesuwin

The app selects the home before the chat starts, so no tool takes home_id
(small models get ids wrong). DISPATCH maps a tool name to fn(df, **args) -> dict.
"""
from src.analytics.anomalies import detect_anomalies
from src.analytics.forecast import forecast
from src.analytics.kpis import compute_kpis
from src.recommend.rules import generate_recommendations
from src.recommend.simulator import simulate_battery, simulate_ev, simulate_extra_pv


def _fn(name, description, properties=None, required=None):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


TOOLS = [
    _fn("get_kpis", "Energy KPIs for the selected home: solar generation, consumption, self-consumption %, "
                    "self-sufficiency %, grid import/export, night baseload, peak demand and bill in ₹."),
    _fn("get_recommendations", "Ranked actions to save energy and money, each with kWh and ₹ saved per month and the evidence."),
    _fn("get_anomalies", "Unusual days and inefficiencies (high night load, consumption spikes, weak solar output) with reasons."),
    _fn("forecast", "Forecast the next N days of consumption ('cons') or solar generation ('gen').",
        {"target": {"type": "string", "enum": ["cons", "gen"]},
         "horizon_days": {"type": "integer", "description": "days ahead, 1-30"}}, ["target"]),
    _fn("simulate_battery", "Yearly savings, self-consumption change and payback for a home battery of a given size.",
        {"capacity_kwh": {"type": "number", "description": "battery size in kWh, e.g. 3, 5, 10"}}, ["capacity_kwh"]),
    _fn("simulate_extra_pv", "Effect of adding more solar panels: extra generation, savings, payback.",
        {"extra_kw": {"type": "number", "description": "extra panel capacity in kW"}}, ["extra_kw"]),
    _fn("simulate_ev", "Effect of buying an electric vehicle charged at home: extra energy, cost, share covered by solar.",
        {"km_per_day": {"type": "number", "description": "average kilometres driven per day"}}, ["km_per_day"]),
]


def _recs(df, **_):
    kpis = compute_kpis(df)
    return {"recommendations": generate_recommendations(df, kpis, detect_anomalies(df))}


def _kpis(df, **_):
    """KPIs plus the complementary shares, so the model never has to compute 100 - x."""
    k = dict(compute_kpis(df))
    if "self_consumption_pct" in k:
        k["exported_share_pct"] = round(100.0 - k["self_consumption_pct"], 2)
    if "self_sufficiency_pct" in k:
        k["grid_share_pct"] = round(100.0 - k["self_sufficiency_pct"], 2)
    return k


MAX_ANOMALIES_FOR_LLM = 8


def _anomalies(df, **_):
    """Counts by type plus the most severe events only: a 3-year history can hold
    hundreds of flagged days, far too much text for a 7B model's context."""
    items = detect_anomalies(df)
    counts = {}
    for a in items:
        counts[a["type"]] = counts.get(a["type"], 0) + 1
    top = sorted(items, key=lambda a: a.get("score", 0), reverse=True)[:MAX_ANOMALIES_FOR_LLM]
    return {"total_flagged_days": len(items), "count_by_type": counts,
            "most_severe": sorted(top, key=lambda a: a["date"])}


DISPATCH = {
    "get_kpis": _kpis,
    "get_recommendations": _recs,
    "get_anomalies": _anomalies,
    "forecast": lambda df, target="cons", horizon_days=14, **a: forecast(df, int(horizon_days), target),
    "simulate_battery": lambda df, capacity_kwh, **a: simulate_battery(df, float(capacity_kwh)),
    "simulate_extra_pv": lambda df, extra_kw, **a: simulate_extra_pv(df, float(extra_kw)),
    "simulate_ev": lambda df, km_per_day, **a: simulate_ev(df, float(km_per_day)),
}

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


DISPATCH = {
    "get_kpis": lambda df, **a: compute_kpis(df),
    "get_recommendations": _recs,
    "get_anomalies": lambda df, **a: {"anomalies": detect_anomalies(df)},
    "forecast": lambda df, target="cons", horizon_days=14, **a: forecast(df, int(horizon_days), target),
    "simulate_battery": lambda df, capacity_kwh, **a: simulate_battery(df, float(capacity_kwh)),
    "simulate_extra_pv": lambda df, extra_kw, **a: simulate_extra_pv(df, float(extra_kw)),
    "simulate_ev": lambda df, km_per_day, **a: simulate_ev(df, float(km_per_day)),
}

"""The data contract between modules. DO NOT change without telling the whole team.

tests/test_contracts.py checks every module against these definitions.
Rules for every function that returns a dict:
  * numbers rounded to at most 2 decimals (the LLM guardrail matches on them)
  * plain Python types only (float, int, str, bool, list, dict) - no numpy / pandas types
  * units in key names: _kwh, _kw, _inr, _pct, _years
"""

HALFHOURLY_COLUMNS = [
    "home_id",     # int, Ausgrid customer number 1-300 (0 = user upload)
    "ts",          # datetime, START of the 30-minute interval
    "gen_kwh",     # solar generation (GG)
    "gc_kwh",      # general consumption (GC)
    "cl_kwh",      # controlled load (CL), 0 if the home has none
    "cons_kwh",    # gc_kwh + cl_kwh
    "self_kwh",    # min(gen_kwh, cons_kwh)
    "import_kwh",  # max(cons_kwh - gen_kwh, 0)
    "export_kwh",  # max(gen_kwh - cons_kwh, 0)
    "valid",       # bool, False for flagged / over-capacity readings
]

HOMES_COLUMNS = ["home_id", "pv_kw", "postcode", "has_cl", "n_days"]

KPI_KEYS = [
    "period", "days", "gen_kwh", "cons_kwh", "self_kwh", "import_kwh", "export_kwh",
    "self_consumption_pct", "self_sufficiency_pct", "baseload_kw", "peak_kw", "peak_hour",
    "avg_daily_gen_kwh", "avg_daily_cons_kwh", "grid_cost_inr", "export_credit_inr", "net_bill_inr",
]

ANOMALY_KEYS = ["date", "type", "value", "expected", "unit", "score", "reason"]
ANOMALY_TYPES = ["high_night_load", "consumption_spike", "low_generation"]

FORECAST_KEYS = ["target", "horizon_days", "dates", "p10", "p50", "p90", "total_p50_kwh", "method"]
FORECAST_METHODS = ["chronos-2", "chronos-bolt", "seasonal_baseline"]

BATTERY_KEYS = [
    "capacity_kwh", "power_kw", "import_avoided_kwh", "savings_inr_year",
    "self_consumption_pct_before", "self_consumption_pct_after", "cost_inr", "payback_years",
]
EXTRA_PV_KEYS = ["extra_kw", "extra_gen_kwh_year", "import_avoided_kwh", "extra_export_kwh",
                 "savings_inr_year", "cost_inr", "payback_years", "self_sufficiency_pct_after"]
LOAD_SHIFT_KEYS = ["kwh_per_day", "shifted_kwh_year", "savings_inr_year", "self_consumption_pct_after"]
EV_KEYS = ["km_per_day", "ev_kwh_year", "solar_share_pct", "extra_import_kwh_year", "extra_cost_inr_year"]

RECOMMENDATION_KEYS = ["id", "title", "category", "action", "kwh_saved_month", "inr_saved_month",
                       "effort", "evidence"]
RECOMMENDATION_CATEGORIES = ["utilization", "inefficiency", "planning"]

CHAT_KEYS = ["reply", "tool_calls", "verified", "unverified_numbers", "latency_s"]
GUARDRAIL_KEYS = ["ok", "bad", "ratio"]

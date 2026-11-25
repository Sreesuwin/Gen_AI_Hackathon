"""Rule library -> ranked recommendations.   OWNER: Divya

STUB. Real rules R1-R6 in README 6.2. Each rule returns a dict with RECOMMENDATION_KEYS
and an "evidence" dict of the numbers it used. Sort by inr_saved_month, largest first.
"""
import pandas as pd


def generate_recommendations(df: pd.DataFrame, kpis: dict, anomalies: list[dict]) -> list[dict]:
    # STUB - fixed examples
    return [
        {"id": "R2", "title": "Cut always-on night load", "category": "inefficiency",
         "action": "Check devices left on overnight (old fridge, pumps, chargers).",
         "kwh_saved_month": 86.4, "inr_saved_month": 691.2, "effort": "low",
         "evidence": {"baseload_kw": 0.62, "median_home_baseload_kw": 0.26}},
        {"id": "R5", "title": "Battery option", "category": "planning",
         "action": "A 5 kWh battery would cover much of your evening grid use; payback is long at current prices.",
         "kwh_saved_month": 100.9, "inr_saved_month": 504.3, "effort": "high",
         "evidence": {"capacity_kwh": 5, "payback_years": 24.8}},
        {"id": "R1", "title": "Run flexible loads at midday", "category": "utilization",
         "action": "Run the washing machine, dishwasher and water pump between 10 am and 2 pm.",
         "kwh_saved_month": 36.5, "inr_saved_month": 182.5, "effort": "low",
         "evidence": {"self_consumption_pct": 52.6, "evening_import_kwh_day": 6.1, "midday_export_kwh_day": 2.8}},
    ]

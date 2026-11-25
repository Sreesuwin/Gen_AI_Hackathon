"""Inefficiency and anomaly detection.   OWNER: Shashikiran

STUB. Real version (README 5.4): daily features -> IsolationForest(contamination=0.03)
-> largest z-score names the type and reason; plus a 14-day yield_ratio < 0.75 panel rule.
"""
import pandas as pd


def detect_anomalies(df: pd.DataFrame) -> list[dict]:
    # STUB - fixed examples, one of each type
    return [
        {"date": "2013-03-14", "type": "high_night_load", "value": 0.92, "expected": 0.31, "unit": "kW",
         "score": 0.81, "reason": "Night load 3x higher than this home's usual level"},
        {"date": "2013-01-08", "type": "consumption_spike", "value": 41.2, "expected": 19.5, "unit": "kWh",
         "score": 0.74, "reason": "Daily consumption more than double the usual level"},
        {"date": "2012-11-20", "type": "low_generation", "value": 0.62, "expected": 0.9, "unit": "ratio",
         "score": 0.66, "reason": "Solar output 31% below a normal clear day for 14 days: possible dust, shading or fault"},
    ]

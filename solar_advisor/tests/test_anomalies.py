"""Tests for M2 Anomaly Detection and PurePythonIsolationForest fallback."""

from __future__ import annotations

import builtins
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analytics.anomalies import (
    BACKEND_FALLBACK,
    BACKEND_SKLEARN,
    build_daily_features,
    detect_anomalies,
    detect_anomalies_with_backend,
)
from src.analytics.isolation_forest_fallback import PurePythonIsolationForest
from src.pipeline.load import load_home

EXPECTED_KEYS = {"date", "type", "value", "expected", "unit", "score", "reason"}
VALID_TYPES = {"high_night_load", "consumption_spike", "low_generation"}


def _make_synthetic_home(
    n_days: int = 60,
    start: str = "2025-01-01",
    base_night_kwh: float = 0.15,
    base_day_kwh: float = 0.25,
    base_eve_kwh: float = 0.40,
    daily_peak_gen_kwh: float = 0.80,
) -> pd.DataFrame:
    """Create a synthetic half-hourly home dataset conforming to schema 1.1."""
    ts = pd.date_range(start, periods=n_days * 48, freq="30min")
    hours = ts.hour

    cons = np.full(len(ts), base_day_kwh, dtype=float)
    cons[(hours >= 1) & (hours < 5)] = base_night_kwh
    cons[(hours >= 18) & (hours < 22)] = base_eve_kwh

    gen = np.zeros(len(ts), dtype=float)
    day_mask = (hours >= 7) & (hours <= 17)
    gen[day_mask] = daily_peak_gen_kwh * np.sin(np.pi * (hours[day_mask] - 7) / 10.0)

    self_u = np.minimum(gen, cons)
    imp = np.maximum(cons - gen, 0.0)
    exp = np.maximum(gen - cons, 0.0)

    return pd.DataFrame(
        {
            "home_id": 999,
            "ts": ts,
            "gen_kwh": gen,
            "gc_kwh": cons,
            "cl_kwh": 0.0,
            "cons_kwh": cons,
            "self_kwh": self_u,
            "import_kwh": imp,
            "export_kwh": exp,
            "valid": True,
        }
    )


class TestAnomalies(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.home1 = load_home(1)

    def test_pure_python_isolation_forest_unit_behavior(self) -> None:
        rng = np.random.default_rng(7)
        normal = rng.normal(loc=0.0, scale=0.2, size=(100, 4))
        outliers = np.array([[8.0, 8.0, 8.0, 8.0], [9.0, -7.0, 6.0, 5.0]])
        X = np.vstack([normal, outliers])

        model = PurePythonIsolationForest(
            n_estimators=100,
            max_samples=256,
            contamination=0.03,
            random_state=42,
        )
        preds = model.fit_predict(X)
        scores = model.anomaly_scores(X)

        self.assertEqual(preds.shape, (102,))
        self.assertTrue(set(np.unique(preds)).issubset({-1, 1}))
        self.assertEqual(preds[-1], -1)
        self.assertEqual(preds[-2], -1)
        self.assertGreater(float(scores[-1]), float(np.median(scores[:-2])))

        # Constant features and empty array safety
        const_preds = PurePythonIsolationForest(random_state=42).fit_predict(np.ones((20, 4)))
        self.assertTrue((const_preds == 1).all())

        empty_model = PurePythonIsolationForest(random_state=42).fit(np.empty((0, 4)))
        self.assertEqual(empty_model.predict(np.empty((0, 4))).size, 0)

    def test_no_silent_fallback_when_sklearn_blocked(self) -> None:
        df = _make_synthetic_home(n_days=20)
        real_import = builtins.__import__

        def block_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sklearn"):
                raise ImportError("Simulated blocked sklearn dependency")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=block_sklearn):
            with self.assertRaises(RuntimeError):
                detect_anomalies(df)  # default allow_fallback=False

            anoms, backend = detect_anomalies_with_backend(df, allow_fallback=True)
            self.assertEqual(backend, BACKEND_FALLBACK)
            self.assertIsInstance(anoms, list)

    def test_backend_identification(self) -> None:
        df = _make_synthetic_home(n_days=30)
        anoms, backend = detect_anomalies_with_backend(df, allow_fallback=True)
        self.assertIn(backend, {BACKEND_SKLEARN, BACKEND_FALLBACK})
        self.assertEqual(anoms, [])

    def test_daily_features_shape_and_columns(self) -> None:
        daily = build_daily_features(self.home1)
        self.assertEqual(len(daily), 1096)
        for col in ("daily_cons_kwh", "night_load_kw", "peak_kw", "evening_share", "yield_ratio"):
            self.assertIn(col, daily.columns)
            self.assertFalse(daily[col].isna().any())

    def test_normal_household_consumption_has_no_false_positives(self) -> None:
        df = _make_synthetic_home(n_days=60)
        anoms = detect_anomalies(df, allow_fallback=True)
        self.assertEqual(anoms, [])

    def test_unusual_night_consumption_detected(self) -> None:
        df = _make_synthetic_home(n_days=60)
        # Add realistic day-to-day daytime/evening variation so night_load_kw is not tied in z-score
        rng = np.random.default_rng(123)
        day_hours = df["ts"].dt.hour.between(8, 21)
        df.loc[day_hours, "cons_kwh"] = df.loc[day_hours, "cons_kwh"] + rng.uniform(
            0.0, 0.35, size=int(day_hours.sum())
        )
        df.loc[day_hours, "gc_kwh"] = df.loc[day_hours, "cons_kwh"]

        # Inject a 4x night load anomaly on 2025-01-21 (0.60 kWh/slot = 1.2 kW vs 0.3 kW)
        target_mask = (df["ts"].dt.strftime("%Y-%m-%d") == "2025-01-21") & (
            df["ts"].dt.hour.between(1, 4)
        )
        df.loc[target_mask, "cons_kwh"] = 0.60
        df.loc[target_mask, "gc_kwh"] = 0.60

        anoms = detect_anomalies(df, allow_fallback=True)
        night_anoms = [a for a in anoms if a["type"] == "high_night_load"]
        self.assertTrue(any(a["date"] == "2025-01-21" for a in night_anoms))
        sample = next(a for a in night_anoms if a["date"] == "2025-01-21")
        self.assertEqual(sample["unit"], "kW")
        self.assertGreater(sample["value"], sample["expected"])

    def test_sudden_consumption_spike_detected(self) -> None:
        df = _make_synthetic_home(n_days=60)
        spike_mask = df["ts"] == pd.Timestamp("2025-02-10 13:00:00")
        df.loc[spike_mask, "cons_kwh"] = 8.0
        df.loc[spike_mask, "gc_kwh"] = 8.0

        anoms = detect_anomalies(df, allow_fallback=True)
        spikes = [a for a in anoms if a["type"] == "consumption_spike"]
        self.assertTrue(any(a["date"] == "2025-02-10" for a in spikes))

    def test_low_solar_generation_detected(self) -> None:
        df = _make_synthetic_home(n_days=60)
        low_mask = (df["ts"] >= pd.Timestamp("2025-01-31")) & (
            df["ts"] < pd.Timestamp("2025-02-20")
        )
        df.loc[low_mask, "gen_kwh"] = df.loc[low_mask, "gen_kwh"] * 0.2

        anoms = detect_anomalies(df, allow_fallback=True)
        low_gen = [a for a in anoms if a["type"] == "low_generation"]
        self.assertGreater(len(low_gen), 0)
        self.assertIn("possible dust, shading or fault", low_gen[0]["reason"].lower())
        self.assertEqual(low_gen[0]["unit"], "kWh")

    def test_missing_values_and_invalid_rows_handled_safely(self) -> None:
        df = _make_synthetic_home(n_days=40)
        df.iloc[10, df.columns.get_loc("cons_kwh")] = np.nan
        df.iloc[15, df.columns.get_loc("gen_kwh")] = np.nan
        df.iloc[20, df.columns.get_loc("valid")] = False
        anoms = detect_anomalies(df, allow_fallback=True)
        self.assertIsInstance(anoms, list)

    def test_empty_and_insufficient_history(self) -> None:
        empty = self.home1.iloc[0:0].copy()
        self.assertEqual(detect_anomalies(empty, allow_fallback=True), [])

        short = _make_synthetic_home(n_days=4)
        self.assertEqual(detect_anomalies(short, allow_fallback=True), [])

    def test_home1_integration_json_and_reproducibility(self) -> None:
        r1, b1 = detect_anomalies_with_backend(self.home1, allow_fallback=True)
        r2, b2 = detect_anomalies_with_backend(self.home1, allow_fallback=True)
        self.assertEqual(b1, b2)
        self.assertEqual(r1, r2)
        self.assertGreater(len(r1), 0)

        for item in r1:
            self.assertEqual(set(item.keys()), EXPECTED_KEYS)
            self.assertIn(item["type"], VALID_TYPES)
            self.assertIs(type(item["date"]), str)
            self.assertIs(type(item["type"]), str)
            self.assertIs(type(item["unit"]), str)
            self.assertIs(type(item["reason"]), str)
            self.assertIs(type(item["value"]), float)
            self.assertIs(type(item["expected"]), float)
            self.assertIs(type(item["score"]), float)

        serialized = json.dumps(r1)
        self.assertIsInstance(serialized, str)


if __name__ == "__main__":
    unittest.main()

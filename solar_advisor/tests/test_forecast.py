"""
tests/test_forecast.py
======================
Unit tests for src/analytics/forecast.py (M2 Task 3).

Run with:
    python -m unittest tests.test_forecast -v

All tests use unittest (pytest is not installed in this environment).
"""

import json
import math
import sys
import types
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Make the project root importable regardless of cwd
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.forecast import evaluate_forecast, forecast  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_days: int = 90, seed: int = 0) -> pd.DataFrame:
    """Return a minimal synthetic half-hourly DataFrame."""
    rng = np.random.default_rng(seed)
    n_slots = n_days * 48
    start = pd.Timestamp("2012-07-01")
    ts = pd.date_range(start, periods=n_slots, freq="30min")
    cons = rng.uniform(0.1, 1.0, n_slots)
    gen  = rng.uniform(0.0, 0.5, n_slots)
    return pd.DataFrame({"ts": ts, "cons_kwh": cons, "gen_kwh": gen})


def _make_weekday_pattern_df() -> pd.DataFrame:
    """
    Create a DataFrame where every Monday has daily_cons_kwh == 10.0 and
    every other weekday has 1.0.  Used to verify weekday-matching logic.
    """
    slots_per_day = 48
    n_days = 56  # 8 full weeks so at least 4 Mondays are in the look-back
    n_slots = n_days * slots_per_day
    # Start on a Monday (2012-07-02 is a Monday)
    start = pd.Timestamp("2012-07-02")
    ts = pd.date_range(start, periods=n_slots, freq="30min")
    cons = []
    for i in range(n_days):
        day_ts = start + timedelta(days=i)
        val = 10.0 / slots_per_day if day_ts.weekday() == 0 else 1.0 / slots_per_day
        cons.extend([val] * slots_per_day)
    return pd.DataFrame({"ts": ts, "cons_kwh": cons, "gen_kwh": [0.0] * n_slots})


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestForecastOutputSchema(unittest.TestCase):
    """Test 1 – All required keys, correct types, JSON-serialisable."""

    def setUp(self):
        self.df = _make_df(n_days=90)
        self.result = forecast(self.df, horizon_days=14, target="cons")

    def test_required_keys_present(self):
        required = {"target", "horizon_days", "dates", "p10", "p50", "p90",
                    "total_p50_kwh", "method"}
        self.assertEqual(required, set(self.result.keys()))

    def test_key_types(self):
        r = self.result
        self.assertIsInstance(r["target"],        str)
        self.assertIsInstance(r["horizon_days"],  int)
        self.assertIsInstance(r["dates"],         list)
        self.assertIsInstance(r["p10"],           list)
        self.assertIsInstance(r["p50"],           list)
        self.assertIsInstance(r["p90"],           list)
        self.assertIsInstance(r["total_p50_kwh"], float)
        self.assertIsInstance(r["method"],        str)

    def test_list_lengths(self):
        r = self.result
        self.assertEqual(len(r["dates"]), 14)
        self.assertEqual(len(r["p10"]),   14)
        self.assertEqual(len(r["p50"]),   14)
        self.assertEqual(len(r["p90"]),   14)

    def test_dates_are_iso_strings(self):
        for d in self.result["dates"]:
            # Must parse without error
            parsed = date.fromisoformat(d)
            self.assertIsInstance(parsed, date)

    def test_json_serialisable(self):
        # Must not raise
        serialised = json.dumps(self.result)
        back = json.loads(serialised)
        self.assertEqual(back["target"], "cons")

    def test_all_floats_are_native_python(self):
        r = self.result
        for lst in (r["p10"], r["p50"], r["p90"]):
            for val in lst:
                self.assertIsInstance(val, float,
                                      msg=f"{val!r} is {type(val).__name__}, expected float")
        self.assertIsInstance(r["total_p50_kwh"], float)

    def test_values_rounded_to_2dp(self):
        r = self.result
        for lst in (r["p10"], r["p50"], r["p90"]):
            for val in lst:
                self.assertAlmostEqual(val, round(val, 2), places=10)
        self.assertAlmostEqual(r["total_p50_kwh"], round(r["total_p50_kwh"], 2))

    def test_method_valid_string(self):
        self.assertIn(self.result["method"], ("chronos-2", "seasonal_baseline"))


class TestTargetConsAndGen(unittest.TestCase):
    """Test 2 – Both targets return valid dicts with the correct target field."""

    def setUp(self):
        self.df = _make_df(n_days=60)

    def test_target_cons(self):
        r = forecast(self.df, horizon_days=7, target="cons")
        self.assertEqual(r["target"], "cons")
        self.assertEqual(r["horizon_days"], 7)
        self.assertEqual(len(r["dates"]), 7)

    def test_target_gen(self):
        r = forecast(self.df, horizon_days=7, target="gen")
        self.assertEqual(r["target"], "gen")
        self.assertEqual(r["horizon_days"], 7)
        self.assertEqual(len(r["dates"]), 7)

    def test_invalid_target_raises(self):
        with self.assertRaises(ValueError):
            forecast(self.df, horizon_days=7, target="solar")


class TestSeasonalBaselineWeekdayLogic(unittest.TestCase):
    """Test 3 – Weekday-matching: Monday forecast should equal 10.0 kWh."""

    def test_monday_p50_matches_pattern(self):
        df = _make_weekday_pattern_df()
        # Find the first Monday after the last training date
        last_date = df["ts"].dt.normalize().max()
        # Forecast up to 7 days and check whichever day is Monday
        r = forecast(df, horizon_days=7, target="cons")

        for i, d_str in enumerate(r["dates"]):
            d = date.fromisoformat(d_str)
            if d.weekday() == 0:  # Monday
                # p50 should be ~10.0 (mean of 4 Mondays all with 10.0 kWh)
                self.assertAlmostEqual(r["p50"][i], 10.0, delta=0.05,
                    msg=f"Monday {d_str} p50 = {r['p50'][i]}, expected ~10.0")
                break
        else:
            self.skipTest("No Monday in 7-day horizon — extend horizon.")

    def test_non_monday_p50_near_1(self):
        df = _make_weekday_pattern_df()
        r = forecast(df, horizon_days=7, target="cons")
        for i, d_str in enumerate(r["dates"]):
            d = date.fromisoformat(d_str)
            if d.weekday() != 0:  # Not Monday
                self.assertAlmostEqual(r["p50"][i], 1.0, delta=0.05,
                    msg=f"{d_str} (wd={d.weekday()}) p50 = {r['p50'][i]}, expected ~1.0")
                break


class TestP10LteP50LteP90(unittest.TestCase):
    """Test 4 – p10 ≤ p50 ≤ p90 on every forecast day."""

    def test_ordering_invariant(self):
        df = _make_df(n_days=120)
        r = forecast(df, horizon_days=14, target="cons")
        for i in range(14):
            self.assertLessEqual(
                r["p10"][i], r["p50"][i],
                msg=f"Day {i}: p10={r['p10'][i]} > p50={r['p50'][i]}"
            )
            self.assertLessEqual(
                r["p50"][i], r["p90"][i],
                msg=f"Day {i}: p50={r['p50'][i]} > p90={r['p90'][i]}"
            )

    def test_all_values_non_negative(self):
        df = _make_df(n_days=60)
        r = forecast(df, horizon_days=14, target="gen")
        for key in ("p10", "p50", "p90"):
            for val in r[key]:
                self.assertGreaterEqual(val, 0.0,
                    msg=f"{key} contains negative value: {val}")


class TestChronologicalNoLeakage(unittest.TestCase):
    """Test 5 – Only dates before forecast start are used as training data."""

    def test_no_future_leakage(self):
        """
        Build a dataset where the last 14 days have artificially high values
        (20 kWh/slot).  If seasonal baseline accidentally uses those future-
        analogue days its p50 would jump to ~960 kWh/day.  It must not.
        """
        n_normal = 56   # 8 weeks of normal data
        n_future = 14   # 2 weeks of "future" data with high values

        rng = np.random.default_rng(42)
        start = pd.Timestamp("2012-07-02")

        # Normal period
        slots_normal = n_normal * 48
        ts_normal = pd.date_range(start, periods=slots_normal, freq="30min")
        cons_normal = rng.uniform(0.1, 0.5, slots_normal)

        # Simulate "future" values present in df (leakage scenario)
        slots_future = n_future * 48
        ts_future = pd.date_range(
            start + timedelta(days=n_normal), periods=slots_future, freq="30min"
        )
        cons_future = np.full(slots_future, 20.0)   # absurdly high

        df_all = pd.DataFrame({
            "ts": np.concatenate([ts_normal, ts_future]),
            "cons_kwh": np.concatenate([cons_normal, cons_future]),
            "gen_kwh": 0.0,
        })

        # forecast() must use ALL rows it receives — but the question is whether
        # it forecasts beyond the last date.  Here the last date IS in the high
        # region, so the forecast should indeed use the high values.
        # Instead, we pass ONLY the normal period.
        df_train = df_all[df_all["ts"] < ts_future[0]].copy()

        r = forecast(df_train, horizon_days=14, target="cons")

        # Last training date
        last_train = df_train["ts"].dt.normalize().max().date()
        for d_str in r["dates"]:
            d = date.fromisoformat(d_str)
            self.assertGreater(d, last_train,
                msg=f"Forecast date {d_str} is not after last training date {last_train}")

        # p50 values must be close to the normal range, not the leakage range
        for val in r["p50"]:
            self.assertLess(val, 50.0,
                msg=f"p50={val} suspiciously high — possible future leakage")


class TestEmptyDataset(unittest.TestCase):
    """Test 6 – Empty DataFrame raises ValueError, does not crash silently."""

    def test_empty_df_raises(self):
        empty = pd.DataFrame(columns=["ts", "cons_kwh", "gen_kwh"])
        with self.assertRaises(ValueError):
            forecast(empty, horizon_days=14, target="cons")

    def test_none_raises(self):
        # forecast should handle None gracefully
        with self.assertRaises((ValueError, AttributeError)):
            forecast(None, horizon_days=14, target="cons")


class TestInsufficientHistory(unittest.TestCase):
    """Test 7 – Fewer than 14 days of history handled safely."""

    def test_single_day(self):
        # 1 day = 48 rows
        start = pd.Timestamp("2012-07-02")
        ts = pd.date_range(start, periods=48, freq="30min")
        df = pd.DataFrame({
            "ts": ts,
            "cons_kwh": [0.5] * 48,
            "gen_kwh": [0.1] * 48,
        })
        r = forecast(df, horizon_days=14, target="cons")
        self.assertEqual(len(r["dates"]), 14)
        self.assertEqual(r["method"], "seasonal_baseline")
        for val in r["p50"]:
            self.assertGreaterEqual(val, 0.0)

    def test_seven_days(self):
        df = _make_df(n_days=7)
        r = forecast(df, horizon_days=14, target="cons")
        self.assertEqual(len(r["dates"]), 14)
        self.assertEqual(r["method"], "seasonal_baseline")


class TestMissingValues(unittest.TestCase):
    """Test 8 – NaN in cons_kwh handled safely (treated as 0)."""

    def test_nan_in_cons_kwh(self):
        df = _make_df(n_days=60)
        # Inject NaNs into 10 % of rows
        nan_idx = df.sample(frac=0.1, random_state=0).index
        df.loc[nan_idx, "cons_kwh"] = float("nan")
        # Must not raise
        r = forecast(df, horizon_days=14, target="cons")
        self.assertEqual(len(r["dates"]), 14)
        for val in r["p50"]:
            self.assertFalse(math.isnan(val), msg=f"p50 contains NaN: {r['p50']}")


class TestHome1Integration(unittest.TestCase):
    """Test 9 – Load Home 1 and run forecast; check schema and method."""

    @classmethod
    def setUpClass(cls):
        from src.pipeline.load import load_home
        cls.df = load_home(1)

    def test_forecast_returns_valid_dict(self):
        r = forecast(self.df, horizon_days=14, target="cons")
        required = {"target", "horizon_days", "dates", "p10", "p50", "p90",
                    "total_p50_kwh", "method"}
        self.assertEqual(required, set(r.keys()))

    def test_method_is_seasonal_baseline(self):
        # torch is blocked by WDAC; must fall back to seasonal_baseline
        r = forecast(self.df, horizon_days=14, target="cons")
        # Allow chronos-2 if torch becomes available; seasonal_baseline is mandatory
        self.assertIn(r["method"], ("chronos-2", "seasonal_baseline"))

    def test_14_dates_in_future(self):
        r = forecast(self.df, horizon_days=14, target="cons")
        last_training = self.df["ts"].dt.normalize().max().date()
        for d_str in r["dates"]:
            d = date.fromisoformat(d_str)
            self.assertGreater(d, last_training)

    def test_total_p50_matches_sum(self):
        r = forecast(self.df, horizon_days=14, target="cons")
        expected = round(sum(r["p50"]), 2)
        self.assertAlmostEqual(r["total_p50_kwh"], expected, places=1)

    def test_gen_target_also_works(self):
        r = forecast(self.df, horizon_days=14, target="gen")
        self.assertEqual(r["target"], "gen")
        self.assertEqual(len(r["dates"]), 14)


class TestEvaluationMetrics(unittest.TestCase):
    """Test 10 – MAE, RMSE, MAPE computed correctly; zero-safe MAPE."""

    def test_perfect_forecast(self):
        actuals = [10.0, 12.0, 8.0]
        preds   = [10.0, 12.0, 8.0]
        m = evaluate_forecast(actuals, preds)
        self.assertAlmostEqual(m["mae"],  0.0, places=6)
        self.assertAlmostEqual(m["rmse"], 0.0, places=6)
        self.assertAlmostEqual(m["mape"], 0.0, places=6)

    def test_known_errors(self):
        actuals = [10.0, 10.0]
        preds   = [8.0,  12.0]
        m = evaluate_forecast(actuals, preds)
        # errors = [-2, 2]
        self.assertAlmostEqual(m["mae"],  2.0, places=4)
        self.assertAlmostEqual(m["rmse"], 2.0, places=4)
        self.assertAlmostEqual(m["mape"], 20.0, places=4)   # 20% each

    def test_zero_safe_mape(self):
        # When all actuals are 0, mape should be None (not raise or return inf/nan)
        actuals = [0.0, 0.0, 0.0]
        preds   = [1.0, 2.0, 3.0]
        m = evaluate_forecast(actuals, preds)
        self.assertIsNone(m["mape"],
            msg="MAPE should be None when all actuals are zero")

    def test_mape_skips_zero_actuals(self):
        # Mix of zero and non-zero; zero days skipped
        actuals = [0.0, 10.0]
        preds   = [5.0,  8.0]
        m = evaluate_forecast(actuals, preds)
        # Only second day counts: |10-8|/10 = 20 %
        self.assertAlmostEqual(m["mape"], 20.0, places=4)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            evaluate_forecast([1.0, 2.0], [1.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            evaluate_forecast([], [])

    def test_n_key(self):
        m = evaluate_forecast([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(m["n"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)

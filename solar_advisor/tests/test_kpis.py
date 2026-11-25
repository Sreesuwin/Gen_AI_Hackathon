"""Tests for M2 KPI Engine (src/analytics/kpis.py)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from src.analytics.kpis import compute_kpis, daily_profile, monthly_summary
from src.config import EXPORT_TARIFF, IMPORT_TARIFF
from src.pipeline.load import load_home

EXPECTED_KPI_KEYS = {
    "period",
    "days",
    "gen_kwh",
    "cons_kwh",
    "self_kwh",
    "import_kwh",
    "export_kwh",
    "self_consumption_pct",
    "self_sufficiency_pct",
    "baseload_kw",
    "peak_kw",
    "peak_hour",
    "avg_daily_gen_kwh",
    "avg_daily_cons_kwh",
    "grid_cost_inr",
    "export_credit_inr",
    "net_bill_inr",
}


class TestKpis(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.home1_full = load_home(1)
        cls.sample_home1 = pd.read_parquet(Path("data/sample/sample_home.parquet"))

    def test_compute_kpis_schema_and_types_full_home1(self) -> None:
        kpis = compute_kpis(self.home1_full)
        self.assertEqual(set(kpis.keys()), EXPECTED_KPI_KEYS)
        self.assertEqual(kpis["period"], "2010-07-01 to 2013-06-30")
        self.assertEqual(kpis["days"], 1096)

        # Native Python types & JSON compatibility
        for k, v in kpis.items():
            self.assertIn(type(v), (str, int, float), f"Non-native type for {k}: {type(v)}")
        serialized = json.dumps(kpis)
        self.assertIsInstance(serialized, str)

        # Energy balance and tariff math
        self.assertAlmostEqual(kpis["cons_kwh"], kpis["self_kwh"] + kpis["import_kwh"], delta=0.05)
        self.assertAlmostEqual(kpis["gen_kwh"], kpis["self_kwh"] + kpis["export_kwh"], delta=0.05)
        self.assertAlmostEqual(
            kpis["net_bill_inr"],
            round(kpis["grid_cost_inr"] - kpis["export_credit_inr"], 2),
            places=2,
        )

    def test_compute_kpis_sample_home1_and_tariffs(self) -> None:
        kpis = compute_kpis(self.sample_home1)
        self.assertEqual(kpis["period"], "2012-07-01 to 2013-06-30")
        self.assertEqual(kpis["days"], 365)
        self.assertAlmostEqual(
            kpis["grid_cost_inr"],
            round(float(self.sample_home1["import_kwh"].sum()) * IMPORT_TARIFF, 2),
            places=2,
        )
        self.assertAlmostEqual(
            kpis["export_credit_inr"],
            round(float(self.sample_home1["export_kwh"].sum()) * EXPORT_TARIFF, 2),
            places=2,
        )

    def test_baseload_and_peak_power_exact_formula(self) -> None:
        # 2 days, 48 slots each: night (01:00..04:30) = 0.25 kWh/slot -> 0.5 kW
        # daytime = 0.5 kWh/slot -> 1.0 kW; one single spike = 5.0 kWh -> 10.0 kW (1 of 96 < 99th pct)
        ts = pd.date_range("2025-01-01 00:00:00", periods=96, freq="30min")
        cons = pd.Series(0.5, index=range(96), dtype=float)
        night_idx = (ts.hour >= 1) & (ts.hour < 5)
        cons.loc[night_idx] = 0.25
        cons.iloc[30] = 5.0  # single spike (~1% of 96)
        df = pd.DataFrame(
            {
                "home_id": 1,
                "ts": ts,
                "gen_kwh": 0.0,
                "gc_kwh": cons,
                "cl_kwh": 0.0,
                "cons_kwh": cons,
                "self_kwh": 0.0,
                "import_kwh": cons,
                "export_kwh": 0.0,
                "valid": True,
            }
        )
        kpis = compute_kpis(df)
        self.assertEqual(kpis["baseload_kw"], 0.5)
        self.assertLess(kpis["peak_kw"], 10.0)
        self.assertGreater(kpis["peak_kw"], 1.0)

    def test_daily_profile_home1(self) -> None:
        prof = daily_profile(self.home1_full)
        self.assertEqual(list(prof.columns), ["hour", "gen_kw", "cons_kw"])
        self.assertEqual(len(prof), 24)
        self.assertEqual(prof["hour"].tolist(), list(range(24)))
        # Night hours should have zero solar generation
        self.assertTrue((prof.loc[prof["hour"].isin([0, 1, 2, 3, 22, 23]), "gen_kw"] == 0.0).all())
        # Midday peak solar should be positive
        self.assertGreater(float(prof.loc[prof["hour"] == 12, "gen_kw"].iloc[0]), 1.0)

    def test_monthly_summary_home1(self) -> None:
        monthly = monthly_summary(self.home1_full)
        self.assertEqual(
            list(monthly.columns),
            ["month", "gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_consumption_pct"],
        )
        self.assertEqual(len(monthly), 36)
        self.assertEqual(monthly["month"].iloc[0], "2010-07")
        self.assertEqual(monthly["month"].iloc[-1], "2013-06")
        self.assertTrue((monthly["self_consumption_pct"] >= 0.0).all())
        self.assertTrue((monthly["self_consumption_pct"] <= 100.0).all())

    def test_empty_dataset_handling(self) -> None:
        empty = self.home1_full.iloc[0:0].copy()
        kpis = compute_kpis(empty)
        self.assertEqual(set(kpis.keys()), EXPECTED_KPI_KEYS)
        self.assertEqual(kpis["days"], 0)
        self.assertEqual(kpis["net_bill_inr"], 0.0)
        json.dumps(kpis)

        prof = daily_profile(empty)
        self.assertEqual(list(prof.columns), ["hour", "gen_kw", "cons_kw"])
        self.assertTrue(prof.empty)

        monthly = monthly_summary(empty)
        self.assertEqual(
            list(monthly.columns),
            ["month", "gen_kwh", "cons_kwh", "import_kwh", "export_kwh", "self_consumption_pct"],
        )
        self.assertTrue(monthly.empty)


if __name__ == "__main__":
    unittest.main()

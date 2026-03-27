import os
import unittest
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

import numpy as np
import pandas as pd

from optimize_plan import build_tariff_prices, plan_battery_dispatch, align_interval_prices


@contextmanager
def patched_env(overrides):
    original = {}
    missing = object()
    for key, value in overrides.items():
        original[key] = os.environ.get(key, missing)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, old in original.items():
            if old is missing:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


class OptimizePlanTests(unittest.TestCase):
    def test_build_tariff_prices_uses_asymmetric_pricing(self):
        with patched_env(
            {
                "GRID_TRANSFER_EUR_PER_KWH": "0.05",
                "ELECTRICITY_TAX_EUR_PER_KWH": "0.03",
                "IMPORT_FIXED_ADDERS_EUR_PER_KWH": "0.01",
                "IMPORT_VAT_MULTIPLIER": "1.24",
                "EXPORT_DEDUCTION_EUR_PER_KWH": "0.02",
            }
        ):
            market = np.array([0.10, 0.20])
            import_prices, export_prices = build_tariff_prices(market)

        expected_import = (market + 0.05 + 0.03 + 0.01) * 1.24
        expected_export = np.maximum(0.0, market - 0.02)

        np.testing.assert_allclose(import_prices, expected_import, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(export_prices, expected_export, rtol=1e-9, atol=1e-9)

    def test_align_interval_prices_ffill_hourly_to_15min(self):
        # Hourly data
        raw_today = [
            {"start": "2026-03-26T00:00:00+00:00", "value": 0.10},
            {"start": "2026-03-26T01:00:00+00:00", "value": 0.20},
        ]
        # Target 15-min intervals
        prediction_timestamps = [
            datetime(2026, 3, 26, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 3, 26, 0, 15, tzinfo=timezone.utc),
            datetime(2026, 3, 26, 0, 30, tzinfo=timezone.utc),
            datetime(2026, 3, 26, 0, 45, tzinfo=timezone.utc),
            datetime(2026, 3, 26, 1, 0, tzinfo=timezone.utc),
        ]
        
        with patched_env({"PLAN_INTERVAL_MINUTES": "15"}):
            aligned = align_interval_prices(raw_today, [], prediction_timestamps)
        
        # All 00:xx should be 0.10, 01:00 should be 0.20
        expected = [0.10, 0.10, 0.10, 0.10, 0.20]
        np.testing.assert_allclose(aligned, expected)

    def test_align_interval_prices_handles_float_list(self):
        # List of floats (assuming hourly from midnight)
        raw_today = [0.10, 0.20, 0.30]
        now_date = datetime.now().date()
        prediction_timestamps = [
            pd.to_datetime(now_date, utc=True) + timedelta(hours=0),
            pd.to_datetime(now_date, utc=True) + timedelta(hours=1),
        ]
        
        aligned = align_interval_prices(raw_today, [], prediction_timestamps)
        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0], 0.10)
        self.assertEqual(aligned[1], 0.20)

    def test_solar_surplus_charges_battery_as_charge_solar(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "2",
                "BATTERY_MAX_DISCHARGE_KW": "2",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
                "PLAN_INTERVAL_HOURS": "1.0",
            }
        ):
            # 3.0 kWh solar surplus
            predictions = np.array([0.0])
            solar = np.array([3.0])
            import_prices = np.array([0.20])
            export_prices = np.array([0.10])

            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        self.assertEqual(plan[0]["battery_action"], "charge_solar")
        self.assertAlmostEqual(plan[0]["charge_from_solar_kwh"], 2.0) # limited by 2kW charge rate

    def test_high_import_price_discharges_battery_as_discharge_load(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "40",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "80",
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "false",
                "PLAN_INTERVAL_HOURS": "1.0",
            }
        ):
            # Load 4.0, Price high. Need 2+ points for percentile thresholds.
            predictions = np.array([4.0, 4.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.80, 0.80])
            export_prices = np.array([0.10, 0.10])

            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        self.assertEqual(plan[0]["battery_action"], "discharge_load")
        self.assertAlmostEqual(plan[0]["discharge_to_load_kwh"], 4.0)

    def test_soc_stays_within_bounds(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "40",
                "BATTERY_MIN_SOC_PCT": "20",
                "BATTERY_MAX_SOC_PCT": "80",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "8",
                "BATTERY_MAX_DISCHARGE_KW": "8",
                "BATTERY_CHARGE_EFFICIENCY": "0.95",
                "BATTERY_DISCHARGE_EFFICIENCY": "0.95",
                "BATTERY_ALLOW_EXPORT": "true",
                "PLAN_INTERVAL_HOURS": "1.0",
            }
        ):
            predictions = np.array([0.0] * 12 + [8.0] * 12)
            solar = np.array([10.0] * 12 + [0.0] * 12)
            import_prices = np.array([0.08] * 12 + [0.40] * 12)
            export_prices = np.array([0.05] * 12 + [0.20] * 12)

            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        min_soc_kwh = 40.0 * 0.20
        max_soc_kwh = 40.0 * 0.80
        for row in plan:
            self.assertGreaterEqual(row["soc_kwh"], min_soc_kwh - 1e-9)
            self.assertLessEqual(row["soc_kwh"], max_soc_kwh + 1e-9)

    def test_export_arbitrage_is_disabled_when_flag_false(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "40",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "90",
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "false",
                "PLAN_INTERVAL_HOURS": "1.0",
            }
        ):
            # No load, no solar. High export value would encourage export arbitrage if enabled.
            predictions = np.array([0.0, 0.0, 0.0, 0.0])
            solar = np.array([0.0, 0.0, 0.0, 0.0])
            import_prices = np.array([0.20, 0.20, 0.20, 0.20])
            export_prices = np.array([0.60, 0.70, 0.80, 0.90])

            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        self.assertTrue(all(row["battery_action"] == "idle" for row in plan))
        self.assertTrue(all(abs(row["grid_export_kwh"]) < 1e-9 for row in plan))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations
import os
import unittest
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

import numpy as np
import pandas as pd

from optimize_plan import build_tariff_prices, plan_battery_dispatch, align_interval_prices, plan_gshp_dispatch, optimize, compute_effective_cost
import sqlite3
import os
import json
from unittest.mock import patch, MagicMock

class OptimizeArchivingTests(unittest.TestCase):
    def setUp(self):
        # Use test-specific paths from environment (set by conftest.py)
        self.db_file = os.getenv('TEST_DB_PATH', 'test_hepo.db')
        self.predictions_file = os.getenv('TEST_PREDICTIONS_FILE', 'future_predictions.json')
        self.plan_file = os.getenv('TEST_PLAN_FILE', 'optimization_plan.json')
        
        # Create directories if they don't exist (only if not in current directory)
        for path in [self.db_file, self.predictions_file, self.plan_file]:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        
        # Setup dummy predictions
        self.predictions_data = [
            {
                "timestamp": "2026-04-05T13:15:00+03:00",
                "predicted_baseload": 2.0,
                "solar_forecast": 0.5,
                "outside_temp": 5.0,
                "is_sauna_active": 0,
                "is_fallback_price": 0
            }
        ]
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions_data, f)

    def tearDown(self):
        # No cleanup needed - conftest.py handles it via tmp_path fixture
        pass

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.get_db_connection')
    def test_optimize_archives_to_db(self, mock_db, mock_prices, mock_ha):
        # Mocking external calls
        mock_db.side_effect = lambda: sqlite3.connect(self.db_file)
        mock_prices.return_value = ([0.1], [0], "Nordpool", False, False, None)
        mock_ha.return_value = {"state": "50.0"} # acc_temp
        
        # Run optimize
        # We need to ensure we don't crash on other HA calls
        mock_ha.side_effect = lambda x: {"state": "0"} if x == "sensor.mlp_teho" else {"state": "50.0"}

        optimize()
        
        # Verify DB
        self.assertTrue(os.path.exists(self.db_file))
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        
        # Check if version column exists and is not 'unknown' (assuming git is available in test env)
        cur.execute("PRAGMA table_info(predictions)")
        columns = [c[1] for c in cur.fetchall()]
        self.assertIn('version', columns)
        self.assertIn('battery_action', columns)
        self.assertIn('import_price', columns)

        cur.execute("""
            SELECT 
                predicted_usage_kw, version, battery_action, battery_power_kw, 
                battery_soc_pct, import_price, export_price, grid_import_kwh, grid_export_kwh 
            FROM predictions LIMIT 1
        """)
        row = cur.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        # Predicted usage should be at least baseload (2.0)
        # GSHP might be 0 or more depending on prices/temp
        self.assertGreaterEqual(row[0], 2.0)
        self.assertNotEqual(row[1], "")
        
        # New battery columns should be populated (not None)
        self.assertIsNotNone(row[2]) # battery_action
        self.assertIsNotNone(row[5]) # import_price
        self.assertIsNotNone(row[6]) # export_price
        self.assertIsNotNone(row[7]) # grid_import_kwh

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.get_db_connection')
    def test_optimize_uses_live_battery_soc(self, mock_db, mock_prices, mock_ha):
        """When sensor.be_soc is available, the plan should start from that SOC, not 50%."""
        # Write zero-load predictions so the battery idles and SOC stays at initial value
        zero_load_predictions = [
            {
                "timestamp": "2026-04-05T13:15:00+03:00",
                "predicted_baseload": 0.0,
                "solar_forecast": 0.0,
                "outside_temp": 5.0,
                "is_sauna_active": 0,
                "is_fallback_price": 0,
                "ev_position": 0
            },
            {
                "timestamp": "2026-04-05T13:30:00+03:00",
                "predicted_baseload": 0.0,
                "solar_forecast": 0.0,
                "outside_temp": 5.0,
                "is_sauna_active": 0,
                "is_fallback_price": 0,
                "ev_position": 0
            },
            {
                "timestamp": "2026-04-05T13:45:00+03:00",
                "predicted_baseload": 0.0,
                "solar_forecast": 0.0,
                "outside_temp": 5.0,
                "is_sauna_active": 0,
                "is_fallback_price": 0,
                "ev_position": 0
            }
        ]
        with open(self.predictions_file, 'w') as f:
            json.dump(zero_load_predictions, f)

        mock_db.side_effect = lambda: sqlite3.connect(self.db_file)
        # Prices: first interval is neither cheapest import nor best export, so battery idles
        mock_prices.return_value = ([0.15, 0.1, 0.2], [0, 0, 0], "Nordpool", False, False, None)

        def ha_side_effect(entity_id):
            if entity_id == 'sensor.mlp_teho':
                return {"state": "0"}
            if entity_id == 'sensor.be_soc':
                return {"state": "73.0"}
            if entity_id == 'sensor.mlp_varaajan_lampotila':
                return {"state": "55.0"}  # GSHP at max temp, stays off
            if entity_id == 'input_boolean.battery_allow_export':
                return {"state": "off"}
            return {"state": "50.0"}
        mock_ha.side_effect = ha_side_effect

        optimize()

        self.assertTrue(os.path.exists(self.db_file))
        conn = sqlite3.connect(self.db_file)
        cur = conn.cursor()
        cur.execute("SELECT battery_soc_pct FROM predictions LIMIT 1")
        row = cur.fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertIsNotNone(row[0])
        self.assertAlmostEqual(row[0], 73.0, places=1,
            msg="Plan should start from live HA battery SOC (73%), not default 50%")


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

    def test_build_tariff_prices_skips_fees_when_inclusive(self):
        with patched_env(
            {
                "GRID_TRANSFER_EUR_PER_KWH": "0.05",
                "ELECTRICITY_TAX_EUR_PER_KWH": "0.03",
                "IMPORT_VAT_MULTIPLIER": "1.0",
            }
        ):
            market = np.array([0.10])
            # is_inclusive=True should skip 0.05 and 0.03
            import_prices, _ = build_tariff_prices(market, is_inclusive=True)
            
            self.assertEqual(import_prices[0], 0.10)

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
            aligned, is_fallback = align_interval_prices(raw_today, [], prediction_timestamps)
        
        # All 00:xx should be 0.10, 01:00 should be 0.20
        expected = [0.10, 0.10, 0.10, 0.10, 0.20]
        np.testing.assert_allclose(aligned, expected)
        self.assertFalse(any(is_fallback))

    def test_align_interval_prices_handles_float_list(self):
        # List of floats (assuming hourly from midnight)
        raw_today = [0.10, 0.20, 0.30]
        now_date = datetime.now().date()
        prediction_timestamps = [
            pd.to_datetime(now_date, utc=True) + timedelta(hours=0),
            pd.to_datetime(now_date, utc=True) + timedelta(hours=1),
        ]
        
        aligned, is_fallback = align_interval_prices(raw_today, [], prediction_timestamps)
        self.assertEqual(len(aligned), 2)
        self.assertEqual(aligned[0], 0.10)
        self.assertEqual(aligned[1], 0.20)

    def test_align_interval_prices_handles_15min_float_list(self):
        # List of floats with 15-minute spacing (96 values per day)
        raw_today = [0.10] * 4 + [0.20] * 4 + [0.30] * 88  # 96 values total
        now_date = datetime.now().date()
        prediction_timestamps = [
            pd.to_datetime(now_date, utc=True) + timedelta(minutes=0),
            pd.to_datetime(now_date, utc=True) + timedelta(minutes=15),
            pd.to_datetime(now_date, utc=True) + timedelta(minutes=30),
            pd.to_datetime(now_date, utc=True) + timedelta(minutes=45),
            pd.to_datetime(now_date, utc=True) + timedelta(minutes=60),
        ]

        aligned, is_fallback = align_interval_prices(raw_today, [], prediction_timestamps, interval_minutes=15)
        self.assertEqual(len(aligned), 5)
        # First hour (0-45 min) should all be 0.10
        for i in range(4):
            self.assertEqual(aligned[i], 0.10, f"index {i} should be 0.10")
        # 60 min should be 0.20
        self.assertEqual(aligned[4], 0.20)

    def test_align_interval_prices_24h_fallback(self):
        # Data for today 08:00
        raw_today = [
            {"start": "2026-03-26T08:00:00+00:00", "value": 0.15},
        ]
        # Prediction for tomorrow 08:00
        prediction_timestamps = [
            datetime(2026, 3, 27, 8, 0, tzinfo=timezone.utc),
        ]
        
        aligned, is_fallback = align_interval_prices(raw_today, [], prediction_timestamps, interval_minutes=60)
        
        # Should fallback to 0.15 from 24h ago
        self.assertEqual(aligned[0], 0.15)
        # Should be flagged as fallback
        self.assertTrue(is_fallback[0])

    def test_solar_surplus_charges_battery_when_storing_is_better(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "5",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "2",
                "BATTERY_MAX_DISCHARGE_KW": "2",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # 3.0 kWh solar surplus, but future load at 0.20 makes storing worthwhile
            # Battery is 5kWh at 50% = 2.5kWh, room = 2.0kWh, so it charges from solar
            # Current import is NOT the cheapest, so grid charge won't happen
            predictions = np.array([0.0, 2.0])
            solar = np.array([3.0, 0.0])
            import_prices = np.array([0.20, 0.20])
            export_prices = np.array([0.20, 0.20])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        self.assertEqual(plan[0]["battery_action"], "charge_solar")
        self.assertAlmostEqual(plan[0]["charge_from_solar_kwh"], 2.0) # limited by 2kW charge rate

    def test_high_price_solar_exported_instead_of_charging(self):
        """When export prices are high and round-trip efficiency makes storing
        solar less valuable than exporting, solar should go to grid."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "30",
                "BATTERY_MAX_CHARGE_KW": "2",
                "BATTERY_MAX_DISCHARGE_KW": "2",
                "BATTERY_CHARGE_EFFICIENCY": "0.95",
                "BATTERY_DISCHARGE_EFFICIENCY": "0.95",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # Interval 0: high export (0.24), solar surplus 3.0 kWh
            # Interval 1: cheap import (0.10), load 2.0 kWh
            # Interval 2: high import (0.25), load 2.0 kWh
            predictions = np.array([0.0, 2.0, 2.0])
            solar = np.array([3.0, 0.0, 0.0])
            import_prices = np.array([0.24, 0.10, 0.25])
            export_prices = np.array([0.24, 0.10, 0.25])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: battery at 3.0 kWh (30%), available output = 1.9 kWh
        # opportunity_cost = 0.25 (only enough for interval 2)
        # store_value = 0.25 * 0.95 * 0.95 = 0.2256
        # current_export = 0.24 > 0.2256
        # So solar should be exported, not stored
        self.assertEqual(plan[0]["battery_action"], "idle")
        self.assertAlmostEqual(plan[0]["charge_from_solar_kwh"], 0.0)
        self.assertAlmostEqual(plan[0]["grid_export_kwh"], 3.0)  # solar goes to grid

        # Interval 1: cheap import, should grid charge
        self.assertEqual(plan[1]["battery_action"], "charge_grid")

        # Interval 2: high import, should discharge
        self.assertEqual(plan[2]["battery_action"], "discharge_load")

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
            }
        ):
            # Load 4.0, Price high. Need 2+ points for percentile thresholds.
            predictions = np.array([4.0, 4.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.80, 0.80])
            export_prices = np.array([0.10, 0.10])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
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
            }
        ):
            predictions = np.array([0.0] * 12 + [8.0] * 12)
            solar = np.array([10.0] * 12 + [0.0] * 12)
            import_prices = np.array([0.08] * 12 + [0.40] * 12)
            export_prices = np.array([0.05] * 12 + [0.20] * 12)

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
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
            }
        ):
            # No load, no solar. High export value would encourage export arbitrage if enabled.
            predictions = np.array([0.0, 0.0, 0.0, 0.0])
            solar = np.array([0.0, 0.0, 0.0, 0.0])
            import_prices = np.array([0.20, 0.20, 0.20, 0.20])
            export_prices = np.array([0.60, 0.70, 0.80, 0.90])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        self.assertTrue(all(row["battery_action"] == "idle" for row in plan))
        self.assertTrue(all(abs(row["grid_export_kwh"]) < 1e-9 for row in plan))

    def test_summer_no_grid_charge_when_solar_covers_load(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
                "MAIN_FUSE_SIZE_A": "25",
            }
        ):
            # 24 hours: night load 1, day load 1 with 5 solar surplus
            predictions = np.array([1.0]*6 + [1.0]*13 + [1.0]*5)  # 24h, load 1 kWh/h
            solar = np.array([0.0]*6 + [5.0]*13 + [0.0]*5)  # solar 5 kWh/h during day
            import_prices = np.array([0.05]*6 + [0.30]*13 + [0.30]*5)
            export_prices = np.array([0.03]*6 + [0.20]*13 + [0.20]*5)
            committed = np.zeros(24)

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices, committed)

        # Night hours (0-5): should NOT grid charge (discharging to cover load is OK)
        for i in range(6):
            self.assertAlmostEqual(plan[i]["charge_from_grid_kwh"], 0.0, places=5, msg=f"Hour {i} should not grid charge")
        
        # In a high-solar summer day, the battery should NEVER charge from the grid
        # (solar covers everything; grid charging would just displace free solar)
        for i in range(24):
            self.assertAlmostEqual(plan[i]["charge_from_grid_kwh"], 0.0, places=5, msg=f"Hour {i} should not grid charge")
        
        # Evening hours: should discharge while energy is available.
        # With export enabled, remaining inverter capacity may also be exported when
        # current export is the best remaining price. Once battery hits min_soc it goes idle.
        for i in range(19, 24):
            self.assertIn(plan[i]["battery_action"], ("discharge_load", "discharge_mixed", "follow"), f"Hour {i} unexpected action")

    def test_grid_capacity_limits_battery_charge(self):
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "10",  # almost empty
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
                "MAIN_FUSE_SIZE_A": "25",
            }
        ):
            # Hour 0: cheap, house load 2 kWh, EV 11 kWh. Total non-battery = 13 kWh.
            # Fuse limit: 17.25 kWh. Available for battery: 4.25 kWh.
            # Battery max charge: 5 kWh.
            predictions = np.array([2.0, 2.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.05, 0.30])
            export_prices = np.array([0.03, 0.20])
            committed = np.array([11.0, 0.0])  # EV charging 11 kWh in first hour

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices, committed)

        # Hour 0: should charge from grid but limited by fuse
        self.assertEqual(plan[0]["battery_action"], "charge_grid")
        self.assertAlmostEqual(plan[0]["charge_from_grid_kwh"], 4.25, places=2)
        
        # Hour 1: should discharge to cover load and export the remaining capacity
        self.assertEqual(plan[1]["battery_action"], "discharge_mixed")
        self.assertAlmostEqual(plan[1]["discharge_to_load_kwh"], 2.0, places=2)
        self.assertGreater(plan[1]["discharge_to_export_kwh"], 0.0)

    def test_grid_charge_allowed_when_profitable_and_pv_insufficient(self):
        """Grid charging is allowed when profitable and future solar surplus won't fill the battery."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # Hour 0: cheap night, no solar, small load
            # Hour 1-3: expensive day, modest solar (surplus 3 kWh total, room 4 kWh)
            predictions = np.array([1.0, 1.0, 1.0, 1.0])
            solar = np.array([0.0, 2.0, 2.0, 2.0])
            import_prices = np.array([0.05, 0.30, 0.30, 0.30])
            export_prices = np.array([0.03, 0.20, 0.20, 0.20])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Hour 0 should grid-charge because it's cheap and profitable compared to
        # later expensive hours, and solar surplus (3 kWh) < battery room (4 kWh).
        self.assertEqual(plan[0]["battery_action"], "charge_grid", "Hour 0 should grid charge when PV is insufficient")
        self.assertGreater(plan[0]["charge_from_grid_kwh"], 0.0)

    def test_no_grid_charge_when_solar_can_fill_battery(self):
        """No grid charging when future solar surplus is enough to fill the battery."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # Hour 0: cheap night, no solar, small load
            # Hour 1-3: expensive day, large solar (surplus 9 kWh total, room 4 kWh)
            predictions = np.array([1.0, 1.0, 1.0, 1.0])
            solar = np.array([0.0, 5.0, 5.0, 5.0])
            import_prices = np.array([0.05, 0.30, 0.30, 0.30])
            export_prices = np.array([0.03, 0.20, 0.20, 0.20])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Solar surplus (9 kWh) > battery room (4 kWh), so no grid charge needed.
        self.assertEqual(plan[0]["battery_action"], "follow", "Hour 0 should not grid charge when solar can fill battery")
        self.assertAlmostEqual(plan[0]["charge_from_grid_kwh"], 0.0, places=5)

    def test_no_pressure_discharge_at_cheap_intervals(self):
        """Battery should not discharge at cheap intervals just to make room for solar."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "80",
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # Hour 0: cheap (0.05), no load, no solar. Best future price is 0.20.
            # Hour 1: expensive (0.20), no load, no solar.
            predictions = np.array([0.0, 0.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.05, 0.20])
            export_prices = np.array([0.03, 0.20])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # At hour 0, current import (0.05) < best future value (0.20).
        # Discharging now would waste energy that is more valuable later.
        self.assertAlmostEqual(plan[0]["discharge_to_load_kwh"], 0.0, places=5, msg="Should not discharge to load at cheap interval")
        self.assertAlmostEqual(plan[0]["discharge_to_export_kwh"], 0.0, places=5, msg="Should not discharge to export at cheap interval")

    def test_discharge_only_when_current_price_higher_than_future(self):
        """Battery should discharge to load only when current import >= best future value."""
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
            }
        ):
            # Load 4.0, Price 0.80. Both intervals same.
            predictions = np.array([4.0, 4.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.80, 0.80])
            export_prices = np.array([0.10, 0.10])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # current_import == best_future_value, so discharging is allowed.
        self.assertEqual(plan[0]["battery_action"], "discharge_load")
        self.assertAlmostEqual(plan[0]["discharge_to_load_kwh"], 4.0)

    def test_discharge_mixed_when_import_and_export_both_high(self):
        """When import is high and current export is the best remaining,
        battery should cover load AND export remaining inverter capacity."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "50",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "80",
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # High load, high prices now. Cheap prices later.
            predictions = np.array([1.3, 1.3, 1.3])
            solar = np.array([0.0, 0.0, 0.0])
            import_prices = np.array([0.12, 0.06, 0.06])
            export_prices = np.array([0.12, 0.06, 0.06])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: import 0.12 >= best_future 0.06 → discharge to load 1.3 kWh
        # current_export 0.12 >= max_future_export 0.06 → export remaining capacity
        self.assertEqual(plan[0]["battery_action"], "discharge_mixed")
        self.assertAlmostEqual(plan[0]["discharge_to_load_kwh"], 1.3)
        self.assertAlmostEqual(plan[0]["discharge_to_export_kwh"], 8.7)  # 10 - 1.3
        self.assertAlmostEqual(plan[0]["battery_power_kw"], -10.0)

    def test_export_arbitrage_when_future_import_cheaper(self):
        """Export now at high export price when future import is cheap enough
        to make export-now + import-later profitable."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "50",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "80",
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # No load. High export now (0.12), cheap import later (0.05).
            # Not the best future export (0.13 tomorrow), but arbitrage is profitable.
            predictions = np.array([0.0, 0.0, 0.0])
            solar = np.array([0.0, 0.0, 0.0])
            import_prices = np.array([0.12, 0.05, 0.05])
            export_prices = np.array([0.12, 0.13, 0.10])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: current_export 0.12 is NOT best (0.13 later), but
        # 0.12 > min_future_import 0.05 / 1.0 → arbitrage profitable
        self.assertEqual(plan[0]["battery_action"], "discharge_export")
        self.assertAlmostEqual(plan[0]["discharge_to_export_kwh"], 10.0)
        self.assertAlmostEqual(plan[0]["battery_power_kw"], -10.0)

    def test_no_export_when_arbitrage_not_profitable(self):
        """If future import is not cheap enough, don't export just because
        current export is decent."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "50",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "80",
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # Export now 0.08. Future export is better (0.09) and future import
            # is not cheap enough (0.10) to make arbitrage profitable.
            predictions = np.array([0.0, 0.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.10, 0.10])
            export_prices = np.array([0.08, 0.09])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: current_export 0.08 is NOT best (0.09 later).
        # 0.08 > 0.10 / 1.0 is False → no arbitrage. No profitable grid charge either.
        self.assertEqual(plan[0]["battery_action"], "idle")
        self.assertAlmostEqual(plan[0]["discharge_to_export_kwh"], 0.0)

    def test_battery_preserves_capacity_for_expensive_night(self):
        """Battery should not dump energy at cheap evening prices when
        expensive night hours are ahead."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "90",
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # 8 x 15-min intervals. Constant 1.0 kWh load per interval, no solar.
            # Prices ramp up to a peak, then fall.
            predictions = np.array([1.0] * 8)
            solar = np.array([0.0] * 8)
            import_prices = np.array([0.13, 0.15, 0.18, 0.22, 0.25, 0.24, 0.20, 0.10])
            export_prices = np.array([0.13, 0.15, 0.18, 0.22, 0.25, 0.24, 0.20, 0.10])

            with patched_env({"PLAN_INTERVAL_MINUTES": "15"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # The cheap ramp-up intervals (0.13-0.18) should preserve energy
        # for the more expensive hours ahead.
        for i in range(3):
            self.assertEqual(plan[i]["battery_action"], "follow",
                             f"Interval {i} (price={import_prices[i]}) should not discharge")
            self.assertAlmostEqual(plan[i]["discharge_to_export_kwh"], 0.0, places=5,
                                   msg=f"Interval {i} should not export")
            self.assertAlmostEqual(plan[i]["discharge_to_load_kwh"], 0.0, places=5,
                                   msg=f"Interval {i} should not discharge to load")

        # At interval 3 (price 0.22) the marginal future opportunity is only 0.10,
        # so discharging at 0.22 is still profitable. The battery may discharge.
        self.assertIn(plan[3]["battery_action"], ("follow", "discharge_mixed", "discharge_load", "discharge_export"),
                      f"Interval 3 (price={import_prices[3]}) unexpected action")

        # At the peak interval (0.25) the battery should definitely discharge.
        self.assertIn(plan[4]["battery_action"], ("discharge_mixed", "discharge_load", "discharge_export"))

        # Because it stayed idle during the cheap ramp-up, the battery entered
        # interval 3 with full SOC.  Even if it discharges some at 0.22, it
        # should still have enough energy left to cover the 0.25 peak.
        self.assertGreaterEqual(plan[4]["soc_kwh"], 2.0,
                                "SOC should be preserved enough to cover the peak price interval")

    def test_lookahead_window_respects_max_lookahead_hours(self):
        """When a distant peak is beyond the max_lookahead_hours, the battery
        should not reserve energy for it."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "90",
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "true",
            }
        ):
            # 24 hourly intervals: price 0.30 for first 8h, 0.10 for next 8h, 0.50 for last 8h
            predictions = np.array([1.0] * 24)
            solar = np.array([0.0] * 24)
            import_prices = np.array([0.30] * 8 + [0.10] * 8 + [0.50] * 8)
            export_prices = np.array([0.30] * 8 + [0.10] * 8 + [0.50] * 8)

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                # With max_lookahead_hours=8, the 0.50 peak is beyond the window
                # so the battery sees only 0.30 and discharges at hour 0
                plan_short = plan_battery_dispatch(
                    predictions, solar, import_prices, export_prices,
                    max_lookahead_hours=8.0
                )
                # With max_lookahead_hours=24, the 0.50 peak is visible
                # so the battery preserves energy for the peak and stays idle at hour 0
                plan_long = plan_battery_dispatch(
                    predictions, solar, import_prices, export_prices,
                    max_lookahead_hours=24.0
                )

        # Short lookahead: battery discharges at hour 0 because it only sees 0.30 ahead
        self.assertGreater(plan_short[0]["discharge_to_load_kwh"], plan_long[0]["discharge_to_load_kwh"],
                           "Short lookahead should discharge more at hour 0")
        # Short lookahead: battery ends up needing to charge from grid at hour 8 (cheap price)
        # because it discharged too early. Long lookahead preserves energy.
        self.assertGreater(plan_short[8]["charge_from_grid_kwh"], plan_long[8]["charge_from_grid_kwh"],
                           "Short lookahead should charge from grid at cheap hour 8")

    def test_local_arbitrage_15min_spread(self):
        """Battery should grid-charge in cheap period to discharge in imminent expensive period,
        even if a cheaper period exists further out. Near-term opportunities take precedence."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "10",  # At minimum
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "false",
                "BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH": "0.005",
            }
        ):
            # Three 15-min intervals: cheap (0.091), expensive (0.119), cheaper (0.050)
            # Load 1.0 kWh each interval
            predictions = np.array([1.0, 1.0, 1.0])
            solar = np.array([0.0, 0.0, 0.0])
            import_prices = np.array([0.091, 0.119, 0.050])
            export_prices = np.array([0.050, 0.050, 0.050])

            with patched_env({"PLAN_INTERVAL_MINUTES": "15"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0 (cheap 0.091): should grid-charge to prepare for expensive interval 1
        # Margin check: 0.119 - 0.091 = 0.028 > 0.005, so profitable
        self.assertEqual(plan[0]["battery_action"], "charge_grid",
                        "Interval 0 should grid-charge for profitable near-term discharge")
        self.assertGreater(plan[0]["charge_from_grid_kwh"], 0.0)

        # Interval 1 (expensive 0.119): should discharge to load (profitable vs 0.091)
        # After charging in interval 0, SOC should be sufficient
        self.assertIn(plan[1]["battery_action"], ("discharge_load", "discharge_mixed", "follow"),
                      "Interval 1 (expensive) should discharge or prepare")

    def test_grid_charge_sized_to_near_term_need(self):
        """Grid charge should be sized to actual near-term load, not max power."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "20",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "10",  # At minimum
                "BATTERY_MAX_CHARGE_KW": "10",
                "BATTERY_MAX_DISCHARGE_KW": "10",
                "BATTERY_CHARGE_EFFICIENCY": "0.95",
                "BATTERY_DISCHARGE_EFFICIENCY": "0.95",
                "BATTERY_ALLOW_EXPORT": "false",
                "BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH": "0.005",
            }
        ):
            # Hourly intervals: cheap now (0.05), expensive soon (0.20), cheap later (0.05)
            # Load 2.0 kWh/h for 2h, then no load
            predictions = np.array([2.0, 2.0, 0.0])
            solar = np.array([0.0, 0.0, 0.0])
            import_prices = np.array([0.05, 0.20, 0.05])
            export_prices = np.array([0.03, 0.10, 0.03])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: should charge enough for interval 1's load (~2 kWh output needed)
        # Accounting for round-trip efficiency: 2.0 / (0.95 * 0.95) ≈ 2.22 kWh input
        self.assertEqual(plan[0]["battery_action"], "charge_grid")
        # Should be less than max_charge (10 kWh) and sized to need
        self.assertGreater(plan[0]["charge_from_grid_kwh"], 1.0)
        self.assertLess(plan[0]["charge_from_grid_kwh"], 5.0,
                       "Charge should be sized to near-term need, not max power")

    def test_prefer_nearer_cheap_window_over_distant(self):
        """When no near-term expensive load, don't charge at distant cheap intervals."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "50",
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "false",
                "BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH": "0.005",
            }
        ):
            # Prices: 0.05 (cheap), 0.20 (expensive), 0.20, 0.03 (cheaper, far away)
            # No load at all
            predictions = np.array([0.0, 0.0, 0.0, 0.0])
            solar = np.array([0.0, 0.0, 0.0, 0.0])
            import_prices = np.array([0.05, 0.20, 0.20, 0.03])
            export_prices = np.array([0.03, 0.10, 0.10, 0.02])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0 (0.05): no profitable near-term discharge (no load before 0.03)
        # Should NOT grid-charge just because 0.03 exists in future without pressure
        self.assertEqual(plan[0]["battery_action"], "idle",
                        "Should not charge at cheap interval without near-term discharge need")
        self.assertAlmostEqual(plan[0]["charge_from_grid_kwh"], 0.0, places=5)

    def test_no_charge_on_unprofitable_spread(self):
        """Reject small unprofitable spreads (round-trip cost > profit)."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "10",  # At minimum
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "0.95",
                "BATTERY_DISCHARGE_EFFICIENCY": "0.95",
                "BATTERY_ALLOW_EXPORT": "false",
                "BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH": "0.005",
            }
        ):
            # Tiny 1% spread: 0.100 -> 0.101. With ~5% round-trip loss, unprofitable.
            # Load 1.0 kWh in interval 1.
            predictions = np.array([1.0, 1.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.100, 0.101])
            export_prices = np.array([0.050, 0.051])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: spread is too small to overcome round-trip loss
        self.assertEqual(plan[0]["battery_action"], "follow",
                        "Should not charge on unprofitable micro-spread")
        self.assertAlmostEqual(plan[0]["charge_from_grid_kwh"], 0.0, places=5)

    def test_solar_inclusive_arbitrage(self):
        """Battery should grid-charge for near-term arbitrage even with solar present."""
        with patched_env(
            {
                "BATTERY_CAPACITY_KWH": "10",
                "BATTERY_MIN_SOC_PCT": "10",
                "BATTERY_MAX_SOC_PCT": "90",
                "BATTERY_INITIAL_SOC_PCT": "10",  # At minimum
                "BATTERY_MAX_CHARGE_KW": "5",
                "BATTERY_MAX_DISCHARGE_KW": "5",
                "BATTERY_CHARGE_EFFICIENCY": "1.0",
                "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
                "BATTERY_ALLOW_EXPORT": "false",
                "BATTERY_GRID_CHARGE_MIN_MARGIN_EUR_PER_KWH": "0.005",
            }
        ):
            # Interval 0: cheap import (0.083), some solar (2.0 kW), small load (1.0 kW)
            # Interval 1: expensive import (0.105), less solar (1.5 kW), large load (3.0 kW)
            # Interval 2: cheap import (0.080), good solar (3.0 kW)
            # Even though solar exists, interval 0 is cheaper than interval 1.
            # Should grid-charge in interval 0 to prepare for expensive interval 1 load.
            predictions = np.array([1.0, 3.0, 0.0])
            solar = np.array([2.0, 1.5, 3.0])
            import_prices = np.array([0.083, 0.105, 0.080])
            export_prices = np.array([0.050, 0.050, 0.050])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
                plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # Interval 0: should grid-charge despite solar being present
        # Spread: 0.105 - 0.083 = 0.022 > 0.005 margin, profitable
        self.assertEqual(plan[0]["battery_action"], "charge_mixed",  # solar + grid
                        "Should grid-charge for near-term arbitrage even with solar")
        self.assertGreater(plan[0]["charge_from_grid_kwh"], 0.0)

        # Interval 1: should discharge to load (expensive period)
        self.assertIn(plan[1]["battery_action"], ("discharge_load", "discharge_mixed", "follow"),
                      "Interval 1 (expensive) should discharge")


class EffectiveCostTests(unittest.TestCase):
    def test_effective_cost_when_grid_importing(self):
        """When grid is importing, extra load increases import → cost = import_price."""
        entry = {
            'grid_import_kwh': 1.0,
            'grid_export_kwh': 0.0,
            'battery_action': 'discharge_load',
            'import_unit_price': 0.20,
            'export_unit_price': 0.10,
            'net_load_without_battery_kwh': 2.0,
        }
        self.assertAlmostEqual(compute_effective_cost(entry), 0.20)

    def test_effective_cost_when_grid_exporting(self):
        """When grid is exporting, extra load reduces export → cost = export_price."""
        entry = {
            'grid_import_kwh': 0.0,
            'grid_export_kwh': 1.0,
            'battery_action': 'charge_solar',
            'import_unit_price': 0.20,
            'export_unit_price': 0.10,
            'net_load_without_battery_kwh': -3.0,
        }
        self.assertAlmostEqual(compute_effective_cost(entry), 0.10)

    def test_effective_cost_when_battery_discharging_no_grid(self):
        """When battery discharges and net grid is zero, cost = import_price (conservative)."""
        entry = {
            'grid_import_kwh': 0.0,
            'grid_export_kwh': 0.0,
            'battery_action': 'discharge_load',
            'import_unit_price': 0.25,
            'export_unit_price': 0.10,
            'net_load_without_battery_kwh': 1.0,
        }
        self.assertAlmostEqual(compute_effective_cost(entry), 0.25)

    def test_effective_cost_when_charging_from_solar_no_grid(self):
        """When battery charges from solar and net grid is zero, solar is free → cost = 0."""
        entry = {
            'grid_import_kwh': 0.0,
            'grid_export_kwh': 0.0,
            'battery_action': 'charge_solar',
            'import_unit_price': 0.20,
            'export_unit_price': 0.10,
            'net_load_without_battery_kwh': -2.0,
        }
        self.assertAlmostEqual(compute_effective_cost(entry), 0.0)

    def test_effective_cost_when_idle_with_solar_surplus(self):
        """When idle with solar surplus, extra load consumes free solar → cost = 0."""
        entry = {
            'grid_import_kwh': 0.0,
            'grid_export_kwh': 0.0,
            'battery_action': 'idle',
            'import_unit_price': 0.20,
            'export_unit_price': 0.10,
            'net_load_without_battery_kwh': -1.0,
        }
        self.assertAlmostEqual(compute_effective_cost(entry), 0.0)

    def test_effective_cost_when_idle_no_surplus(self):
        """When idle with no surplus, extra load causes import → cost = import_price."""
        entry = {
            'grid_import_kwh': 0.0,
            'grid_export_kwh': 0.0,
            'battery_action': 'idle',
            'import_unit_price': 0.20,
            'export_unit_price': 0.10,
            'net_load_without_battery_kwh': 0.0,
        }
        self.assertAlmostEqual(compute_effective_cost(entry), 0.20)

    def test_effective_cost_on_realistic_plan_entries(self):
        """effective_cost should be computable on realistic full plan entries."""
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
                "PLAN_INTERVAL_MINUTES": "60",
            }
        ):
            predictions = np.array([1.0, 1.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.10, 0.30])
            export_prices = np.array([0.05, 0.20])
            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        for entry in plan:
            # Simulate the full entry as built by optimize()
            full_entry = {
                'import_unit_price': 0.10,
                'export_unit_price': 0.05,
                **entry,
            }
            cost = compute_effective_cost(full_entry)
            self.assertIsInstance(cost, float)
            self.assertGreaterEqual(cost, 0.0)


class GSHPPlanTests(unittest.TestCase):
    def test_gshp_starts_at_min_temp(self):
        # Initial temp is 65.0C. High loss, but lookahead window (8h) shouldn't hit 45.0 yet.
        prediction_timestamps = [datetime.now()] * 40
        outside_temps = [10.0] * 40
        import_prices = [0.20] * 40
        solar_forecast_kw = [0.0] * 40
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "65.0",
            "GSHP_MIN_TEMP": "45.0",
            "GSHP_IS_RUNNING": "false",
            "GSHP_HEAT_LOSS_K": "0.1",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            is_sauna_active = [0] * len(prediction_timestamps)
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # Should stay STOP for the first several intervals
        self.assertEqual(plan[0]["gshp_intent"], "STOP")
        self.assertEqual(plan[10]["gshp_intent"], "STOP")

    def test_gshp_preheats_during_cheap_prices(self):
        # Temp is 47C (safe). But next hour is expensive.
        prediction_timestamps = [datetime.now()] * 10
        outside_temps = [0.0] * 10
        # Price is cheap now (0.05), but spikes to 0.50 later
        import_prices = [0.05, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
        solar_forecast_kw = [0.0] * 10
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "47.0",
            "GSHP_MIN_TEMP": "45.0",
            "GSHP_IS_RUNNING": "false",
            "GSHP_HEAT_LOSS_K": "0.1",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            is_sauna_active = [0] * len(prediction_timestamps)
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # Should start immediately to take advantage of the 0.05 price
        self.assertEqual(plan[0]["gshp_intent"], "START")

    def test_gshp_stops_at_max_temp(self):
        prediction_timestamps = [datetime.now()] * 4
        outside_temps = [20.0] * 4 # No heat loss (baseline disabled for this test)
        import_prices = [0.20] * 4
        solar_forecast_kw = [0.0] * 4
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "54.9",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_IS_RUNNING": "true",
            "GSHP_ELECTRIC_POWER_KW": "4.0",
            "GSHP_COP": "3.5",
            "GSHP_BASELINE_DEMAND_KW": "0.0",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            is_sauna_active = [0] * len(prediction_timestamps)
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # First interval it's already running and stays running until it crosses 55
        self.assertEqual(plan[0]["gshp_intent"], "START")
        # After one interval of 14kW heat into 500L, it definitely hits 55
        self.assertEqual(plan[1]["gshp_intent"], "STOP")

    def test_gshp_accounts_for_layering_drop(self):
        prediction_timestamps = [datetime.now()] * 2
        outside_temps = [10.0] * 2
        import_prices = [0.01] * 2 # Force start
        solar_forecast_kw = [0.0] * 2
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "50.0",
            "GSHP_INITIAL_TEMP_DROP": "3.0",
            "GSHP_IS_RUNNING": "false",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            is_sauna_active = [0] * len(prediction_timestamps)
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # Initial temp 50. After start, sim_temp should include the -3.0 drop
        # plus the heat gain from the interval. 
        # Heat gain: (14kW * 0.25h) / 0.58 kWh/C = ~6C.
        # Net: 50 - 3 + 6 = 53.
        self.assertLess(plan[0]["gshp_temp_sim"], 55.0)
        self.assertGreater(plan[0]["gshp_temp_sim"], 51.0) # 50 - 3 + some gain

    def test_gshp_uses_solar_arbitrage(self):
        # Temp is high (54C). 
        # Import price is high now (0.30), but we have high solar (10kW).
        # Export price is lower (0.10).
        prediction_timestamps = [datetime.now() + timedelta(minutes=15*i) for i in range(4)]
        outside_temps = [10.0] * 4
        import_prices = [0.30] * 4
        export_prices = [0.10] * 4
        solar_forecast_kw = [10.0] * 4
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "54.0",
            "GSHP_IS_RUNNING": "false",
            "PLAN_INTERVAL_MINUTES": "15",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_MAX_TEMP": "60.0"
        }):
            plan = plan_gshp_dispatch(prediction_timestamps, [0]*4, outside_temps, import_prices, export_prices, solar_forecast_kw)
            
        # Should start immediately because effective price is export_price (0.10) 
        # which is cheaper than import_price (0.30)
        self.assertEqual(plan[0]["gshp_intent"], "START")

    def test_gshp_power_ramp(self):
        # Verify that power increases as temp increases
        prediction_timestamps = [datetime.now() + timedelta(minutes=15*i) for i in range(20)]
        outside_temps = [20.0] * 20 # No heat loss (baseline disabled for this test)
        import_prices = [0.01] * 20 # Force start
        solar_forecast_kw = [0.0] * 20
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "42.0",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_POWER_MIN_KW": "3.4",
            "GSHP_POWER_MAX_KW": "4.2",
            "GSHP_IS_RUNNING": "true",
            "GSHP_BASELINE_DEMAND_KW": "0.0",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            is_sauna_active = [0] * len(prediction_timestamps)
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # First interval should be near 3.4kW (at 42C)
        self.assertAlmostEqual(plan[0]["gshp_electric_kw"], 3.4)
        
        # Power should increase in subsequent intervals as temp rises
        for i in range(1, len(plan)):
            if plan[i]["gshp_intent"] == "START" and plan[i-1]["gshp_intent"] == "START":
                # Only check if it didn't hit max_temp in this interval (which would drop actual power)
                if plan[i]["gshp_temp_sim"] < 55.0:
                    self.assertGreaterEqual(plan[i]["gshp_electric_kw"], plan[i-1]["gshp_electric_kw"])

    def test_gshp_strategic_stop_before_max_temp(self):
        # Current temp is 51C (Safe, min is 42C, max 55C).
        # Price now is high (0.25), but drops to 0.15 in 1 hour.
        prediction_timestamps = [datetime.now(timezone.utc) + timedelta(minutes=15*i) for i in range(20)]
        outside_temps = [0.0] * 20
        # Price: 0.25 for 4 intervals (1h), then 0.15
        import_prices = [0.25] * 4 + [0.15] * 16
        is_sauna_active = [0] * 20
        solar_forecast_kw = [0.0] * 20

        with patched_env({
            "GSHP_INITIAL_TEMP": "51.0",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_IS_RUNNING": "true",
            "GSHP_COP": "3.5",
            "GSHP_ELECTRIC_POWER_KW": "4.0",
            "GSHP_HEAT_LOSS_K": "0.1",
            "PLAN_INTERVAL_MINUTES": "15",
            "GSHP_STRATEGIC_STOP_DIFF_EUR": "0.05"
        }):
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)

        # Should STOP at index 0 because 0.25 is >= 0.15 + 0.05
        self.assertEqual(plan[0]['gshp_intent'], "STOP")

    def test_gshp_heating_efficiency_impact(self):
        prediction_timestamps = [datetime.now(timezone.utc)] * 2
        outside_temps = [20.0] * 2 # No loss (baseline disabled for this test)
        import_prices = [0.01] * 2 # Force start
        solar_forecast_kw = [0.0] * 2
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "45.0",
            "GSHP_MIN_TEMP": "40.0",
            "GSHP_MAX_TEMP": "60.0",
            "GSHP_IS_RUNNING": "true",
            "GSHP_ELECTRIC_POWER_KW": "4.0",
            "GSHP_COP": "3.5",
            "GSHP_HEATING_EFFICIENCY": "0.4",
            "GSHP_BASELINE_DEMAND_KW": "0.0",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            is_sauna_active = [0] * 2
            plan = plan_gshp_dispatch(prediction_timestamps, is_sauna_active, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # Expected temp: 45.0 + ~2.41 = 47.41
        self.assertAlmostEqual(plan[0]["gshp_temp_sim"], 47.41, places=2)

    def test_gshp_strategic_stop_for_single_interval(self):
        # Current temp is 43.1C (Min 42.0 + buffer 1.0). 
        # Price now is 0.15, price in next interval is 0.10.
        # It will hit min_temp in next interval, so intervals_to_min = 1.
        prediction_timestamps = [datetime.now(timezone.utc) + timedelta(minutes=15*i) for i in range(4)]
        outside_temps = [10.0] * 4
        import_prices = [0.15, 0.10, 0.10, 0.10]
        solar_forecast_kw = [0.0] * 4
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "43.1",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_IS_RUNNING": "true",
            "GSHP_STRATEGIC_STOP_DIFF_EUR": "0.02"
        }):
            plan = plan_gshp_dispatch(prediction_timestamps, [0]*4, outside_temps, import_prices, import_prices, solar_forecast_kw)
            
        # SHOULD stop at index 0 because 0.15 >= 0.10 + 0.02
        self.assertEqual(plan[0]['gshp_intent'], "STOP")

    def test_gshp_preheats_more_with_solar(self):
        # Temp is 54.0C (Max 55.0). 
        # Import price 0.20, Export price 0.10.
        # Now we have solar (10kW), so effective price is 0.10.
        prediction_timestamps = [datetime.now(timezone.utc) + timedelta(minutes=15*i) for i in range(4)]
        outside_temps = [10.0] * 4
        import_prices = [0.20] * 4
        export_prices = [0.10] * 4
        solar_forecast_kw = [10.0] * 4
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "54.0",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_IS_RUNNING": "false"
        }):
            plan = plan_gshp_dispatch(prediction_timestamps, [0]*4, outside_temps, import_prices, export_prices, solar_forecast_kw)
            
        # SHOULD start pre-heating because buffer_margin is 0.0 for solar, 
        # so 54.0 < (55.0 - 0.0) is true.
        self.assertEqual(plan[0]["gshp_intent"], "START")

    def test_gshp_cools_realistically_in_summer(self):
        # Even when outside temp is 20C (no space heating need), the accumulator
        # still cools due to DHW, circulation, and tank standby losses.
        # Real-world data shows ~1.7C/hour cooling at 20C outside.
        prediction_timestamps = [datetime.now(timezone.utc) + timedelta(minutes=15*i) for i in range(20)]
        outside_temps = [20.0] * 20
        # Prices always decrease, so preheating is never triggered (future is always cheaper)
        import_prices = [0.30 - 0.01 * i for i in range(20)]
        solar_forecast_kw = [0.0] * 20
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "45.0",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_IS_RUNNING": "false",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            plan = plan_gshp_dispatch(prediction_timestamps, [0]*20, outside_temps, import_prices, import_prices, solar_forecast_kw)
        
        # GSHP should stay OFF for the first few intervals
        self.assertEqual(plan[0]["gshp_intent"], "STOP")
        
        # After 1 hour (4 intervals), should cool by at least 1.0C
        # (Real data shows ~1.7C/hour; model should be in that ballpark)
        self.assertLess(plan[3]["gshp_temp_sim"], 44.0)


    @patch('optimize_plan.get_ha_state')
    def test_ha_allow_export_on_overrides_env_false(self, mock_ha):
        # HA switch ON should allow export even when env says false
        def ha_side_effect(entity_id):
            if entity_id == 'input_boolean.battery_allow_export':
                return {"state": "on"}
            return None
        mock_ha.side_effect = ha_side_effect

        with patched_env({
            "BATTERY_CAPACITY_KWH": "40",
            "BATTERY_MIN_SOC_PCT": "10",
            "BATTERY_MAX_SOC_PCT": "90",
            "BATTERY_INITIAL_SOC_PCT": "80",
            "BATTERY_MAX_CHARGE_KW": "10",
            "BATTERY_MAX_DISCHARGE_KW": "10",
            "BATTERY_CHARGE_EFFICIENCY": "1.0",
            "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
            "BATTERY_ALLOW_EXPORT": "false",
            "PLAN_INTERVAL_MINUTES": "60",
        }):
            # High export price, should discharge to export if allowed
            predictions = np.array([1.0, 1.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.10, 0.10])
            export_prices = np.array([0.50, 0.50])
            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # With 80% SOC and high export prices, we should see discharge_export
        actions = [p["battery_action"] for p in plan]
        self.assertIn("discharge_export", actions)

    @patch('optimize_plan.get_ha_state')
    def test_ha_allow_export_off_overrides_env_true(self, mock_ha):
        # HA switch OFF should block export even when env says true
        def ha_side_effect(entity_id):
            if entity_id == 'input_boolean.battery_allow_export':
                return {"state": "off"}
            return None
        mock_ha.side_effect = ha_side_effect

        with patched_env({
            "BATTERY_CAPACITY_KWH": "40",
            "BATTERY_MIN_SOC_PCT": "10",
            "BATTERY_MAX_SOC_PCT": "90",
            "BATTERY_INITIAL_SOC_PCT": "80",
            "BATTERY_MAX_CHARGE_KW": "10",
            "BATTERY_MAX_DISCHARGE_KW": "10",
            "BATTERY_CHARGE_EFFICIENCY": "1.0",
            "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
            "BATTERY_ALLOW_EXPORT": "true",
            "PLAN_INTERVAL_MINUTES": "60",
        }):
            predictions = np.array([1.0, 1.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.10, 0.10])
            export_prices = np.array([0.50, 0.50])
            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # discharge_export should NOT appear when HA switch is off
        actions = [p["battery_action"] for p in plan]
        self.assertNotIn("discharge_export", actions)

    @patch('optimize_plan.get_ha_state')
    def test_ha_allow_export_unavailable_falls_back_to_env(self, mock_ha):
        # Unavailable HA switch should fall back to env value
        mock_ha.return_value = None

        with patched_env({
            "BATTERY_CAPACITY_KWH": "40",
            "BATTERY_MIN_SOC_PCT": "10",
            "BATTERY_MAX_SOC_PCT": "90",
            "BATTERY_INITIAL_SOC_PCT": "80",
            "BATTERY_MAX_CHARGE_KW": "10",
            "BATTERY_MAX_DISCHARGE_KW": "10",
            "BATTERY_CHARGE_EFFICIENCY": "1.0",
            "BATTERY_DISCHARGE_EFFICIENCY": "1.0",
            "BATTERY_ALLOW_EXPORT": "false",
            "PLAN_INTERVAL_MINUTES": "60",
        }):
            predictions = np.array([1.0, 1.0])
            solar = np.array([0.0, 0.0])
            import_prices = np.array([0.10, 0.10])
            export_prices = np.array([0.50, 0.50])
            plan = plan_battery_dispatch(predictions, solar, import_prices, export_prices)

        # With env false and HA unavailable, discharge_export should be blocked
        actions = [p["battery_action"] for p in plan]
        self.assertNotIn("discharge_export", actions)


if __name__ == "__main__":
    unittest.main()

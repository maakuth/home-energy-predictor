import os
import unittest
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager

import numpy as np
import pandas as pd

from optimize_plan import build_tariff_prices, plan_battery_dispatch, align_interval_prices, plan_gshp_dispatch, optimize
import sqlite3
import os
import json
from unittest.mock import patch, MagicMock

class OptimizeArchivingTests(unittest.TestCase):
    def setUp(self):
        self.db_file = 'hepo.db'
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
        
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
        with open('future_predictions.json', 'w') as f:
            json.dump(self.predictions_data, f)

    def tearDown(self):
        if os.path.exists(self.db_file):
            os.remove(self.db_file)
        if os.path.exists('future_predictions.json'):
            os.remove('future_predictions.json')
        if os.path.exists('optimization_plan.json'):
            os.remove('optimization_plan.json')

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    def test_optimize_archives_to_db(self, mock_prices, mock_ha):
        # Mocking external calls
        mock_prices.return_value = ([0.1], [0], "Nordpool")
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

        cur.execute("SELECT predicted_usage_kw, version FROM predictions LIMIT 1")
        row = cur.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        # Predicted usage should be at least baseload (2.0)
        # GSHP might be 0 or more depending on prices/temp
        self.assertGreaterEqual(row[0], 2.0)
        self.assertNotEqual(row[1], "")


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
            }
        ):
            # 3.0 kWh solar surplus
            predictions = np.array([0.0])
            solar = np.array([3.0])
            import_prices = np.array([0.20])
            export_prices = np.array([0.10])

            with patched_env({"PLAN_INTERVAL_MINUTES": "60"}):
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
        outside_temps = [20.0] * 4 # No heat loss
        import_prices = [0.20] * 4
        solar_forecast_kw = [0.0] * 4
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "54.9",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_IS_RUNNING": "true",
            "GSHP_ELECTRIC_POWER_KW": "4.0",
            "GSHP_COP": "3.5",
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
        outside_temps = [20.0] * 20 # No heat loss
        import_prices = [0.01] * 20 # Force start
        solar_forecast_kw = [0.0] * 20
        
        with patched_env({
            "GSHP_INITIAL_TEMP": "42.0",
            "GSHP_MIN_TEMP": "42.0",
            "GSHP_MAX_TEMP": "55.0",
            "GSHP_POWER_MIN_KW": "3.4",
            "GSHP_POWER_MAX_KW": "4.2",
            "GSHP_IS_RUNNING": "true",
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
        outside_temps = [20.0] * 2 # No loss
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
        self.assertEqual(plan[0]['gshp_intent'], "START")


if __name__ == "__main__":
    unittest.main()

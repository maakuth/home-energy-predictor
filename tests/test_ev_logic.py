from __future__ import annotations
import unittest
import json
import os
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from optimize_plan import optimize

class TestEVLogic(unittest.TestCase):
    def setUp(self):
        # Use test-specific paths from environment (set by conftest.py)
        self.predictions_file = os.getenv('TEST_PREDICTIONS_FILE', 'future_predictions.json')
        self.plan_file = os.getenv('TEST_PLAN_FILE', 'optimization_plan.json')
        self.db_file = os.getenv('TEST_DB_PATH', 'hepo.db')
        
        # Create directories if they don't exist (only if not in current directory)
        for path in [self.predictions_file, self.plan_file, self.db_file]:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        
        # Dummy data: 4 intervals. 
        # Prices: [10, 5, 20, 15] -> index 1 is cheapest, then 3.
        # ev_position: [0, 1, 0, 1] -> Car home only at index 1 and 3.
        self.predictions_data = [
            {"timestamp": "2026-04-15T10:00:00Z", "predicted_baseload": 1.0, "solar_forecast": 0.0, "outside_temp": 10.0, "ev_position": 0},
            {"timestamp": "2026-04-15T10:15:00Z", "predicted_baseload": 1.0, "solar_forecast": 0.0, "outside_temp": 10.0, "ev_position": 1},
            {"timestamp": "2026-04-15T10:30:00Z", "predicted_baseload": 1.0, "solar_forecast": 0.0, "outside_temp": 10.0, "ev_position": 0},
            {"timestamp": "2026-04-15T10:45:00Z", "predicted_baseload": 1.0, "solar_forecast": 0.0, "outside_temp": 10.0, "ev_position": 1},
        ]
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions_data, f)
            
    def tearDown(self):
        # No cleanup needed - conftest.py handles it via tmp_path fixture
        pass

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.build_tariff_prices')
    def test_ev_charging_restricted_to_home(self, mock_build_tariff, mock_fetch_prices, mock_ha):
        # Mock prices such that index 0 is cheapest, but car is AWAY.
        # Prices: [0.01, 0.10, 0.05, 0.20]
        # Cheapest is 0 (Away), then 2 (Away), then 1 (Home), then 3 (Home).
        mock_build_tariff.return_value = (np.array([0.01, 0.10, 0.05, 0.20]), np.array([0.0, 0.0, 0.0, 0.0]))
        mock_fetch_prices.return_value = ([0.01, 0.10, 0.05, 0.20], [0,0,0,0], "Mock", False, False)
        
        mock_ha.return_value = {"state": "unknown"} # for various sensors
        
        # Set EV_CHARGE_HOURS to 0.25 (1 interval)
        with patch.dict(os.environ, {"EV_CHARGE_HOURS": "0.25", "PLAN_INTERVAL_MINUTES": "15"}):
            optimize()
            
        with open(self.plan_file, 'r') as f:
            plan = json.load(f)
            
        # Car is home at index 1 and 3. Cheapest of those is index 1 (price 0.10 < 0.20).
        # Even though index 0 and 2 are cheaper, car is away there.
        self.assertEqual(plan[1]['planned_ev_kw'], 7.0, "Should charge at index 1 (cheapest home interval)")
        self.assertEqual(plan[0]['planned_ev_kw'], 0.0, "Should NOT charge at index 0 (car away)")
        self.assertEqual(plan[2]['planned_ev_kw'], 0.0, "Should NOT charge at index 2 (car away)")
        self.assertEqual(plan[3]['planned_ev_kw'], 0.0, "Should NOT charge at index 3")

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.build_tariff_prices')
    def test_ev_charging_none_if_never_home(self, mock_build_tariff, mock_fetch_prices, mock_ha):
        # All intervals AWAY
        for p in self.predictions_data:
            p['ev_position'] = 0
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions_data, f)

        mock_build_tariff.return_value = (np.array([0.01, 0.10, 0.05, 0.20]), np.array([0.0, 0.0, 0.0, 0.0]))
        mock_fetch_prices.return_value = ([0.01, 0.10, 0.05, 0.20], [0,0,0,0], "Mock", False, False)
        mock_ha.return_value = {"state": "unknown"}
        
        with patch.dict(os.environ, {"EV_CHARGE_HOURS": "1.0", "PLAN_INTERVAL_MINUTES": "15"}):
            optimize()
            
        with open(self.plan_file, 'r') as f:
            plan = json.load(f)
            
        self.assertTrue(all(p['planned_ev_kw'] == 0.0 for p in plan), "Should not charge if never home")

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.build_tariff_prices')
    def test_ev_charging_based_on_soc_deficit(self, mock_build_tariff, mock_fetch_prices, mock_ha):
        # Setup: 4 intervals (1h total). Car at home all the time.
        # Prices: [0.10, 0.05, 0.08, 0.20]. Cheapest are index 1, then 2, then 0.
        # SoC: 20%. Target 80%. Deficit = 60% of 60kWh = 36kWh.
        # Charger: 7kW * 0.25h = 1.75 kWh per slot.
        # Slots needed: 36 / 1.75 = 20.5 -> 21 slots.
        # Since we only have 4 intervals total, it should take ALL 4 intervals.
        
        for p in self.predictions_data:
            p['ev_position'] = 1
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions_data, f)

        mock_build_tariff.return_value = (np.array([0.10, 0.05, 0.08, 0.20]), np.array([0.0, 0.0, 0.0, 0.0]))
        mock_fetch_prices.return_value = ([0.10, 0.05, 0.08, 0.20], [0,0,0,0], "Mock", False, False)
        
        mock_ha.return_value = {"state": "20"} # EV SoC = 20%
        
        with patch.dict(os.environ, {
            "EV_TARGET_SOC_PCT": "80", 
            "EV_CAPACITY_KWH": "60",
            "PLAN_INTERVAL_MINUTES": "15"
        }):
            optimize()
            
        with open(self.plan_file, 'r') as f:
            plan = json.load(f)
            
        # Verify it charges in all 4 slots because deficit is massive
        self.assertTrue(all(p['planned_ev_kw'] == 7.0 for p in plan), "Should charge in all 4 intervals to meet energy demand")

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.build_tariff_prices')
    def test_ev_no_charging_when_at_target_soc(self, mock_build_tariff, mock_fetch_prices, mock_ha):
        # Setup: Car home, SoC = 85%, Target = 80%.
        for p in self.predictions_data:
            p['ev_position'] = 1
        with open(self.predictions_file, 'w') as f:
            json.dump(self.predictions_data, f)

        mock_build_tariff.return_value = (np.array([0.10, 0.05, 0.08, 0.20]), np.array([0.0, 0.0, 0.0, 0.0]))
        mock_fetch_prices.return_value = ([0.10, 0.05, 0.08, 0.20], [0,0,0,0], "Mock", False, False)
        
        mock_ha.return_value = {"state": "85"} # SoC already above target
        
        with patch.dict(os.environ, {"EV_TARGET_SOC_PCT": "80"}):
            optimize()
            
        with open(self.plan_file, 'r') as f:
            plan = json.load(f)
            
        self.assertTrue(all(p['planned_ev_kw'] == 0.0 for p in plan), "Should not plan any charging when SoC is above target")

if __name__ == '__main__':
    unittest.main()

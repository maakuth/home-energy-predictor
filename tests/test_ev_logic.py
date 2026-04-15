import unittest
import json
import os
import numpy as np
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from optimize_plan import optimize

class TestEVLogic(unittest.TestCase):
    def setUp(self):
        self.predictions_file = 'future_predictions.json'
        self.plan_file = 'optimization_plan.json'
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
        for f in [self.predictions_file, self.plan_file, 'hepo.db']:
            if os.path.exists(f):
                os.remove(f)

    @patch('optimize_plan.get_ha_state')
    @patch('optimize_plan.fetch_market_prices')
    @patch('optimize_plan.build_tariff_prices')
    def test_ev_charging_restricted_to_home(self, mock_build_tariff, mock_fetch_prices, mock_ha):
        # Mock prices such that index 0 is cheapest, but car is AWAY.
        # Prices: [0.01, 0.10, 0.05, 0.20]
        # Cheapest is 0 (Away), then 2 (Away), then 1 (Home), then 3 (Home).
        mock_build_tariff.return_value = (np.array([0.01, 0.10, 0.05, 0.20]), np.array([0.0, 0.0, 0.0, 0.0]))
        mock_fetch_prices.return_value = ([0.01, 0.10, 0.05, 0.20], [0,0,0,0], "Mock")
        
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
        mock_fetch_prices.return_value = ([0.01, 0.10, 0.05, 0.20], [0,0,0,0], "Mock")
        mock_ha.return_value = {"state": "unknown"}
        
        with patch.dict(os.environ, {"EV_CHARGE_HOURS": "1.0", "PLAN_INTERVAL_MINUTES": "15"}):
            optimize()
            
        with open(self.plan_file, 'r') as f:
            plan = json.load(f)
            
        self.assertTrue(all(p['planned_ev_kw'] == 0.0 for p in plan), "Should not charge if never home")

if __name__ == '__main__':
    unittest.main()

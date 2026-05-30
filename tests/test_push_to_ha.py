import unittest
from unittest.mock import patch, MagicMock, mock_open
import os
import sqlite3
import json
from push_to_ha import push_accuracy, push_plan

class TestPushToHA(unittest.TestCase):
    
    @patch('push_to_ha.push_ha_state')
    @patch('push_to_ha.get_db_connection')
    @patch('push_to_ha.db_exists')
    def test_push_accuracy_success(self, mock_exists, mock_connect, mock_push):
        # Setup
        mock_exists.return_value = True
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_cur = mock_conn.cursor.return_value
        
        # Mock database row: mae_kw, bias_kw, model_version, period_days
        mock_cur.fetchone.return_value = (0.123, -0.05, "1.2.3", 7)
        
        # Execute
        push_accuracy()
        
        # Verify
        mock_push.assert_called_once()
        args, kwargs = mock_push.call_args
        self.assertEqual(args[0], 'sensor.hepo_accuracy')
        self.assertEqual(args[1], '0.123')
        self.assertEqual(args[2]['bias'], -0.05)
        self.assertEqual(args[2]['model_version'], '1.2.3')
        self.assertEqual(args[2]['period_days'], 7)

    @patch('push_to_ha.push_ha_state')
    @patch('push_to_ha.db_exists')
    def test_push_accuracy_no_db(self, mock_exists, mock_push):
        mock_exists.return_value = False
        push_accuracy()
        mock_push.assert_not_called()


class TestPushPlan(unittest.TestCase):
    
    def setUp(self):
        """Clean up test file if it exists."""
        if os.path.exists('optimization_plan.json'):
            os.remove('optimization_plan.json')
    
    def tearDown(self):
        """Clean up test file after test."""
        if os.path.exists('optimization_plan.json'):
            os.remove('optimization_plan.json')
    
    @patch('push_to_ha.push_ha_state')
    def test_push_plan_with_battery_intent(self, mock_push):
        """Test that battery intent is correctly pushed with reversed sign."""
        # Setup mock plan data
        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'gshp_intent': 'RUN',
                'gshp_temp_simulated': 48.5,
                'leaf_intent': 'OFF',
                'battery_power_kw': 2.5,  # Charging: positive in plan
                'battery_action': 'charge_solar',
                'soc_pct': 60.0
            }
        ]
        # Expand to 96 intervals for 24h
        plan_data.extend([plan_data[0].copy() for _ in range(95)])
        
        # Write test plan to file
        with open('optimization_plan.json', 'w') as f:
            json.dump(plan_data, f)
        
        # Execute
        push_plan()
        
        # Verify battery intent was pushed with correct properties
        calls = mock_push.call_args_list
        battery_intent_call = None
        for call in calls:
            args = call[0]
            if len(args) > 0 and args[0] == 'sensor.hepo_battery_intent':
                battery_intent_call = call
                break
        
        self.assertIsNotNone(battery_intent_call, "Battery intent push not found")
        args = battery_intent_call[0]
        
        # Verify entity ID
        self.assertEqual(args[0], 'sensor.hepo_battery_intent')
        
        # Verify sign reversal: battery_power_kw=2.5 (charging) → intent=-2500W (charge)
        self.assertEqual(int(args[1]), -2500)
        
        # Verify attributes
        attrs = args[2]
        self.assertEqual(attrs['battery_action'], 'charge_solar')
        self.assertEqual(attrs['battery_soc_pct'], 60.0)
        self.assertEqual(attrs['unit_of_measurement'], 'W')
        self.assertEqual(attrs['device_class'], 'power')

    @patch('push_to_ha.push_ha_state')
    def test_push_plan_battery_discharge_intent(self, mock_push):
        """Test battery discharge intent (negative battery_power_kw → positive intent)."""
        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'gshp_intent': 'STOP',
                'gshp_temp_simulated': 52.0,
                'leaf_intent': 'OFF',
                'battery_power_kw': -3.0,  # Discharging: negative in plan
                'battery_action': 'discharge_load',
                'soc_pct': 45.0
            }
        ]
        # Expand to 96 intervals
        plan_data.extend([plan_data[0].copy() for _ in range(95)])
        
        # Write test plan to file
        with open('optimization_plan.json', 'w') as f:
            json.dump(plan_data, f)
        
        # Execute
        push_plan()
        
        # Verify battery intent
        calls = mock_push.call_args_list
        battery_intent_call = None
        for call in calls:
            args = call[0]
            if len(args) > 0 and args[0] == 'sensor.hepo_battery_intent':
                battery_intent_call = call
                break
        
        self.assertIsNotNone(battery_intent_call)
        args = battery_intent_call[0]
        
        # battery_power_kw=-3.0 (discharging) → intent=3000W (discharge/provide power)
        self.assertEqual(int(args[1]), 3000)
        self.assertEqual(args[2]['battery_action'], 'discharge_load')
        self.assertEqual(args[2]['battery_soc_pct'], 45.0)

if __name__ == '__main__':
    unittest.main()

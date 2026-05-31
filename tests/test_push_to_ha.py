import unittest
from unittest.mock import patch, MagicMock
import os
import sqlite3
import json
from push_to_ha import push_accuracy, push_plan
from utils.battery_utils import push_battery_control

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
        """Use test-specific plan file from environment."""
        # Use test-specific plan file path from environment (set by conftest.py)
        self.plan_file = os.getenv('TEST_PLAN_FILE', 'optimization_plan.json')
        
        # Create directory if it doesn't exist (only if not in current directory)
        dir_path = os.path.dirname(self.plan_file)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
    
    def tearDown(self):
        """No cleanup needed - conftest.py handles it via tmp_path fixture."""
        pass
    
    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_plan_with_battery_intent(self, mock_service, mock_battery_available):
        """Test that battery control is correctly pushed with reversed sign."""
        mock_battery_available.return_value = True  # Battery is available
        mock_service.return_value = {}  # Mock successful service call (returns dict)
        
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
        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)
        
        # Execute
        push_plan()
        
        # Verify battery control was pushed with correct service call
        calls = mock_service.call_args_list
        battery_control_call = None
        for call in calls:
            kwargs = call[1]
            if kwargs.get('service') == 'set_value' and kwargs.get('service_data', {}).get('entity_id') == 'number.hoymiles_remote_control_hoymiles_battery_power':
                battery_control_call = call
                break
        
        self.assertIsNotNone(battery_control_call, "Battery control service call not found")
        
        # Verify sign reversal: battery_power_kw=2.5 (charging) → control=-2500W (charge)
        service_data = battery_control_call[1]['service_data']
        self.assertEqual(service_data['entity_id'], 'number.hoymiles_remote_control_hoymiles_battery_power')
        self.assertEqual(int(service_data['value']), -2500)

    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_plan_battery_discharge_intent(self, mock_service, mock_battery_available):
        """Test battery discharge control (negative battery_power_kw → positive control)."""
        mock_battery_available.return_value = True  # Battery is available
        mock_service.return_value = {}  # Mock successful service call (returns dict)
        
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
        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)
        
        # Execute
        push_plan()
        
        # Verify battery control service call
        calls = mock_service.call_args_list
        battery_control_call = None
        for call in calls:
            kwargs = call[1]
            if kwargs.get('service') == 'set_value' and kwargs.get('service_data', {}).get('entity_id') == 'number.hoymiles_remote_control_hoymiles_battery_power':
                battery_control_call = call
                break
        
        self.assertIsNotNone(battery_control_call)
        
        # battery_power_kw=-3.0 (discharging) → control=3000W (discharge/provide power)
        service_data = battery_control_call[1]['service_data']
        self.assertEqual(int(service_data['value']), 3000)

if __name__ == '__main__':
    unittest.main()

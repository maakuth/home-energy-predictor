import unittest
from unittest.mock import patch, MagicMock
import os
import sqlite3
import json
from push_to_ha import push_accuracy, push_plan
from utils.battery_utils import (
    push_battery_control, is_battery_available, compute_load_following_setpoint,
    get_current_plan_entry,
    BATTERY_CONTROL_ENTITY_ID
)

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
            if kwargs.get('service') == 'set_value' and kwargs.get('service_data', {}).get('entity_id') == BATTERY_CONTROL_ENTITY_ID:
                battery_control_call = call
                break
        
        self.assertIsNotNone(battery_control_call, "Battery control service call not found")
        
        # Verify sign reversal: battery_power_kw=2.5 (charging) → control=-2500W (charge)
        service_data = battery_control_call[1]['service_data']
        self.assertEqual(service_data['entity_id'], BATTERY_CONTROL_ENTITY_ID)
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
            if kwargs.get('service') == 'set_value' and kwargs.get('service_data', {}).get('entity_id') == BATTERY_CONTROL_ENTITY_ID:
                battery_control_call = call
                break
        
        self.assertIsNotNone(battery_control_call)
        
        # battery_power_kw=-3.0 (discharging) → control=3000W (discharge/provide power)
        service_data = battery_control_call[1]['service_data']
        self.assertEqual(int(service_data['value']), 3000)

    @patch('push_to_ha.push_ha_state')
    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_effective_cost(self, mock_service, mock_battery_available, mock_push):
        """Test that effective_cost sensor is pushed to HA."""
        mock_battery_available.return_value = True
        mock_service.return_value = {}
        
        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.08,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'idle',
                'soc_pct': 50.0
            }
        ]
        plan_data.extend([plan_data[0].copy() for _ in range(95)])
        
        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)
        
        push_plan()
        
        # Find the effective_cost push call
        effective_cost_call = None
        for call in mock_push.call_args_list:
            args, kwargs = call
            if args[0] == 'sensor.hepo_effective_cost':
                effective_cost_call = call
                break
        
        self.assertIsNotNone(effective_cost_call, "effective_cost sensor push not found")
        self.assertEqual(effective_cost_call[0][1], '0.0800')
        self.assertEqual(effective_cost_call[0][2]['unit_of_measurement'], 'EUR/kWh')

    @patch('push_to_ha.push_ha_state')
    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_low_cost_signal_on(self, mock_service, mock_battery_available, mock_push):
        """Test that low_cost_signal is ON when current cost is below threshold."""
        mock_battery_available.return_value = True
        mock_service.return_value = {}
        
        # Create a plan with varying effective_cost: most are 0.20, first is 0.05
        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.05,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'idle',
                'soc_pct': 50.0
            }
        ]
        # Rest are expensive
        for _ in range(95):
            plan_data.append({
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.20,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'idle',
                'soc_pct': 50.0
            })
        
        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)
        
        push_plan()
        
        # Find the low_cost_signal push call
        low_cost_call = None
        for call in mock_push.call_args_list:
            args, kwargs = call
            if args[0] == 'sensor.hepo_low_cost_signal':
                low_cost_call = call
                break
        
        self.assertIsNotNone(low_cost_call, "low_cost_signal sensor push not found")
        self.assertEqual(low_cost_call[0][1], 'ON')
        self.assertEqual(low_cost_call[0][2]['icon'], 'mdi:flash')
        self.assertIsNotNone(low_cost_call[0][2]['low_cost_threshold'])

    @patch('push_to_ha.push_ha_state')
    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_low_cost_signal_off(self, mock_service, mock_battery_available, mock_push):
        """Test that low_cost_signal is OFF when current cost is above threshold."""
        mock_battery_available.return_value = True
        mock_service.return_value = {}
        
        # Create a plan with varying effective_cost: most are 0.05, first is 0.20
        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.20,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'idle',
                'soc_pct': 50.0
            }
        ]
        # Rest are cheap
        for _ in range(95):
            plan_data.append({
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.05,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'idle',
                'soc_pct': 50.0
            })
        
        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)
        
        push_plan()
        
        # Find the low_cost_signal push call
        low_cost_call = None
        for call in mock_push.call_args_list:
            args, kwargs = call
            if args[0] == 'sensor.hepo_low_cost_signal':
                low_cost_call = call
                break
        
        self.assertIsNotNone(low_cost_call, "low_cost_signal sensor push not found")
        self.assertEqual(low_cost_call[0][1], 'OFF')

class TestBatteryAvailability(unittest.TestCase):
    @patch('utils.battery_utils.get_ha_state')
    def test_available_when_soc_sensor_online(self, mock_get_state):
        """Battery should be available when SoC sensor has a valid state."""
        mock_get_state.return_value = {'state': '26.6'}
        self.assertTrue(is_battery_available())
        mock_get_state.assert_called_once_with('sensor.be_soc')

    @patch('utils.battery_utils.get_ha_state')
    def test_unavailable_when_soc_sensor_unknown(self, mock_get_state):
        """Battery should be unavailable when SoC sensor is unknown."""
        mock_get_state.return_value = {'state': 'unknown'}
        self.assertFalse(is_battery_available())
        mock_get_state.assert_called_once_with('sensor.be_soc')

    @patch('utils.battery_utils.get_ha_state')
    def test_unavailable_when_soc_sensor_none(self, mock_get_state):
        """Battery should be unavailable when SoC sensor returns None."""
        mock_get_state.return_value = None
        self.assertFalse(is_battery_available())
        mock_get_state.assert_called_once_with('sensor.be_soc')

    @patch('utils.battery_utils.get_ha_state')
    def test_unavailable_when_soc_sensor_unavailable(self, mock_get_state):
        """Battery should be unavailable when SoC sensor state is 'unavailable'."""
        mock_get_state.return_value = {'state': 'unavailable'}
        self.assertFalse(is_battery_available())
        mock_get_state.assert_called_once_with('sensor.be_soc')


class TestGetCurrentPlanEntry(unittest.TestCase):
    def test_matches_current_interval(self):
        """Should return the entry whose timestamp matches the current 15-min slot."""
        from datetime import datetime, timezone
        now = datetime(2026, 6, 10, 9, 46, 30, tzinfo=timezone.utc)
        plan = [
            {'timestamp': '2026-06-10T09:30:00+00:00', 'battery_action': 'charge_solar'},
            {'timestamp': '2026-06-10T09:45:00+00:00', 'battery_action': 'discharge_export'},
            {'timestamp': '2026-06-10T10:00:00+00:00', 'battery_action': 'idle'},
        ]
        with patch('utils.battery_utils.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            mock_dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
            result = get_current_plan_entry(plan)
            # 09:46:30 rounds down to 09:45 slot
            self.assertEqual(result['battery_action'], 'discharge_export')

    def test_matches_exact_boundary(self):
        """At exact 15-min boundary, should return that slot."""
        from datetime import datetime, timezone
        now = datetime(2026, 6, 10, 9, 45, 0, tzinfo=timezone.utc)
        plan = [
            {'timestamp': '2026-06-10T09:30:00+00:00', 'battery_action': 'charge_solar'},
            {'timestamp': '2026-06-10T09:45:00+00:00', 'battery_action': 'discharge_export'},
        ]
        with patch('utils.battery_utils.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            mock_dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
            result = get_current_plan_entry(plan)
            self.assertEqual(result['battery_action'], 'discharge_export')

    def test_fallback_to_first(self):
        """If no entry matches current time, fallback to plan[0]."""
        from datetime import datetime, timezone
        now = datetime(2026, 6, 10, 9, 46, 30, tzinfo=timezone.utc)
        plan = [
            {'timestamp': '2026-06-10T10:00:00+00:00', 'battery_action': 'idle'},
        ]
        with patch('utils.battery_utils.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            mock_dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
            result = get_current_plan_entry(plan)
            self.assertEqual(result['battery_action'], 'idle')

    def test_empty_plan(self):
        """Empty plan should return None."""
        self.assertIsNone(get_current_plan_entry([]))


class TestLoadFollowing(unittest.TestCase):
    def test_charge_solar_limited_by_surplus(self):
        """charge_solar should be limited to actual solar surplus."""
        # Planned: charge 5kW from solar
        # Actual: solar=3kW, load=2kW -> grid = load - solar + battery = -1kW
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=5.0,
            planned_action='charge_solar',
            solar_kw=3.0,
            grid_w=-1000.0,
            battery_w=0.0,
            gshp_kw=0.0,
            leaf_kw=0.0
        )
        self.assertAlmostEqual(adjusted, 1.0)
        self.assertIn('1.00kW', msg)

    def test_charge_solar_no_surplus(self):
        """charge_solar should drop to 0 when there is no solar surplus."""
        # Planned: charge 5kW
        # Actual: solar=2kW, load=3kW -> grid = load - solar + battery = 1kW (import)
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=5.0,
            planned_action='charge_solar',
            solar_kw=2.0,
            grid_w=1000.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, 0.0)
        self.assertIn('0.00kW', msg)

    def test_charge_solar_follows_plan_when_enough_surplus(self):
        """charge_solar should follow plan when surplus exceeds plan."""
        # Planned: charge 2kW
        # Actual: solar=5kW, load=1kW -> grid = load - solar + battery = -4kW (export)
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=2.0,
            planned_action='charge_solar',
            solar_kw=5.0,
            grid_w=-4000.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, 2.0)
        self.assertEqual(msg, '')  # No adjustment needed

    def test_discharge_load_limited_by_actual_load(self):
        """discharge_load should be limited to actual house load."""
        # Planned: discharge 5kW
        # Actual: load=2kW, solar=0, battery already discharging 2kW
        # grid = load - solar + battery = 2 - 0 + (-2) = 0
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=-5.0,
            planned_action='discharge_load',
            solar_kw=0.0,
            grid_w=0.0,
            battery_w=-2000.0
        )
        self.assertAlmostEqual(adjusted, -2.0)
        self.assertIn('2.00kW', msg)

    def test_discharge_load_follows_plan_when_load_large(self):
        """discharge_load should follow plan when actual load exceeds plan."""
        # Planned: discharge 2kW
        # Actual: load=5kW, solar=0, battery already discharging 3kW
        # grid = load - solar + battery = 5 - 0 + (-3) = 2kW (importing)
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=-2.0,
            planned_action='discharge_load',
            solar_kw=0.0,
            grid_w=2000.0,
            battery_w=-3000.0
        )
        self.assertAlmostEqual(adjusted, -2.0)
        self.assertEqual(msg, '')

    def test_idle_opportunistic_charge_on_export(self):
        """idle should opportunistically charge when exporting to grid."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='idle',
            solar_kw=5.0,
            grid_w=-3000.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, 3.0)
        self.assertIn('opportunistic charge', msg)

    def test_idle_opportunistic_discharge_on_import(self):
        """idle should opportunistically discharge when importing from grid."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='idle',
            solar_kw=0.0,
            grid_w=4000.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, -4.0)
        self.assertIn('opportunistic discharge', msg)

    def test_idle_stays_idle_in_deadband(self):
        """idle should stay at 0 when grid flow is within deadband."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='idle',
            solar_kw=2.0,
            grid_w=-200.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, 0.0)
        self.assertEqual(msg, '')

    def test_charge_mixed_unchanged(self):
        """charge_mixed should not be adjusted by load following."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=5.0,
            planned_action='charge_mixed',
            solar_kw=2.0,
            grid_w=0.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, 5.0)
        self.assertEqual(msg, '')

    def test_clamped_to_max_battery_kw(self):
        """Adjusted setpoint should be clamped to max_battery_kw."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='idle',
            solar_kw=0.0,
            grid_w=-15000.0,  # 15kW export
            battery_w=0.0,
            max_battery_kw=10.0
        )
        self.assertAlmostEqual(adjusted, 10.0)


if __name__ == '__main__':
    unittest.main()

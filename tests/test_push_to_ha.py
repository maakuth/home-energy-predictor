from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock
import os
import sqlite3
import json
from push_to_ha import push_accuracy, push_plan
from utils.battery_utils import (
    is_battery_available, compute_load_following_setpoint,
    get_current_plan_entry, compute_net_metering_setpoint,
    adjust_charge_solar_for_real_time,
    smooth_planned_setpoint,
    apply_ramp_rate,
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
                'battery_action': 'follow',
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
                'battery_action': 'follow',
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
                'battery_action': 'follow',
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
                'battery_action': 'follow',
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
                'battery_action': 'follow',
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

    @patch('push_to_ha.push_ha_state')
    @patch('utils.battery_utils.is_battery_available')
    @patch('utils.battery_utils.call_ha_service')
    def test_push_period_balance(self, mock_service, mock_battery_available, mock_push):
        """Test that period power balance sensor is pushed with import/export attributes."""
        mock_battery_available.return_value = True
        mock_service.return_value = {}

        plan_data = [
            {
                'predicted_usage_kwh': 0.5,
                'effective_cost': 0.08,
                'gshp_intent': 'STOP',
                'leaf_intent': 'OFF',
                'battery_power_kw': 0.0,
                'battery_action': 'follow',
                'soc_pct': 50.0,
                'grid_import_kwh': 2.5,
                'grid_export_kwh': 0.8,
            }
        ]
        plan_data.extend([plan_data[0].copy() for _ in range(95)])

        with open(self.plan_file, 'w') as f:
            json.dump(plan_data, f)

        push_plan()

        # Find the period balance push call
        balance_call = None
        for call in mock_push.call_args_list:
            args, kwargs = call
            if args[0] == 'sensor.hepo_period_balance':
                balance_call = call
                break

        self.assertIsNotNone(balance_call, "period balance sensor push not found")
        self.assertEqual(balance_call[0][1], '1.700')  # 2.5 - 0.8
        self.assertEqual(balance_call[0][2]['import_kwh'], 2.5)
        self.assertEqual(balance_call[0][2]['export_kwh'], 0.8)
        self.assertEqual(balance_call[0][2]['net_kw'], 6.8)  # 1.7 * 4

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
            {'timestamp': '2026-06-10T10:00:00+00:00', 'battery_action': 'follow'},
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
            {'timestamp': '2026-06-10T10:00:00+00:00', 'battery_action': 'follow'},
        ]
        with patch('utils.battery_utils.datetime') as mock_dt:
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            mock_dt.fromisoformat.side_effect = lambda s: datetime.fromisoformat(s)
            result = get_current_plan_entry(plan)
            self.assertEqual(result['battery_action'], 'follow')

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

    def test_charge_solar_scales_to_actual_surplus(self):
        """charge_solar should charge at actual surplus up to max_battery_kw."""
        # Planned: charge 2kW
        # Actual: solar=5kW, load=1kW -> surplus = 4kW
        # New behavior: charge at min(max_battery_kw, surplus) = 4kW
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=2.0,
            planned_action='charge_solar',
            solar_kw=5.0,
            grid_w=-4000.0,
            battery_w=0.0
        )
        self.assertAlmostEqual(adjusted, 4.0)  # Surplus = 4kW, max_battery_kw=10
        self.assertIn('planned 2.00kW -> adjusted 4.00kW', msg)

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

    def test_follow_opportunistic_charge_on_export(self):
        """follow should opportunistically charge when exporting to grid,
        but capped at BATTERY_FOLLOW_MAX_KW (default 2 kW)."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='follow',
            solar_kw=5.0,
            grid_w=-3000.0,
            battery_w=0.0
        )
        # 3 kW export exceeds follow cap → stays at 0
        self.assertAlmostEqual(adjusted, 0.0)
        self.assertEqual(msg, '')

    def test_follow_opportunistic_discharge_on_import(self):
        """follow should opportunistically discharge when importing from grid,
        but capped at BATTERY_FOLLOW_MAX_KW (default 2 kW)."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='follow',
            solar_kw=0.0,
            grid_w=4000.0,
            battery_w=0.0
        )
        # 4 kW import exceeds follow cap → stays at 0
        self.assertAlmostEqual(adjusted, 0.0)
        self.assertEqual(msg, '')

    def test_follow_stays_put_in_deadband(self):
        """follow should stay at 0 when grid flow is within deadband."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='follow',
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
            planned_action='follow',
            solar_kw=0.0,
            grid_w=-15000.0,  # 15kW export
            battery_w=0.0,
            max_battery_kw=10.0
        )
        # Follow caps at BATTERY_FOLLOW_MAX_KW (default 2 kW). 15 kW exceeds cap → stays at 0.
        self.assertAlmostEqual(adjusted, 0.0)

class TestPhaseCapping(unittest.TestCase):
    def test_no_phase_cap_needed(self):
        """Should not cap if currents are well within limits."""
        # planned 5kW charge, phase currents 10A each (fuse 25A)
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=5.0,
            planned_action='charge_mixed',
            solar_kw=0.0,
            grid_w=5000.0,
            battery_w=5000.0,
            phase_currents=[10.0, 10.0, 10.0]
        )
        self.assertAlmostEqual(adjusted, 5.0)
        self.assertEqual(msg, '')

    def test_import_phase_cap(self):
        """Should cap charge if one phase hits import limit."""
        # planned 10kW charge (approx 14.5A per phase)
        # current Ip = 20A. If we add 10kW charge, Ip becomes 20 + 14.5 = 34.5A (>25A)
        # Max extra current allowed: 25 - 20 = 5A
        # Max extra power: 5A * 3 * 230V = 3450W
        # Current battery_w = 10000W
        # Max allowed P = 10000 + 3450 = 13450W (wait, Ip includes current battery contribution)
        # If Ip = 20A and battery_w = 5kW (approx 7.2A per phase)
        # Then non-battery load on phase is 20 - 7.2 = 12.8A
        # Max battery contribution: 25 - 12.8 = 12.2A
        # Max battery power: 12.2 * 3 * 230 = 8418W = 8.42kW
        
        # Test case: Ip=20A, battery_w=0, planned=10kW charge
        # Max extra: (25 - 20) * 3 * 230 = 3450W = 3.45kW
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=10.0,
            planned_action='charge_mixed',
            solar_kw=0.0,
            grid_w=10000.0,
            battery_w=0.0,
            phase_currents=[20.0, 10.0, 10.0]
        )
        self.assertAlmostEqual(adjusted, 3.45)
        self.assertIn('phase cap', msg)

    def test_export_phase_cap(self):
        """Should cap discharge if one phase hits export limit."""
        # planned 10kW discharge (-14.5A per phase)
        # Ip = -20A (exporting). Max allowed export -25A.
        # Max extra export current: -25 - (-20) = -5A
        # Max extra export power: -5A * 3 * 230V = -3450W = -3.45kW
        # Current battery_w = 0
        # New setpoint should be -3.45kW
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=-10.0,
            planned_action='discharge_export',
            solar_kw=0.0,
            grid_w=-10000.0,
            battery_w=0.0,
            phase_currents=[-20.0, -10.0, -10.0]
        )
        self.assertAlmostEqual(adjusted, -3.45)
        self.assertIn('phase cap', msg)

    def test_forced_discharge_to_assist_import_overload(self):
        """Should discharge to protect the fuse when phase is overloaded."""
        # Ip = 30A (exceeding 25A fuse). battery_w = 0.
        # Max P_extra = (25 - 30) * 3 * 230 = -3450W = -3.45kW
        # So it MUST discharge AT LEAST 3.45kW.
        # Follow cap prevents opportunistic full discharge; phase cap enforces minimum.
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='follow',
            solar_kw=0.0,
            grid_w=20000.0,
            battery_w=0.0,
            phase_currents=[30.0, 10.0, 10.0]
        )
        # Phase cap forces -3.45 kW discharge to protect fuse
        self.assertAlmostEqual(adjusted, -3.45)
        self.assertIn('phase cap', msg)

    def test_forced_discharge_from_zero(self):
        """Should force discharge if house load exceeds fuse and no opportunistic discharge happens."""
        # Ip = 30A, but grid_w is small (maybe sensors are inconsistent or we are in a deadband)
        # We'll use a planned_action that doesn't have opportunistic discharge to be sure.
        # Actually, let's just use follow with grid_w in deadband.
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=0.0,
            planned_action='follow',
            solar_kw=0.0,
            grid_w=200.0, # Within follow deadband (500W)
            battery_w=0.0,
            phase_currents=[30.0, 10.0, 10.0]
        )
        self.assertAlmostEqual(adjusted, -3.45)
        self.assertIn('Phase cap applied', msg)

    def test_partial_phase_data(self):
        """Should handle None in phase currents."""
        adjusted, msg = compute_load_following_setpoint(
            planned_battery_kw=5.0,
            planned_action='charge_mixed',
            solar_kw=0.0,
            grid_w=5000.0,
            battery_w=0.0,
            phase_currents=[20.0, None, 10.0]
        )
        self.assertAlmostEqual(adjusted, 3.45) # Still caps based on phase 1


class TestNetMeteringBatteryControl(unittest.TestCase):
    """Tests for net metering-aware battery control using cumulative energy sensors."""

    def setUp(self):
        """Set up test environment with mock state file."""
        self.test_state_file = '/tmp/hepo_net_metering_test.json'
        os.environ['HEPO_NET_METERING_STATE_FILE'] = self.test_state_file
        # Clean up any existing test state
        if os.path.exists(self.test_state_file):
            os.remove(self.test_state_file)

    def tearDown(self):
        """Clean up test state file."""
        if os.path.exists(self.test_state_file):
            os.remove(self.test_state_file)
        if 'HEPO_NET_METERING_STATE_FILE' in os.environ:
            del os.environ['HEPO_NET_METERING_STATE_FILE']

    def test_net_metering_discharges_more_when_imported_too_much(self):
        """When actual net import > planned, battery should discharge more."""
        # Interval: 15 min, elapsed: 5 min, remaining: 10 min
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # Second call: actual net import = 0.5, planned = 0.1
        # Deviation = 0.5 - 0.1 = 0.4 kWh
        # Correction = -0.4 / (10/60) = -2.4 kW
        # Planned battery = -1.0 kW (discharging 1kW)
        # Adjusted = -1.0 + (-2.4) = -3.4 kW
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.5,
            cumulative_export_kwh=0.0,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, -3.4, places=1)
        self.assertIn('correction', log)

    def test_net_metering_charges_less_when_exported_too_much(self):
        """When actual net export > planned, battery should charge less (or discharge more)."""
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=2.0,
            planned_action='charge_solar',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # Second call: actual net export = 0.3, planned = 0.0
        # Deviation = -0.3 - 0 = -0.3 kWh
        # Correction = -(-0.3) / (10/60) = +1.8 kW
        # Planned battery = +2.0 kW (charging)
        # Adjusted = 2.0 + 1.8 = 3.8 kW (charge more to absorb the export)
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=2.0,
            planned_action='charge_solar',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.3,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, 3.8, places=1)
        self.assertIn('correction', log)

    def test_net_metering_no_adjustment_when_on_track(self):
        """When actual matches planned, no correction needed."""
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=-1.5,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.375,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # Second call: actual matches planned
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=-1.5,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.375,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.375,
            cumulative_export_kwh=0.0,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, -1.5, places=2)
        self.assertEqual(log, '')

    def test_net_metering_resets_state_on_new_interval(self):
        """When a new interval starts, cumulative readings should be captured."""
        # First call: interval starts, captures baseline
        adjusted1, log1 = compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=100.0,
            cumulative_export_kwh=50.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # At interval start, no correction possible (no time elapsed)
        self.assertAlmostEqual(adjusted1, -1.0, places=2)
        self.assertIn('baseline', log1)

        # Second call: 5 minutes later, cumulative import increased by 0.5
        adjusted2, log2 = compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=100.5,
            cumulative_export_kwh=50.0,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        # Net import = 0.5, planned = 0.1, deviation = 0.4
        # Correction = -0.4 / (10/60) = -2.4 kW
        # Adjusted = -1.0 - 2.4 = -3.4 kW
        self.assertAlmostEqual(adjusted2, -3.4, places=1)

    def test_net_metering_clamped_to_max_battery_power(self):
        """Adjusted power should not exceed physical battery limits."""
        # Large deviation would suggest huge correction
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
            max_battery_kw=10.0,
        )
        # Second call: large deviation
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=5.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=5,
            interval_minutes=15,
            max_battery_kw=10.0,
        )
        # Should not exceed max_battery_kw in magnitude
        self.assertLessEqual(abs(adjusted), 10.0)
        self.assertIn('clamped', log)

    def test_net_metering_handles_zero_elapsed(self):
        """At interval start with zero elapsed, return planned power."""
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.1,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=100.0,
            cumulative_export_kwh=50.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, -1.0, places=2)

    def test_net_metering_reduces_discharge_when_exporting_too_much(self):
        """When battery discharges too much towards load, reduce discharge to match plan.
        
        This is the 'other direction' — the battery was discharging more than needed,
        causing net export. The correction should reduce discharge (or even charge)
        to bring the quarterly average back to the planned net energy.
        """
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # Second call: battery discharged too much, causing 0.3 kWh export
        # Planned net = 0, actual net = -0.3 (export)
        # Deviation = -0.3 - 0 = -0.3
        # Correction = -(-0.3) / (10/60) = +1.8 kW
        # Adjusted = -1.0 + 1.8 = +0.8 kW (charge to absorb excess export)
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=-1.0,
            planned_action='discharge_load',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.3,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, 0.8, places=1)
        self.assertIn('correction', log)

    def test_net_metering_under_export_does_not_increase_discharge(self):
        """Under-export: actual export < planned export → don't discharge more to match target.
        
        The plan expects net export (0.4 kWh export, 0 import). Actual is
        only 0.1 kWh export (battery absorbed some solar). The correction
        would ask the battery to discharge more to hit the export target,
        but that's uneconomical when export price is low.
        """
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=-2.0,
            planned_action='discharge_export',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.4,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # Second call: actual export = 0.1, planned export = 0.4
        # Planned net = 0 - 0.4 = -0.4
        # Actual net = 0 - 0.1 = -0.1
        # Deviation = -0.1 - (-0.4) = 0.3
        # Correction = -0.3 / (10/60) = -1.8
        # Without guard: adjusted = -2.0 + (-1.8) = -3.8
        # With guard: deviation>0 & planned_net<0 → cap at planned_battery_kw
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=-2.0,
            planned_action='discharge_export',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.4,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.1,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        # Guard should cap adjustment, not go below planned -2.0
        self.assertGreaterEqual(adjusted, -2.0)
        self.assertNotAlmostEqual(adjusted, -3.8, places=1)

    def test_net_metering_under_import_does_not_increase_charge(self):
        """Under-import: actual import < planned import → don't charge more to match target.
        
        The plan expects net import (0.4 kWh import, 0 export). Actual is
        only 0.1 kWh import. The correction would ask the battery to charge
        more (grid-charge) to hit the import target, but paying retail
        import price to match a plan number is wasteful.
        """
        # First call: capture baseline
        compute_net_metering_setpoint(
            planned_battery_kw=2.0,
            planned_action='charge_grid',
            planned_grid_import_kwh=0.4,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.0,
            cumulative_export_kwh=0.0,
            elapsed_minutes=0,
            interval_minutes=15,
        )
        # Second call: actual import = 0.1, planned import = 0.4
        # Planned net = 0.4 - 0 = 0.4
        # Actual net = 0.1 - 0 = 0.1
        # Deviation = 0.1 - 0.4 = -0.3
        # Correction = -(-0.3) / (10/60) = 1.8
        # Without guard: adjusted = 2.0 + 1.8 = 3.8
        # With guard: deviation<0 & planned_net>0 → cap at planned_battery_kw
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=2.0,
            planned_action='charge_grid',
            planned_grid_import_kwh=0.4,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=0.1,
            cumulative_export_kwh=0.0,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        # Guard should cap adjustment, not go above planned 2.0
        self.assertLessEqual(adjusted, 2.0)
        self.assertNotAlmostEqual(adjusted, 3.8, places=1)

    def test_net_metering_idle_passes_through(self):
        """Idle action passes planned setpoint through regardless of deviation."""
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=0.0,
            planned_action='idle',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.950,
            cumulative_import_kwh=100.5,
            cumulative_export_kwh=50.0,
            elapsed_minutes=5,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, 0.0, places=2)
        self.assertEqual(log, '')

    def test_net_metering_follow_passes_through(self):
        """Follow action passes planned setpoint through regardless of deviation."""
        adjusted, log = compute_net_metering_setpoint(
            planned_battery_kw=2.0,
            planned_action='follow',
            planned_grid_import_kwh=0.0,
            planned_grid_export_kwh=0.0,
            cumulative_import_kwh=101.0,
            cumulative_export_kwh=50.0,
            elapsed_minutes=7,
            interval_minutes=15,
        )
        self.assertAlmostEqual(adjusted, 2.0, places=2)
        self.assertEqual(log, '')


class TestAdjustChargeSolarRealTime(unittest.TestCase):
    """Test adjust_charge_solar_for_real_time adjustments."""

    def setUp(self):
        self.kwargs = dict(
            planned_battery_kw=5.0,
            planned_action='charge_solar',
            solar_kw=2.0,
        )

    def test_switches_to_discharge_when_no_surplus_and_adequate_soc(self):
        """No solar surplus + adequate SoC => discharge to cover load."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -2.0)

    def test_keeps_charge_when_surplus_present(self):
        """Actual solar surplus exists => stay charge_solar."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=-2000.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'charge_solar')
        self.assertAlmostEqual(adjusted, 5.0)

    def test_keeps_charge_when_surplus_is_negligible(self):
        """Tiny negative surplus (>= -0.1) shouldn't trigger switch."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=100.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'charge_solar')

    def test_keeps_charge_when_soc_too_low(self):
        """SoC close to minimum => don't risk over-discharge."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0, battery_soc_pct=13.0,
        )
        self.assertEqual(action, 'charge_solar')
        self.assertAlmostEqual(adjusted, 5.0)

    def test_keeps_original_when_soc_is_none(self):
        """Unavailable SoC sensor => conservative, keep plan."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0, battery_soc_pct=None,
        )
        self.assertEqual(action, 'charge_solar')

    def test_ignores_non_charge_solar_actions(self):
        """Only charge_solar action is affected."""
        for action in ('discharge_load', 'charge_grid', 'idle', 'follow'):
            adjusted, adj_action = adjust_charge_solar_for_real_time(
                planned_battery_kw=5.0, planned_action=action,
                solar_kw=2.0, grid_w=2000.0, battery_w=0.0, battery_soc_pct=50.0,
            )
            self.assertEqual(adj_action, action)
            self.assertAlmostEqual(adjusted, 5.0)

    def test_discharge_capped_to_battery_max(self):
        """Discharge is capped to max_battery_kw even for large net load."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=15000.0, battery_w=0.0,
            battery_soc_pct=50.0, max_battery_kw=6.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -6.0)

    def test_discharge_respects_actual_load(self):
        """Discharge matches the actual net load, not more."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=500.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -0.5)

    def test_min_soc_threshold_blocks_discharge(self):
        """Custom min_soc_pct blocks discharge when SoC within margin."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0,
            battery_soc_pct=22.0, min_soc_pct=18.0,
        )
        self.assertEqual(action, 'charge_solar')

    def test_min_soc_threshold_respected(self):
        """Custom (lower) min_soc_pct lowers the discharge threshold."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0,
            battery_soc_pct=12.0, min_soc_pct=5.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -2.0)


if __name__ == '__main__':
    """Test adjust_charge_solar_for_real_time adjustments."""

    def setUp(self):
        self.kwargs = dict(
            planned_battery_kw=5.0,
            planned_action='charge_solar',
            solar_kw=2.0,
        )

    def test_switches_to_discharge_when_no_surplus_and_adequate_soc(self):
        """No solar surplus + adequate SoC => discharge to cover load."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -2.0)

    def test_keeps_charge_when_surplus_present(self):
        """Actual solar surplus exists => stay charge_solar."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=-2000.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'charge_solar')
        self.assertAlmostEqual(adjusted, 5.0)

    def test_keeps_charge_when_surplus_is_negligible(self):
        """Tiny negative surplus (>= -0.1) shouldn't trigger switch."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=100.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'charge_solar')

    def test_keeps_charge_when_soc_too_low(self):
        """SoC close to minimum => don't risk over-discharge."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0, battery_soc_pct=13.0,
        )
        self.assertEqual(action, 'charge_solar')
        self.assertAlmostEqual(adjusted, 5.0)

    def test_keeps_original_when_soc_is_none(self):
        """Unavailable SoC sensor => conservative, keep plan."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0, battery_soc_pct=None,
        )
        self.assertEqual(action, 'charge_solar')

    def test_ignores_non_charge_solar_actions(self):
        """Only charge_solar action is affected."""
        for action in ('discharge_load', 'charge_grid', 'idle', 'follow'):
            adjusted, adj_action = adjust_charge_solar_for_real_time(
                planned_battery_kw=5.0, planned_action=action,
                solar_kw=2.0, grid_w=2000.0, battery_w=0.0, battery_soc_pct=50.0,
            )
            self.assertEqual(adj_action, action)
            self.assertAlmostEqual(adjusted, 5.0)

    def test_discharge_capped_to_battery_max(self):
        """Discharge is capped to max_battery_kw even for large net load."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=15000.0, battery_w=0.0,
            battery_soc_pct=50.0, max_battery_kw=6.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -6.0)

    def test_discharge_respects_actual_load(self):
        """Discharge matches the actual net load, not more."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=500.0, battery_w=0.0, battery_soc_pct=50.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -0.5)

    def test_min_soc_threshold_respected(self):
        """Custom min_soc_pct raises the threshold."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0,
            battery_soc_pct=20.0, min_soc_pct=18.0,
        )
        self.assertEqual(action, 'discharge_load')
        self.assertAlmostEqual(adjusted, -2.0)

    def test_min_soc_threshold_blocks_discharge(self):
        """Custom min_soc_pct blocks discharge when inside margin."""
        adjusted, action = adjust_charge_solar_for_real_time(
            **self.kwargs, grid_w=2000.0, battery_w=0.0,
            battery_soc_pct=22.0, min_soc_pct=18.0,
        )
        self.assertEqual(action, 'charge_solar')


class TestSetpointSmoothing(unittest.TestCase):
    """Tests for smooth_planned_setpoint interval-boundary smoothing."""

    def setUp(self):
        self.test_state_file = '/tmp/hepo_setpoint_smooth_test.json'
        self.env_var = 'HEPO_SETPOINT_SMOOTH_STATE_FILE'
        os.environ[self.env_var] = self.test_state_file
        self._clean_state()

    def tearDown(self):
        self._clean_state()
        if self.env_var in os.environ:
            del os.environ[self.env_var]

    def _clean_state(self):
        if os.path.exists(self.test_state_file):
            os.remove(self.test_state_file)

    def _make_plan(self, powers: list[float]) -> list[dict]:
        import datetime
        base = datetime.datetime.now(datetime.timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        plan = []
        for i, p in enumerate(powers):
            ts = (base + datetime.timedelta(minutes=15 * i)).isoformat()
            plan.append({'timestamp': ts, 'battery_power_kw': p, 'battery_action': 'follow'})
        return plan

    def test_first_call_returns_planned(self):
        """First call with no prior state returns the planned value unchanged."""
        plan = self._make_plan([2.0, 1.0])
        result = smooth_planned_setpoint(
            planned_battery_kw=2.0, planned_action='follow',
            actual_battery_w=1500, plan=plan,
        )
        self.assertAlmostEqual(result, 2.0)

    def test_within_same_interval_returns_stored_smoothed(self):
        """Subsequent calls in the same interval return the previously computed smoothed value."""
        plan = self._make_plan([2.0, 1.0])
        smooth_planned_setpoint(
            planned_battery_kw=2.0, planned_action='follow',
            actual_battery_w=1500, plan=plan,
        )
        result = smooth_planned_setpoint(
            planned_battery_kw=2.0, planned_action='follow',
            actual_battery_w=1600, plan=plan,
        )
        self.assertAlmostEqual(result, 2.0)

    def test_delta_preserves_planner_intent(self):
        """Delta adjusts for planned load changes (GSHP, Leaf turning on/off)."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_now = int(now.timestamp()) // 900

        plan = self._make_plan([2.0, 5.0])
        state = {
            'interval_index': interval_now - 1,
            'power_sum_kw': 1.8,
            'power_count': 1,
            'prior_planned_kw': 2.0,
            'smoothed_setpoint_kw': 2.0,
        }
        with open(self.test_state_file, 'w') as f:
            json.dump(state, f)

        result = smooth_planned_setpoint(
            planned_battery_kw=5.0, planned_action='follow',
            actual_battery_w=3000, plan=plan,
        )
        # expected: 1.8 + (5.0 - 2.0) = 4.8
        self.assertAlmostEqual(result, 4.8)

    def test_delta_from_idle_to_charge(self):
        """Transition from idle (0) to charging uses full planned power as delta."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_now = int(now.timestamp()) // 900

        plan = self._make_plan([0.0, 3.0])
        state = {
            'interval_index': interval_now - 1,
            'power_sum_kw': 0.1,
            'power_count': 1,
            'prior_planned_kw': 0.0,
            'smoothed_setpoint_kw': 0.0,
        }
        with open(self.test_state_file, 'w') as f:
            json.dump(state, f)

        result = smooth_planned_setpoint(
            planned_battery_kw=3.0, planned_action='follow',
            actual_battery_w=500, plan=plan,
        )
        # expected: 0.1 + (3.0 - 0.0) = 3.1
        self.assertAlmostEqual(result, 3.1)

    def test_clamps_to_max_battery_kw(self):
        """Smoothed output is clamped to max_battery_kw."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_now = int(now.timestamp()) // 900

        plan = self._make_plan([0.0, 50.0])
        state = {
            'interval_index': interval_now - 1,
            'power_sum_kw': 0.0,
            'power_count': 1,
            'prior_planned_kw': 0.0,
            'smoothed_setpoint_kw': 0.0,
        }
        with open(self.test_state_file, 'w') as f:
            json.dump(state, f)

        result = smooth_planned_setpoint(
            planned_battery_kw=50.0, planned_action='follow',
            actual_battery_w=100, plan=plan,
            max_battery_kw=10.0,
        )
        self.assertAlmostEqual(result, 10.0)

    def test_clamps_to_negative_max_battery_kw(self):
        """Smoothed output is clamped to -max_battery_kw."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_now = int(now.timestamp()) // 900

        plan = self._make_plan([0.0, -50.0])
        state = {
            'interval_index': interval_now - 1,
            'power_sum_kw': 0.0,
            'power_count': 1,
            'prior_planned_kw': 0.0,
            'smoothed_setpoint_kw': 0.0,
        }
        with open(self.test_state_file, 'w') as f:
            json.dump(state, f)

        result = smooth_planned_setpoint(
            planned_battery_kw=-50.0, planned_action='follow',
            actual_battery_w=-100, plan=plan,
            max_battery_kw=10.0,
        )
        self.assertAlmostEqual(result, -10.0)

    def test_first_call_clamps_to_max_battery_kw(self):
        """First call also clamps to max_battery_kw."""
        plan = self._make_plan([20.0, 5.0])
        result = smooth_planned_setpoint(
            planned_battery_kw=20.0, planned_action='follow',
            actual_battery_w=15000, plan=plan,
            max_battery_kw=10.0,
        )
        self.assertAlmostEqual(result, 10.0)

    def test_idle_resets_to_zero(self):
        """Transition from charge to idle forces smoothed setpoint to 0."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_now = int(now.timestamp()) // 900

        plan = self._make_plan([4.0, 0.0])
        state = {
            'interval_index': interval_now - 1,
            'power_sum_kw': 3.5,
            'power_count': 1,
            'prior_planned_kw': 4.0,
            'smoothed_setpoint_kw': 4.0,
        }
        with open(self.test_state_file, 'w') as f:
            json.dump(state, f)

        result = smooth_planned_setpoint(
            planned_battery_kw=0.0, planned_action='idle',
            actual_battery_w=3500, plan=plan,
        )
        # idle forces reset to 0 regardless of prior avg
        self.assertAlmostEqual(result, 0.0)

    def test_idle_to_idle_stays_zero(self):
        """Consecutive idle intervals keep setpoint at 0."""
        import datetime
        now = datetime.datetime.now(datetime.timezone.utc)
        interval_now = int(now.timestamp()) // 900

        plan = self._make_plan([0.0, 0.0])
        state = {
            'interval_index': interval_now - 1,
            'power_sum_kw': 0.0,
            'power_count': 1,
            'prior_planned_kw': 0.0,
            'smoothed_setpoint_kw': 0.0,
        }
        with open(self.test_state_file, 'w') as f:
            json.dump(state, f)

        result = smooth_planned_setpoint(
            planned_battery_kw=0.0, planned_action='idle',
            actual_battery_w=100, plan=plan,
        )
        self.assertAlmostEqual(result, 0.0)


class TestRampRateLimiter(unittest.TestCase):
    """Tests for apply_ramp_rate battery power smoothing."""

    def test_disabled_passthrough(self):
        """ramp_rate=0 returns target unchanged."""
        result = apply_ramp_rate(target_setpoint_kw=5.0, actual_battery_kw=2.0, ramp_rate_kw_per_min=0.0)
        self.assertAlmostEqual(result, 5.0)

    def test_limits_large_change_positive(self):
        """Large charge increase is ramped."""
        result = apply_ramp_rate(
            target_setpoint_kw=10.0, actual_battery_kw=0.0,
            ramp_rate_kw_per_min=3.0, interval_seconds=20.0,
        )
        # max change = 3.0 * (20/60) = 1.0 kW
        self.assertAlmostEqual(result, 1.0)

    def test_limits_large_change_negative(self):
        """Large discharge increase (more negative) is ramped."""
        result = apply_ramp_rate(
            target_setpoint_kw=-10.0, actual_battery_kw=0.0,
            ramp_rate_kw_per_min=3.0, interval_seconds=20.0,
        )
        self.assertAlmostEqual(result, -1.0)

    def test_small_change_passthrough(self):
        """Small change within ramp limit passes through unchanged."""
        result = apply_ramp_rate(
            target_setpoint_kw=0.5, actual_battery_kw=0.0,
            ramp_rate_kw_per_min=3.0, interval_seconds=20.0,
        )
        # max change = 1.0 kW, 0.5 < 1.0, so passthrough
        self.assertAlmostEqual(result, 0.5)

    def test_flip_from_charge_to_discharge_is_ramped(self):
        """Flipping from +5kW charge to -5kW discharge is ramped over cycles."""
        cycle1 = apply_ramp_rate(5.0, 0.0, 3.0, 20.0)   # actual=0, target=+5
        self.assertAlmostEqual(cycle1, 1.0)                # can only move 1 kW
        cycle2 = apply_ramp_rate(5.0, cycle1, 3.0, 20.0) # actual=1, target=+5
        self.assertAlmostEqual(cycle2, 2.0)                # can only move 1 kW

        # Flip to discharge
        cycle3 = apply_ramp_rate(-5.0, cycle2, 3.0, 20.0) # actual=2, target=-5
        self.assertAlmostEqual(cycle3, 1.0)                 # can only move 1 kW towards -5
        cycle4 = apply_ramp_rate(-5.0, cycle3, 3.0, 20.0) # actual=1, target=-5
        self.assertAlmostEqual(cycle4, 0.0)                 # crosses zero

    def test_zero_actual_battery(self):
        """When battery sensor reads 0, ramp from zero."""
        result = apply_ramp_rate(
            target_setpoint_kw=3.0, actual_battery_kw=0.0,
            ramp_rate_kw_per_min=6.0, interval_seconds=10.0,
        )
        # max change = 6.0 * (10/60) = 1.0 kW
        self.assertAlmostEqual(result, 1.0)

    def test_custom_interval(self):
        """Custom interval_seconds changes max step size."""
        # 60 second cycle with 2 kW/min ramp = 2 kW max change
        result = apply_ramp_rate(
            target_setpoint_kw=5.0, actual_battery_kw=0.0,
            ramp_rate_kw_per_min=2.0, interval_seconds=60.0,
        )
        self.assertAlmostEqual(result, 2.0)

    def test_negative_ramp_rate_disables(self):
        """Negative ramp_rate is treated as disabled (passthrough)."""
        result = apply_ramp_rate(target_setpoint_kw=3.0, actual_battery_kw=0.0, ramp_rate_kw_per_min=-1.0)
        self.assertAlmostEqual(result, 3.0)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock, call
import json
import os
import tempfile
import shutil
from datetime import datetime, timedelta, timezone


class TestRunOften(unittest.TestCase):
    """Test the run_often.py orchestration flow."""

    @classmethod
    def setUpClass(cls):
        cls.orig_cwd = os.getcwd()

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        state_dir = os.path.join(self.test_dir, 'state')
        os.makedirs(state_dir, exist_ok=True)
        os.chdir(self.test_dir)

        self._make_plan_file(state_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_plan_file(self, state_dir: str, extra_fields: dict | None = None):
        """Create a minimal optimization_plan.json."""
        now = datetime.now(timezone.utc)
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        plan_entry = {
            'timestamp': slot.isoformat(),
            'battery_power_kw': 1.5,
            'battery_action': 'charge',
            'soc_pct': 50.0,
            'grid_import_kwh': 2.0,
            'grid_export_kwh': 0.0,
        }
        if extra_fields:
            plan_entry.update(extra_fields)
        plan = [plan_entry]
        with open(os.path.join(state_dir, 'optimization_plan.json'), 'w') as f:
            json.dump(plan, f)

    def _mock_ha_state(self, values: dict[str, str]) -> MagicMock:
        """Create a get_ha_state mock that returns a 'state' dict."""
        def side_effect(entity_id: str):
            val = values.get(entity_id, '0.0')
            return {'state': str(val)}
        return MagicMock(side_effect=side_effect)

    @patch('run_often.call_ha_service')
    def test_resistive_heater_control_follows_on_intent(self, mock_service):
        from run_often import control_resistive_heater
        now = datetime.now().astimezone()
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

        control_resistive_heater(
            {'timestamp': slot.isoformat(), 'resistive_heater_intent': 'ON'},
            accumulator_temp=50.0,
        )

        mock_service.assert_called_once_with(
            'switch', 'turn_on',
            {'entity_id': 'switch.mlp_vastus_output_0'},
            return_response=False,
        )

    @patch('run_often.call_ha_service')
    def test_resistive_heater_control_fails_closed(self, mock_service):
        from run_often import control_resistive_heater

        control_resistive_heater(
            {'resistive_heater_intent': 'ON'},
            accumulator_temp=None,
        )

        mock_service.assert_called_once_with(
            'switch', 'turn_off',
            {'entity_id': 'switch.mlp_vastus_output_0'},
            return_response=False,
        )

    @patch('run_often.call_ha_service')
    def test_resistive_heater_control_rejects_stale_plan(self, mock_service):
        from run_often import control_resistive_heater
        stale = datetime.now().astimezone() - timedelta(hours=1)

        control_resistive_heater(
            {'timestamp': stale.isoformat(), 'resistive_heater_intent': 'ON'},
            accumulator_temp=50.0,
        )

        self.assertEqual(mock_service.call_args.args[1], 'turn_off')

    @patch('run_often.call_ha_service')
    def test_resistive_heater_stays_off_after_reaching_interval_cutoff(self, mock_service):
        from run_often import control_resistive_heater
        now = datetime.now().astimezone()
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        plan = {
            'timestamp': slot.isoformat(),
            'resistive_heater_intent': 'ON',
            'gshp_temp_simulated': 52.0,
        }

        control_resistive_heater(plan, accumulator_temp=52.0)
        control_resistive_heater(plan, accumulator_temp=51.5)

        self.assertEqual([item.args[1] for item in mock_service.call_args_list], ['turn_off', 'turn_off'])

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_load_following_flow(self, mock_get_ha, mock_push):
        """Normal flow with BATTERY_NET_METERING=0 calls load following."""
        mock_get_ha.side_effect = lambda eid: {'state': '50.0'}
        import os as os_module
        with patch.dict(os.environ, {'BATTERY_NET_METERING': '0'}):
            from run_often import main
            main()

        # push_battery_control should be called with a negative power (discharge)
        mock_push.assert_called_once()
        args = mock_push.call_args
        self.assertEqual(args[1]['battery_action'], 'charge',
                         "Should pass through the plan's action")
        self.assertIsInstance(args[1]['battery_power_w'], int)
        self.assertEqual(args[1]['battery_soc_pct'], 50.0)

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_net_metering_flow(self, mock_get_ha, mock_push):
        """With BATTERY_NET_METERING=1, net metering branch is taken."""
        mock_get_ha.side_effect = lambda eid: {'state': '50.0'}
        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push.assert_called_once()

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_idle_absorbs_cheap_unexpected_solar_export(self, mock_get_ha, mock_push):
        state_dir = os.path.join(self.test_dir, 'state')
        self._make_plan_file(state_dir, extra_fields={
            'battery_power_kw': 0.0,
            'battery_action': 'idle',
            'export_unit_price': 0.003,
            'grid_import_kwh': 0.0,
        })

        def state_side_effect(entity_id: str):
            values = {
                'sensor.be_soc': '17.5',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '-4.0',
                'sensor.solar_plant_real_power_kw_2': '5.0',
                'sensor.cumulative_active_import': '100.0',
                'sensor.cumulative_active_export': '50.0',
                'input_select.hepo_battery_action': 'auto',
            }
            return {'state': values.get(entity_id, '0.0')}

        mock_get_ha.side_effect = state_side_effect
        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '1',
            'BATTERY_RAMP_RATE_KW_PER_MIN': '0',
            'BATTERY_IDLE_SOLAR_CHARGE_MAX_EXPORT_PRICE': '0.02',
        }):
            from run_often import main
            main()

        args = mock_push.call_args.kwargs
        self.assertEqual(args['battery_action'], 'net_metering')
        self.assertEqual(args['battery_power_w'], -4000)

    @patch('run_often.push_ha_state')
    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_push_period_balance(self, mock_get_ha, mock_push_battery, mock_push_state):
        """Period balance sensor is pushed with actual interval import/export values."""
        # Set up cumulative meters that differ from baseline to get non-zero deltas
        def state_side_effect(eid: str):
            vals = {
                'sensor.be_soc': '26.6',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.8',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        # Pre-seed net metering state with baselines that differ from current readings
        net_state = {
            'interval_start': 97.5,
            'import_start': 47.5,
            'export_start': 50.0,
            'planned_battery_kw': 0.0,
        }
        with open(os.path.join(self.test_dir, 'state', 'net_metering_state.json'), 'w') as f:
            json.dump(net_state, f)

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push_state.assert_called_once()
        args, kwargs = mock_push_state.call_args
        self.assertEqual(args[0], 'sensor.hepo_period_balance')
        self.assertEqual(args[1], '1.700')  # 2.5 - 0.8
        self.assertEqual(args[2]['import_kwh'], 2.5)
        self.assertEqual(args[2]['export_kwh'], 0.8)
        self.assertEqual(args[2]['net_kw'], 6.8)  # 1.7 * 4
        self.assertEqual(args[2]['target_net_kwh'], 2.0)  # from plan: 2.0 - 0.0

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_graceful_when_sensors_unavailable(self, mock_get_ha, mock_push):
        """When HA sensors return 'unavailable', main() should not crash."""
        mock_get_ha.side_effect = lambda eid: {'state': 'unavailable'}
        with patch.dict(os.environ, {'BATTERY_NET_METERING': '0'}):
            from run_often import main
            main()

        mock_push.assert_called_once()

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_graceful_when_plan_missing(self, mock_get_ha, mock_push):
        """When optimization_plan.json is missing, main() returns without push."""
        os.remove(os.path.join(self.test_dir, 'state', 'optimization_plan.json'))
        mock_get_ha.side_effect = lambda eid: {'state': '50.0'}
        with patch.dict(os.environ, {'BATTERY_NET_METERING': '0'}):
            from run_often import main
            main()

        mock_push.assert_not_called()

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_no_current_plan_entry(self, mock_get_ha, mock_push):
        """When no entry matches current time, first entry is used (fallback)."""
        # Create plan with only future timestamps
        now = datetime.now(timezone.utc)
        far_future = now.replace(hour=(now.hour + 3) % 24)
        state_dir = os.path.join(self.test_dir, 'state')
        plan = [{
            'timestamp': far_future.isoformat(),
            'battery_power_kw': 2.0,
            'battery_action': 'discharge',
            'soc_pct': 60.0,
        }]
        with open(os.path.join(state_dir, 'optimization_plan.json'), 'w') as f:
            json.dump(plan, f)

        mock_get_ha.side_effect = lambda eid: {'state': '50.0'}
        with patch.dict(os.environ, {'BATTERY_NET_METERING': '0'}):
            from run_often import main
            main()

        mock_push.assert_called_once()

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_soc_none_does_not_crash(self, mock_get_ha, mock_push):
        """When SoC sensor is 'unknown', soc_pct=None but flow continues."""
        def state_side_effect(eid: str):
            if eid == 'sensor.be_soc':
                return {'state': 'unknown'}
            return {'state': '50.0'}
        mock_get_ha.side_effect = state_side_effect
        with patch.dict(os.environ, {'BATTERY_NET_METERING': '0'}):
            from run_often import main
            main()

        mock_push.assert_called_once()

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_follow_in_net_metering_uses_load_following(self, mock_get_ha, mock_push):
        """When net metering is enabled and action is 'follow', load-following
        logic should be used (not net metering PI controller).

        Regression test: 'follow' was passed through unchanged by
        compute_net_metering_setpoint, causing the battery to blindly
        follow wrong SARIMA predictions and export excess energy.
        """
        now = datetime.now(timezone.utc)
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        plan = [{
            'timestamp': slot.isoformat(),
            'battery_power_kw': -4.0,
            'battery_action': 'follow',
            'soc_pct': 29.5,
            'grid_import_kwh': 1.0,
            'grid_export_kwh': 0.0,
        }]
        state_dir = os.path.join(self.test_dir, 'state')
        with open(os.path.join(state_dir, 'optimization_plan.json'), 'w') as f:
            json.dump(plan, f)

        def state_side_effect(eid: str):
            vals = {
                'sensor.be_soc': '29.5',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '1.0',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '100.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '1',
            'BATTERY_RAMP_RATE_KW_PER_MIN': '0',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        battery_w = args[1]['battery_power_w']
        # Load-following should discharge ~1 kW to zero out grid import,
        # NOT follow the plan's -4.0 kW discharge.
        self.assertGreater(battery_w, 0, "Battery should discharge")
        self.assertLess(battery_w, 2500,
                        "Should discharge ~1 kW to follow load, NOT 4 kW")
        self.assertGreater(battery_w, 500,
                           "Should discharge at least 0.5 kW to follow load")

    @patch('run_often.apply_discharge_budget', return_value=(-1.0, ''))
    @patch('run_often.accumulate_interval_discharge', return_value=0.0)
    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_discharge_budget_uses_configured_interval(
        self, mock_get_ha, mock_push, mock_accumulate, mock_apply,
    ):
        """Budget accounting must use the same interval as the planner."""
        state_dir = os.path.join(self.test_dir, 'state')
        self._make_plan_file(state_dir, extra_fields={
            'battery_power_kw': -1.0,
            'battery_action': 'discharge_load',
            'discharge_budget_kwh': 1.0,
        })
        mock_get_ha.side_effect = lambda eid: {'state': '50.0'}

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '0',
            'PLAN_INTERVAL_MINUTES': '60',
        }):
            from run_often import main
            main()

        mock_accumulate.assert_called_once_with(50.0, interval_minutes=60)
        self.assertEqual(mock_apply.call_args.kwargs['interval_minutes'], 60)


    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_solar_forecast_fallback_when_real_time_unavailable(self, mock_get_ha, mock_push):
        """When solar sensor is unavailable, fall back to solar_forecast_kw from plan."""
        state_dir = os.path.join(self.test_dir, 'state')
        self._make_plan_file(state_dir, extra_fields={'solar_forecast_kw': 3.5})

        def state_side_effect(eid: str):
            vals = {
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': 'unavailable',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '0',
            'SOLAR_FALLBACK_TO_FORECAST': 'true',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        battery_w = args[1]['battery_power_w']
        # With solar_kw=3.5 and grid_w=0 / battery_w=0,
        # load_following should see enough solar and NOT discharge.
        # Solar 3.5kW > load 0kW → surplus → battery should charge (negative power).
        self.assertLess(battery_w, 0, "Battery should charge with solar surplus from forecast")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_solar_forecast_fallback_disabled(self, mock_get_ha, mock_push):
        """When SOLAR_FALLBACK_TO_FORECAST=false, unavailable solar stays 0."""
        state_dir = os.path.join(self.test_dir, 'state')
        self._make_plan_file(state_dir, extra_fields={'solar_forecast_kw': 3.5})

        def state_side_effect(eid: str):
            vals = {
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': 'unavailable',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '0',
            'SOLAR_FALLBACK_TO_FORECAST': 'false',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        # When fallback is disabled: solar_kw stays 0.
        # plan says charge at 1.5kW; push_battery_control uses convention negative=charge.
        args = mock_push.call_args
        battery_w = args[1]['battery_power_w']
        self.assertLess(battery_w, 0, "Battery should charge per plan when fallback is disabled (negative=charge)")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_solar_entity_read_from_env(self, mock_get_ha, mock_push):
        """Solar production entity is read from SOLAR_PRODUCTION_ENTITY env var."""
        state_dir = os.path.join(self.test_dir, 'state')
        self._make_plan_file(state_dir)

        called_entities: list[str] = []

        def state_side_effect(eid: str):
            called_entities.append(eid)
            return {'state': '0.0'}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {
            'SOLAR_PRODUCTION_ENTITY': 'sensor.solar_plant_real_power_kw_2',
            'BATTERY_NET_METERING': '0',
        }):
            from run_often import main
            main()

        self.assertIn('sensor.solar_plant_real_power_kw_2', called_entities,
                      "Solar entity should be read from SOLAR_PRODUCTION_ENTITY env var")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_solar_entity_default(self, mock_get_ha, mock_push):
        """Without SOLAR_PRODUCTION_ENTITY, the legacy solar entity is used."""
        state_dir = os.path.join(self.test_dir, 'state')
        self._make_plan_file(state_dir)

        called_entities: list[str] = []

        def state_side_effect(eid: str):
            called_entities.append(eid)
            return {'state': '0.0'}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '0'}):
            from run_often import main
            main()

        self.assertIn('sensor.solarh_63038_real_power_kw', called_entities,
                      "Default solar entity should be the legacy sensor")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_manual_override_auto(self, mock_get_ha, mock_push):
        """When input_select is 'auto', normal flow proceeds (net metering active)."""
        def state_side_effect(eid: str):
            vals = {
                'input_select.hepo_battery_action': 'auto',
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        self.assertEqual(args[1]['battery_action'], 'net_metering',
                         "Normal net metering flow when override is 'auto'")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_manual_override_idle(self, mock_get_ha, mock_push):
        """When input_select is 'idle', battery is forced to 0W and load-following is used."""
        def state_side_effect(eid: str):
            vals = {
                'input_select.hepo_battery_action': 'idle',
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        self.assertEqual(args[1]['battery_power_w'], 0,
                         "Battery should be forced to 0W when override is 'idle'")
        self.assertEqual(args[1]['battery_action'], 'idle',
                         "Battery action should be overridden to 'idle'")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_manual_override_charge_solar_skips_net_metering(self, mock_get_ha, mock_push):
        """When input_select is 'charge_solar', net metering is skipped even if enabled."""
        now = datetime.now(timezone.utc)
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        plan = [{
            'timestamp': slot.isoformat(),
            'battery_power_kw': 0.0,
            'battery_action': 'follow',
            'soc_pct': 50.0,
            'grid_import_kwh': 1.0,
            'grid_export_kwh': 0.0,
        }]
        state_dir = os.path.join(self.test_dir, 'state')
        with open(os.path.join(state_dir, 'optimization_plan.json'), 'w') as f:
            json.dump(plan, f)

        def state_side_effect(eid: str):
            vals = {
                'input_select.hepo_battery_action': 'charge_solar',
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '-0.5',
                'sensor.solarh_63038_real_power_kw': '2.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        self.assertEqual(args[1]['battery_action'], 'charge_solar',
                         "Battery action should be overridden to 'charge_solar'")
        battery_w = args[1]['battery_power_w']
        self.assertLess(battery_w, 0,
                        "Battery should charge (negative power = charge) from solar surplus")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_manual_override_unknown_falls_back(self, mock_get_ha, mock_push):
        """When input_select state is 'unknown', normal flow proceeds (no override)."""
        def state_side_effect(eid: str):
            vals = {
                'input_select.hepo_battery_action': 'unknown',
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        self.assertEqual(args[1]['battery_action'], 'net_metering',
                         "Normal net metering flow when override is 'unknown'")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_manual_override_unavailable_falls_back(self, mock_get_ha, mock_push):
        """When input_select state is 'unavailable', normal flow proceeds (no override)."""
        def state_side_effect(eid: str):
            vals = {
                'input_select.hepo_battery_action': 'unavailable',
                'sensor.be_soc': '50.0',
                'sensor.be_stat_batt_power': '0.0',
                'sensor.sahkokauppa_20s': '0.0',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': None,
                'sensor.current_phase_2': None,
                'sensor.current_phase_3': None,
                'sensor.cumulative_active_import': '50.0',
                'sensor.cumulative_active_export': '50.0',
            }
            return {'state': vals.get(eid, '0.0')}
        mock_get_ha.side_effect = state_side_effect

        with patch.dict(os.environ, {'BATTERY_NET_METERING': '1'}):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        self.assertEqual(args[1]['battery_action'], 'net_metering',
                         "Normal net metering flow when override is 'unavailable'")

    def _make_net_metering_plan(self, battery_kw: float = 10.0, action: str = 'charge_grid'):
        """Create a plan entry for the current interval with a net-energy target."""
        now = datetime.now(timezone.utc)
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        plan = [{
            'timestamp': slot.isoformat(),
            'battery_power_kw': battery_kw,
            'battery_action': action,
            'soc_pct': 54.0,
            'grid_import_kwh': 3.760,
            'grid_export_kwh': 0.0,
        }]
        with open(os.path.join(self.test_dir, 'state', 'optimization_plan.json'), 'w') as f:
            json.dump(plan, f)

    def _phase_side_effect(self, phases, battery_w='9480.0', soc='54.0'):
        """Return a get_ha_state side effect with the given phase currents (Amps)."""
        def side_effect(eid: str):
            vals = {
                'sensor.be_soc': soc,
                'sensor.be_stat_batt_power': battery_w,
                'sensor.sahkokauppa_20s': '18.7',
                'sensor.solarh_63038_real_power_kw': '0.0',
                'sensor.mlp_teho': '0.0',
                'sensor.tasmota_energy_power_3': '0.0',
                'sensor.current_phase_1': str(phases[0]) if phases[0] is not None else None,
                'sensor.current_phase_2': str(phases[1]) if phases[1] is not None else None,
                'sensor.current_phase_3': str(phases[2]) if phases[2] is not None else None,
                'sensor.cumulative_active_import': '95895.0',
                'sensor.cumulative_active_export': '17047.0',
            }
            return {'state': vals.get(eid, '0.0')}
        return side_effect

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_fuse_cap_limits_net_metering_charge(self, mock_get_ha, mock_push):
        """Net metering must not charge at full power when a phase exceeds the fuse.

        Regression test for the 05:20 incident: L2/L3 were ~32A against the 25A
        fuse limit while the inverter was instructed to charge at 10kW. The fuse
        cap must limit the charge to the safe headroom instead.
        """
        self._make_net_metering_plan(battery_kw=10.0, action='charge_grid')
        # Phases from the incident log (L1 15.9A, L2 32.4A, L3 32.1A) with the
        # battery already charging at 9480W.
        mock_get_ha.side_effect = self._phase_side_effect([15.9, 32.4, 32.1], battery_w='9480.0')

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '1',
            'BATTERY_RAMP_RATE_KW_PER_MIN': '0',
            'MAIN_FUSE_SIZE_A': '25',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        battery_control_w = args[1]['battery_power_w']
        # Safe max charge = 9480 + (25 - 32.4) * 3 * 230 = 4374W
        self.assertEqual(battery_control_w, -4374,
                         "Charge must be capped to the fuse-limited headroom, not 10kW")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_fuse_cap_forces_discharge_on_phase_overload(self, mock_get_ha, mock_push):
        """When non-battery load alone exceeds the fuse, discharge is forced.

        Even though the plan wants to charge at 10kW, a phase at 30A (over the
        25A fuse) leaves no room to charge: the safe setpoint becomes negative
        and the battery must discharge to protect the fuse.
        """
        self._make_net_metering_plan(battery_kw=10.0, action='charge_grid')
        mock_get_ha.side_effect = self._phase_side_effect([30.0, 10.0, 10.0], battery_w='0.0')

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '1',
            'BATTERY_RAMP_RATE_KW_PER_MIN': '0',
            'MAIN_FUSE_SIZE_A': '25',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        battery_control_w = args[1]['battery_power_w']
        # Safe max = 0 + (25 - 30) * 3 * 230 = -3450W -> forced discharge.
        self.assertEqual(battery_control_w, 3450,
                         "Battery must discharge to protect the fuse even though the plan charges")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_fuse_cap_applies_after_ramp_limiter(self, mock_get_ha, mock_push):
        """The fuse cap must win over the ramp limiter.

        Even with a ramp rate enabled, the commanded setpoint must not end up
        above the fuse-limited headroom after ramping toward the planned charge.
        """
        self._make_net_metering_plan(battery_kw=10.0, action='charge_grid')
        mock_get_ha.side_effect = self._phase_side_effect([15.9, 32.4, 32.1], battery_w='9480.0')

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '1',
            'BATTERY_RAMP_RATE_KW_PER_MIN': '3.0',
            'MAIN_FUSE_SIZE_A': '25',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        battery_control_w = args[1]['battery_power_w']
        # Ramp allows moving toward 10kW, but the fuse cap must clamp to 4374W.
        self.assertEqual(battery_control_w, -4374,
                         "Fuse cap must be applied after the ramp limiter")

    @patch('run_often.push_battery_control')
    @patch('run_often.get_ha_state')
    def test_fuse_cap_respects_soc_floor(self, mock_get_ha, mock_push):
        """The fuse cap must not discharge below the configured SoC floor."""
        self._make_net_metering_plan(battery_kw=10.0, action='charge_grid')
        # Phase overloaded but battery is at the 10% floor.
        mock_get_ha.side_effect = self._phase_side_effect([30.0, 10.0, 10.0], battery_w='0.0', soc='10.0')

        with patch.dict(os.environ, {
            'BATTERY_NET_METERING': '1',
            'BATTERY_RAMP_RATE_KW_PER_MIN': '0',
            'MAIN_FUSE_SIZE_A': '25',
            'BATTERY_MIN_SOC_PCT': '10.0',
        }):
            from run_often import main
            main()

        mock_push.assert_called_once()
        args = mock_push.call_args
        battery_control_w = args[1]['battery_power_w']
        # SoC at floor blocks the forced discharge -> idle.
        self.assertEqual(battery_control_w, 0,
                         "Fuse cap must not discharge below the SoC floor")


if __name__ == '__main__':
    unittest.main()

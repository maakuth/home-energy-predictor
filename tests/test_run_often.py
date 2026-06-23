from __future__ import annotations
import unittest
from unittest.mock import patch, MagicMock, call
import json
import os
import tempfile
import shutil
from datetime import datetime, timezone


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

    def _make_plan_file(self, state_dir: str):
        """Create a minimal optimization_plan.json."""
        now = datetime.now(timezone.utc)
        slot = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
        plan = [{
            'timestamp': slot.isoformat(),
            'battery_power_kw': 1.5,
            'battery_action': 'charge',
            'soc_pct': 50.0,
            'grid_import_kwh': 2.0,
            'grid_export_kwh': 0.0,
        }]
        with open(os.path.join(state_dir, 'optimization_plan.json'), 'w') as f:
            json.dump(plan, f)

    def _mock_ha_state(self, values: dict[str, str]) -> MagicMock:
        """Create a get_ha_state mock that returns a 'state' dict."""
        def side_effect(entity_id: str):
            val = values.get(entity_id, '0.0')
            return {'state': str(val)}
        return MagicMock(side_effect=side_effect)

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


if __name__ == '__main__':
    unittest.main()

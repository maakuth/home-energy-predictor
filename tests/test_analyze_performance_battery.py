from __future__ import annotations
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from analyze_performance import fetch_actuals

class TestBatteryAwareActuals(unittest.TestCase):
    """Test that battery power is correctly subtracted from actual_usage in performance analysis."""

    def test_battery_discharging_increases_actual_usage(self):
        """When battery discharges (negative power), actual home load increases."""
        with patch('analyze_performance.fetch_states_history') as mock_fetch:
            ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(4)]
            
            # Battery discharging at 2000W = -2kW (negative = discharging in our convention)
            # But be_stat_batt_power: positive = charging
            # So discharging = -2000W
            mock_fetch.return_value = {
                'sensor.sahkokauppa_nyt': pd.DataFrame({'state': [1.0, 1.0, 1.0, 1.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.solarh_63038_real_power_kw': pd.DataFrame({'state': [0.5, 0.5, 0.5, 0.5]}, index=pd.DatetimeIndex(ts)),
                'sensor.mlp_teho': pd.DataFrame({'state': [0.0, 0.0, 0.0, 0.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.be_stat_batt_power': pd.DataFrame({'state': [-2000.0, -2000.0, -2000.0, -2000.0]}, index=pd.DatetimeIndex(ts))
            }
            
            df_actual = fetch_actuals(days=1)
            
            # actual_usage = grid + solar - battery_kw
            # = 1.0 + 0.5 - (-2000/1000) = 1.5 + 2.0 = 3.5 kW
            expected = [3.5, 3.5, 3.5, 3.5]
            np.testing.assert_array_almost_equal(df_actual['actual_usage'].values, expected)

    def test_battery_charging_decreases_actual_usage(self):
        """When battery charges (positive power), actual home load decreases."""
        with patch('analyze_performance.fetch_states_history') as mock_fetch:
            ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(4)]
            
            # Battery charging at 3000W = +3kW
            mock_fetch.return_value = {
                'sensor.sahkokauppa_nyt': pd.DataFrame({'state': [2.0, 2.0, 2.0, 2.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.solarh_63038_real_power_kw': pd.DataFrame({'state': [5.0, 5.0, 5.0, 5.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.mlp_teho': pd.DataFrame({'state': [0.0, 0.0, 0.0, 0.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.be_stat_batt_power': pd.DataFrame({'state': [3000.0, 3000.0, 3000.0, 3000.0]}, index=pd.DatetimeIndex(ts))
            }
            
            df_actual = fetch_actuals(days=1)
            
            # actual_usage = 2.0 + 5.0 - (3000/1000) = 7.0 - 3.0 = 4.0 kW
            expected = [4.0, 4.0, 4.0, 4.0]
            np.testing.assert_array_almost_equal(df_actual['actual_usage'].values, expected)

    def test_battery_idle_no_effect(self):
        """When battery is idle (0W), actual_usage is unchanged."""
        with patch('analyze_performance.fetch_states_history') as mock_fetch:
            ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(4)]
            
            mock_fetch.return_value = {
                'sensor.sahkokauppa_nyt': pd.DataFrame({'state': [1.0, 1.0, 1.0, 1.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.solarh_63038_real_power_kw': pd.DataFrame({'state': [0.5, 0.5, 0.5, 0.5]}, index=pd.DatetimeIndex(ts)),
                'sensor.mlp_teho': pd.DataFrame({'state': [0.0, 0.0, 0.0, 0.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.be_stat_batt_power': pd.DataFrame({'state': [0.0, 0.0, 0.0, 0.0]}, index=pd.DatetimeIndex(ts))
            }
            
            df_actual = fetch_actuals(days=1)
            
            # actual_usage = 1.0 + 0.5 - 0 = 1.5 kW
            expected = [1.5, 1.5, 1.5, 1.5]
            np.testing.assert_array_almost_equal(df_actual['actual_usage'].values, expected)

    def test_battery_sensor_missing_defaults_to_zero(self):
        """If battery sensor is not available, actual_usage should still work (backward compat)."""
        with patch('analyze_performance.fetch_states_history') as mock_fetch:
            ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(4)]
            
            # No battery sensor in returned data
            mock_fetch.return_value = {
                'sensor.sahkokauppa_nyt': pd.DataFrame({'state': [1.0, 1.0, 1.0, 1.0]}, index=pd.DatetimeIndex(ts)),
                'sensor.solarh_63038_real_power_kw': pd.DataFrame({'state': [0.5, 0.5, 0.5, 0.5]}, index=pd.DatetimeIndex(ts)),
                'sensor.mlp_teho': pd.DataFrame({'state': [0.0, 0.0, 0.0, 0.0]}, index=pd.DatetimeIndex(ts))
            }
            
            df_actual = fetch_actuals(days=1)
            
            # actual_usage = 1.0 + 0.5 - 0 = 1.5 kW (battery defaults to 0)
            expected = [1.5, 1.5, 1.5, 1.5]
            np.testing.assert_array_almost_equal(df_actual['actual_usage'].values, expected)


if __name__ == '__main__':
    unittest.main()

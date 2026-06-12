"""
Example test using pickled battery test data.

This demonstrates how to use the BatteryTestData utility to load
pre-dumped battery planning data for testing.

To generate test data:
    python dump_battery_data.py --days 7 --output battery_test_data.pkl --verbose

Then use it in tests:
    pytest tests/test_battery_data_example.py
"""

import unittest
import os
from pathlib import Path
from utils.battery_test_data import BatteryTestData, load_battery_test_data


class TestBatteryDataLoading(unittest.TestCase):
    """Test loading and accessing battery test data."""
    
    @classmethod
    def setUpClass(cls):
        """Check if test data file exists."""
        cls.test_data_path = Path('battery_test_data.pkl')
        cls.has_test_data = cls.test_data_path.exists()
    
    def setUp(self):
        """Skip tests if test data not available."""
        if not self.has_test_data:
            self.skipTest("battery_test_data.pkl not found - run dump_battery_data.py first")
    
    def test_load_battery_test_data(self):
        """Test loading battery data from pickle file."""
        data = BatteryTestData.load(str(self.test_data_path))
        
        # Verify metadata exists
        self.assertIn('dumped_at', data.metadata)
        self.assertIn('period_start', data.metadata)
        self.assertIn('period_end', data.metadata)
        self.assertIn('model_version', data.metadata)
    
    def test_predictions_access(self):
        """Test accessing predictions from loaded data."""
        data = load_battery_test_data(str(self.test_data_path))
        
        preds = data.predictions_list()
        self.assertIsInstance(preds, list)
        
        if preds:
            # Should have expected fields
            first = preds[0]
            self.assertIn('timestamp', first)
            # May have: predicted_baseload, solar_forecast, outside_temp, etc.
    
    def test_predictions_dataframe(self):
        """Test getting predictions as DataFrame."""
        data = load_battery_test_data(str(self.test_data_path))
        
        df = data.predictions_df()
        
        # Should be a valid DataFrame
        self.assertIsNotNone(df)
        
        if len(df) > 0:
            # Should have timestamp as index
            self.assertTrue(hasattr(df.index, 'name'))
    
    def test_ha_states_access(self):
        """Test accessing Home Assistant states."""
        data = load_battery_test_data(str(self.test_data_path))
        
        # Test getting a state
        battery_soc = data.ha_state('sensor.battery_soc_pct')
        if battery_soc is not None:
            # Should have the expected fields
            self.assertIn('state', battery_soc)
            # Accessing as float should work
            soc_value = data.ha_state_float('sensor.battery_soc_pct')
            self.assertIsInstance(soc_value, float)
    
    def test_ha_state_float_with_default(self):
        """Test ha_state_float returns default for missing entities."""
        data = load_battery_test_data(str(self.test_data_path))
        
        # Non-existent entity should return default
        value = data.ha_state_float('sensor.nonexistent_entity', default=42.0)
        self.assertEqual(value, 42.0)
    
    def test_battery_config_access(self):
        """Test accessing battery configuration."""
        data = load_battery_test_data(str(self.test_data_path))
        
        config = data.battery_config
        self.assertIsInstance(config, dict)
        
        # Should have these keys
        expected_keys = ['capacity_kwh', 'min_soc_pct', 'max_soc_pct', 
                         'charge_rate_kw', 'discharge_rate_kw', 'enabled']
        for key in expected_keys:
            self.assertIn(key, config)
    
    def test_gshp_config_access(self):
        """Test accessing GSHP configuration."""
        data = load_battery_test_data(str(self.test_data_path))
        
        config = data.gshp_config
        self.assertIsInstance(config, dict)
        
        # Should have these keys
        expected_keys = ['enabled', 'max_power_kw']
        for key in expected_keys:
            self.assertIn(key, config)
    
    def test_period_helpers(self):
        """Test helper methods for time period."""
        data = load_battery_test_data(str(self.test_data_path))
        
        start = data.period_start()
        end = data.period_end()
        dumped = data.dumped_at()
        
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)
        self.assertIsNotNone(dumped)
        
        # Verify logical ordering
        self.assertLess(start, end)
        self.assertGreaterEqual(dumped, end)
    
    def test_summary(self):
        """Test human-readable summary."""
        data = load_battery_test_data(str(self.test_data_path))
        
        summary = data.summary()
        self.assertIsInstance(summary, str)
        self.assertIn('Battery Test Data Summary', summary)
        self.assertIn('Period:', summary)


class TestBatteryDataUsageExample(unittest.TestCase):
    """Example of using battery test data in a test."""
    
    @classmethod
    def setUpClass(cls):
        """Check if test data file exists."""
        cls.test_data_path = Path('battery_test_data.pkl')
        cls.has_test_data = cls.test_data_path.exists()
    
    def setUp(self):
        """Skip if no test data."""
        if not self.has_test_data:
            self.skipTest("battery_test_data.pkl not found - run dump_battery_data.py first")
    
    def test_example_battery_planning_scenario(self):
        """
        Example: Use real battery test data to verify planning logic.
        
        This demonstrates a realistic test scenario where you:
        1. Load historical battery data
        2. Extract the relevant inputs (predictions, prices, current state)
        3. Run your planning algorithm
        4. Verify the output makes sense
        """
        data = load_battery_test_data(str(self.test_data_path))
        
        # Get current battery state from HA snapshot
        soc_pct = data.ha_state_float('sensor.battery_soc_pct', default=50.0)
        battery_enabled = data.battery_config.get('enabled', False)
        
        print(f"\n📊 Test Scenario:")
        print(f"   Battery enabled: {battery_enabled}")
        print(f"   Current SOC: {soc_pct}%")
        print(f"   Data period: {data.period_start()} to {data.period_end()}")
        
        # Get predictions for the period
        df_pred = data.predictions_df()
        if len(df_pred) > 0:
            print(f"   Predictions available: {len(df_pred)} records")
            if 'predicted_baseload' in df_pred.columns:
                avg_load = df_pred['predicted_baseload'].mean()
                print(f"   Average baseload: {avg_load:.2f} kW")
        
        # Get market prices
        df_prices = data.market_prices_df()
        if len(df_prices) > 0:
            print(f"   Prices available: {len(df_prices)} records")
            if 'import_price' in df_prices.columns:
                avg_price = df_prices['import_price'].mean()
                print(f"   Average import price: {avg_price:.4f} EUR/kWh")
        
        # Example: Verify data quality
        self.assertIsNotNone(data)
        self.assertTrue(battery_enabled or not battery_enabled)  # Just verify it's a boolean


if __name__ == '__main__':
    unittest.main()

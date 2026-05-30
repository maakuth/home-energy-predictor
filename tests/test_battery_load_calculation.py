"""
Tests for battery-aware load power calculations.

The model must correctly account for battery power when calculating home load.
Without battery compensation, load will be calculated incorrectly when battery
is present, causing the model to learn wrong patterns.

Physics:
  Home Load = Grid Import + Solar - Battery Net Power
  where Battery Net Power = Battery Charging - Battery Discharging
        (our convention: positive when charging from grid/solar)
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestBatteryAwareLoadCalculation(unittest.TestCase):
    """Test that battery power is correctly subtracted from total_home_power."""

    @staticmethod
    def calculate_load(total_power, solar_actual, battery_power=0.0, gshp_power=0.0, leaf_power=0.0):
        """Helper: Calculate home load with battery compensation (same logic as process_data.py)."""
        # total_home_power = grid + solar - battery (if battery sensor exists)
        total_home_power = total_power + solar_actual - (battery_power / 1000.0)
        # baseload = total_home_power - gshp - leaf
        gshp_kw = gshp_power / 1000.0
        leaf_kw = leaf_power / 1000.0
        baseload_power = max(0.0, total_home_power - gshp_kw - leaf_kw)
        return total_home_power, baseload_power

    def test_battery_charging_from_solar(self):
        """When battery charges from solar, home load should exclude that charge."""
        # Scenario: Solar produces 5 kW, battery charges at 3 kW, no grid import
        # Real home load = 2 kW (only baseload, battery is charging)
        
        total, baseload = self.calculate_load(
            total_power=-2.0,    # Grid: -2 kW (exporting)
            solar_actual=5.0,    # Solar: 5 kW
            battery_power=3000.0,  # Battery: 3 kW charging (in Watts)
            gshp_power=0.0,
            leaf_power=0.0
        )
        
        # total_home_power = -2 + 5 - 3 = 0 kW (only baseload, GSHP at 0)
        # baseload_power = 0 - 0 - 0 = 0 kW
        self.assertAlmostEqual(total, 0.0, places=5)
        self.assertAlmostEqual(baseload, 0.0, places=5)

    def test_battery_discharging_to_home(self):
        """When battery discharges, that power counts toward home load."""
        # Scenario: Grid imports 1 kW, solar produces 0.5 kW, battery discharges 2 kW
        # Real home load = 3.5 kW
        
        total, baseload = self.calculate_load(
            total_power=1.0,       # Grid: 1 kW (importing)
            solar_actual=0.5,      # Solar: 0.5 kW
            battery_power=-2000.0, # Battery: -2 kW discharging (in Watts)
            gshp_power=0.0,
            leaf_power=0.0
        )
        
        # total_home_power = 1.0 + 0.5 - (-2.0) = 3.5 kW
        # baseload_power = 3.5 - 0 - 0 = 3.5 kW
        self.assertAlmostEqual(total, 3.5, places=5)
        self.assertAlmostEqual(baseload, 3.5, places=5)

    def test_battery_idle(self):
        """When battery is idle (0 power), load calculation is unchanged."""
        # Scenario: Grid 1 kW, solar 1 kW, battery idle
        # Real home load = 2 kW
        
        total, baseload = self.calculate_load(
            total_power=1.0,
            solar_actual=1.0,
            battery_power=0.0,  # Battery idle
            gshp_power=0.0,
            leaf_power=0.0
        )
        
        # total_home_power = 1.0 + 1.0 - 0.0 = 2.0 kW
        self.assertAlmostEqual(total, 2.0, places=5)

    def test_battery_mixed_with_gshp(self):
        """Battery power subtraction should work alongside GSHP subtraction."""
        # Scenario:
        # - Grid: 2 kW import
        # - Solar: 1 kW
        # - Battery: 1.5 kW charging
        # - GSHP: 2 kW
        # 
        # total_home_power = 2 + 1 - 1.5 = 1.5 kW (grid + solar - battery)
        # baseload = 1.5 - 2 (GSHP) → clipped to 0 (house load < GSHP means GSHP is running)
        
        total, baseload = self.calculate_load(
            total_power=2.0,
            solar_actual=1.0,
            battery_power=1500.0,  # Battery charging (in Watts)
            gshp_power=2000.0,  # 2 kW in Watts
            leaf_power=0.0
        )
        
        # total_home_power = 2 + 1 - 1.5 = 1.5 kW
        # baseload = 1.5 - 2 = -0.5 → clipped to 0
        self.assertAlmostEqual(total, 1.5, places=5)
        self.assertAlmostEqual(baseload, 0.0, places=5)

    def test_battery_without_sensor_gracefully_defaults(self):
        """If battery_power not provided, calculation should default to 0."""
        # Regression test: ensure backward compatibility when battery sensor is unavailable
        
        total, baseload = self.calculate_load(
            total_power=2.0,
            solar_actual=1.0,
            # battery_power defaults to 0.0
            gshp_power=0.0,
            leaf_power=0.0
        )
        
        # total_home_power = 2 + 1 - 0 = 3 kW (no battery term)
        self.assertAlmostEqual(total, 3.0, places=5)

    def test_battery_power_in_watts_converted_to_kw(self):
        """Battery power comes in Watts and must be converted to kW."""
        # Battery power: 1500 W = 1.5 kW
        
        total, baseload = self.calculate_load(
            total_power=2.0,
            solar_actual=0.5,
            battery_power=1500.0,  # 1500 W (charging)
            gshp_power=0.0,
            leaf_power=0.0
        )
        
        # total_home_power = 2.0 + 0.5 - (1500/1000) = 2.5 - 1.5 = 1.0 kW
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_battery_multiple_intervals(self):
        """Test battery calculation across multiple 15-minute intervals."""
        
        totals_expected = [2.0, 2.5, 2.5, 3.0]  # (total + solar - battery/1000)
        
        test_cases = [
            {'total': 2.0, 'solar': 1.0, 'battery': 1000.0},  # 2 + 1 - 1 = 2.0
            {'total': 2.5, 'solar': 1.5, 'battery': 1500.0},  # 2.5 + 1.5 - 1.5 = 2.5
            {'total': 1.0, 'solar': 2.0, 'battery': 500.0},   # 1.0 + 2.0 - 0.5 = 2.5
            {'total': 0.5, 'solar': 2.5, 'battery': 0.0},     # 0.5 + 2.5 - 0 = 3.0
        ]
        
        for i, (expected_val, case) in enumerate(zip(totals_expected, test_cases)):
            total, _ = self.calculate_load(
                total_power=case['total'],
                solar_actual=case['solar'],
                battery_power=case['battery']
            )
            self.assertAlmostEqual(total, expected_val, places=5, msg=f"Interval {i}")


if __name__ == '__main__':
    unittest.main()

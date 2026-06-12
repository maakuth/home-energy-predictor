"""
Parametrized replay tests for battery planners over realistic fixture data.

Tests each planner against all available fixtures, verifying:
- No SoC constraint violations
- Valid plan structure
- Performance relative to baseline (no battery)
- No future data leakage
"""

import unittest
import os
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from battery_planners import BatteryPlannerFactory
from tests.battery_planner_replay import (
    BatteryReplaySimulator,
    load_fixture,
    get_fixtures,
)


class TestBatteryPlannerReplay(unittest.TestCase):
    """Base test class for battery planner replay."""
    
    @classmethod
    def setUpClass(cls):
        """Discover available fixtures and planners."""
        cls.fixtures = get_fixtures()
        cls.planner_names = BatteryPlannerFactory.names()
        cls.battery_config = {
            'capacity_kwh': 50.0,
            'min_soc_pct': 10.0,
            'max_soc_pct': 90.0,
            'initial_soc_pct': 10.0,
        }
    
    def setUp(self):
        """Skip if no fixtures available."""
        if not self.fixtures:
            self.skipTest("No battery test fixtures found in tests/fixtures/")
        if not self.planner_names:
            self.skipTest("No battery planners registered")


@pytest.mark.parametrize(
    "fixture_path,planner_name",
    [
        (fixture, planner)
        for fixture in get_fixtures()
        for planner in BatteryPlannerFactory.names()
    ],
    ids=lambda x: f"{Path(x[0]).stem}-{x[1]}" if isinstance(x, tuple) else str(x)
)
class TestBatteryPlannerReplayParametrized:
    """Parametrized replay tests for each fixture and planner combination."""
    
    @staticmethod
    def _print_planner_score(result, fixture_name, planner_name):
        """Print a summary of planner performance."""
        print(f"\n{'='*60}")
        print(f"Planner Score: {planner_name} on {fixture_name}")
        print(f"{'='*60}")
        print(f"  Intervals run:      {result['intervals_run']}")
        print(f"  Baseline cost:      {result['cost_no_battery_eur']:.2f} EUR")
        print(f"  With battery:       {result['cost_with_battery_eur']:.2f} EUR")
        print(f"  Savings:            {result['savings_eur']:.2f} EUR")
        print(f"  Savings %:          {result['savings_pct']:.1f}%")
        print(f"  Final SoC:          {result['final_soc_pct']:.1f}%")
        print(f"  SoC violations:     {result['soc_violations']}")
        print(f"{'='*60}\n")
    
    def test_planner_replay_no_violations(self, fixture_path, planner_name):
        """Planner should respect SoC constraints throughout simulation."""
        fixture = load_fixture(fixture_path)
        simulator = BatteryReplaySimulator(fixture)
        
        if simulator.measurements_df is None or simulator.measurements_df.empty:
            pytest.skip(f"Fixture {Path(fixture_path).stem} has no measurement data")
        
        if simulator.predictions_df is None or simulator.predictions_df.empty:
            pytest.skip(f"Fixture {Path(fixture_path).stem} has no prediction archive")
        
        planner = BatteryPlannerFactory.create(planner_name)
        
        result = simulator.simulate_battery_control(
            planner=planner,
            planner_type=planner_name,
            battery_capacity_kwh=50.0,
            battery_min_soc_pct=10.0,
            battery_max_soc_pct=90.0,
            battery_initial_soc_pct=10.0,
            max_planks=96,
        )
        
        assert result['success'], f"Replay failed: {result.get('error', 'Unknown error')}"
        assert result['soc_violations'] == 0, \
            f"SoC constraint violations: {result['soc_violation_details']}"
        assert result['intervals_run'] > 0, "No intervals were simulated"
        
        # Print planner score
        self._print_planner_score(result, Path(fixture_path).stem, planner_name)
    
    def test_planner_replay_finite_cost(self, fixture_path, planner_name):
        """Planner output should not produce NaN or infinite costs."""
        fixture = load_fixture(fixture_path)
        simulator = BatteryReplaySimulator(fixture)
        
        if simulator.measurements_df is None or simulator.measurements_df.empty:
            pytest.skip(f"Fixture {Path(fixture_path).stem} has no measurement data")
        
        if simulator.predictions_df is None or simulator.predictions_df.empty:
            pytest.skip(f"Fixture {Path(fixture_path).stem} has no prediction archive")
        
        planner = BatteryPlannerFactory.create(planner_name)
        
        result = simulator.simulate_battery_control(
            planner=planner,
            planner_type=planner_name,
            battery_capacity_kwh=50.0,
            battery_min_soc_pct=10.0,
            battery_max_soc_pct=90.0,
            battery_initial_soc_pct=10.0,
            max_planks=96,
        )
        
        assert result['success']
        assert np.isfinite(result['cost_with_battery_eur']), \
            f"Cost with battery is not finite: {result['cost_with_battery_eur']}"
        assert np.isfinite(result['cost_no_battery_eur']), \
            f"Baseline cost is not finite: {result['cost_no_battery_eur']}"
        
        # Print planner score
        self._print_planner_score(result, Path(fixture_path).stem, planner_name)
    
    def test_planner_replay_not_worse_than_baseline(self, fixture_path, planner_name):
        """Planner cost should not significantly exceed no-battery baseline.
        
        We allow a small tolerance for simulation imprecision, but excessive
        degradation (>10% worse) indicates a problem.
        """
        fixture = load_fixture(fixture_path)
        simulator = BatteryReplaySimulator(fixture)
        
        if simulator.measurements_df is None or simulator.measurements_df.empty:
            pytest.skip(f"Fixture {Path(fixture_path).stem} has no measurement data")
        
        planner = BatteryPlannerFactory.create(planner_name)
        
        result = simulator.simulate_battery_control(
            planner=planner,
            planner_type=planner_name,
            battery_capacity_kwh=50.0,
            battery_min_soc_pct=10.0,
            battery_max_soc_pct=90.0,
            battery_initial_soc_pct=10.0,
            max_planks=96,
        )
        
        assert result['success']
        
        baseline_cost = result['cost_no_battery_eur']
        planner_cost = result['cost_with_battery_eur']
        
        # Allow cost to be up to 10% worse than baseline (measurement noise, etc)
        max_acceptable_cost = baseline_cost * 1.10
        
        assert planner_cost <= max_acceptable_cost, \
            f"Planner cost {planner_cost:.2f} EUR exceeds baseline by 10%: {baseline_cost:.2f} EUR"
        
        # Print planner score
        self._print_planner_score(result, Path(fixture_path).stem, planner_name)
    
    def test_planner_output_structure(self, fixture_path, planner_name):
        """Planner output should have correct structure and valid values."""
        fixture = load_fixture(fixture_path)
        simulator = BatteryReplaySimulator(fixture)
        planner = BatteryPlannerFactory.create(planner_name)
        
        # Get a visible plan
        if simulator.measurements_df is None or simulator.measurements_df.empty:
            pytest.skip("Fixture has no measurement data")
        
        planning_time = simulator.measurements_df.index[0]
        if not isinstance(planning_time, (datetime, pd.Timestamp)):
            pytest.skip("Invalid planning time")
        
        predictions, solar, import_prices, export_prices, timestamps = \
            simulator.get_planner_horizon(planning_time, 96)
        
        if len(predictions) == 0:
            pytest.skip("No visible forecasts in fixture for first interval")
        
        os.environ['BATTERY_INITIAL_SOC_PCT'] = '50'
        os.environ['BATTERY_CAPACITY_KWH'] = '50'
        os.environ['BATTERY_MIN_SOC_PCT'] = '10'
        os.environ['BATTERY_MAX_SOC_PCT'] = '90'
        
        plan = planner.plan(
            predictions_kwh=predictions,
            solar_kwh=solar,
            import_prices=import_prices,
            export_prices=export_prices,
            prediction_timestamps=timestamps,
            allow_export=True
        )
        
        assert plan is not None, "Planner returned None"
        assert len(plan) > 0, "Planner returned empty plan"
        assert len(plan) == len(predictions), \
            f"Plan length {len(plan)} != predictions length {len(predictions)}"
        
        # Check first entry structure
        entry = plan[0]
        
        # All required fields should exist
        required_fields = [
            'timestamp', 'battery_action', 'battery_power_kw',
            'soc_kwh', 'soc_pct', 'grid_import_kwh', 'grid_export_kwh'
        ]
        for field in required_fields:
            assert hasattr(entry, field), f"Missing field: {field}"
        
        # Values should be finite
        assert np.isfinite(entry.battery_power_kw), "battery_power_kw is not finite"
        assert np.isfinite(entry.soc_kwh), "soc_kwh is not finite"
        assert np.isfinite(entry.soc_pct), "soc_pct is not finite"
        assert np.isfinite(entry.grid_import_kwh), "grid_import_kwh is not finite"
        assert np.isfinite(entry.grid_export_kwh), "grid_export_kwh is not finite"
        
        # Grid import/export should be non-negative
        assert entry.grid_import_kwh >= 0, "grid_import_kwh is negative"
        assert entry.grid_export_kwh >= 0, "grid_export_kwh is negative"


class TestBatteryReplaySimulatorBasics(unittest.TestCase):
    """Unit tests for the replay simulator itself."""
    
    def test_simulator_initialization(self):
        """Simulator should initialize correctly with fixture data."""
        if not get_fixtures():
            self.skipTest("No fixtures available")
        
        fixture = load_fixture(get_fixtures()[0])
        simulator = BatteryReplaySimulator(fixture)
        
        # Old fixtures may not have measurements; that's OK
        self.assertIsNotNone(simulator.predictions_df, "Predictions should be loaded")
    
    def test_visible_predictions_respects_generated_at(self):
        """Visible predictions should only include generated_at <= planning_time."""
        if not get_fixtures():
            self.skipTest("No fixtures available")
        
        fixture = load_fixture(get_fixtures()[0])
        simulator = BatteryReplaySimulator(fixture)
        
        if simulator.predictions_df is None or simulator.predictions_df.empty:
            self.skipTest("No prediction data in fixture")
        
        # Pick a planning time from the middle
        first_pred_time = simulator.predictions_df.index.get_level_values(0)[0]
        if not isinstance(first_pred_time, pd.Timestamp):
            self.skipTest("Could not extract valid prediction timestamp")
        
        planning_time = pd.Timestamp(first_pred_time) + pd.Timedelta(hours=12)
        
        visible = simulator.get_visible_predictions(planning_time)
        
        if not visible.empty and 'generated_at' in visible.columns:
            # All visible forecasts should have been generated before or at planning_time
            for _, row in visible.iterrows():
                generated_at = row.get('generated_at')
                if generated_at is not None:
                    gen_dt = pd.to_datetime(generated_at, utc=True)
                    assert gen_dt <= planning_time, \
                        f"Generated at {generated_at} is after planning time {planning_time}"


if __name__ == '__main__':
    unittest.main()

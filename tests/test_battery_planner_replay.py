from __future__ import annotations
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

from battery_planners import BatteryPlannerFactory, BatteryPlannerContext
from battery_planners.base import BatteryPlanEntry
from tests.battery_planner_replay import (
    BatteryReplaySimulator,
    execute_plan_entry,
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
        savings = result['savings_pct']
        soc = result['final_soc_pct']
        cost = result['cost_with_battery_eur']
        base = result['cost_no_battery_eur']
        viol = result['soc_violations']
        print(f"  {planner_name:22s} {fixture_name:5s}  savings={savings:6.1f}%  soc={soc:5.1f}%  cost={cost:.3f}  base={base:.3f}  viol={viol}")
    
    @pytest.mark.slow
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
            max_planks=len(simulator.measurements_df) if simulator.measurements_df is not None else 96,
        )
        
        assert result['success'], f"Replay failed: {result.get('error', 'Unknown error')}"
        assert result['soc_violations'] == 0, \
            f"SoC constraint violations: {result['soc_violation_details']}"
        assert result['intervals_run'] > 0, "No intervals were simulated"
        
        # Print planner score
        self._print_planner_score(result, Path(fixture_path).stem, planner_name)
    
    @pytest.mark.slow
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
            max_planks=len(simulator.measurements_df) if simulator.measurements_df is not None else 96,
        )
        
        assert result['success']
        assert np.isfinite(result['cost_with_battery_eur']), \
            f"Cost with battery is not finite: {result['cost_with_battery_eur']}"
        assert np.isfinite(result['cost_no_battery_eur']), \
            f"Baseline cost is not finite: {result['cost_no_battery_eur']}"
        
        # Print planner score
        self._print_planner_score(result, Path(fixture_path).stem, planner_name)
    
    @pytest.mark.slow
    def test_planner_replay_not_worse_than_baseline(self, fixture_path, planner_name):
        """Planner cost should not catastrophically exceed baseline.
        
        Because battery round-trip losses (~10%) require price spreads that may
        not always be present, a planner may show small losses. We allow up to
        100% degradation (cost <= 2x baseline) to filter truly broken planners.
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
            max_planks=len(simulator.measurements_df) if simulator.measurements_df is not None else 96,
        )
        
        assert result['success']
        
        baseline_cost = result['cost_no_battery_eur']
        planner_cost = result['cost_with_battery_eur']
        
        # Allow cost to be up to 2x baseline (100% worse)
        max_acceptable_cost = baseline_cost * 2.0
        
        assert planner_cost <= max_acceptable_cost, \
            f"Planner cost {planner_cost:.2f} EUR exceeds baseline by >100%: {baseline_cost:.2f} EUR"
        
        # Print planner score
        self._print_planner_score(result, Path(fixture_path).stem, planner_name)
    
    @pytest.mark.slow
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
        
        predictions, solar, import_prices, export_prices, timestamps, _solar_p10 = \
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
    
    def test_planner_replay_quick(self, fixture_path, planner_name):
        """Quick 24h (96 interval) sanity check for rapid iteration."""
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

        assert result['success'], f"Quick replay failed: {result.get('error', 'Unknown error')}"
        assert result['soc_violations'] == 0, \
            f"SoC violations in quick replay: {result.get('soc_violation_details', [])}"
        # Allow large negative savings — some planners may show losses in adverse conditions.
        # This test is just a sanity check that the planner runs without crashes.
        assert np.isfinite(result['savings_pct']), "Savings is not finite"
        assert np.isfinite(result['cost_with_battery_eur']), "Cost is not finite"

        self._print_planner_score(result, Path(fixture_path).stem, planner_name)

    @pytest.mark.slow
    def test_planner_replay_with_context(self, fixture_path, planner_name):
        """Planner should accept a context dict during replay and still pass constraints."""
        fixture = load_fixture(fixture_path)
        simulator = BatteryReplaySimulator(fixture)
        
        if simulator.measurements_df is None or simulator.measurements_df.empty:
            pytest.skip(f"Fixture {Path(fixture_path).stem} has no measurement data")
        
        planner = BatteryPlannerFactory.create(planner_name)
        
        horizon = len(simulator.measurements_df) if simulator.measurements_df is not None else 96
        context: BatteryPlannerContext = {
            'outside_temps': np.zeros(horizon),
            'is_sauna_active': np.zeros(horizon, dtype=int),
            'tomorrow_valid': False,
        }
        
        result = simulator.simulate_battery_control(
            planner=planner,
            planner_type=planner_name,
            battery_capacity_kwh=50.0,
            battery_min_soc_pct=10.0,
            battery_max_soc_pct=90.0,
            battery_initial_soc_pct=10.0,
            max_planks=horizon,
            context=context,
        )
        
        assert result['success'], f"Replay with context failed: {result.get('error', 'Unknown error')}"
        assert result['soc_violations'] == 0, \
            f"SoC constraint violations with context: {result.get('soc_violation_details', [])}"


class TestSolarP10Passthrough(unittest.TestCase):
    """The replay harness must surface archived solar_forecast_p10_kw to planners.

    Production feeds p10 via BatteryPlannerContext['solar_p10_kwh'] (kWh per
    interval); the replay harness must do the same when the fixture's
    predictions archive carries the column, and omit it otherwise.
    """

    def _make_simulator(self, with_p10: bool) -> BatteryReplaySimulator:
        from utils.battery_test_data import BatteryTestData

        t0 = pd.Timestamp('2026-07-15 00:00', tz='UTC')
        n = 4
        measurements = [
            {
                'timestamp': (t0 + pd.Timedelta(minutes=15 * i)).isoformat(),
                'total_power_kw': 0.5,
                'solar_actual_kw': 1.0,
            }
            for i in range(n)
        ]
        archive = []
        for i in range(n):
            row = {
                'target_timestamp': (t0 + pd.Timedelta(minutes=15 * i)).isoformat(),
                'generated_at': (t0 - pd.Timedelta(hours=1)).isoformat(),
                'predicted_usage_kw': 0.5,
                'solar_forecast_kw': 2.0,
                'import_price': 0.10,
                'export_price': 0.02,
            }
            if with_p10:
                row['solar_forecast_p10_kw'] = 0.8
            archive.append(row)
        raw = {'history': {'measurements': measurements, 'predictions_archive': archive}}
        return BatteryReplaySimulator(BatteryTestData(raw))

    class _SpyPlanner:
        def __init__(self):
            self.seen_context = None
            self.seen_solar = None

        def plan(self, predictions_kwh, solar_kwh, import_prices, export_prices,
                 prediction_timestamps, committed_load_kwh=None, allow_export=True,
                 initial_soc_pct=None, context=None):
            self.seen_context = dict(context) if context else {}
            self.seen_solar = np.asarray(solar_kwh, dtype=float).copy()
            return [
                BatteryPlanEntry(
                    timestamp=str(ts), battery_action='idle', battery_power_kw=0.0,
                    charge_from_solar_kwh=0.0, charge_from_grid_kwh=0.0,
                    discharge_to_load_kwh=0.0, discharge_to_export_kwh=0.0,
                    soc_kwh=5.0, soc_pct=50.0, grid_import_kwh=0.0,
                    grid_export_kwh=0.0, estimated_hour_cost=0.0,
                    estimated_hour_savings=0.0, net_load_without_battery_kwh=0.0,
                )
                for ts in prediction_timestamps
            ]

    def _run(self, simulator, context=None):
        spy = self._SpyPlanner()
        result = simulator.simulate_battery_control(
            planner=spy, planner_type='spy',
            battery_capacity_kwh=10.0, battery_min_soc_pct=10.0,
            battery_max_soc_pct=90.0, battery_initial_soc_pct=50.0,
            max_planks=4, context=context,
        )
        assert result['success'], f"replay failed: {result.get('error')}"
        return spy

    def test_p10_reaches_planner_context_in_kwh(self):
        spy = self._run(self._make_simulator(with_p10=True))
        assert 'solar_p10_kwh' in spy.seen_context, \
            f"context keys: {sorted(spy.seen_context)}"
        # 0.8 kW * 0.25 h = 0.2 kWh per interval
        np.testing.assert_allclose(
            spy.seen_context['solar_p10_kwh'], [0.2, 0.2, 0.2, 0.2], atol=1e-9)
        # p50 series unchanged: 2.0 kW * 0.25 h = 0.5 kWh
        np.testing.assert_allclose(spy.seen_solar, [0.5, 0.5, 0.5, 0.5], atol=1e-9)

    def test_p10_omitted_when_archive_lacks_column(self):
        spy = self._run(self._make_simulator(with_p10=False))
        assert 'solar_p10_kwh' not in spy.seen_context

    def test_caller_context_p10_wins_over_fixture(self):
        override = np.full(4, 9.9)
        spy = self._run(self._make_simulator(with_p10=True),
                        context={'solar_p10_kwh': override})
        np.testing.assert_allclose(spy.seen_context['solar_p10_kwh'], override)


class TestPriceWindowRobustness(unittest.TestCase):
    """get_planner_horizon must return equal-length arrays regardless of how
    much price data the fixture carries (the aug fixture was the first with a
    real market_prices table — wider than the prediction window and with
    duplicated timestamps from the dump's per-generation price join)."""

    def _make_simulator(self, n_pred_intervals: int, price_rows: list[dict]):
        from utils.battery_test_data import BatteryTestData

        t0 = pd.Timestamp('2026-07-15 10:00', tz='UTC')
        measurements = [
            {
                'timestamp': (t0 + pd.Timedelta(minutes=15 * i)).isoformat(),
                'total_power_kw': 0.5,
                'solar_actual_kw': 1.0,
            }
            for i in range(n_pred_intervals)
        ]
        archive = [
            {
                'target_timestamp': (t0 + pd.Timedelta(minutes=15 * i)).isoformat(),
                'generated_at': (t0 - pd.Timedelta(hours=1)).isoformat(),
                'predicted_usage_kw': 0.5,
                'solar_forecast_kw': 2.0,
                'import_price': 0.10,
                'export_price': 0.02,
            }
            for i in range(n_pred_intervals)
        ]
        raw = {
            'history': {'measurements': measurements, 'predictions_archive': archive},
            'market_prices': price_rows,
        }
        return BatteryReplaySimulator(BatteryTestData(raw)), t0

    def _price_rows(self, day: str, n: int, duplicate: int = 1) -> list[dict]:
        base = pd.Timestamp(f'2026-07-{day} 00:00', tz='UTC')
        rows = []
        for i in range(n):
            row = {
                'timestamp': (base + pd.Timedelta(minutes=15 * i)).isoformat(),
                'import_price': 0.10,
                'export_price': 0.02,
            }
            rows.extend([dict(row) for _ in range(duplicate)])
        return rows

    def test_arrays_equal_length_when_price_window_wider_than_predictions(self):
        # 4 prediction intervals at 10:00, but prices through end of day (56).
        sim, t0 = self._make_simulator(4, self._price_rows('15', 96))
        horizon = sim.get_planner_horizon(t0, 96)
        lengths = {len(a) for a in horizon[:5]}
        assert lengths == {96}, f"horizon array lengths differ: {lengths}"

    def test_duplicate_price_timestamps_are_deduplicated(self):
        sim, t0 = self._make_simulator(4, self._price_rows('15', 96, duplicate=3))
        assert sim.prices_df.index.is_unique
        horizon = sim.get_planner_horizon(t0, 96)
        import_prices = horizon[2]
        assert len(import_prices) == 96
        np.testing.assert_allclose(import_prices[:4], 0.10)


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


class TestReplayIntervalAccounting(unittest.TestCase):
    def _entry(self, charge=0.0, discharge=0.0):
        return BatteryPlanEntry(
            timestamp='t0', battery_action='idle', battery_power_kw=0.0,
            charge_from_solar_kwh=charge, charge_from_grid_kwh=0.0,
            discharge_to_load_kwh=discharge, discharge_to_export_kwh=0.0,
            soc_kwh=0.0, soc_pct=0.0, grid_import_kwh=0.0, grid_export_kwh=0.0,
            estimated_hour_cost=0.0, estimated_hour_savings=0.0,
            net_load_without_battery_kwh=0.0,
        )

    def test_soc_uses_battery_efficiencies(self):
        result = execute_plan_entry(
            soc_kwh=10.0, entry=self._entry(charge=2.0),
            min_soc_kwh=5.0, max_soc_kwh=20.0,
            charge_efficiency=0.9, discharge_efficiency=0.8,
            actual_load_kwh=0.0, actual_solar_kwh=2.0,
        )

        self.assertAlmostEqual(result['soc_kwh'], 11.8)

    def test_violation_is_recorded_before_soc_is_clamped(self):
        result = execute_plan_entry(
            soc_kwh=10.0, entry=self._entry(discharge=2.0),
            min_soc_kwh=9.0, max_soc_kwh=20.0,
            charge_efficiency=0.9, discharge_efficiency=0.8,
            actual_load_kwh=2.0, actual_solar_kwh=0.0,
        )

        self.assertTrue(result['soc_violation'])
        self.assertAlmostEqual(result['raw_soc_kwh'], 7.5)
        self.assertAlmostEqual(result['soc_kwh'], 9.0)


if __name__ == '__main__':
    unittest.main()

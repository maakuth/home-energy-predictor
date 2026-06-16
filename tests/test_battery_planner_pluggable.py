"""Tests for pluggable battery planner architecture."""

import os
import unittest
import numpy as np
from battery_planners import (
    BatteryPlanner,
    BatteryPlanEntry,
    BatteryPlannerContext,
    BatteryPlannerFactory,
    HeuristicBatteryPlanner,
)


class TestBatteryPlannerInterface(unittest.TestCase):
    """Test that BatteryPlanner is properly abstracted."""
    
    def test_battery_planner_is_abstract(self):
        """BatteryPlanner should not be instantiable."""
        with self.assertRaises(TypeError):
            BatteryPlanner()
    
    def test_heuristic_planner_is_concrete(self):
        """HeuristicBatteryPlanner should be instantiable."""
        planner = HeuristicBatteryPlanner()
        self.assertIsInstance(planner, BatteryPlanner)
    
    def test_battery_plan_entry_conversion_to_dict(self):
        """BatteryPlanEntry should convert to dict for compatibility."""
        entry = BatteryPlanEntry(
            timestamp="2024-01-01T00:00:00",
            battery_action="charge_solar",
            battery_power_kw=5.0,
            charge_from_solar_kwh=1.25,
            charge_from_grid_kwh=0.0,
            discharge_to_load_kwh=0.0,
            discharge_to_export_kwh=0.0,
            soc_kwh=45.0,
            soc_pct=112.5,
            grid_import_kwh=0.0,
            grid_export_kwh=0.0,
            estimated_hour_cost=0.5,
            estimated_hour_savings=2.0,
            net_load_without_battery_kwh=2.0,
        )
        
        d = entry.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d['battery_action'], 'charge_solar')
        self.assertEqual(d['battery_power_kw'], 5.0)
        self.assertEqual(d['soc_pct'], 112.5)


class TestBatteryPlannerFactory(unittest.TestCase):
    """Test planner factory registration and creation."""
    
    def test_factory_creates_heuristic_by_name(self):
        """Factory should create heuristic planner when explicitly requested."""
        planner = BatteryPlannerFactory.create('heuristic')
        self.assertIsInstance(planner, HeuristicBatteryPlanner)
    
    def test_factory_is_case_insensitive(self):
        """Factory should handle case-insensitive planner names."""
        planner1 = BatteryPlannerFactory.create('HEURISTIC')
        planner2 = BatteryPlannerFactory.create('Heuristic')
        planner3 = BatteryPlannerFactory.create('heuristic')
        
        self.assertIsInstance(planner1, HeuristicBatteryPlanner)
        self.assertIsInstance(planner2, HeuristicBatteryPlanner)
        self.assertIsInstance(planner3, HeuristicBatteryPlanner)
    
    def test_factory_raises_on_unknown_planner(self):
        """Factory should raise ValueError for unknown planner type."""
        with self.assertRaises(ValueError) as cm:
            BatteryPlannerFactory.create('nonexistent_planner')
        self.assertIn('Unknown battery planner type', str(cm.exception))
        self.assertIn('heuristic', str(cm.exception))


class TestHeuristicPlannerBasic(unittest.TestCase):
    """Test basic functionality of HeuristicBatteryPlanner."""
    
    def test_planner_generates_plan_with_correct_length(self):
        """Planner should generate a plan with one entry per interval."""
        planner = HeuristicBatteryPlanner()
        
        horizon = 10
        predictions = np.ones(horizon) * 2.0  # 2 kW baseload
        solar = np.zeros(horizon)
        import_prices = np.ones(horizon) * 0.15
        export_prices = np.ones(horizon) * 0.05
        
        plan = planner.plan(
            predictions, solar, import_prices, export_prices,
            prediction_timestamps=[f"interval_{i}" for i in range(horizon)]
        )
        
        self.assertEqual(len(plan), horizon)
        self.assertTrue(all(isinstance(entry, BatteryPlanEntry) for entry in plan))
    
    def test_planner_produces_dict_convertible_output(self):
        """Planner output should be convertible to dicts."""
        planner = HeuristicBatteryPlanner()
        
        predictions = np.array([2.0, 2.0, 2.0])
        solar = np.array([0.0, 0.0, 0.0])
        import_prices = np.array([0.15, 0.15, 0.15])
        export_prices = np.array([0.05, 0.05, 0.05])
        
        plan = planner.plan(
            predictions, solar, import_prices, export_prices,
            prediction_timestamps=["t0", "t1", "t2"]
        )
        
        # Convert to dicts (as done in optimize_plan.py)
        plan_dicts = [entry.to_dict() for entry in plan]
        
        self.assertEqual(len(plan_dicts), 3)
        self.assertIn('battery_action', plan_dicts[0])
        self.assertIn('soc_pct', plan_dicts[0])
        self.assertIn('grid_import_kwh', plan_dicts[0])


class TestPlannerSwappability(unittest.TestCase):
    """Test that planners can be easily swapped."""
    
    def test_multiple_planner_instances_are_independent(self):
        """Creating multiple planner instances should not affect each other."""
        planner1 = BatteryPlannerFactory.create('heuristic')
        planner2 = BatteryPlannerFactory.create('heuristic')
        
        # They should be different instances
        self.assertIsNot(planner1, planner2)
        
        # But both should work
        self.assertIsInstance(planner1, BatteryPlanner)
        self.assertIsInstance(planner2, BatteryPlanner)


class TestBatteryPlannerContext(unittest.TestCase):
    """Test that the extensible context dict is accepted and ignored gracefully."""
    
    def test_planner_accepts_context_dict(self):
        """Planner should accept a BatteryPlannerContext and still produce valid output."""
        planner = HeuristicBatteryPlanner()
        
        predictions = np.array([2.0, 2.0, 2.0])
        solar = np.array([0.0, 0.0, 0.0])
        import_prices = np.array([0.15, 0.15, 0.15])
        export_prices = np.array([0.05, 0.05, 0.05])
        
        context: BatteryPlannerContext = {
            'outside_temps': np.array([-5.0, -3.0, 0.0]),
            'is_sauna_active': np.array([0, 1, 0]),
            'tomorrow_valid': True,
        }
        
        plan = planner.plan(
            predictions, solar, import_prices, export_prices,
            prediction_timestamps=["t0", "t1", "t2"],
            context=context,
        )
        
        self.assertEqual(len(plan), 3)
        self.assertTrue(all(isinstance(entry, BatteryPlanEntry) for entry in plan))
    
    def test_planner_ignores_unknown_context_keys(self):
        """Planner should ignore keys in context that it does not recognise."""
        planner = HeuristicBatteryPlanner()
        
        predictions = np.array([2.0, 2.0])
        solar = np.array([0.0, 0.0])
        import_prices = np.array([0.15, 0.15])
        export_prices = np.array([0.05, 0.05])
        
        # Pass a context with a completely made-up key
        context: BatteryPlannerContext = {
            'future_alien_invasion': np.array([0, 0]),  # type: ignore[typeddict-unknown-key]
        }
        
        plan = planner.plan(
            predictions, solar, import_prices, export_prices,
            prediction_timestamps=["t0", "t1"],
            context=context,
        )
        
        self.assertEqual(len(plan), 2)


class TestNemotronLinprogAdaptiveHorizon(unittest.TestCase):
    """Test adaptive horizon behavior of NemotronLinprogPlanner."""

    def setUp(self):
        self._env_backup = {}
        for key in ['BATTERY_LP_HORIZON', 'BATTERY_LP_HORIZON_FALLBACK',
                     'BATTERY_LP_DISCOUNT', 'BATTERY_INITIAL_SOC_PCT',
                     'BATTERY_CAPACITY_KWH', 'BATTERY_MIN_SOC_PCT',
                     'BATTERY_MAX_SOC_PCT', 'PLAN_INTERVAL_MINUTES']:
            self._env_backup[key] = os.environ.pop(key, None)

        os.environ['BATTERY_LP_HORIZON'] = '40'
        os.environ['BATTERY_LP_HORIZON_FALLBACK'] = '20'
        os.environ['BATTERY_LP_DISCOUNT'] = '1.0'
        os.environ['BATTERY_INITIAL_SOC_PCT'] = '10'
        os.environ['BATTERY_CAPACITY_KWH'] = '50'
        os.environ['BATTERY_MIN_SOC_PCT'] = '10'
        os.environ['BATTERY_MAX_SOC_PCT'] = '90'
        os.environ['PLAN_INTERVAL_MINUTES'] = '15'

    def tearDown(self):
        for key, val in self._env_backup.items():
            if val is not None:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)

    def _make_price_scenario(self, n=50):
        """Prices: cheap 0-15, moderate 15-25, expensive 25-35, moderate 35+."""
        predictions = np.ones(n) * 2.0  # 2 kW baseload
        solar = np.zeros(n)
        import_prices = np.concatenate([
            np.ones(15) * 0.05,
            np.ones(11) * 0.12,
            np.ones(11) * 0.30,
            np.ones(n - 37) * 0.12,
        ])
        export_prices = import_prices * 0.5
        timestamps = [f"interval_{i}" for i in range(n)]
        return predictions, solar, import_prices, export_prices, timestamps

    def test_tomorrow_valid_charges_for_far_future_peak(self):
        """With tomorrow_valid=True, LP should charge for peaks beyond fallback horizon."""
        data = self._make_price_scenario(50)
        planner_true = BatteryPlannerFactory.create('nemotron-linprog')
        context = {
            'tomorrow_valid': True,
            'outside_temps': np.zeros(50),
            'is_sauna_active': np.zeros(50, dtype=int),
        }
        plan_true = planner_true.plan(*data, context=context)

        planner_false = BatteryPlannerFactory.create('nemotron-linprog')
        context['tomorrow_valid'] = False
        plan_false = planner_false.plan(*data, context=context)

        # At index 19 (end of fallback horizon), long-horizon plan should
        # have charged for the far-future peak, while fallback only sees
        # moderate prices and stays at min SoC
        self.assertGreater(
            plan_true[19].soc_pct,
            plan_false[19].soc_pct + 1.0,
            "tomorrow_valid=True should charge for peaks beyond fallback horizon"
        )

    def test_no_context_equals_tomorrow_false(self):
        """Without context, LP should use fallback horizon (safe default)."""
        data = self._make_price_scenario(50)
        planner = BatteryPlannerFactory.create('nemotron-linprog')

        plan_no_ctx = planner.plan(*data)

        planner2 = BatteryPlannerFactory.create('nemotron-linprog')
        context = {
            'tomorrow_valid': False,
            'outside_temps': np.zeros(50),
            'is_sauna_active': np.zeros(50, dtype=int),
        }
        plan_with_ctx = planner2.plan(*data, context=context)

        # Both should produce identical behavior (same SoC at index 19)
        self.assertAlmostEqual(
            plan_no_ctx[19].soc_pct,
            plan_with_ctx[19].soc_pct,
            places=4,
            msg="No context should behave like tomorrow_valid=False"
        )


if __name__ == '__main__':
    unittest.main()

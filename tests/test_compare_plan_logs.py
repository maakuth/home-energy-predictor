from __future__ import annotations
import unittest
from pathlib import Path
from utils.compare_plan_logs import (
    OftenLogParser,
    FrequentLogParser,
    build_comparison,
    compute_attribution,
    print_attribution,
    print_trending,
)


class TestAttribution(unittest.TestCase):
    """Tests for the error attribution and trending modes."""

    maxDiff = None

    def test_compute_attribution_all_over_predicted(self):
        """All three components over-predict → positive energy error."""
        comparison = [
            {
                'time': '06-15 00:00',
                'plan_soc': 50.0, 'actual_soc': 60.0, 'soc_diff': 10.0,
                'plan_gshp': 4.0, 'actual_gshp': 2.0, 'gshp_diff': -2.0,
                'plan_solar': 0.0, 'actual_solar': 0.0, 'solar_diff': 0.0,
                'plan_grid': 5.0, 'plan_baseload': 3.0,
                'actual_baseload': 2.0,
                'actual_grid_kw': 4.0, 'actual_batt_kw': 0.0,
            },
            {
                'time': '06-15 00:15',
                'plan_soc': 50.0, 'actual_soc': 60.0, 'soc_diff': 10.0,
                'plan_gshp': 4.0, 'actual_gshp': 2.0, 'gshp_diff': -2.0,
                'plan_solar': 0.0, 'actual_solar': 0.0, 'solar_diff': 0.0,
                'plan_grid': 5.0, 'plan_baseload': 3.0,
                'actual_baseload': 2.0,
                'actual_grid_kw': 4.0, 'actual_batt_kw': 0.0,
            },
        ]
        attr = compute_attribution(comparison, battery_capacity_kwh=50.0, interval_hours=0.25)

        self.assertEqual(attr['periods'], 2)

        # GSHP over-prediction: (4-2)*0.25 = 0.5 kWh per interval, 2 intervals = 1.0 kWh
        self.assertAlmostEqual(attr['gshp_kwh'], 1.0, places=4)
        # Baseload over-prediction: (3-2)*0.25 = 0.25 kWh per interval, 2 intervals = 0.5 kWh
        self.assertAlmostEqual(attr['baseload_kwh'], 0.5, places=4)
        # Solar: 0
        self.assertAlmostEqual(attr['solar_kwh'], 0.0, places=4)

        # Total: 1.5 kWh
        self.assertAlmostEqual(attr['total_err_kwh'], 1.5, places=4)

        # SoC impact: 1.5 kWh / 50 kWh * 100 = 3.0%
        self.assertAlmostEqual(attr['soc_impact_pct'], 3.0, places=4)

        # Percentage: GSHP 66.7%, Baseload 33.3%, Solar 0%
        self.assertAlmostEqual(attr['gshp_pct'], 66.6667, places=1)
        self.assertAlmostEqual(attr['baseload_pct'], 33.3333, places=1)
        self.assertAlmostEqual(attr['solar_pct'], 0.0, places=1)

    def test_compute_attribution_partial_data(self):
        """Some entries missing actual_baseload → skip them."""
        comparison = [
            {
                'time': '06-15 00:00',
                'plan_soc': 50.0, 'actual_soc': 60.0, 'soc_diff': 10.0,
                'plan_gshp': 4.0, 'actual_gshp': 2.0, 'gshp_diff': -2.0,
                'plan_solar': 0.0, 'actual_solar': 0.0, 'solar_diff': 0.0,
                'plan_grid': 5.0, 'plan_baseload': 3.0,
                'actual_baseload': None,
                'actual_grid_kw': None, 'actual_batt_kw': None,
            },
            {
                'time': '06-15 00:15',
                'plan_soc': 50.0, 'actual_soc': 60.0, 'soc_diff': 10.0,
                'plan_gshp': 4.0, 'actual_gshp': 2.0, 'gshp_diff': -2.0,
                'plan_solar': 0.0, 'actual_solar': 0.0, 'solar_diff': 0.0,
                'plan_grid': 5.0, 'plan_baseload': 3.0,
                'actual_baseload': 2.0,
                'actual_grid_kw': 4.0, 'actual_batt_kw': 0.0,
            },
        ]
        attr = compute_attribution(comparison, battery_capacity_kwh=50.0, interval_hours=0.25)

        # Only 1 valid period (the second one)
        self.assertEqual(attr['periods'], 1)
        self.assertAlmostEqual(attr['gshp_kwh'], 0.5, places=4)

    def test_compute_attribution_zero_capacity(self):
        """Zero battery capacity → soc_impact is 0."""
        comparison = [
            {
                'time': '06-15 00:00',
                'plan_soc': 50.0, 'actual_soc': 60.0, 'soc_diff': 10.0,
                'plan_gshp': 4.0, 'actual_gshp': 2.0, 'gshp_diff': -2.0,
                'plan_solar': 0.0, 'actual_solar': 0.0, 'solar_diff': 0.0,
                'plan_grid': 5.0, 'plan_baseload': 3.0,
                'actual_baseload': 2.0,
                'actual_grid_kw': 4.0, 'actual_batt_kw': 0.0,
            },
        ]
        attr = compute_attribution(comparison, battery_capacity_kwh=0.0, interval_hours=0.25)
        self.assertEqual(attr['soc_impact_pct'], 0.0)

    def test_compute_attribution_solar_error(self):
        """Solar under-forecast → negative solar error (more solar = less grid)."""
        comparison = [
            {
                'time': '06-15 12:00',
                'plan_soc': 50.0, 'actual_soc': 55.0, 'soc_diff': 5.0,
                'plan_gshp': 0.0, 'actual_gshp': 0.0, 'gshp_diff': 0.0,
                'plan_solar': 1.0, 'actual_solar': 3.0, 'solar_diff': 2.0,
                'plan_grid': 0.0, 'plan_baseload': 2.0,
                'actual_baseload': 1.5,
                'actual_grid_kw': -1.0, 'actual_batt_kw': 0.0,
            },
        ]
        attr = compute_attribution(comparison, battery_capacity_kwh=50.0, interval_hours=0.25)

        # Solar: (3.0 - 1.0) * 0.25 = 0.5 kWh (inverted: more actual solar)
        self.assertAlmostEqual(attr['solar_kwh'], 0.5, places=4)

    def test_print_attribution_no_crash(self):
        """print_attribution should not crash with varied input."""
        comparison = [
            {
                'time': '06-15 00:00',
                'plan_soc': 50.0, 'actual_soc': 60.0, 'soc_diff': 10.0,
                'plan_gshp': 4.0, 'actual_gshp': 2.0, 'gshp_diff': -2.0,
                'plan_solar': 0.0, 'actual_solar': 0.0, 'solar_diff': 0.0,
                'plan_grid': 5.0, 'plan_baseload': 3.0,
                'actual_baseload': 2.0,
                'actual_grid_kw': 4.0, 'actual_batt_kw': 0.0,
            },
        ]
        # Should not raise
        print_attribution(comparison, battery_capacity_kwh=50.0)

    def test_compute_attribution_empty(self):
        """Empty list → zero results."""
        attr = compute_attribution([])
        self.assertEqual(attr['periods'], 0)
        self.assertEqual(attr['total_err_kwh'], 0.0)


class TestTrending(unittest.TestCase):
    """Tests for the trending mode (uses hermetic fixture logs)."""

    @staticmethod
    def _fixture(name: str) -> str:
        return str(Path(__file__).parent.parent / 'tests' / 'fixtures' / name)

    def test_trending_first_plan(self):
        """Trending should process plan 0 from the fixture logs without error."""
        import io
        from contextlib import redirect_stdout

        often = OftenLogParser(self._fixture('often_log_fixture.txt'))
        freq = FrequentLogParser(self._fixture('frequent_log_fixture.txt'))
        self.assertTrue(freq.plans, "fixture should contain at least one plan")

        buf = io.StringIO()
        with redirect_stdout(buf):
            # Should not raise
            print_trending(often, freq, every_n=1, battery_capacity_kwh=50.0)
        out = buf.getvalue()
        self.assertIn("PLAN TRENDING", out)

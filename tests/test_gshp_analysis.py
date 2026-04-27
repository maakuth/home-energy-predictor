import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from analyze_performance import summarize_gshp_performance

class TestGSHPAnalysis(unittest.TestCase):
    def test_summarize_gshp_performance_solar(self):
        # Setup data: 1 hour, 4 intervals of 15 min
        ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(4)]
        df = pd.DataFrame({
            'gshp_actual_kw': [4.0, 4.0, 0.0, 0.0],
            'solar_actual':   [10.0, 10.0, 10.0, 10.0],
            'actual_usage':   [5.0, 5.0, 1.0, 1.0], # total = baseload + gshp. so baseload = 1.0
            'import_price':   [0.1, 0.1, 0.1, 0.1]
        }, index=ts)
        
        # Baseload = total - gshp = 5 - 4 = 1.0
        # Solar for GSHP = solar - baseload = 10 - 1 = 9.0
        # gshp_from_solar = min(4.0, 9.0) = 4.0
        # All GSHP should be from solar.
        
        res = summarize_gshp_performance(df)
        self.assertEqual(res['gshp_solar_pct'], 100.0)
        self.assertEqual(res['gshp_avg_price'], 0.0)
        self.assertEqual(res['gshp_total_kwh'], 2.0) # (4kW * 0.25h) * 2 = 2kWh

    def test_summarize_gshp_performance_grid(self):
        # Setup data: 1 hour, no solar
        ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(4)]
        df = pd.DataFrame({
            'gshp_actual_kw': [4.0, 4.0, 4.0, 4.0],
            'solar_actual':   [0.0, 0.0, 0.0, 0.0],
            'actual_usage':   [5.0, 5.0, 5.0, 5.0],
            'import_price':   [0.1, 0.2, 0.3, 0.4]
        }, index=ts)
        
        # gshp_solar_pct should be 0.0
        # gshp_avg_price should be avg of 0.1, 0.2, 0.3, 0.4 = 0.25
        
        res = summarize_gshp_performance(df)
        self.assertEqual(res['gshp_solar_pct'], 0.0)
        self.assertAlmostEqual(res['gshp_avg_price'], 0.25)
        self.assertEqual(res['gshp_total_kwh'], 4.0)

    def test_summarize_gshp_performance_partial_solar(self):
        # Setup data: 1 interval
        ts = [datetime(2024, 1, 1, 12, 0) + timedelta(minutes=15*i) for i in range(2)]
        df = pd.DataFrame({
            'gshp_actual_kw': [4.0, 4.0],
            'solar_actual':   [2.0, 2.0],
            'actual_usage':   [5.0, 5.0], # baseload = 1.0
            'import_price':   [0.1, 0.1]
        }, index=ts)
        
        # baseload = 1.0
        # solar_for_gshp = 2.0 - 1.0 = 1.0
        # gshp_from_solar = min(4.0, 1.0) = 1.0
        # gshp_solar_pct = 1.0 / 4.0 = 25.0%
        
        res = summarize_gshp_performance(df)
        self.assertEqual(res['gshp_solar_pct'], 25.0)

if __name__ == '__main__':
    unittest.main()

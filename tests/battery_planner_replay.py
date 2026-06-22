from __future__ import annotations
"""
Battery planner replay harness for realistic testing.

This module provides utilities to replay fixture data through a battery planner,
simulating the passage of time and respecting the constraints that planners
should only see forecasts and prices that would have been available at each
planning time.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle

from utils.battery_test_data import BatteryTestData
from battery_planners import BatteryPlannerFactory, BatteryPlanner


class BatteryReplaySimulator:
    """Simulates battery planner performance over historical data.
    
    Enforces time-aware visibility:
    - Forecasts only visible if generated_at <= planning_time
    - Spot prices only visible through end of current day until 15:00 local time,
      then through end of next day
    - No future measurement leakage
    """
    
    def __init__(self, fixture_data: BatteryTestData):
        """Initialize simulator with fixture data.
        
        Args:
            fixture_data: BatteryTestData instance from loaded fixture
        """
        self.data = fixture_data
        self.measurements_df = None
        self.predictions_df = None
        self.prices_df = None
        
        self._load_and_normalize_data()
    
    def _load_and_normalize_data(self):
        """Load and normalize fixture data into DataFrames."""
        # Load measurements (actual HA sensor readings)
        measurements = self.data._data.get('history', {}).get('measurements', [])
        if measurements:
            self.measurements_df = pd.DataFrame(measurements)
            if 'timestamp' in self.measurements_df.columns:
                self.measurements_df['timestamp'] = pd.to_datetime(
                    self.measurements_df['timestamp'], utc=True
                )
                self.measurements_df = self.measurements_df.set_index('timestamp').sort_index()
        
        # Load prediction archive with generated_at preserved
        predictions_archive = self.data._data.get('history', {}).get('predictions_archive', [])
        if predictions_archive:
            self.predictions_df = pd.DataFrame(predictions_archive)
            
            # Parse timestamps
            for col in ['target_timestamp', 'generated_at']:
                if col in self.predictions_df.columns:
                    self.predictions_df[col] = pd.to_datetime(
                        self.predictions_df[col], utc=True
                    )
            
            # Create multi-index on target_timestamp and generated_at
            if 'target_timestamp' in self.predictions_df.columns and \
               'generated_at' in self.predictions_df.columns:
                self.predictions_df = self.predictions_df.set_index(
                    ['target_timestamp', 'generated_at']
                ).sort_index()
        
        # Load market prices
        market_prices = self.data.market_prices_list()
        if market_prices:
            self.prices_df = pd.DataFrame(market_prices)
            if 'timestamp' in self.prices_df.columns:
                self.prices_df['timestamp'] = pd.to_datetime(
                    self.prices_df['timestamp'], utc=True
                )
                self.prices_df = self.prices_df.set_index('timestamp').sort_index()
    
    def get_visible_predictions(self, planning_time) -> pd.DataFrame:
        """Get predictions visible at planning_time.
        
        Returns only predictions where generated_at <= planning_time AND target_timestamp >= planning_time.
        For each target_timestamp, returns only the latest forecast generated
        before or at planning_time.
        """
        if self.predictions_df is None or self.predictions_df.empty:
            return pd.DataFrame()
        
        # Filter to predictions generated before or at planning_time AND target is in the future
        visible = self.predictions_df[
            (self.predictions_df.index.get_level_values('generated_at') <= planning_time) &
            (self.predictions_df.index.get_level_values('target_timestamp') >= planning_time)
        ]
        
        if visible.empty:
            return pd.DataFrame()
        
        # For each target_timestamp, keep only the latest generated_at
        visible = visible.reset_index()
        latest_per_target = visible.loc[
            visible.groupby('target_timestamp')['generated_at'].idxmax()
        ]
        
        return latest_per_target.set_index('target_timestamp').sort_index()
    
    def get_visible_prices(self, planning_time) -> np.ndarray:
        """Get market prices visible at planning_time.
        
        Before 15:00 local time: prices available through end of current day.
        At or after 15:00 local time: prices available through end of next day.
        
        Falls back to prices from visible predictions if market prices unavailable.
        """
        prices = None
        
        # First try market prices table
        if self.prices_df is not None and not self.prices_df.empty:
            local_hour = planning_time.hour
            
            if local_hour < 15:
                # Prices available through end of current day
                cutoff = planning_time.replace(hour=23, minute=59, second=59)
            else:
                # Prices available through end of next day
                cutoff = (planning_time + timedelta(days=1)).replace(
                    hour=23, minute=59, second=59
                )
            
            visible_prices = self.prices_df[
                (self.prices_df.index >= planning_time) &
                (self.prices_df.index <= cutoff)
            ]
            
            if not visible_prices.empty and 'import_price' in visible_prices.columns:
                try:
                    prices = np.asarray(visible_prices['import_price'])
                except:
                    prices = None
        
        # Fall back to import_price from predictions if available
        if prices is None or len(prices) == 0:
            visible_preds = self.get_visible_predictions(planning_time)
            if not visible_preds.empty and 'import_price' in visible_preds.columns:
                try:
                    prices = np.asarray(visible_preds['import_price'])
                except:
                    prices = None
        
        return prices if prices is not None else np.array([])
    
    def get_planner_horizon(
        self, planning_time, planner_output_length: int
    ):
        """Build planner input horizon (predictions and prices visible at planning_time).
        
        Returns:
            (predictions_kwh, solar_forecast_kwh, import_prices, export_prices, timestamps)
        """
        visible_preds = self.get_visible_predictions(planning_time)
        
        if visible_preds.empty:
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        # Convert kW → kWh (APIs expect kWh per interval)
        interval_hours = 0.25
        pred_series = visible_preds.get('predicted_usage_kw', pd.Series())
        predictions_kwh = np.asarray(pred_series, dtype=float)[:planner_output_length] * interval_hours
        
        solar_series = visible_preds.get('solar_forecast_kw', pd.Series())
        solar_kwh = np.asarray(solar_series, dtype=float)[:planner_output_length] * interval_hours
        
        # Get prices from visible window
        visible_prices = self.get_visible_prices(planning_time)
        import_prices = visible_prices[:planner_output_length] if len(visible_prices) > 0 else np.array([])
        
        export_series = visible_preds.get('export_price', pd.Series())
        export_prices = np.asarray(export_series)[:planner_output_length]
        
        # Build timestamp array
        timestamps = visible_preds.index.tolist()[:planner_output_length]
        
        # Pad if necessary (planner expects full length)
        if len(predictions_kwh) < planner_output_length:
            pad_len = planner_output_length - len(predictions_kwh)
            predictions_kwh = np.pad(predictions_kwh, (0, pad_len), mode='edge')
            solar_kwh = np.pad(solar_kwh, (0, pad_len), mode='edge')
            import_prices = np.pad(import_prices, (0, pad_len), mode='edge') if len(import_prices) > 0 else np.ones(planner_output_length) * 0.15
            export_prices = np.pad(export_prices, (0, pad_len), mode='edge')
            timestamps.extend([timestamps[-1] + timedelta(minutes=15*i) for i in range(1, pad_len+1)])
        
        return predictions_kwh, solar_kwh, import_prices, export_prices, timestamps
    
    def get_measurements_at(self, timestamp) -> Dict[str, float]:
        """Get actual measurements at timestamp (15-min interval)."""
        if self.measurements_df is None or self.measurements_df.empty:
            return {}
        
        # Round to nearest 15-min interval
        rounded = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
        
        if rounded in self.measurements_df.index:
            row = self.measurements_df.loc[rounded]
            # Handle NaN values by replacing with 0
            return {
                'total_power_kw': float(row.get('total_power_kw', 0.0)) if not pd.isna(row.get('total_power_kw')) else 0.0,
                'solar_actual_kw': float(row.get('solar_actual_kw', 0.0)) if not pd.isna(row.get('solar_actual_kw')) else 0.0,
                'gshp_power_kw': float(row.get('gshp_power_kw', 0.0)) if not pd.isna(row.get('gshp_power_kw')) else 0.0,
                'leaf_power_kw': float(row.get('leaf_power_kw', 0.0)) if not pd.isna(row.get('leaf_power_kw')) else 0.0,
            }
        
        return {}
    
    def simulate_battery_control(
        self,
        planner: BatteryPlanner,
        planner_type: str,
        battery_capacity_kwh: float = 50.0,
        battery_min_soc_pct: float = 10.0,
        battery_max_soc_pct: float = 90.0,
        battery_initial_soc_pct: float = 10.0,
        max_planks: int = 96,  # 24 hours at 15-min intervals
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run step-by-step battery planner simulation.
        
        Args:
            planner: BatteryPlanner instance to test
            planner_type: Name of planner (for reporting)
            battery_capacity_kwh: Battery capacity (kWh)
            battery_min_soc_pct: Minimum SoC (%)
            battery_max_soc_pct: Maximum SoC (%)
            battery_initial_soc_pct: Initial SoC (%)
            max_planks: Maximum planning intervals to simulate
        
        Returns:
            Dict with replay metrics: success, soc_violations, cost, savings, etc.
        """
        import os
        from dotenv import load_dotenv
        
        load_dotenv(override=True)
        
        # Configure battery parameters for this test
        os.environ['BATTERY_CAPACITY_KWH'] = str(battery_capacity_kwh)
        os.environ['BATTERY_MIN_SOC_PCT'] = str(battery_min_soc_pct)
        os.environ['BATTERY_MAX_SOC_PCT'] = str(battery_max_soc_pct)
        os.environ['PLAN_INTERVAL_MINUTES'] = '15'
        
        if self.measurements_df is None or self.measurements_df.empty:
            return {
                'success': False,
                'error': 'No measurement data in fixture',
                'planner_type': planner_type,
            }
        
        # Start simulation from first measurement
        current_time = self.measurements_df.index[0]
        end_time = self.measurements_df.index[-1]
        soc_kwh = (battery_capacity_kwh * battery_initial_soc_pct / 100.0)
        
        # Metrics
        intervals_run = 0
        soc_violations = []
        cost_with_battery = 0.0
        cost_no_battery = 0.0
        battery_actions = []
        
        # Simulation loop
        while current_time < end_time and intervals_run < max_planks:
            # Compute current SoC to pass directly to planner
            current_soc_pct = (soc_kwh / battery_capacity_kwh) * 100.0
            
            # Get planner horizon
            predictions, solar, import_prices, export_prices, timestamps = self.get_planner_horizon(
                current_time, max_planks
            )
            
            if len(predictions) == 0:
                break
            
            try:
                # Run planner
                ts_strings = []
                for ts in timestamps:
                    if hasattr(ts, 'isoformat'):
                        ts_strings.append(ts.isoformat())
                    else:
                        ts_strings.append(str(ts))
                
                plan = planner.plan(
                    predictions_kwh=predictions,
                    solar_kwh=solar,
                    import_prices=import_prices,
                    export_prices=export_prices,
                    prediction_timestamps=ts_strings,
                    allow_export=True,
                    initial_soc_pct=current_soc_pct,
                    context=context,
                )
                
                if plan is None or len(plan) == 0:
                    break
                
                # Execute first interval only
                entry = plan[0]
                
                # Get actual measurements for this interval
                measurements = self.get_measurements_at(current_time)
                # actual_load = grid power (total_power_kw) + solar generation
                # If solar_actual_kw not available, just use grid power (conservative)
                actual_solar = measurements.get('solar_actual_kw', 0.0)
                actual_load_kw = measurements.get('total_power_kw', 0.0) + actual_solar
                
                # Convert load to kWh for the interval (15 min = 0.25 hours)
                interval_hours = 0.25
                actual_load_kwh = actual_load_kw * interval_hours
                
                # Apply battery action and compute grid exchange
                battery_charge_kwh = entry.charge_from_solar_kwh + entry.charge_from_grid_kwh
                battery_discharge_kwh = entry.discharge_to_load_kwh + entry.discharge_to_export_kwh
                
                # Update SoC (simplified: assume efficiency)
                soc_kwh = soc_kwh + battery_charge_kwh - battery_discharge_kwh
                
                # Clamp to valid range
                soc_kwh = np.clip(
                    soc_kwh,
                    battery_capacity_kwh * battery_min_soc_pct / 100.0,
                    battery_capacity_kwh * battery_max_soc_pct / 100.0
                )
                
                current_soc_pct_after = (soc_kwh / battery_capacity_kwh) * 100.0
                
                # Check for violations
                if current_soc_pct_after < battery_min_soc_pct or current_soc_pct_after > battery_max_soc_pct:
                    soc_violations.append({
                        'timestamp': current_time.isoformat(),
                        'soc_pct': current_soc_pct_after,
                        'min': battery_min_soc_pct,
                        'max': battery_max_soc_pct,
                    })
                
                # Compute costs
                import_price = import_prices[0] if len(import_prices) > 0 else 0.15
                export_price = export_prices[0] if len(export_prices) > 0 else 0.05
                
                # Realized grid exchange (both in kWh)
                # Battery charging from grid increases import, discharging reduces it
                grid_import = max(0.0, actual_load_kwh + battery_charge_kwh - battery_discharge_kwh)
                grid_export = max(0.0, battery_discharge_kwh - actual_load_kwh - battery_charge_kwh)
                
                interval_cost_battery = grid_import * import_price - grid_export * export_price
                interval_cost_no_battery = actual_load_kwh * import_price
                
                cost_with_battery += interval_cost_battery
                cost_no_battery += interval_cost_no_battery
                
                battery_actions.append({
                    'timestamp': current_time.isoformat(),
                    'action': entry.battery_action,
                    'soc_pct': current_soc_pct_after,
                    'cost_eur': interval_cost_battery,
                })
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e),
                    'planner_type': planner_type,
                    'intervals_run': intervals_run,
                }
            
            # Move to next interval
            current_time += timedelta(minutes=15)
            intervals_run += 1
        
        savings = cost_no_battery - cost_with_battery
        
        return {
            'success': True,
            'planner_type': planner_type,
            'intervals_run': intervals_run,
            'soc_violations': len(soc_violations),
            'soc_violation_details': soc_violations[:5],  # First 5 for logging
            'cost_with_battery_eur': cost_with_battery,
            'cost_no_battery_eur': cost_no_battery,
            'savings_eur': savings,
            'savings_pct': (savings / cost_no_battery * 100.0) if cost_no_battery > 0 else 0.0,
            'final_soc_pct': (soc_kwh / battery_capacity_kwh) * 100.0,
            'battery_actions_sample': battery_actions[:5],
        }


def load_fixture(fixture_path: str) -> BatteryTestData:
    """Load fixture pickle file."""
    if not Path(fixture_path).exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    
    return BatteryTestData.load(fixture_path)


def get_fixtures() -> List[str]:
    """Get all fixture files in tests/fixtures."""
    fixture_dir = Path(__file__).parent / 'fixtures'
    return sorted([str(p) for p in fixture_dir.glob('*.pkl')])

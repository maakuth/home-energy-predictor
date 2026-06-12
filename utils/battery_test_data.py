"""
Utilities for loading and using pickled battery test data.

This module provides helpers to deserialize battery planning data that was
dumped using dump_battery_data.py for use in tests.
"""

import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timezone


class BatteryTestData:
    """
    Container for battery planning test data loaded from pickle file.
    
    Provides convenient access to the various data sources (HA states,
    predictions, prices, history, configuration).
    
    Example:
        data = BatteryTestData.load('battery_test_data.pkl')
        
        # Access Home Assistant states
        soc = data.ha_state('sensor.battery_soc_pct')
        
        # Get predictions as DataFrame
        df_pred = data.predictions_df()
        
        # Get historical data
        history = data.history('sensor.solar_power_kw')
        
        # Access configuration
        print(data.battery_config)
    """
    
    def __init__(self, raw_data: Dict[str, Any]):
        """Initialize with raw pickle data dictionary."""
        self._data = raw_data
    
    @classmethod
    def load(cls, path: str) -> 'BatteryTestData':
        """
        Load battery test data from pickle file.
        
        Args:
            path: Path to pickle file created by dump_battery_data.py
            
        Returns:
            BatteryTestData instance
            
        Raises:
            FileNotFoundError: If pickle file doesn't exist
            ValueError: If pickle file format is invalid
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Battery test data file not found: {path}")
        
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Could not deserialize pickle file {path}: {e}")
        
        if not isinstance(data, dict) or 'metadata' not in data:
            raise ValueError(f"Invalid battery test data format in {path}")
        
        return cls(data)
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata about when and how the data was dumped."""
        return self._data.get('metadata', {})
    
    @property
    def battery_config(self) -> Dict[str, Any]:
        """Get battery configuration used when data was dumped."""
        return self._data.get('battery_config', {})
    
    @property
    def gshp_config(self) -> Dict[str, Any]:
        """Get GSHP (ground source heat pump) configuration."""
        return self._data.get('gshp_config', {})
    
    def ha_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get Home Assistant entity state snapshot.
        
        Args:
            entity_id: Home Assistant entity ID (e.g., 'sensor.battery_soc_pct')
            
        Returns:
            Dictionary with 'state', 'attributes', 'last_updated', or None if not found
        """
        ha_states = self._data.get('ha_states', {})
        return ha_states.get(entity_id)
    
    def ha_state_value(self, entity_id: str, default: Any = None) -> Any:
        """
        Get Home Assistant entity state value (the 'state' field).
        
        Args:
            entity_id: Home Assistant entity ID
            default: Default value if entity not found
            
        Returns:
            State value or default
        """
        state_data = self.ha_state(entity_id)
        if state_data is None:
            return default
        return state_data.get('state', default)
    
    def ha_state_float(self, entity_id: str, default: float = 0.0) -> float:
        """
        Get Home Assistant entity state as float.
        
        Args:
            entity_id: Home Assistant entity ID
            default: Default value if not found or unparseable
            
        Returns:
            Float value or default
        """
        value = self.ha_state_value(entity_id)
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def predictions_list(self) -> List[Dict[str, Any]]:
        """Get list of predictions (baseload, solar, temperature, etc)."""
        return self._data.get('predictions', [])
    
    def predictions_df(self) -> pd.DataFrame:
        """
        Get predictions as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: timestamp, predicted_baseload, solar_forecast,
                                   outside_temp, is_sauna_active, is_fallback_price
        """
        preds = self.predictions_list()
        if not preds:
            return pd.DataFrame()
        
        df = pd.DataFrame(preds)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        
        return df
    
    def market_prices_list(self) -> List[Dict[str, Any]]:
        """Get list of market price data."""
        return self._data.get('market_prices', [])
    
    def market_prices_df(self) -> pd.DataFrame:
        """
        Get market prices as a pandas DataFrame.
        
        Returns:
            DataFrame with columns: timestamp, import_price, export_price
        """
        prices = self.market_prices_list()
        if not prices:
            return pd.DataFrame()
        
        df = pd.DataFrame(prices)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        
        return df
    
    def history(self, entity_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get historical state changes for an entity.
        
        Args:
            entity_id: Home Assistant entity ID
            
        Returns:
            List of history entries with timestamp/value/state, or None if not found
        """
        history = self._data.get('history', {})
        return history.get(entity_id)
    
    def history_df(self, entity_id: str) -> pd.DataFrame:
        """
        Get entity history as a pandas DataFrame.
        
        Args:
            entity_id: Home Assistant entity ID
            
        Returns:
            DataFrame with timestamp as index and 'value'/'state' columns
        """
        hist = self.history(entity_id)
        if not hist:
            return pd.DataFrame()
        
        df = pd.DataFrame(hist)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        
        # Convert numeric values
        if 'value' in df.columns:
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        return df
    
    def archive_predictions(self) -> List[Dict[str, Any]]:
        """
        Get archived predictions from database history.
        
        These are the predictions that were actually stored in the database
        during the time period.
        
        Returns:
            List of archived prediction records
        """
        history = self._data.get('history', {})
        return history.get('predictions_archive', [])
    
    def archive_predictions_df(self) -> pd.DataFrame:
        """
        Get archived predictions as a pandas DataFrame.
        
        Returns:
            DataFrame with columns from database schema
        """
        archive = self.archive_predictions()
        if not archive:
            return pd.DataFrame()
        
        df = pd.DataFrame(archive)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            df = df.set_index('timestamp')
        
        return df
    
    def period_start(self) -> Optional[datetime]:
        """Get the start of the data collection period."""
        start_str = self.metadata.get('period_start')
        if start_str:
            return datetime.fromisoformat(start_str.replace('Z', '+00:00'))
        return None
    
    def period_end(self) -> Optional[datetime]:
        """Get the end of the data collection period."""
        end_str = self.metadata.get('period_end')
        if end_str:
            return datetime.fromisoformat(end_str.replace('Z', '+00:00'))
        return None
    
    def dumped_at(self) -> Optional[datetime]:
        """Get when this data was dumped."""
        dump_str = self.metadata.get('dumped_at')
        if dump_str:
            return datetime.fromisoformat(dump_str.replace('Z', '+00:00'))
        return None
    
    def model_version(self) -> Optional[str]:
        """Get the model version when data was dumped."""
        return self.metadata.get('model_version')
    
    def summary(self) -> str:
        """Get a human-readable summary of the data."""
        lines = [
            f"Battery Test Data Summary",
            f"  Dumped: {self.dumped_at()}",
            f"  Period: {self.period_start()} to {self.period_end()}",
            f"  Model version: {self.model_version()}",
            f"  Predictions: {len(self.predictions_list())} records",
            f"  Market prices: {len(self.market_prices_list())} records",
            f"  Archive records: {len(self.archive_predictions())} records",
            f"  Battery enabled: {self.battery_config.get('enabled', False)}",
            f"  Battery capacity: {self.battery_config.get('capacity_kwh', 0)} kWh",
            f"  GSHP enabled: {self.gshp_config.get('enabled', False)}",
        ]
        return "\n".join(lines)


def load_battery_test_data(path: str) -> BatteryTestData:
    """
    Convenience function to load battery test data.
    
    Args:
        path: Path to pickle file
        
    Returns:
        BatteryTestData instance
        
    Example:
        data = load_battery_test_data('battery_test_data.pkl')
        df_pred = data.predictions_df()
        df_prices = data.market_prices_df()
    """
    return BatteryTestData.load(path)

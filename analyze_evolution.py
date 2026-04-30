import os
import sqlite3
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta, timezone
from analyze_performance import fetch_actuals
from utils.git_utils import get_model_version
from utils.sqlite_utils import get_db_connection, db_exists

def fetch_all_predictions(days=7):
    """Fetch ALL archived predictions from the SQLite DB for the given period."""
    if not db_exists():
        print(f"⚠️ Database not found.")
        return pd.DataFrame()

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        conn = get_db_connection()
        query = f"""
            SELECT target_timestamp, generated_at, predicted_usage_kw, 
                   battery_action, version, grid_import_kwh, import_price, 
                   grid_export_kwh, export_price
            FROM predictions
            WHERE target_timestamp >= '{cutoff}'
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return df
            
        df['target_timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
        df['generated_at'] = pd.to_datetime(df['generated_at'], utc=True)
        
        # Calculate lead time in hours
        df['lead_time_hours'] = (df['target_timestamp'] - df['generated_at']).dt.total_seconds() / 3600.0
        
        return df
    except Exception as e:
        print(f"Error reading archived predictions: {e}")
        return pd.DataFrame()

def analyze_evolution(days=7):
    print(f"=== Plan Evolution Analysis (Last {days} days) ===")
    
    df_actual = fetch_actuals(days=days)
    if df_actual.empty:
        print("⚠️ No actuals found. Cannot analyze accuracy.")
        return

    df_all = fetch_all_predictions(days=days)
    if df_all.empty:
        print("⚠️ No archived predictions found.")
        return

    # Join predictions with actuals
    # Note: df_actual index is 'timestamp' (rounded to 15min)
    # predictions 'target_timestamp' should also be rounded to 15min if not already
    df_all['target_timestamp_rounded'] = df_all['target_timestamp'].dt.round('15min')
    
    df_merged = df_all.merge(df_actual, left_on='target_timestamp_rounded', right_index=True)
    
    if df_merged.empty:
        print("⚠️ No overlapping data between predictions and actuals.")
        return

    df_merged['error'] = df_merged['predicted_usage_kw'] - df_merged['actual_usage']
    
    # Calculate Planned Cost: (grid_import * import_price) - (grid_export * export_price)
    # We use 15min intervals (0.25h) to convert kWh
    # Wait, grid_import_kwh in the DB is already in kWh (presumably)
    # Let's check optimize_plan.py to be sure.
    # In optimize_plan.py: item.get('grid_import_kwh') 
    # and grid_import_kwh = max(0, net_load - discharge_kw) * interval_hours
    # So it is indeed kWh.
    
    df_merged['planned_cost'] = (df_merged['grid_import_kwh'] * df_merged['import_price']) - \
                                (df_merged['grid_export_kwh'] * df_merged['export_price'])

    # Group by lead time bins
    # Bins: 0-1h, 1-3h, 3-6h, 6-12h, 12-24h, 24-48h
    bins = [0, 1, 3, 6, 12, 24, 48, 168]
    labels = ['<1h', '1-3h', '3-6h', '6-12h', '12-24h', '24-48h', '>48h']
    df_merged['lead_time_bin'] = pd.cut(df_merged['lead_time_hours'], bins=bins, labels=labels)

    print("\nAccuracy by Lead Time (Prediction Horizon):")
    summary = df_merged.groupby('lead_time_bin', observed=True).agg({
        'error': ['count', 'mean', lambda x: x.abs().mean()]
    })
    summary.columns = ['Samples', 'Bias (kW)', 'MAE (kW)']
    print(summary)

    print("\nPlanned Cost Evolution (Average planned cost per 15min interval):")
    cost_summary = df_merged.groupby('lead_time_bin', observed=True).agg({
        'planned_cost': ['mean', 'std']
    })
    cost_summary.columns = ['Avg Planned Cost (€)', 'StdDev (€)']
    print(cost_summary)

    # Stability Analysis: For each target_timestamp, how much does predicted_usage_kw change?
    print("\nStability Analysis (Prediction changes as we get closer):")
    # Only consider target_timestamps that have at least 2 versions
    multi_version = df_all.groupby('target_timestamp').filter(lambda x: len(x) > 1)
    if not multi_version.empty:
        # For each target_timestamp, calculate the Std Dev of predictions
        stability = multi_version.groupby('target_timestamp').agg({
            'predicted_usage_kw': ['count', 'std', 'max', 'min']
        })
        stability.columns = ['Versions', 'StdDev', 'Max', 'Min']
        stability['Range'] = stability['Max'] - stability['Min']
        
        print(f"Average StdDev of predictions for same timestamp: {stability['StdDev'].mean():.3f} kW")
        print(f"Average Range (Max-Min) for same timestamp:     {stability['Range'].mean():.3f} kW")
        
        # Action Stability: How often does battery_action change?
        # (Simplified: % of timestamps where battery_action was NOT constant across all versions)
        action_changes = multi_version.groupby('target_timestamp')['battery_action'].nunique()
        changed_actions_pct = (action_changes > 1).mean() * 100
        print(f"Timestamps where Battery Action evolved:        {changed_actions_pct:.1f}%")

    # Version comparison: Does the latest version of the model perform better?
    current_version = get_model_version()
    versions = df_merged['version'].unique()
    if len(versions) > 1:
        print("\nAccuracy by Model Version:")
        v_summary = df_merged.groupby('version').agg({
            'error': ['count', lambda x: x.abs().mean()]
        })
        v_summary.columns = ['Samples', 'MAE (kW)']
        print(v_summary.sort_values('version', ascending=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    args = parser.parse_args()
    
    analyze_evolution(days=args.days)

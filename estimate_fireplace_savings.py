from __future__ import annotations
import os
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import sqlite3
from typing import Any
from dotenv import load_dotenv

load_dotenv(override=True)

def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)

def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate fireplace cost savings.")
    parser.add_argument('--days', type=int, default=None, help='Number of recent days to analyze. If omitted, analyzes all available data.')
    args = parser.parse_args()

    print("Loading processed data...")
    try:
        df = pd.read_csv('processed_data.csv', index_col=0)
    except FileNotFoundError:
        print("processed_data.csv not found.")
        return

    df.index = pd.to_datetime(df.index, utc=True)
    
    if args.days is not None:
        cutoff_date = df.index.max() - pd.Timedelta(days=args.days)
        print(f"Filtering data to the last {args.days} days (since {cutoff_date})...")
        df = df[df.index >= cutoff_date]

    
    # We need to detect the interval
    if len(df) > 1:
        median_interval = pd.Series(df.index).diff().median()
        interval_minutes = int(round(median_interval.total_seconds() / 60.0))
    else:
        interval_minutes = 15
    
    interval_hours = interval_minutes / 60.0

    print(f"Data resolution: {interval_minutes} minutes")

    if 'is_fireplace_active' not in df.columns:
        print("is_fireplace_active column not found in processed_data.csv.")
        return

    active_df = df[df['is_fireplace_active'] == 1].copy()
    if active_df.empty:
        print("No fireplace active periods found.")
        return

    print(f"Found {len(active_df)} intervals where fireplace was active.")

    # Constants from optimize_plan.py
    cop = get_env_float('GSHP_COP', 3.5)
    reservoir_l = get_env_float('GSHP_RESERVOIR_LITERS', 500)
    kwh_per_degree = (reservoir_l * 4.18) / 3600.0 
    heat_loss_k = get_env_float('GSHP_HEAT_LOSS_K', 0.135)
    baseline_demand_kw = get_env_float('GSHP_BASELINE_DEMAND_KW', 1.0)

    # Calculate thermal energy for each interval
    # 1. House heat demand during this interval
    # baseline_demand_kw + max(0, (20.0 - o_temp) * heat_loss_k) in kW, multiplied by interval_hours

    active_df['outside_temp'] = active_df['outside_temp'].fillna(5.0)
    active_df['heat_demand_kw'] = baseline_demand_kw + np.maximum(0, (20.0 - active_df['outside_temp']) * heat_loss_k)
    active_df['heat_demand_kwh'] = active_df['heat_demand_kw'] * interval_hours

    # 2. Accumulator heating
    # The accumulator temp difference is in acc_roc.
    active_df['acc_roc'] = active_df['acc_roc'].fillna(0)
    # only consider positive rate of change (heating up)
    active_df['acc_heating_kwh'] = np.maximum(0, active_df['acc_roc']) * kwh_per_degree

    # Total thermal energy
    active_df['total_thermal_kwh'] = active_df['heat_demand_kwh'] + active_df['acc_heating_kwh']

    # Electrical savings
    active_df['electrical_savings_kwh'] = active_df['total_thermal_kwh'] / cop

    # Connect to DB to get prices
    try:
        conn = sqlite3.connect('hepo.db')
        # Load prices
        prices_df = pd.read_sql_query("SELECT target_timestamp, import_price FROM predictions WHERE import_price IS NOT NULL", conn)
        prices_df['target_timestamp'] = pd.to_datetime(prices_df['target_timestamp'], utc=True)
        # Drop duplicates in case there are multiple predictions for the same target timestamp
        prices_df = prices_df.drop_duplicates(subset=['target_timestamp'], keep='last')
        prices_df = prices_df.set_index('target_timestamp')
    except Exception as e:
        print(f"Error loading prices from DB: {e}")
        prices_df = pd.DataFrame()

    # Default price if missing
    default_price = 0.10

    # Join prices to active_df
    active_df = active_df.join(prices_df, how='left')
    active_df['import_price'] = active_df['import_price'].fillna(default_price)

    # Financial savings
    active_df['financial_savings_eur'] = active_df['electrical_savings_kwh'] * active_df['import_price']

    # Summary
    total_hours = len(active_df) * interval_hours
    total_thermal_kwh = active_df['total_thermal_kwh'].sum()
    total_electrical_kwh = active_df['electrical_savings_kwh'].sum()
    total_financial_eur = active_df['financial_savings_eur'].sum()

    print("\n=== Fireplace Savings Estimate ===")
    print(f"Total time active:      {total_hours:.1f} hours")
    print(f"Thermal energy produced: {total_thermal_kwh:.1f} kWh")
    print(f"Electrical energy saved: {total_electrical_kwh:.1f} kWh")
    print(f"Estimated cost savings:  {total_financial_eur:.2f} €")
    print("==================================")

if __name__ == '__main__':
    main()

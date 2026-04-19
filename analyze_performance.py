import os
import json
import pandas as pd
import numpy as np
import sqlite3
import argparse
import xgboost as xgb
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from utils.ha_utils import push_ha_state
from utils.db_utils import fetch_states_history
from utils.git_utils import get_git_version

load_dotenv(override=True)

def fetch_actuals(days=7):
    """Fetch actual grid meter and solar data from HA/PostgreSQL."""
    print(f"Fetching actuals for last {days} days from PostgreSQL...")
    entities = {
        'sensor.sahkokauppa_nyt': 'total_power',
        'sensor.solarh_63038_real_power_kw': 'solar_actual'
    }
    
    hist_data = fetch_states_history(list(entities.keys()), hours=days*24)
    
    all_resampled = []
    for eid, col_name in entities.items():
        df = hist_data.get(eid)
        if df is not None and not df.empty:
            df = df.rename(columns={'state': col_name})
            all_resampled.append(df.set_index('timestamp').resample('15min').mean())
    
    if not all_resampled:
        print("⚠️ No data fetched from PostgreSQL.")
        return pd.DataFrame(columns=['actual_usage'])

    df_actual = pd.concat(all_resampled, axis=1).fillna(0)
    # Total home power is grid_meter + solar_production
    df_actual['actual_usage'] = df_actual.get('total_power', 0) + df_actual.get('solar_actual', 0)
    return df_actual[['actual_usage']]

def get_archived_predictions(version=None, include_battery=False):
    """Load predictions that were actually made in real-time from the SQLite DB."""
    db_file = 'hepo.db'
    if not os.path.exists(db_file):
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_file)
        # Filter by version first to get the latest prediction made BY THIS VERSION
        where_clause = f"WHERE version = '{version}'" if version else ""
        
        cols = "target_timestamp, predicted_usage_kw as predicted_usage, is_fallback_price, version"
        if include_battery:
            cols += ", battery_action, battery_power_kw, battery_soc_pct, import_price, export_price, grid_import_kwh, grid_export_kwh"

        query = f"""
            SELECT {cols}
            FROM (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY generated_at DESC) as rn
                FROM predictions
                {where_clause}
            )
            WHERE rn = 1
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['target_timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
        return df.set_index('target_timestamp')
    except Exception as e:
        print(f"Error reading archived predictions: {e}")
        return pd.DataFrame()

def summarize_battery_performance(df_merged):
    """
    Analyzes how well the battery plan performed compared to a 'no battery' baseline.
    Uses the archived plan context (what we THOUGHT would happen).
    """
    if 'battery_action' not in df_merged.columns or df_merged['battery_action'].isnull().all():
        return

    print("\n--- BATTERY PLAN EVALUATION (Hindsight vs. Foresight) ---")
    
    # Filter to periods where battery was active
    active = df_merged[df_merged['battery_action'] != 'idle'].copy()
    if active.empty:
        print("  No battery activity found in this period.")
        return

    # Calculate Savings: (Planned Discharge * Actual Price) - (Planned Charge * Actual Price)
    # We use planned energy because we want to evaluate the logic of the plan.
    # Note: interval_hours is needed to convert power to energy.
    interval_hours = (df_merged.index[1] - df_merged.index[0]).total_seconds() / 3600.0 if len(df_merged) > 1 else 0.25
    
    planned_savings = (active['import_price'] * (active['predicted_usage'] - active['grid_import_kwh']) * interval_hours).sum()
    
    charge_hours = active[active['battery_action'].str.contains('charge', na=False)]
    discharge_hours = active[active['battery_action'].str.contains('discharge', na=False)]
    
    print(f"  Planned Savings:      {planned_savings:+.2f} €")
    if not charge_hours.empty:
        print(f"  Avg Charge Price:    {charge_hours['import_price'].mean():.4f} €/kWh")
    if not discharge_hours.empty:
        print(f"  Avg Discharge Value: {discharge_hours['import_price'].mean():.4f} €/kWh")
    
    if not charge_hours.empty and not discharge_hours.empty:
        spread = discharge_hours['import_price'].mean() - charge_hours['import_price'].mean()
        print(f"  Planned Spread:      {spread:+.4f} €/kWh")

def backtest_current_model(df_actual):
    """
    Run a 'hindsight' prediction using the CURRENT model on PAST features.
    This tells us if the NEW model is better at predicting the past than the OLD model was.
    """
    model_path = 'energy_model.json'
    features_path = 'model_features.json'
    processed_path = 'processed_data.csv'

    if not all(os.path.exists(p) for p in [model_path, features_path, processed_path]):
        print("⚠️ Model files or processed_data.csv not found. Skipping hindsight backtest.")
        return pd.DataFrame()

    try:
        # Load model and features
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        with open(features_path, 'r') as f:
            features = json.load(f)

        # Load processed features
        df_proc = pd.read_csv(processed_path, index_col=0)
        df_proc.index = pd.to_datetime(df_proc.index, utc=True)
        
        # Only predict for timestamps we have actuals for
        common_idx = df_proc.index.intersection(df_actual.index)
        if common_idx.empty:
            return pd.DataFrame()
            
        X = df_proc.loc[common_idx, features]
        # In HEPO, the model predicts Baseload (Power - GSHP). 
        # But analyze_performance compares against Total Power.
        # So we must add the GSHP power back if we want to compare against the 'total actual'.
        # However, for pure model accuracy backtesting, it's cleaner to just compare Baseload vs Baseload.
        # But hepo.db stores the *entire planned usage* (baseload + gshp).
        # Let's stick to comparing what was ARCHIVED in hepo.db vs what the NEW model would ARCHIVE.
        
        preds = model.predict(X)
        
        # For simplicity, we assume GSHP usage was exactly as it happened in history
        # (Since we're evaluating the ML model, not the optimization strategy here)
        gshp_col = 'gshp_power'
        gshp_val = (df_proc.loc[common_idx, gshp_col] / 1000.0) if gshp_col in df_proc.columns else 0.0
        
        df_hindsight = pd.DataFrame({
            'hindsight_usage': preds + gshp_val
        }, index=common_idx)
        
        return df_hindsight
    except Exception as e:
        print(f"Error during hindsight backtest: {e}")
        return pd.DataFrame()

def analyze(days=2, do_backtest=False):
    print(f"=== Starting Performance Analysis (Window: {days} days) ===")
    
    df_actual = fetch_actuals(days=days)
    if df_actual.empty:
        return

    current_version = get_git_version()
    df_archived = get_archived_predictions(version=current_version, include_battery=True)
    
    # 1. Real-time Analysis (What actually happened)
    # Scope to current model version
    if not df_archived.empty:
        comparison = df_archived.join(df_actual, how='inner')
    else:
        print(f"⚠️ No archived predictions found for current version ({current_version}) in the last {days} days.")
        comparison = pd.DataFrame()
    
    if comparison.empty:
        print("No overlapping data found between archived predictions (current version) and actuals.")
    else:
        # Filter out fallback prices
        comparison_clean = comparison[comparison.get('is_fallback_price', 0) == 0]
        
        # 3-Hour Analysis
        res_3h = comparison_clean.resample('3h').mean(numeric_only=True).dropna()
        if not res_3h.empty:
            res_3h['error'] = res_3h['predicted_usage'] - res_3h['actual_usage']
            mae = res_3h['error'].abs().mean()
            bias = res_3h['error'].mean()
            print(f"\nREAL-TIME ARCHIVED PERFORMANCE (3-Hour Avg, Version: {current_version}):")
            print(f"  MAE:  {mae:.3f} kW")
            print(f"  Bias: {bias:+.3f} kW (Positive means over-predicting)")
            
            # Push to HA
            push_ha_state('sensor.hepo_accuracy', f"{mae:.3f}", {
                'friendly_name': 'HEPO Real-time MAE (3h)',
                'unit_of_measurement': 'kW',
                'bias': float(bias),
                'sample_count': len(res_3h),
                'model_version': current_version
            })

        # Battery Evaluation
        summarize_battery_performance(comparison_clean)

    # 2. Hindsight Backtest (How would the current model have done?)
    if do_backtest:
        df_hindsight = backtest_current_model(df_actual)
        if not df_hindsight.empty:
            # Join hindsight with actuals
            hindsight_comp = df_hindsight.join(df_actual, how='inner')
            h_res_3h = hindsight_comp.resample('3h').mean(numeric_only=True).dropna()
            
            if not h_res_3h.empty:
                h_res_3h['error'] = h_res_3h['hindsight_usage'] - h_res_3h['actual_usage']
                h_mae = h_res_3h['error'].abs().mean()
                h_bias = h_res_3h['error'].mean()
                
                print(f"\nHINDSIGHT PERFORMANCE (Current Model on same history):")
                print(f"  MAE:  {h_mae:.3f} kW")
                print(f"  Bias: {h_bias:+.3f} kW")
                
                if not comparison.empty and not res_3h.empty:
                    improvement = ((mae - h_mae) / mae) * 100 if mae > 0 else 0
                    print(f"  Improvement vs Real-time: {improvement:+.1f}%")

    print("\n✅ Analysis complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=2, help='Number of days to look back')
    parser.add_argument('--backtest', action='store_true', help='Compare with currently trained model in hindsight')
    args = parser.parse_args()
    
    analyze(days=args.days, do_backtest=args.backtest)

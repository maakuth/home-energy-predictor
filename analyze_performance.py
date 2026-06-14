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
from utils.sqlite_utils import get_db_connection, db_exists
from utils.git_utils import get_model_version

load_dotenv(override=True)

def fetch_actuals(days=7):
    """Fetch actual grid meter and solar data from HA/PostgreSQL."""
    print(f"Fetching actuals for last {days} days from PostgreSQL...")
    entities = {
        'sensor.sahkokauppa_nyt': 'total_power',
        'sensor.solarh_63038_real_power_kw': 'solar_actual',
        'sensor.mlp_teho': 'gshp_actual_w',
        'sensor.be_stat_batt_power': 'battery_actual_w'
    }
    
    hist_data = fetch_states_history(list(entities.keys()), hours=days*24)
    
    all_resampled = []
    for eid, col_name in entities.items():
        df = hist_data.get(eid)
        if df is not None and not df.empty:
            df = df.rename(columns={'state': col_name})
            if not isinstance(df.index, pd.DatetimeIndex):
                df = df.set_index('timestamp')
            all_resampled.append(df.resample('15min').mean())
    
    if not all_resampled:
        print("⚠️ No data fetched from PostgreSQL.")
        return pd.DataFrame(columns=['actual_usage', 'solar_actual', 'gshp_actual_kw'])

    df_actual = pd.concat(all_resampled, axis=1).fillna(0)
    # Total home power is grid_meter + solar_production - battery_net_power
    battery_kw = df_actual.get('battery_actual_w', 0) / 1000.0
    df_actual['actual_usage'] = df_actual.get('total_power', 0) + df_actual.get('solar_actual', 0) - battery_kw
    df_actual['gshp_actual_kw'] = df_actual.get('gshp_actual_w', 0) / 1000.0
    return df_actual[['actual_usage', 'solar_actual', 'gshp_actual_kw']]

def get_archived_predictions(version=None, include_battery=False):
    """Load predictions that were actually made in real-time from the SQLite DB."""
    if not db_exists():
        return pd.DataFrame()

    try:
        conn = get_db_connection()
        # Filter by version first to get the latest prediction made BY THIS VERSION
        where_clause = f"WHERE version = '{version}'" if version else ""
        
        cols = "target_timestamp, predicted_usage_kw as predicted_usage, solar_forecast_kw, is_fallback_price, version"
        if include_battery:
            cols += ", battery_action, battery_power_kw, battery_soc_pct, import_price, export_price, grid_import_kwh, grid_export_kwh, charge_from_solar_kwh, charge_from_grid_kwh, discharge_to_load_kwh, discharge_to_export_kwh, planned_gshp_kw"

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

def summarize_gshp_performance(df_merged):
    """
    Analyzes how well the GSHP load was timed relative to solar and spot prices.
    """
    if 'gshp_actual_kw' not in df_merged.columns or 'import_price' not in df_merged.columns:
        return {}

    # Calculate intervals in hours
    if len(df_merged) < 2: return {}
    interval_hours = (df_merged.index[1] - df_merged.index[0]).total_seconds() / 3600.0
    
    gshp_kw = df_merged['gshp_actual_kw']
    prices = df_merged['import_price']
    solar_kw = df_merged['solar_actual']
    total_kw = df_merged['actual_usage']
    
    # Baseload (excluding GSHP)
    baseload_kw = (total_kw - gshp_kw).clip(lower=0)
    
    # Solar available for GSHP
    solar_for_gshp = (solar_kw - baseload_kw).clip(lower=0)
    gshp_from_solar = np.minimum(gshp_kw, solar_for_gshp)
    
    gshp_total_kwh = (gshp_kw * interval_hours).sum()
    if gshp_total_kwh < 0.1: # Negligible usage
        return {}
        
    gshp_solar_kwh = (gshp_from_solar * interval_hours).sum()
    gshp_solar_pct = (gshp_solar_kwh / gshp_total_kwh) * 100
    
    # Weighted average price for GSHP (excluding solar-covered part)
    gshp_from_grid_kw = gshp_kw - gshp_from_solar
    gshp_grid_kwh = (gshp_from_grid_kw * interval_hours).sum()
    gshp_cost_eur = (gshp_from_grid_kw * prices * interval_hours).sum()
    
    gshp_avg_price = (gshp_cost_eur / gshp_grid_kwh) if gshp_grid_kwh > 0.01 else 0
    
    # Calculate spot-only price paid for GSHP for fair comparison
    if 'export_price' in df_merged.columns and gshp_grid_kwh > 0.01:
        gshp_spot_cost_eur = (gshp_from_grid_kw * df_merged['export_price'] * interval_hours).sum()
        gshp_avg_spot_paid = gshp_spot_cost_eur / gshp_grid_kwh
    else:
        gshp_avg_spot_paid = gshp_avg_price # Fallback if no export_price

    # Market average price for the same period (for comparison)
    # We use export_price as it's the raw spot price (or closest to it)
    market_avg_price = df_merged['export_price'].mean() if 'export_price' in df_merged.columns else prices.mean()
    
    print(f"\n--- GSHP LOAD TIMING EVALUATION ---")
    print(f"  Total GSHP Energy:    {gshp_total_kwh:.2f} kWh")
    print(f"  Solar Utilization:    {gshp_solar_pct:.1f}% ({gshp_solar_kwh:.2f} kWh)")
    print(f"  Avg Grid Price Paid:  {gshp_avg_price:.4f} €/kWh (incl. fees)")
    print(f"  Avg Spot Price Paid:  {gshp_avg_spot_paid:.4f} €/kWh")
    print(f"  Market Avg Spot Price: {market_avg_price:.4f} €/kWh")
    if market_avg_price > 0 and gshp_avg_spot_paid > 0:
        savings_vs_avg = (1 - gshp_avg_spot_paid / market_avg_price) * 100
        print(f"  Timing Efficiency:    {savings_vs_avg:+.1f}% vs market average")

    return {
        'gshp_avg_price': float(gshp_avg_price),
        'gshp_solar_pct': float(gshp_solar_pct),
        'gshp_total_kwh': float(gshp_total_kwh),
        'market_avg_price': float(market_avg_price)
    }

def summarize_battery_performance(df_merged):
    """
    Analyzes how well the battery plan performed compared to a 'no battery' baseline.
    Uses the archived plan context (what we THOUGHT would happen).
    """
    if 'battery_action' not in df_merged.columns or df_merged['battery_action'].isnull().all():
        return {}

    start_ts = df_merged.index.min()
    end_ts = df_merged.index.max()
    duration_hours = (end_ts - start_ts).total_seconds() / 3600
    print(f"\n--- BATTERY PLAN EVALUATION (Hindsight vs. Foresight) ---")
    print(f"  Period: {start_ts.strftime('%Y-%m-%d %H:%M')} to {end_ts.strftime('%Y-%m-%d %H:%M')} ({duration_hours:.1f}h)")
    
    # Filter to periods where battery was active
    active = df_merged[df_merged['battery_action'] != 'idle'].copy()
    if active.empty:
        print("  No battery activity found in this period.")
        return {}

    # Note: interval_hours is needed to convert power to energy.
    interval_hours = (df_merged.index[1] - df_merged.index[0]).total_seconds() / 3600.0 if len(df_merged) > 1 else 0.25
    
    # More accurate Savings calculation: compare with 'no battery' baseline
    if 'solar_forecast_kw' in active.columns and 'export_price' in active.columns:
        # No battery: net = predicted_usage - solar_forecast_kw
        net_no_batt = active['predicted_usage'] - active['solar_forecast_kw']
        import_no_batt = np.maximum(net_no_batt, 0)
        export_no_batt = np.maximum(-net_no_batt, 0)
        cost_no_batt = (import_no_batt * active['import_price']) - (export_no_batt * active['export_price'])
        
        # With battery: grid_import_kwh and grid_export_kwh are stored per interval
        cost_with_batt = (active['grid_import_kwh'] * active['import_price']) - (active['grid_export_kwh'] * active['export_price'])
        
        # Savings = (Cost No Batt - Cost With Batt) summed over active hours
        planned_savings = ((cost_no_batt - cost_with_batt) * interval_hours).sum()
    else:
        # Fallback to old less-accurate logic if columns are missing
        planned_savings = (active['import_price'] * (active['predicted_usage'] - active['grid_import_kwh']) * interval_hours).sum()
    
    # Calculate Weighted Average Charge Price
    if 'charge_from_grid_kwh' in active.columns and 'charge_from_solar_kwh' in active.columns:
        # Cost of solar charging is the lost export opportunity
        chg_cost = (active['charge_from_grid_kwh'] * active['import_price'] + 
                    active['charge_from_solar_kwh'] * active['export_price']).sum()
        chg_kwh = (active['charge_from_grid_kwh'] + active['charge_from_solar_kwh']).sum()
        avg_charge = chg_cost / chg_kwh if chg_kwh > 0 else None
    else:
        charge_hours = active[active['battery_action'].str.contains('charge', na=False)]
        avg_charge = charge_hours['import_price'].mean() if not charge_hours.empty else None

    # Calculate Weighted Average Discharge Value
    if 'discharge_to_load_kwh' in active.columns and 'discharge_to_export_kwh' in active.columns:
        dis_val = (active['discharge_to_load_kwh'] * active['import_price'] + 
                   active['discharge_to_export_kwh'] * active['export_price']).sum()
        dis_kwh = (active['discharge_to_load_kwh'] + active['discharge_to_export_kwh']).sum()
        avg_discharge = dis_val / dis_kwh if dis_kwh > 0 else None
    else:
        discharge_hours = active[active['battery_action'].str.contains('discharge', na=False)]
        avg_discharge = discharge_hours['import_price'].mean() if not discharge_hours.empty else None
    spread = avg_discharge - avg_charge if avg_charge is not None and avg_discharge is not None else None

    print(f"  Planned Savings:      {planned_savings:+.2f} €")
    if avg_charge is not None:
        print(f"  Avg Charge Price:    {avg_charge:.4f} €/kWh")
    if avg_discharge is not None:
        print(f"  Avg Discharge Value: {avg_discharge:.4f} €/kWh")
    if spread is not None:
        print(f"  Planned Spread:      {spread:+.4f} €/kWh")
        
    return {
        'planned_savings_eur': float(planned_savings),
        'avg_charge_price': float(avg_charge) if avg_charge is not None else None,
        'avg_discharge_price': float(avg_discharge) if avg_discharge is not None else None,
        'planned_spread': float(spread) if spread is not None else None
    }

def store_performance_results(results):
    """Store the analysis results into hepo.db for historical tracking and agent access."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS performance_analysis (
                analysis_timestamp TEXT,
                period_days INTEGER,
                mae_kw REAL,
                bias_kw REAL,
                battery_planned_savings_eur REAL,
                battery_avg_charge_price REAL,
                battery_avg_discharge_price REAL,
                battery_planned_spread REAL,
                model_version TEXT,
                gshp_avg_price REAL,
                gshp_solar_pct REAL,
                gshp_total_kwh REAL,
                market_avg_price REAL,
                PRIMARY KEY (analysis_timestamp)
            )
        ''')
        
        # Migration: Add missing columns if they don't exist
        cur.execute("PRAGMA table_info(performance_analysis)")
        cols = [c[1] for c in cur.fetchall()]
        new_cols = {
            'gshp_avg_price': 'REAL',
            'gshp_solar_pct': 'REAL',
            'gshp_total_kwh': 'REAL',
            'market_avg_price': 'REAL'
        }
        for col, col_type in new_cols.items():
            if col not in cols:
                cur.execute(f"ALTER TABLE performance_analysis ADD COLUMN {col} {col_type}")

        cur.execute('''
            INSERT INTO performance_analysis 
            (analysis_timestamp, period_days, mae_kw, bias_kw, 
             battery_planned_savings_eur, battery_avg_charge_price, 
             battery_avg_discharge_price, battery_planned_spread, model_version,
             gshp_avg_price, gshp_solar_pct, gshp_total_kwh, market_avg_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().astimezone().isoformat(),
            results.get('period_days'),
            results.get('mae_kw'),
            results.get('bias_kw'),
            results.get('battery_planned_savings_eur'),
            results.get('battery_avg_charge_price'),
            results.get('battery_avg_discharge_price'),
            results.get('battery_planned_spread'),
            results.get('model_version'),
            results.get('gshp_avg_price'),
            results.get('gshp_solar_pct'),
            results.get('gshp_total_kwh'),
            results.get('market_avg_price')
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error storing analysis results: {e}")

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

    current_version = get_model_version()
    df_archived = get_archived_predictions(version=current_version, include_battery=True)
    
    # 1. Real-time Analysis (What actually happened)
    # Scope to current model version
    if not df_archived.empty:
        comparison = df_archived.join(df_actual, how='inner')
    else:
        print(f"⚠️ No archived predictions found for current version ({current_version}) in the last {days} days.")
        comparison = pd.DataFrame()
    
    results = {
        'period_days': days,
        'model_version': current_version,
        'mae_kw': None,
        'bias_kw': None,
        'battery_planned_savings_eur': None,
        'battery_avg_charge_price': None,
        'battery_avg_discharge_price': None,
        'battery_planned_spread': None,
        'gshp_avg_price': None,
        'gshp_solar_pct': None,
        'gshp_total_kwh': None,
        'market_avg_price': None
    }

    if comparison.empty:
        print("No overlapping data found between archived predictions (current version) and actuals.")
    else:
        # Filter out fallback prices
        comparison_clean = comparison[comparison.get('is_fallback_price', 0) == 0]
        
        # 3-Hour Analysis
        res_3h = comparison_clean.resample('3h').mean(numeric_only=True).dropna()
        if not res_3h.empty:
            res_3h['error'] = res_3h['predicted_usage'] - res_3h['actual_usage']
            mae = float(res_3h['error'].abs().mean())
            bias = float(res_3h['error'].mean())
            
            results['mae_kw'] = mae
            results['bias_kw'] = bias
            
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
        batt_res = summarize_battery_performance(comparison_clean)
        if batt_res:
            results['battery_planned_savings_eur'] = batt_res.get('planned_savings_eur')
            results['battery_avg_charge_price'] = batt_res.get('avg_charge_price')
            results['battery_avg_discharge_price'] = batt_res.get('avg_discharge_price')
            results['battery_planned_spread'] = batt_res.get('planned_spread')

        # GSHP Evaluation
        gshp_res = summarize_gshp_performance(comparison_clean)
        if gshp_res:
            results['gshp_avg_price'] = gshp_res.get('gshp_avg_price')
            results['gshp_solar_pct'] = gshp_res.get('gshp_solar_pct')
            results['gshp_total_kwh'] = gshp_res.get('gshp_total_kwh')
            results['market_avg_price'] = gshp_res.get('market_avg_price')

        # Store results in DB
        if results['mae_kw'] is not None or results['battery_planned_savings_eur'] is not None or results['gshp_total_kwh'] is not None:
            store_performance_results(results)
            print("✅ Performance analysis stored in hepo.db")

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

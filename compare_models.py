from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from sarimax_predictor import load_historical_data as load_actual_baseload
from utils.sqlite_utils import get_db_connection, db_exists
from utils.archive_history import ArchiveHistory, add_history_arguments, history_from_arguments


def get_archived_xgboost_predictions(days: int = 2, *, history: ArchiveHistory | None = None) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()

    conn = get_db_connection()
    end = pd.to_datetime(history.cutoff, unit='s', utc=True) if history is not None else datetime.now(timezone.utc)
    cutoff = (end - timedelta(days=days)).isoformat()
    as_of = (f"AND julianday(generated_at) <= julianday({history.cutoff}, 'unixepoch')"
             if history is not None else '')
    query = f"""
        SELECT target_timestamp, predicted_usage_kw as predicted_baseload
        FROM (
            SELECT target_timestamp, predicted_usage_kw,
                   ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY julianday(generated_at) DESC) as rn
            FROM predictions
            WHERE julianday(target_timestamp) >= julianday('{cutoff}')
            {as_of}
        )
        WHERE rn = 1
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
    return df.set_index('timestamp')[['predicted_baseload']]


def load_sarima_latest_forecast(filename: str = 'state/sarimax_predictions.json') -> pd.DataFrame:
    if not os.path.exists(filename):
        return pd.DataFrame()

    with open(filename, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df.set_index('timestamp')[['predicted_baseload']]


def get_archived_sarima_predictions(days: int = 2, *, history: ArchiveHistory | None = None) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()

    conn = get_db_connection()
    end = pd.to_datetime(history.cutoff, unit='s', utc=True) if history is not None else datetime.now(timezone.utc)
    cutoff = (end - timedelta(days=days)).isoformat()
    as_of = (f"AND julianday(generated_at) <= julianday({history.cutoff}, 'unixepoch')"
             if history is not None else '')
    try:
        query = f"""
            SELECT target_timestamp, predicted_baseload_kw as predicted_baseload
            FROM (
                SELECT target_timestamp, predicted_baseload_kw,
                       ROW_NUMBER() OVER (PARTITION BY target_timestamp ORDER BY julianday(generated_at) DESC) as rn
                FROM sarimax_predictions
                WHERE julianday(target_timestamp) >= julianday('{cutoff}')
                {as_of}
            )
            WHERE rn = 1
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        df['timestamp'] = pd.to_datetime(df['target_timestamp'], utc=True)
        return df.set_index('timestamp')[['predicted_baseload']]
    except Exception:
        conn.close()
        return pd.DataFrame()


def print_hourly_breakdown(label: str, df_comp: pd.DataFrame) -> None:
    """Print hourly MAE/bias breakdown for a joined prediction vs actual DataFrame."""
    if df_comp.empty:
        return
    pred_col = [c for c in df_comp.columns if c != 'actual_baseload'][0]
    df_comp['hour'] = df_comp.index.hour
    print(f"\n  Hourly breakdown ({label}):")
    print(f"  {'Hour':<6} {'Samples':<8} {'MAE (kW)':<10} {'Bias (kW)':<12} {'Mean Pred':<10} {'Mean Act':<10}")
    print(f"  {'-'*56}")
    for h in range(24):
        grp = df_comp[df_comp['hour'] == h]
        if grp.empty:
            continue
        mae = (grp[pred_col] - grp['actual_baseload']).abs().mean()
        bias = (grp[pred_col] - grp['actual_baseload']).mean()
        mean_pred = grp[pred_col].mean()
        mean_act = grp['actual_baseload'].mean()
        print(f"  {h:>2}:00   {len(grp):<8} {mae:<10.4f} {bias:<+12.4f} {mean_pred:<10.3f} {mean_act:<10.3f}")


def compare(days: int = 2, hourly: bool = False, output_csv: Optional[str] = None, show_blend: bool = False,
            *, history: ArchiveHistory | None = None) -> None:
    print(f"=== Model Comparison Analysis (last {days} days) ===")

    measured = pd.DataFrame()
    if history is not None:
        from analyze_performance import fetch_actuals
        measured = fetch_actuals(days, history=history)
        actual_ts = measured.baseload_actual_kw
        print(f'Archive actuals policy: {history.legacy_policy}; legacy eligibility is not source verification.')
        if show_blend:
            print('Archive blend skipped: archived main-model total load and SARIMA baseload have different targets.')
            show_blend = False
    else:
        actual_ts = load_actual_baseload(file_path='state/processed_data.csv', last_n_days=days)
    if actual_ts is None:
        return

    actual_df = pd.DataFrame({'actual_baseload': actual_ts})

    results: dict[str, Any] = {}
    options = {'history': history} if history is not None else {}
    xgb_df = get_archived_xgboost_predictions(days=days, **options)
    sarima_df = get_archived_sarima_predictions(days=days, **options)

    if not xgb_df.empty:
        xgb_actuals = (measured[['actual_usage']].rename(columns={'actual_usage': 'actual_baseload'})
                       if history is not None else actual_df)
        xgb_comp = xgb_df.join(xgb_actuals, how='inner').dropna()
        if not xgb_comp.empty:
            mae = (xgb_comp['predicted_baseload'] - xgb_comp['actual_baseload']).abs().mean()
            bias = (xgb_comp['predicted_baseload'] - xgb_comp['actual_baseload']).mean()
            print('\nXGBoost (Main, total load):' if history is not None else '\nXGBoost (Main):')
            print(f"  MAE:  {mae:.4f} kW  Bias: {bias:+.4f} kW  Samples: {len(xgb_comp)}")
            results['xgb'] = {'mae': float(mae), 'bias': float(bias), 'samples': len(xgb_comp)}
            if history is not None:
                results['xgb']['target'] = 'total_load'
            if hourly:
                print_hourly_breakdown('XGBoost', xgb_comp)
        else:
            print("⚠️ XGBoost: No overlapping actuals yet.")

    if not sarima_df.empty:
        sarima_comp = sarima_df.join(actual_df, how='inner').dropna()
        if not sarima_comp.empty:
            mae = (sarima_comp['predicted_baseload'] - sarima_comp['actual_baseload']).abs().mean()
            bias = (sarima_comp['predicted_baseload'] - sarima_comp['actual_baseload']).mean()
            print(f"\nSARIMA (Benchmark):")
            print(f"  MAE:  {mae:.4f} kW  Bias: {bias:+.4f} kW  Samples: {len(sarima_comp)}")
            results['sarima'] = {'mae': float(mae), 'bias': float(bias), 'samples': len(sarima_comp)}
            if history is not None:
                results['sarima']['target'] = 'baseload'
            if hourly:
                print_hourly_breakdown('SARIMA', sarima_comp)
    else:
        print("⚠️ No archived SARIMA predictions found.")

    if show_blend and not xgb_df.empty and not sarima_df.empty:
        # Align both to actuals, then compute 50/50 blend
        both = xgb_df.join(sarima_df, how='inner', lsuffix='_xgb', rsuffix='_sarima').join(actual_df, how='inner').dropna()
        if not both.empty:
            both['blend'] = 0.5 * both['predicted_baseload_xgb'] + 0.5 * both['predicted_baseload_sarima']
            mae = (both['blend'] - both['actual_baseload']).abs().mean()
            bias = (both['blend'] - both['actual_baseload']).mean()
            print(f"\n50/50 Blend (production):")
            print(f"  MAE:  {mae:.4f} kW  Bias: {bias:+.4f} kW  Samples: {len(both)}")
            results['blend'] = {'mae': float(mae), 'bias': float(bias), 'samples': len(both)}
            if hourly:
                both_hourly = both.copy()
                both_hourly['hour'] = both_hourly.index.hour
                print(f"\n  Hourly breakdown (50/50 Blend):")
                print(f"  {'Hour':<6} {'Samples':<8} {'MAE (kW)':<10} {'Bias (kW)':<12} {'Mean Pred':<10} {'Mean Act':<10}")
                print(f"  {'-'*56}")
                for h in range(24):
                    grp = both_hourly[both_hourly['hour'] == h]
                    if grp.empty:
                        continue
                    mae_h = (grp['blend'] - grp['actual_baseload']).abs().mean()
                    bias_h = (grp['blend'] - grp['actual_baseload']).mean()
                    print(f"  {h:>2}:00   {len(grp):<8} {mae_h:<10.4f} {bias_h:<+12.4f} {grp['blend'].mean():<10.3f} {grp['actual_baseload'].mean():<10.3f}")

    if output_csv and results:
        rows = []
        for model, metrics in results.items():
            row = {'model': model, **metrics}
            if history is not None:
                row.update({'history_cutoff': history.cutoff, 'measurement_policy': history.legacy_policy,
                            'legacy_is_source_verified': False})
            rows.append(row)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to {output_csv}")

    print("\nNote: MAE is only calculated for timestamps where both actual data and a prior prediction exist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare XGBoost vs SARIMA model predictions against actual baseload.')
    parser.add_argument('--days', type=int, default=2, help='Number of days to look back (default: 2)')
    parser.add_argument('--hourly', action='store_true', help='Show hourly breakdown')
    parser.add_argument('--blend', action='store_true', help='Evaluate 50/50 blend used in production')
    parser.add_argument('--output-csv', type=str, default=None, help='Save results to CSV file')
    add_history_arguments(parser)
    args = parser.parse_args()

    compare(days=args.days, hourly=args.hourly, output_csv=args.output_csv, show_blend=args.blend,
            history=history_from_arguments(args))

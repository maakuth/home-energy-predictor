from __future__ import annotations
import pandas as pd
import numpy as np
import xgboost as xgb
import argparse
from sklearn.metrics import mean_absolute_error
import json
import os
from datetime import datetime
from typing import Any
from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract, select_training_rows
from utils.archive_features import load_archive_features, load_archive_training_features
from utils.archive_quality import LEGACY_POLICIES
from utils.git_utils import get_model_version

def train(holdout_days: int = 0, data_path: str = 'state/processed_data.csv', min_coverage_days: float = 14.0,
          frame: pd.DataFrame | None = None, *, _outcome: dict[str, bool] | None = None) -> None:
    if _outcome is not None:
        _outcome['saved'] = False
    print('Loading processed data...')
    df = pd.read_csv(data_path, index_col=0) if frame is None else frame.copy(deep=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    if df.index.has_duplicates:
        print('❌ Duplicate timestamps; keeping existing model.')
        return
    step_seconds = float(df.index.to_series().diff().dt.total_seconds().median())
    print(f"  - Raw rows: {len(df)}")
    
    # Target (Y): aligned home load minus the original shared-heating and Leaf meters.
    target = 'baseload_power'
    
    # Features (X)
    features = list(BASELOAD_FEATURES)

    # A partial/corrupt extract cannot provide a usable model.  Check columns
    # before dropna so this path fails gracefully instead of raising KeyError.
    required_columns = [target, *features]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns: {', '.join(missing_columns)}")
        return

    # Drop rows where critical new features are missing
    # Anchor holdout to the input end, even if recent measurements are rejected.
    dataset_end = df.index.max()
    try:
        df = select_training_rows(df)
    except ValueError as error:
        print(f'❌ {error}; keeping existing model.')
        return
    print(f"  - Rows after dropna: {len(df)}")
    if 'heating_regime' in df:
        print('  - Eligible rows by heating regime:\n' + df['heating_regime'].value_counts().to_string())

    # --- TRUE BACKTEST LOGIC ---
    if holdout_days > 0:
        cutoff = dataset_end - pd.Timedelta(days=holdout_days)
        print(f'  - Excluding everything after {cutoff} for a true hold-out test.')
        df = df[df.index <= cutoff]
        print(f"  - Rows after holdout: {len(df)}")

    if len(df) < 10:
        print("❌ Error: Not enough data points to train a model.")
        return
    if (not np.isfinite(step_seconds) or step_seconds <= 0 or step_seconds > 900
            or not np.isfinite(min_coverage_days) or min_coverage_days <= 0
            or len(df) * step_seconds < min_coverage_days * 86400):
        print(f'❌ Insufficient eligible coverage (requires {min_coverage_days:g} sample-days); keeping existing model.')
        return

    X = df[features]
    y = df[target]

    # Weights: Give more weight to recent data (last 6 months)
    weights = np.where(df['is_extended_complex'] == 1, 3.0, 1.0)
    
    # Temporal Split: No random shuffling for time-series!
    split_idx = int(len(df) * 0.8)
    
    if split_idx == 0 or split_idx == len(df):
        print("⚠️ Warning: Dataset too small for 80/20 split. Using all data for training/testing.")
        X_train = X_test = X
        y_train = y_test = y
        w_train = w_test = weights
    else:
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        w_train = weights[:split_idx]
        w_test = weights[split_idx:]
    
    print(f"  - Training on {len(X_train)} rows, Testing on {len(X_test)} rows.")

    # Model Specification
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=50,
        random_state=42
    )
    
    print('Fitting model...')
    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_test, y_test)],
        sample_weight_eval_set=[w_test],
        verbose=False
    )
    
    # Evaluation
    predictions = model.predict(X_test)
    if len(y_test) > 0:
        mae = mean_absolute_error(y_test, predictions)
        print(f'✅ Model Training Complete. Test Set MAE: {mae:.4f}')
    else:
        print('✅ Model Training Complete (Evaluation skipped due to empty test set).')
    
    # Save model
    os.makedirs('state', exist_ok=True)
    model.get_booster().set_attr(baseload_feature_contract=json.dumps(baseload_feature_contract(), sort_keys=True))
    model.save_model('state/energy_model.json')
    with open('state/model_features.json', 'w') as f:
        json.dump(features, f)
    with open('state/model_training_metadata.json', 'w') as f:
        json.dump({'model_version': get_model_version(),
                   'feature_contract': baseload_feature_contract(),
                   'archive_training': df.attrs.get('archive_training'),
                   'input_mode': 'archive' if 'archive_training' in df.attrs else 'processed-frame-or-csv',
                   'eligible_rows': len(df), 'holdout_days': holdout_days,
                   'first_eligible': str(df.index.min()), 'last_eligible': str(df.index.max()),
                   'eligible_verified_targets': int(df.baseload_target_valid.astype(str).str.lower().eq('true').sum())}, f, indent=2)
    print('Model saved to energy_model.json')
    if _outcome is not None:
        _outcome['saved'] = True

def _epoch(value: str) -> float:
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected ISO timestamp with timezone') from error
    if date.tzinfo is None or date.utcoffset() is None:
        raise argparse.ArgumentTypeError('Timestamp must have a timezone')
    return date.timestamp()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout-days', type=int, default=0, help='Number of recent days to exclude from training')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--data-path', default='state/processed_data.csv', help='Processed training-data CSV path')
    source.add_argument('--archive', help='Explicit read-only measurement archive; requires --from and --to')
    parser.add_argument('--sources', help='Dated source YAML for archive mode; merges with extraction defaults')
    parser.add_argument('--measurement-policy', help='Explicit unit/kind/freshness registry for archive mode')
    parser.add_argument('--feature-chunk-seconds', type=int,
                        help='Project bounded feature chunks before building the fit matrix (archive only)')
    parser.add_argument('--from', dest='start', type=_epoch)
    parser.add_argument('--to', dest='end', type=_epoch)
    parser.add_argument('--interval-seconds', type=int, default=60)
    parser.add_argument('--allow-legacy', action='store_true', help='Read unverified legacy inputs')
    parser.add_argument('--legacy-policy', choices=LEGACY_POLICIES, default='verified-only',
                        help='Explicit target eligibility policy; source verification is never changed')
    parser.add_argument('--preferred-import', type=int, action='append', help='Explicit legacy precedence in archive mode')
    parser.add_argument('--min-coverage-days', type=float, default=14.0, help='Minimum eligible sample-days before fitting (default 14)')
    parser.add_argument('--require-success', action='store_true', help='Exit nonzero when training keeps the old model instead of saving a new one')
    args = parser.parse_args(argv)
    frame = None
    if args.archive:
        if args.start is None or args.end is None or args.start >= args.end:
            parser.error('archive mode requires increasing --from and --to timestamps')
        if args.legacy_policy != 'verified-only' and not args.allow_legacy:
            parser.error('--legacy-policy complete-legacy requires --allow-legacy')
        options: dict[str, Any] = {'sources_path': args.sources} if args.sources else {}
        if args.measurement_policy:
            options['policy_path'] = args.measurement_policy
            options['require_policies'] = True
        loader = load_archive_features
        if args.feature_chunk_seconds is not None:
            if args.feature_chunk_seconds <= 0 or args.feature_chunk_seconds % args.interval_seconds:
                parser.error('feature chunks must be positive and interval-aligned')
            options['chunk_seconds'] = args.feature_chunk_seconds
            loader = load_archive_training_features
        frame = loader(args.archive, args.start, args.end,
            interval_seconds=args.interval_seconds, allow_legacy=args.allow_legacy,
            preferred_imports=args.preferred_import, legacy_policy=args.legacy_policy, **options)
    elif (args.start is not None or args.end is not None or args.sources or args.measurement_policy or args.allow_legacy
          or args.preferred_import or args.legacy_policy != 'verified-only' or args.feature_chunk_seconds is not None):
        parser.error('source/window/legacy options require --archive')
    outcome: dict[str, bool] = {}
    train(holdout_days=args.holdout_days, data_path=args.data_path,
          min_coverage_days=args.min_coverage_days, frame=frame, _outcome=outcome)
    if args.require_success and not outcome.get('saved', False):
        raise SystemExit(1)


if __name__ == '__main__':
    main()

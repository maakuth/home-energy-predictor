from __future__ import annotations

import argparse
import os

import pandas as pd

from utils.feature_processing import (
    diagnose_feature_coverage,
    check_feature_coverage,
    feature_coverage,
    process_frame,
)


def process_data(input_path: str = 'state/raw_data.csv', output_path: str = 'state/processed_data.csv') -> None:
    print('Loading raw data...')
    try:
        raw = pd.read_csv(input_path, index_col=0, low_memory=False)
    except FileNotFoundError:
        print(f'Error: {input_path} not found.')
        return
    df = process_frame(raw)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path)
    print(f'✅ Processing complete. Saved to {output_path}. Shape: {df.shape}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process extracted energy data.')
    parser.add_argument('--input', default='state/raw_data.csv', help='Input raw-data CSV path.')
    parser.add_argument('--output', default='state/processed_data.csv', help='Output processed-data CSV path.')
    args = parser.parse_args()
    process_data(args.input, args.output)

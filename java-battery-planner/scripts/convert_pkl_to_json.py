#!/usr/bin/env python3
"""Convert battery planner .pkl fixtures to .json for Java test suite."""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path


def convert_value(obj):
    """Convert pickle values to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: convert_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_value(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, float):
        if obj != obj:  # NaN check
            return None
        if obj == float('inf'):
            return None
        if obj == float('-inf'):
            return None
        return obj
    if isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    if isinstance(obj, set):
        return list(obj)
    return obj


def convert_pkl_to_json(pkl_path: Path, json_path: Path, verbose: bool = False) -> bool:
    """Convert a single .pkl fixture to .json."""
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            print(f"  Skipping {pkl_path.name}: not a dict (got {type(data).__name__})")
            return False

        json_data = convert_value(data)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        if verbose:
            meta = data.get('metadata', {})
            print(f"  {pkl_path.name} -> {json_path.name}"
                  f"  ({len(data.get('predictions', []))} predictions,"
                  f" {len(data.get('market_prices', []))} prices,"
                  f" period={meta.get('period_start','?')[:10]})")
        return True
    except Exception as e:
        print(f"  Error converting {pkl_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Convert .pkl fixtures to .json')
    parser.add_argument('--input-dir', default='../tests/fixtures',
                        help='Directory containing .pkl files')
    parser.add_argument('--output-dir', default='fixtures',
                        help='Directory for output .json files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show per-file details')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    pkl_files = sorted(input_dir.glob('*.pkl'))
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(pkl_files)} fixture(s)...")
    success = 0
    for pkl_path in pkl_files:
        json_path = output_dir / pkl_path.with_suffix('.json').name
        if convert_pkl_to_json(pkl_path, json_path, verbose=args.verbose):
            success += 1

    print(f"Done: {success}/{len(pkl_files)} converted.")
    if success == 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

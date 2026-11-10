"""Read-only chunked archive profiling; no model fits or operational writes."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time
import tracemalloc
from typing import Any

from utils.archive_features import load_archive_raw
from utils.archive_history import aware_epoch
from utils.measurement_reader import MeasurementReader, interval_index


def profile_archive(path: str | Path, start: float, end: float, *,
                    measurements: list[str], interval_seconds: int = 60,
                    chunk_seconds: int = 86400, **options: Any) -> dict[str, Any]:
    interval_index(start, end, interval_seconds)
    if not measurements or chunk_seconds <= 0 or chunk_seconds % interval_seconds:
        raise ValueError('Select measurements and an aligned positive chunk size')
    with MeasurementReader(path) as reader:
        plans = {
            'legacy': [tuple(row) for row in reader.db.execute('''EXPLAIN QUERY PLAN
                SELECT timestamp,value FROM legacy_intervals WHERE import_id=?
                AND timestamp>=? AND timestamp<? AND measurement=?''',
                (1, start, end, measurements[0]))],
            'recorder': [tuple(row) for row in reader.db.execute('''EXPLAIN QUERY PLAN
                SELECT timestamp,state FROM observations WHERE source_id=? AND timestamp>=?
                AND timestamp<? ORDER BY timestamp,recorder_id''', (1, start, end))],
        }
    rows = max_rows = max_frame_bytes = chunks = 0
    tracing_owned = not tracemalloc.is_tracing()
    if tracing_owned:
        tracemalloc.start()
    started = time.perf_counter()
    try:
        left = start
        while left < end:
            right = min(end, left + chunk_seconds)
            frame = load_archive_raw(path, left, right, measurements=measurements,
                interval_seconds=interval_seconds, chunk_seconds=chunk_seconds, **options)
            rows += len(frame)
            max_rows = max(max_rows, len(frame))
            max_frame_bytes = max(max_frame_bytes, int(frame.memory_usage(index=True, deep=True).sum()))
            chunks += 1
            del frame
            left = right
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if tracing_owned:
            tracemalloc.stop()
    elapsed = time.perf_counter() - started
    return {
        'archive_path': str(Path(path).resolve()), 'start': start, 'end': end,
        'interval_seconds': interval_seconds, 'chunk_seconds': chunk_seconds,
        'measurements': measurements, 'rows': rows, 'chunks': chunks,
        'max_chunk_rows': max_rows, 'max_chunk_dataframe_bytes': max_frame_bytes,
        'elapsed_seconds': elapsed, 'rows_per_second': rows / elapsed if elapsed else None,
        'python_tracemalloc_peak_bytes': peak,
        'process_max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'index_plans': plans, 'training_frame_bounded': False,
        'note': 'Each raw chunk snapshot closes before the next. No frames are concatenated here. '
                'RSS is process high-water, not allocation attributable only to this call. '
                'Tracemalloc does not include every native allocation. Feature/training loaders '
                'still materialize their requested output; this raw profile does not certify bounded fitting memory.',
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--from', dest='start', required=True, type=aware_epoch)
    parser.add_argument('--to', dest='end', required=True, type=aware_epoch)
    parser.add_argument('--measurement', action='append', required=True)
    parser.add_argument('--sources')
    parser.add_argument('--measurement-policy')
    parser.add_argument('--allow-legacy', action='store_true')
    parser.add_argument('--interval-seconds', type=int, default=60)
    parser.add_argument('--chunk-seconds', type=int, default=86400)
    args = parser.parse_args(argv)
    options: dict[str, Any] = {'allow_legacy': args.allow_legacy}
    if args.sources:
        options['sources_path'] = args.sources
    if args.measurement_policy:
        options['policy_path'] = args.measurement_policy
    if not all(math.isfinite(bound) for bound in (args.start, args.end)):
        parser.error('Bounds must be finite')
    report = profile_archive(args.archive, args.start, args.end, measurements=args.measurement,
        interval_seconds=args.interval_seconds, chunk_seconds=args.chunk_seconds, **options)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

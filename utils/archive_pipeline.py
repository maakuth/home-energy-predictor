"""Explicit archive rollout orchestration. Dry-run by default, never CSV fallback."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from typing import Any

from utils.archive_features import BASELOAD_INPUTS, default_sources
from utils.archive_history import ArchiveHistory, add_history_arguments, aware_epoch, history_from_arguments
from utils.measurement_policy import load_policies, require_policy_entries
from utils.measurement_sources import load_sources
from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract

_SOURCES = Path(__file__).resolve().parents[1] / 'config' / 'measurement_sources.yaml'


def _iso(seconds: float) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def _history_flags(history: ArchiveHistory, *, window_start: float | None = None) -> list[str]:
    flags = ['--archive', str(history.path), '--sources', str(history.sources_path or _SOURCES),
             '--legacy-policy', history.legacy_policy]
    if window_start is None:
        flags += ['--cutoff', _iso(history.cutoff)]
    else:
        flags += ['--from', _iso(window_start), '--to', _iso(history.cutoff)]
    if history.allow_legacy:
        flags += ['--allow-legacy']
    for import_id in history.preferred_imports:
        flags += ['--preferred-import', str(import_id)]
    if history.policy_path is not None:
        flags += ['--measurement-policy', str(history.policy_path)]
    return flags


def _required_measurements() -> list[str]:
    names = [*BASELOAD_INPUTS, 'sauna_temp']
    if 'accumulator_temp' in default_sources():
        names.append('accumulator_temp')
    return names


def build_plan(mode: str, history: ArchiveHistory, *, bootstrap_start: float,
               recorder_namespace: str, approve_publish: bool = False,
               training_days: int = 730, sarima_days: int = 30,
               analysis_days: int = 14) -> list[dict[str, Any]]:
    if mode not in {'frequent', 'weekly'}:
        raise ValueError('Archive pipeline mode must be frequent or weekly')
    if (not math.isfinite(bootstrap_start) or bootstrap_start % 60
            or bootstrap_start >= history.cutoff or not recorder_namespace.strip()):
        raise ValueError('Require a minute-aligned bootstrap floor and explicit recorder namespace')
    if any(not isinstance(days, int) or days <= 0 for days in (training_days, sarima_days, analysis_days)):
        raise ValueError('Training/evaluation day windows must be positive integers')
    required = _required_measurements()
    ingest = [sys.executable, '-m', 'utils.measurement_ingest_cli', '--archive', str(history.path),
              '--sources', str(history.sources_path or _SOURCES), '--include-default-sources',
              '--from', _iso(bootstrap_start), '--to', _iso(history.cutoff),
              '--recorder-namespace', recorder_namespace, '--overlap-seconds', '900']
    for name in required:
        ingest += ['--measurement', name]
    plan: list[dict[str, Any]] = [{'name': 'incremental-ingest', 'argv': ingest}]
    if mode == 'weekly':
        plan += [
            {'name': 'train-xgboost', 'argv': [sys.executable, 'train_model.py',
                '--require-success', '--feature-chunk-seconds', '86400',
                *_history_flags(history, window_start=history.cutoff - training_days * 86400)]},
            {'name': 'train-sarima', 'argv': [sys.executable, 'train_sarima.py',
                '--require-success', *_history_flags(history, window_start=history.cutoff - sarima_days * 86400)]},
        ]
    plan += [
        {'name': 'predict', 'argv': [sys.executable, 'predict_future.py', *_history_flags(history)]},
        {'name': 'optimize', 'argv': [sys.executable, 'optimize_plan.py', *_history_flags(history)]},
    ]
    if approve_publish:
        plan.append({'name': 'publish', 'argv': [sys.executable, 'push_to_ha.py']})
    if mode == 'weekly':
        plan += [
            {'name': 'evaluate-performance', 'argv': [sys.executable, 'analyze_performance.py',
                '--days', str(analysis_days), '--backtest', *_history_flags(history)]},
            {'name': 'evaluate-evolution', 'argv': [sys.executable, 'analyze_evolution.py',
                '--days', str(analysis_days), *_history_flags(history)]},
        ]
    return plan


def _verify_context(metadata: dict[str, Any], history: ArchiveHistory) -> None:
    if (metadata.get('history_cutoff', metadata.get('end')) != history.cutoff
            or metadata.get('archive_path') != str(Path(history.path).resolve())
            or metadata.get('legacy_policy') != history.legacy_policy):
        raise ValueError('Artifact metadata does not match the current archive/cutoff/policy; refusing stale output')
    for key, expected in (('allow_legacy', history.allow_legacy), ('preferred_imports', list(history.preferred_imports))):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f'Artifact metadata mismatch for {key}')
    for key, expected in (('sources_path', history.sources_path or _SOURCES),
                          ('policy_path', history.policy_path), ('measurement_policy_path', history.policy_path)):
        if metadata.get(key) is not None and (expected is None or Path(metadata[key]).resolve() != Path(expected).resolve()):
            raise ValueError(f'Artifact metadata mismatch for {key}')
    if metadata.get('policy_sha256') is not None and history.policy_path is not None:
        if metadata['policy_sha256'] != hashlib.sha256(Path(history.policy_path).read_bytes()).hexdigest():
            raise ValueError('Artifact policy hash mismatch')


def _configuration_hashes(history: ArchiveHistory) -> dict[str, str]:
    paths = {'sources': Path(history.sources_path or _SOURCES)}
    if history.policy_path is not None:
        paths['policy'] = Path(history.policy_path)
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _verify_output(step: str, history: ArchiveHistory) -> None:
    if step == 'train-xgboost':
        metadata = json.loads(Path('state/model_training_metadata.json').read_text())
        if (metadata.get('feature_contract') != baseload_feature_contract()
                or json.loads(Path('state/model_features.json').read_text()) != list(BASELOAD_FEATURES)):
            raise ValueError('Incompatible baseload feature contract; retrain before archive rollout')
        _verify_context(metadata.get('archive_training') or {}, history)
        if not Path('state/energy_model.json').is_file():
            raise ValueError('Missing trained XGBoost model artifact')
    elif step == 'train-sarima':
        # Local model parameters are an existing trusted pickle format, not input
        # supplied by an archive or downloaded by this command.
        with Path('state/sarima_model_params.pkl').open('rb') as stream:
            metadata = pickle.load(stream)
        _verify_context(metadata.get('training_metadata', {}).get('archive_training') or {}, history)
    elif step in {'predict', 'optimize'}:
        filename = 'future_predictions' if step == 'predict' else 'optimization_plan'
        variable = 'TEST_PREDICTIONS_FILE' if step == 'predict' else 'TEST_PLAN_FILE'
        artifact = Path(os.getenv(variable, f'state/{filename}.json'))
        metadata = json.loads(artifact.with_name(f'{filename}_metadata.json').read_text())
        _verify_context(metadata, history)
        data = json.loads(artifact.read_text())
        if not data:
            raise ValueError(f'Empty {filename} artifact; refusing to continue')


def _reset_net_metering(cutoff: float) -> None:
    """Preserve frequent-job quarter housekeeping, only under execution lock."""
    state = Path(os.getenv('TEST_TMP_DIR', 'state'))
    state.mkdir(parents=True, exist_ok=True)
    track = state / 'hepo_net_metering_interval'
    interval = int(cutoff // 900)
    if not track.exists() or int(track.read_text().strip()) < interval:
        for path, content in ((state / 'net_metering_state.json', '{}'), (track, str(interval))):
            temporary = path.with_suffix(path.suffix + '.archive-pipeline.tmp')
            temporary.write_text(content)
            temporary.replace(path)


def execute_plan(plan: list[dict[str, Any]], history: ArchiveHistory, *, mode: str,
                 lock_path: str | Path = '/tmp/hepo.lock') -> None:
    """Operational execution only after explicit opt-in. No implicit retries/routes."""
    if not any(step['name'] == 'publish' for step in plan):
        raise ValueError('Full pipeline execution requires rollout approval: active plan files can actuate without HA publication')
    configuration = _configuration_hashes(history)
    with Path(lock_path).open('a') as lock:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if mode == 'frequent' else 0)
        fcntl.flock(lock, flags)
        if mode == 'frequent':
            _reset_net_metering(history.cutoff)
        for step in plan:
            if _configuration_hashes(history) != configuration:
                raise ValueError('Source/policy configuration changed during pipeline run')
            subprocess.run(step['argv'], check=True)
            if _configuration_hashes(history) != configuration:
                raise ValueError('Source/policy configuration changed during pipeline run')
            _verify_output(step['name'], history)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_history_arguments(parser)
    parser.add_argument('--mode', choices=('frequent', 'weekly'), required=True)
    parser.add_argument('--bootstrap-from', type=aware_epoch, required=True,
                        help='Explicit lower recorder query bound; successful old ranges are not re-fetched')
    parser.add_argument('--recorder-namespace', required=True)
    parser.add_argument('--execute', action='store_true', help='Actually ingest, fit, predict and optimize')
    parser.add_argument('--approve-publish', action='store_true',
                        help='Approve active model/plan writes and HA publication; mandatory for full execution')
    parser.add_argument('--training-days', type=int, default=730)
    parser.add_argument('--sarima-days', type=int, default=30)
    parser.add_argument('--analysis-days', type=int, default=14)
    parser.add_argument('--lock-path', default='/tmp/hepo.lock')
    args = parser.parse_args(argv)
    try:
        if not args.archive:
            raise ValueError('Explicit --archive is required; use existing no-argument scripts for CSV rollback')
        if args.cutoff is None:
            args.cutoff = math.floor(time.time() / 60) * 60
        if args.execute:
            if not args.approve_publish:
                raise ValueError('--execute requires --approve-publish: writing an active plan can actuate through run_often.py')
            if not args.sources or not args.measurement_policy:
                raise ValueError('Execution requires explicit --sources and --measurement-policy evidence configuration')
            from dotenv import load_dotenv
            load_dotenv(override=True)
        history = history_from_arguments(args)
        if history is None:
            raise ValueError('Missing archive request')
        if args.execute:
            registry = load_policies(args.measurement_policy)
            require_policy_entries(registry, _required_measurements())
            configured = {**default_sources(), **load_sources(args.sources)}
            if any(name not in configured for name in _required_measurements()):
                raise ValueError('Missing source mappings for required archive consumers')
        plan = build_plan(args.mode, history, bootstrap_start=args.bootstrap_from,
            recorder_namespace=args.recorder_namespace, approve_publish=args.approve_publish,
            training_days=args.training_days, sarima_days=args.sarima_days, analysis_days=args.analysis_days)
        if not args.execute:
            print(json.dumps({'dry_run': True, 'migration_ready': False,
                'cutoff': _iso(history.cutoff), 'will_publish_if_executed': args.approve_publish,
                'execution_requires_rollout_approval': True,
                'commands': plan}, indent=2))
            return
        execute_plan(plan, history, mode=args.mode, lock_path=args.lock_path)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Archive pipeline stopped: {error}\n')


if __name__ == '__main__':
    main()

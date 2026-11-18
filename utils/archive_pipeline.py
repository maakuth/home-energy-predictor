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
import subprocess
import sys
import time
from typing import Any

from utils.archive_features import BASELOAD_INPUTS, default_sources
from utils.archive_history import ArchiveHistory, add_history_arguments, aware_epoch, history_from_arguments
from utils.measurement_policy import load_policies, require_policy_entries
from utils.measurement_sources import load_sources
from utils.baseload_quality import BASELOAD_FEATURES, baseload_feature_contract
from utils.degraded_plan import write_degraded_plan

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
               training_days: int = 730,
               analysis_days: int = 14, model_dir: str | Path | None = None) -> list[dict[str, Any]]:
    if mode not in {'frequent', 'weekly'}:
        raise ValueError('Archive pipeline mode must be frequent or weekly')
    if model_dir is not None and mode != 'frequent':
        raise ValueError('Separate model directory is not supported for weekly training')
    if model_dir is not None and not Path(model_dir).is_absolute():
        raise ValueError('Archive model directory must be absolute')
    if (not math.isfinite(bootstrap_start) or bootstrap_start % 60
            or bootstrap_start >= history.cutoff or not recorder_namespace.strip()):
        raise ValueError('Require a minute-aligned bootstrap floor and explicit recorder namespace')
    if any(not isinstance(days, int) or days <= 0 for days in (training_days, analysis_days)):
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
        ]
    if mode == 'frequent':
        plan += [
            {'name': 'predict', 'argv': [sys.executable, 'predict_future.py', *_history_flags(history),
                                        *(['--model-dir', str(model_dir)] if model_dir is not None else [])]},
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


def _verify_context(metadata: Any, history: ArchiveHistory) -> None:
    if not isinstance(metadata, dict):
        raise ValueError('Artifact metadata must be a JSON object')
    if 'allow_legacy' in metadata and not isinstance(metadata['allow_legacy'], bool):
        raise ValueError('Artifact metadata allow_legacy must be boolean')
    if 'preferred_imports' in metadata and not isinstance(metadata['preferred_imports'], list):
        raise ValueError('Artifact metadata preferred_imports must be an array')
    if metadata.get('policy_sha256') is not None and not isinstance(metadata['policy_sha256'], str):
        raise ValueError('Artifact metadata policy_sha256 must be a string or null')
    if (metadata.get('history_cutoff', metadata.get('end')) != history.cutoff
            or metadata.get('archive_path') != str(Path(history.path).resolve())
            or metadata.get('legacy_policy') != history.legacy_policy):
        raise ValueError('Artifact metadata does not match the current archive/cutoff/policy; refusing stale output')
    for key, expected in (('allow_legacy', history.allow_legacy), ('preferred_imports', list(history.preferred_imports))):
        if key in metadata and metadata[key] != expected:
            raise ValueError(f'Artifact metadata mismatch for {key}')
    for key, expected in (('sources_path', history.sources_path or _SOURCES),
                          ('policy_path', history.policy_path), ('measurement_policy_path', history.policy_path)):
        value = metadata.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(f'Artifact metadata {key} must be a path string or null')
        if value is not None and (expected is None or Path(value).resolve() != Path(expected).resolve()):
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
                 lock_path: str | Path = '/tmp/hepo.lock',
                 writer_lock_path: str | Path | None = None,
                 approve_model_write: bool = False) -> None:
    """Operational execution only after explicit opt-in. No implicit retries/routes."""
    if mode == 'frequent' and not any(step['name'] == 'publish' for step in plan):
        raise ValueError('Full pipeline execution requires rollout approval: active plan files can actuate without HA publication')
    if mode == 'weekly' and (not approve_model_write or not any(step['name'] == 'train-xgboost' for step in plan)):
        raise ValueError('Weekly archive training requires explicit active-model-write approval')
    if mode == 'weekly' and any(step['name'] in {'predict', 'optimize', 'publish'} for step in plan):
        raise ValueError('Weekly archive pipeline must not predict, optimize or publish a plan')
    if mode not in {'frequent', 'weekly'}:
        raise ValueError('Unknown pipeline mode')
    if writer_lock_path is None or not Path(writer_lock_path).is_absolute():
        raise ValueError('Execution requires an absolute shared archive writer lock path')
    writer_path = Path(writer_lock_path).resolve()
    archive_path = Path(history.path).resolve()
    protected = {Path(lock_path).resolve(), *(Path(str(archive_path) + suffix)
                 for suffix in ('', '-wal', '-shm', '-journal'))}
    protected.update(Path(path).resolve() for path in (history.sources_path, history.policy_path) if path is not None)
    if writer_path in protected or writer_path.name == 'hepo.db':
        raise ValueError('Writer lock must not replace the pipeline lock or an input artifact')
    if not plan or plan[0]['name'] != 'incremental-ingest' or any(
            step['name'] == 'incremental-ingest' for step in plan[1:]):
        raise ValueError('Pipeline requires exactly one initial ingestion step')
    configuration = _configuration_hashes(history)
    with Path(lock_path).open('a') as lock:
        flags = fcntl.LOCK_EX | (fcntl.LOCK_NB if mode == 'frequent' else 0)
        fcntl.flock(lock, flags)

        def publish(argv: list[str]) -> None:
            env = dict(os.environ, HEPO_PUBLISH_LOCK_FD=str(lock.fileno()),
                       HEPO_PIPELINE_LOCK_PATH=str(Path(lock_path).resolve()))
            subprocess.run(argv, check=True, env=env, pass_fds=(lock.fileno(),))

        def run_step(step: dict[str, Any]) -> None:
            if _configuration_hashes(history) != configuration:
                raise ValueError('Source/policy configuration changed during pipeline run')
            if step['name'] == 'publish':
                publish(step['argv'])
            else:
                subprocess.run(step['argv'], check=True)
            if _configuration_hashes(history) != configuration:
                raise ValueError('Source/policy configuration changed during pipeline run')
            _verify_output(step['name'], history)

        if mode == 'frequent':
            _reset_net_metering(history.cutoff)
        with writer_path.open('a') as writer:
            # Match the collector's nonblocking shared staging lock. Do not hold
            # it during training, prediction or optimization.
            try:
                fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if mode != 'frequent':
                    raise
                print('⚠️ Archive writer busy; skipping ingestion and attempting prediction '
                      'from the existing archive. Input eligibility remains enforced.', flush=True)
            else:
                try:
                    run_step(plan[0])
                except subprocess.CalledProcessError:
                    if mode != 'frequent':
                        raise
                    if _configuration_hashes(history) != configuration:
                        raise ValueError('Source/policy configuration changed during pipeline run')
                    print('⚠️ Ingestion failed; skipping ingestion and attempting prediction '
                          'from the existing archive. Input eligibility remains enforced.', flush=True)
        for step in plan[1:]:
            try:
                run_step(step)
            except (subprocess.CalledProcessError, OSError, ValueError) as error:
                if mode != 'frequent' or step['name'] not in {'predict', 'optimize'}:
                    raise
                if _configuration_hashes(history) != configuration:
                    raise ValueError('Source/policy configuration changed during pipeline run') from error
                reason = f'{step["name"]} failed: {type(error).__name__}: {error}'
                print(f'⚠️ {reason}; installing local degraded idle plan.', flush=True)
                write_degraded_plan(history, step['name'], reason)
                # Publish only unavailability markers, not degraded numeric data.
                publish([sys.executable, 'push_to_ha.py', '--plan-only'])
                return


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_history_arguments(parser)
    parser.add_argument('--mode', choices=('frequent', 'weekly'), required=True)
    parser.add_argument('--bootstrap-from', type=aware_epoch, required=True,
                        help='Explicit lower recorder query bound; successful old ranges are not re-fetched')
    parser.add_argument('--recorder-namespace', required=True)
    parser.add_argument('--execute', action='store_true', help='Actually ingest, fit, predict and optimize')
    parser.add_argument('--approve-publish', action='store_true',
                        help='Approve active model writes or plan writes/publication; mandatory for execution')
    parser.add_argument('--training-days', type=int, default=730)
    parser.add_argument('--analysis-days', type=int, default=14)
    parser.add_argument('--lock-path', default='/tmp/hepo.lock')
    parser.add_argument('--writer-lock', help='Absolute shared archive writer lock used by the collector')
    parser.add_argument('--model-dir', help='Separate absolute archive model directory (frequent mode only)')
    args = parser.parse_args(argv)
    try:
        if not args.archive:
            raise ValueError('Explicit --archive is required; use existing no-argument scripts for CSV rollback')
        if args.cutoff is None:
            args.cutoff = math.floor(time.time() / 60) * 60
        if args.execute:
            if not args.approve_publish:
                raise ValueError('--execute requires --approve-publish: weekly training writes active models and frequent plans can actuate through run_often.py')
            if not args.sources or not args.measurement_policy:
                raise ValueError('Execution requires explicit --sources and --measurement-policy evidence configuration')
            if not args.writer_lock or not Path(args.writer_lock).is_absolute():
                raise ValueError('Execution requires an absolute --writer-lock shared with the collector')
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
            training_days=args.training_days, analysis_days=args.analysis_days, model_dir=args.model_dir)
        if not args.execute:
            policy_preflight = None
            if args.measurement_policy:
                required = _required_measurements()
                registry = load_policies(args.measurement_policy)
                missing = sorted(set(required) - set(registry))
                policy_preflight = {'required_measurements': required,
                                    'missing_measurements': missing,
                                    'complete': not missing,
                                    'note': 'Uses current process configuration, not the execution dotenv; entry presence is not policy approval or data coverage.'}
            print(json.dumps({'dry_run': True, 'migration_ready': False,
                'cutoff': _iso(history.cutoff), 'will_publish_if_executed': args.approve_publish and args.mode == 'frequent',
                'execution_requires_rollout_approval': True,
                'writer_lock': args.writer_lock,
                'policy_preflight': policy_preflight,
                'commands': plan}, indent=2))
            return
        execute_plan(plan, history, mode=args.mode, lock_path=args.lock_path,
                     writer_lock_path=args.writer_lock, approve_model_write=args.approve_publish)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Archive pipeline stopped: {error}\n')


if __name__ == '__main__':
    main()

from __future__ import annotations

"""Run offline full-fixture planner replays and write a structured report.

This deliberately imports the replay harness rather than operational scripts:
it neither opens Home Assistant nor reads production databases or ``.env``.
"""

import argparse
from pathlib import Path
import sys
from typing import Any

# ``python utils/run_quality_gate.py`` puts only ``utils`` on sys.path.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from battery_planners import BatteryPlannerFactory
from tests.battery_planner_replay import BatteryReplaySimulator, load_fixture
from utils.git_utils import get_model_version
from utils.quality_gates import (
    QUALITY_REPORT_SCHEMA_VERSION,
    compare_quality_reports,
    fixture_digest,
    load_fixture_manifest,
    verify_fixture_manifest,
    write_json,
)


def _safety(result: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    if not result.get('success'):
        violations.append(str(result.get('error', 'replay failed')))
    if result.get('soc_violations', 0):
        violations.append(f"soc_violations={result['soc_violations']}")
    return {'ok': not violations, 'violations': violations}


def build_report(manifest_path: Path, planner_name: str) -> dict[str, Any]:
    manifest = load_fixture_manifest(manifest_path)
    fixture_dir = manifest_path.parent
    errors = verify_fixture_manifest(manifest, fixture_dir)
    if errors:
        raise ValueError('; '.join(errors))

    results: list[dict[str, Any]] = []
    for spec in manifest['fixtures']:
        if not spec.get('included', False):
            continue
        if spec.get('planner', planner_name) != planner_name:
            continue
        fixture_path = fixture_dir / spec['name']
        simulator = BatteryReplaySimulator(load_fixture(str(fixture_path)))
        if simulator.measurements_df is None:
            raise ValueError(f"{spec['name']}: no measurements")
        replay = simulator.simulate_battery_control(
            planner=BatteryPlannerFactory.create(planner_name),
            planner_type=planner_name,
            battery_capacity_kwh=50.0,
            battery_min_soc_pct=10.0,
            battery_max_soc_pct=90.0,
            battery_initial_soc_pct=10.0,
            max_planks=len(simulator.measurements_df),
        )
        if not replay.get('complete'):
            raise ValueError(f"{spec['name']}: replay did not cover the full fixture")
        results.append({
            'fixture': spec['name'],
            'fixture_digest': fixture_digest(fixture_path),
            'planner': planner_name,
            'complete': bool(replay.get('complete')),
            'intervals_attempted': replay.get('intervals_attempted', 0),
            'intervals_run': replay.get('intervals_run', 0),
            'safety': _safety(replay),
            'cost_with_battery_eur': replay.get('cost_with_battery_eur'),
            'cost_no_battery_eur': replay.get('cost_no_battery_eur'),
            'delta_vs_no_battery_eur': (
                None if replay.get('cost_with_battery_eur') is None else
                replay['cost_with_battery_eur'] - replay.get('cost_no_battery_eur', 0.0)
            ),
            'imported_kwh': replay.get('imported_kwh'),
            'exported_kwh': replay.get('exported_kwh'),
            'final_soc_pct': replay.get('final_soc_pct'),
            'action_counts': replay.get('action_counts', {}),
            # Archived forecasts are fixed replay inputs, not a candidate model.
            'forecast_evaluation': {'available': False, 'reason': 'archived replay forecasts'},
        })
    if not results:
        raise ValueError(f'No included fixtures are configured for planner {planner_name!r}')
    return {
        'schema_version': QUALITY_REPORT_SCHEMA_VERSION,
        'evaluator_version': '1',
        'candidate': {'planner': planner_name, 'version': get_model_version()},
        'manifest': manifest_path.name,
        'results': results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Offline model/planner quality replay')
    parser.add_argument('--manifest', type=Path, default=Path('tests/fixtures/quality_manifest.json'))
    parser.add_argument('--planner', default='nemotron-linprog', choices=BatteryPlannerFactory.names())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Approved report to compare (report-only if omitted)')
    args = parser.parse_args()

    report = build_report(args.manifest, args.planner)
    if args.reference:
        import json
        reference = json.loads(args.reference.read_text(encoding='utf-8'))
        report['comparison'] = compare_quality_reports(report, reference, load_fixture_manifest(args.manifest))
    write_json(args.output, report)
    return 1 if args.reference and not report['comparison']['passed'] else 0


if __name__ == '__main__':
    raise SystemExit(main())

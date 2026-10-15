from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from utils.quality_gates import (
    QUALITY_REPORT_SCHEMA_VERSION,
    compare_quality_reports,
    fixture_digest,
    load_fixture_manifest,
    verify_fixture_manifest,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding='utf-8')


def _report(cost: float, *, safety_ok: bool = True, fixture_digest_value: str = 'abc') -> dict:
    return {
        'schema_version': QUALITY_REPORT_SCHEMA_VERSION,
        'evaluator_version': '1',
        'candidate': {'planner': 'heuristic', 'version': '1.34.0'},
        'results': [{
            'fixture': 'jan.pkl',
            'fixture_digest': fixture_digest_value,
            'planner': 'heuristic',
            'complete': True,
            'safety': {'ok': safety_ok, 'violations': []},
            'cost_with_battery_eur': cost,
        }],
    }


def test_manifest_verification_detects_fixture_change(tmp_path: Path):
    fixture = tmp_path / 'jan.pkl'
    fixture.write_bytes(b'fixture v1')
    manifest_path = tmp_path / 'manifest.json'
    _write_json(manifest_path, {
        'schema_version': 1,
        'fixtures': [{
            'name': 'jan.pkl',
            'sha256': fixture_digest(fixture),
            'included': True,
            'weight': 1.0,
        }],
    })

    manifest = load_fixture_manifest(manifest_path)
    assert verify_fixture_manifest(manifest, tmp_path) == []

    fixture.write_bytes(b'fixture v2')
    errors = verify_fixture_manifest(manifest, tmp_path)
    assert errors == ['jan.pkl: SHA-256 mismatch']


def test_comparison_rejects_cost_regression_beyond_fixture_tolerance():
    baseline = _report(10.0)
    candidate = _report(10.06)
    manifest = {
        'fixtures': [{'name': 'jan.pkl', 'included': True, 'weight': 1.0,
                      'tolerance_eur': 0.05}],
        'portfolio_tolerance_eur': 0.10,
    }

    comparison = compare_quality_reports(candidate, baseline, manifest)

    assert not comparison['passed']
    assert comparison['fixtures'][0]['economic_passed'] is False
    assert 'cost regressed by 0.060000 EUR' in comparison['failures'][0]


def test_portfolio_cannot_mask_a_fixture_hard_gate_failure():
    baseline = _report(10.0)
    candidate = _report(9.0, safety_ok=False)
    manifest = {
        'fixtures': [{'name': 'jan.pkl', 'included': True, 'weight': 1.0,
                      'tolerance_eur': 1.0}],
        'portfolio_tolerance_eur': 1.0,
    }

    comparison = compare_quality_reports(candidate, baseline, manifest)

    assert comparison['portfolio']['passed'] is True
    assert comparison['passed'] is False
    assert comparison['fixtures'][0]['safety_passed'] is False


def test_comparison_rejects_incompatible_fixture_digest():
    manifest = {'fixtures': [{'name': 'jan.pkl', 'included': True, 'weight': 1.0}],
                'portfolio_tolerance_eur': 0.0}

    comparison = compare_quality_reports(_report(9.0), _report(10.0, fixture_digest_value='other'), manifest)

    assert not comparison['passed']
    assert comparison['fixtures'][0]['comparable'] is False
    assert 'fixture digest differs' in comparison['failures'][0]


def test_comparison_rejects_different_evaluator_versions():
    candidate = _report(9.0)
    candidate['evaluator_version'] = '2'
    manifest = {'fixtures': [{'name': 'jan.pkl', 'included': True, 'weight': 1.0}],
                'portfolio_tolerance_eur': 0.0}

    comparison = compare_quality_reports(candidate, _report(10.0), manifest)

    assert not comparison['passed']
    assert comparison['failures'] == ['evaluator version differs']


def test_repository_manifest_matches_selected_fixtures():
    root = Path(__file__).parent / 'fixtures'
    manifest = load_fixture_manifest(root / 'quality_manifest.json')

    assert verify_fixture_manifest(manifest, root) == []


def test_comparison_rejects_short_horizon_forecast_degradation():
    baseline = _report(10.0)
    candidate = _report(10.0)
    baseline['results'][0]['forecast_evaluation'] = {
        'available': True, 'lead_time_bins': {'<1h': {'mae_kw': 0.2, 'bias_kw': 0.0}},
    }
    candidate['results'][0]['forecast_evaluation'] = {
        'available': True, 'lead_time_bins': {'<1h': {'mae_kw': 0.4, 'bias_kw': 0.0}},
    }
    manifest = {
        'fixtures': [{'name': 'jan.pkl', 'included': True, 'weight': 1.0,
                      'forecast_tolerances': {'<1h': {'mae_kw': 0.05, 'bias_kw': 0.05}}}],
        'portfolio_tolerance_eur': 0.0,
    }

    comparison = compare_quality_reports(candidate, baseline, manifest)

    assert not comparison['passed']
    assert 'forecast MAE' in comparison['failures'][0]

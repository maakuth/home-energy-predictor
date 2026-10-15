from __future__ import annotations

"""Deterministic validation and comparison for offline planner replays."""

from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any


QUALITY_REPORT_SCHEMA_VERSION = 1


def fixture_digest(path: str | Path) -> str:
    """Return a SHA-256 digest without loading an untrusted pickle."""
    digest = sha256()
    with Path(path).open('rb') as fixture:
        for chunk in iter(lambda: fixture.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixture_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or not isinstance(manifest.get('fixtures'), list):
        raise ValueError(f'Invalid quality fixture manifest: {path}')
    return manifest


def verify_fixture_manifest(manifest: dict[str, Any], fixture_dir: str | Path) -> list[str]:
    """Return deterministic errors for selected missing or modified fixtures."""
    errors: list[str] = []
    root = Path(fixture_dir)
    for fixture in manifest['fixtures']:
        if not fixture.get('included', False):
            continue
        name = fixture.get('name')
        if not isinstance(name, str) or Path(name).name != name:
            errors.append(f'{name!r}: invalid fixture name')
            continue
        path = root / name
        if not path.is_file():
            errors.append(f'{name}: fixture is missing')
        elif fixture_digest(path) != fixture.get('sha256'):
            errors.append(f'{name}: SHA-256 mismatch')
    return errors


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write a stable, reviewable report."""
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _result_index(report: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    results = report.get('results')
    if report.get('schema_version') != QUALITY_REPORT_SCHEMA_VERSION or not isinstance(results, list):
        raise ValueError('Invalid quality report schema')
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise ValueError('Quality report result is not an object')
        key = (str(result.get('fixture')), str(result.get('planner')))
        if key in indexed:
            raise ValueError(f'Duplicate quality result: {key}')
        indexed[key] = result
    return indexed


def _finite_cost(result: dict[str, Any]) -> float | None:
    value = result.get('cost_with_battery_eur')
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def compare_quality_reports(
    candidate: dict[str, Any], baseline: dict[str, Any], manifest: dict[str, Any],
) -> dict[str, Any]:
    """Compare matching replay records without allowing portfolio masking.

    Every selected fixture is a hard gate.  Costs are absolute EUR and may be
    negative, which makes this safe for export-heavy periods.
    """
    candidate_results = _result_index(candidate)
    baseline_results = _result_index(baseline)
    failures: list[str] = []
    fixture_results: list[dict[str, Any]] = []
    weighted_delta = 0.0
    if candidate.get('evaluator_version') != baseline.get('evaluator_version'):
        failures.append('evaluator version differs')

    for spec in manifest.get('fixtures', []):
        if not spec.get('included', False):
            continue
        name = str(spec.get('name'))
        matching_keys = [key for key in baseline_results if key[0] == name]
        if len(matching_keys) != 1:
            failures.append(f'{name}: reference must contain exactly one planner result')
            continue
        key = matching_keys[0]
        reference = baseline_results[key]
        result = candidate_results.get(key)
        item: dict[str, Any] = {'fixture': name, 'planner': key[1], 'comparable': True}
        if result is None:
            item.update(comparable=False, safety_passed=False, economic_passed=False)
            failures.append(f'{name}: candidate result is missing')
            fixture_results.append(item)
            continue
        if result.get('fixture_digest') != reference.get('fixture_digest'):
            item.update(comparable=False, safety_passed=False, economic_passed=False)
            failures.append(f'{name}: fixture digest differs')
            fixture_results.append(item)
            continue

        safety_ok = bool(result.get('complete')) and bool(result.get('safety', {}).get('ok'))
        item['safety_passed'] = safety_ok
        if not safety_ok:
            failures.append(f'{name}: candidate has replay safety or completion failure')

        candidate_cost = _finite_cost(result)
        reference_cost = _finite_cost(reference)
        tolerance = float(spec.get('tolerance_eur', 0.0))
        if candidate_cost is None or reference_cost is None:
            item.update(economic_passed=False, delta_eur=None)
            failures.append(f'{name}: non-finite candidate or reference cost')
        else:
            delta = candidate_cost - reference_cost
            economic_ok = delta <= tolerance
            item.update(candidate_cost_eur=candidate_cost, reference_cost_eur=reference_cost,
                        delta_eur=delta, tolerance_eur=tolerance, economic_passed=economic_ok)
            if not economic_ok:
                failures.append(f'{name}: cost regressed by {delta:.6f} EUR (tolerance {tolerance:.6f} EUR)')
            weighted_delta += delta * float(spec.get('weight', 1.0))

        forecast_tolerances = spec.get('forecast_tolerances', {})
        if forecast_tolerances:
            candidate_forecast = result.get('forecast_evaluation', {})
            reference_forecast = reference.get('forecast_evaluation', {})
            candidate_bins = candidate_forecast.get('lead_time_bins', {})
            reference_bins = reference_forecast.get('lead_time_bins', {})
            for bin_name, tolerances in forecast_tolerances.items():
                candidate_bin = candidate_bins.get(bin_name)
                reference_bin = reference_bins.get(bin_name)
                if candidate_bin is None or reference_bin is None:
                    failures.append(f'{name}: missing forecast metrics for {bin_name}')
                    continue
                mae_tolerance = float(tolerances.get('mae_kw', 0.0))
                bias_tolerance = float(tolerances.get('bias_kw', 0.0))
                candidate_mae = float(candidate_bin['mae_kw'])
                reference_mae = float(reference_bin['mae_kw'])
                candidate_bias = abs(float(candidate_bin['bias_kw']))
                reference_bias = abs(float(reference_bin['bias_kw']))
                if candidate_mae > reference_mae + mae_tolerance:
                    failures.append(f'{name}: forecast MAE regressed in {bin_name}')
                if candidate_bias > reference_bias + bias_tolerance:
                    failures.append(f'{name}: forecast bias regressed in {bin_name}')
        fixture_results.append(item)

    portfolio_tolerance = float(manifest.get('portfolio_tolerance_eur', 0.0))
    portfolio_passed = weighted_delta <= portfolio_tolerance
    if not portfolio_passed:
        failures.append('portfolio: weighted cost regressed by '
                        f'{weighted_delta:.6f} EUR (tolerance {portfolio_tolerance:.6f} EUR)')
    return {
        'schema_version': QUALITY_REPORT_SCHEMA_VERSION,
        'fixtures': fixture_results,
        'portfolio': {'weighted_delta_eur': weighted_delta, 'tolerance_eur': portfolio_tolerance,
                      'passed': portfolio_passed},
        'failures': failures,
        'passed': not failures,
    }

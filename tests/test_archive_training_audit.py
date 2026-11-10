import json

import pandas as pd

from utils.archive_features import BASELOAD_INPUTS
from utils.measurement_archive import MeasurementArchive
from utils.measurement_audit import audit_training_eligibility
from utils.measurement_audit_cli import main
from utils.measurement_sources import parse_sources


def test_training_audit_reports_missing_meters_without_feature_imputation(tmp_path):
    path = tmp_path / 'history.sqlite'
    start = pd.Timestamp('2026-10-21T00:00Z').timestamp()
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.gshp_power', None, start - 86400, start + 900,
                       [(1, start - 86400, '1000', None)])
    sources = parse_sources({name: [{'entity': 'sensor.' + name}] for name in BASELOAD_INPUTS})
    out = audit_training_eligibility(path, start, start + 900, sources=sources)
    assert out['eligible_targets'] == 0
    assert out['eligible_training_rows'] == 0
    assert out['inputs']['battery_power'] == 0
    assert out['inputs']['gshp_power'] == 15
    assert 'battery_power;' in next(iter(out['target_exclusion_reasons']))
    assert out['migration_ready'] is False


def test_cli_can_audit_all_training_inputs_in_one_snapshot(tmp_path, capsys):
    path = tmp_path / 'history.sqlite'
    with MeasurementArchive(path):
        pass
    sources = tmp_path / 'sources.yaml'
    sources.write_text('gshp_power:\n  - entity: sensor.gshp_power\n')
    main(['archive', '--archive', str(path), '--sources', str(sources),
          '--from', '2026-10-21T00:00Z', '--to', '2026-10-21T00:15Z',
          '--training-eligibility'])
    out = json.loads(capsys.readouterr().out)
    assert out['training_eligibility']['eligible_training_rows'] == 0
    assert out['training_eligibility']['legacy_policy'] == 'verified-only'

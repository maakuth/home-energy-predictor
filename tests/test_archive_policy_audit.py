import json

from utils.measurement_archive import MeasurementArchive
from utils.measurement_audit_cli import main


def test_numeric_audit_uses_declared_units_and_reports_verification(tmp_path, capsys):
    path = tmp_path / 'history.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.power', None, 0, 120,
                       [(1, 0., '2', '{"unit_of_measurement":"kW"}')])
    sources = tmp_path / 'sources.yaml'
    sources.write_text('gshp_power:\n  - entity: sensor.power\n')
    policy = tmp_path / 'policy.yaml'
    policy.write_text('gshp_power:\n  kind: numeric\n  input_unit: kW\n  canonical_unit: W\n  sign: 1\n  require_unit: true\n')
    main(['archive', '--archive', str(path), '--sources', str(sources),
          '--measurement-policy', str(policy), '--measurement', 'gshp_power',
          '--from', '1970-01-01T00:00:00Z', '--to', '1970-01-01T00:02:00Z',
          '--interval-seconds', '60'])
    report = json.loads(capsys.readouterr().out)
    assert report['measurements']['gshp_power']['unit_verified_bins'] == 2
    assert len(report['policy_sha256']) == 64

import pandas as pd

from utils.archive_features import load_archive_raw
from utils.measurement_archive import MeasurementArchive
from utils.measurement_policy import parse_policies
from utils.measurement_sources import parse_sources


def test_committed_predecessor_carries_only_with_evidence_and_original_expiry(tmp_path):
    path = tmp_path / 'history.sqlite'
    with MeasurementArchive(path, recorder_namespace='test-recorder') as archive:
        archive.ingest('sensor.power', None, 60, 180, [],
                       predecessor=(1, 30., '1000', '{"unit_of_measurement":"W"}'))
    sources = parse_sources({'gshp_power': [{'entity': 'sensor.power'}]})
    policy = parse_policies({'gshp_power': dict(kind='numeric', input_unit='W', canonical_unit='W',
        sign=1, require_unit=True, max_age_seconds=120)})
    out = load_archive_raw(path, 60, 180, sources=sources, measurements=['gshp_power'],
                           policies=policy, require_policies=True)
    assert out.gshp_power.iloc[0] == 1000
    assert pd.isna(out.gshp_power.iloc[1])
    assert out.gshp_power_coverage_fraction.tolist() == [1., .5]
    assert out.gshp_power_unit_verified.tolist() == [True, False]


def test_source_switch_does_not_promote_old_predecessor_across_mapping_boundary(tmp_path):
    path = tmp_path / 'history.sqlite'
    with MeasurementArchive(path) as archive:
        archive.ingest('sensor.new', None, 60, 120, [], predecessor=(1, 0., '1000', None))
    sources = parse_sources({'gshp_power': [{'entity': 'sensor.new', 'from': '1970-01-01T00:01:00Z'}]})
    out = load_archive_raw(path, 60, 120, sources=sources, measurements=['gshp_power'])
    assert out.gshp_power.isna().all()
    assert not out.gshp_power_source_valid.any()

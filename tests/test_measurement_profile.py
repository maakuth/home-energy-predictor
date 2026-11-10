from unittest.mock import patch

import pandas as pd

from utils.measurement_profile import profile_archive


def test_profile_releases_each_chunk_and_reports_index_plan_without_writes(tmp_path):
    from utils.measurement_archive import MeasurementArchive
    path = tmp_path / 'history.sqlite'
    with MeasurementArchive(path):
        pass
    before = path.read_bytes()
    calls = []

    def load(path, start, end, **options):
        calls.append((start, end))
        frame = pd.DataFrame({'gshp_power': 1000., 'gshp_power_source_valid': True},
                             index=pd.date_range(pd.Timestamp(start, unit='s', tz='UTC'),
                                 pd.Timestamp(end, unit='s', tz='UTC'), freq='min', inclusive='left'))
        return frame

    with patch('utils.measurement_profile.load_archive_raw', side_effect=load):
        report = profile_archive(path, 0., 7200., measurements=['gshp_power'], chunk_seconds=3600)
    assert calls == [(0., 3600.), (3600., 7200.)]
    assert report['rows'] == 120
    assert report['max_chunk_rows'] == 60
    assert report['training_frame_bounded'] is False
    assert report['index_plans']['legacy']
    assert path.read_bytes() == before

from unittest.mock import patch

import pandas as pd
import pytest

from utils.archive_history import ArchiveHistory, add_history_arguments, history_from_arguments


def test_request_forwards_fixed_context_without_live_or_csv_fallback():
    history = ArchiveHistory('archive.sqlite', 172800., sources_path='sources.yaml',
                             allow_legacy=True, legacy_policy='complete-legacy')
    frame = pd.DataFrame()
    with patch('utils.archive_history.load_archive_raw', return_value=frame) as load:
        assert history.raw(86400., 172800., measurements=['gshp_power']) is frame
    assert load.call_args.kwargs['sources_path'] == 'sources.yaml'
    assert load.call_args.kwargs['allow_legacy'] is True
    assert load.call_args.kwargs['measurements'] == ['gshp_power']


@pytest.mark.parametrize('start,end', [(172800., 172860.), (86400., 172801.), (172800., 86400.)])
def test_history_refuses_lookahead_and_unaligned_or_reversed_windows(start, end):
    history = ArchiveHistory('archive.sqlite', 172800.)
    with patch('utils.archive_history.load_archive_raw') as load:
        with pytest.raises(ValueError):
            history.raw(start, end)
    load.assert_not_called()


def test_history_requires_explicit_legacy_policy_opt_in():
    with pytest.raises(ValueError, match='allow_legacy'):
        ArchiveHistory('archive.sqlite', 172800., legacy_policy='complete-legacy')


def test_common_cli_requires_cutoff_and_timezone():
    import argparse
    parser = argparse.ArgumentParser()
    add_history_arguments(parser)
    with pytest.raises(ValueError, match='cutoff'):
        history_from_arguments(parser.parse_args(['--archive', 'a.sqlite']))
    with pytest.raises(SystemExit):
        parser.parse_args(['--archive', 'a.sqlite', '--cutoff', '2026-11-10'])
    request = history_from_arguments(parser.parse_args(['--archive', 'a.sqlite', '--cutoff', '2026-11-10T00:00Z']))
    assert request is not None
    assert request.cutoff == pd.Timestamp('2026-11-10T00:00Z').timestamp()

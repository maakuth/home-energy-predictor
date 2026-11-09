from unittest.mock import MagicMock

import pytest

from utils.measurement_recorder import recorder_events


def test_recorder_reads_stable_bounded_rows_and_raw_attributes():
    cursor = MagicMock()
    cursor.fetchone.return_value = (12,)
    cursor.fetchmany.side_effect = [[(9, 1.5, 'unavailable', '{"wind_speed":3}')], []]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    result = list(recorder_events(connection, 'weather.home', 1, 2, batch_size=20))
    assert result == [(9, 1.5, 'unavailable', '{"wind_speed":3}')]
    query = cursor.execute.call_args_list[1]
    assert 'ORDER BY s.last_updated_ts, s.state_id' in query.args[0]
    assert query.args[1] == (12, 1, 2)
    cursor.close.assert_called_once()


def test_missing_entity_does_not_claim_successful_empty_fetch():
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    connection = MagicMock()
    connection.cursor.return_value = cursor
    with pytest.raises(ValueError, match='not found'):
        list(recorder_events(connection, 'sensor.missing', 1, 2))
    cursor.close.assert_called_once()

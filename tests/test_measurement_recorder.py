from unittest.mock import MagicMock

import pytest

from utils.measurement_recorder import recorder_events


def test_recorder_reads_stable_bounded_rows_and_raw_attributes():
    cursor = MagicMock()
    cursor.fetchone.return_value = (12,)
    cursor.fetchall.side_effect = [[(9, 1.5, 'unavailable', '{"wind_speed":3}')], []]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    result = list(recorder_events(connection, 'weather.home', 1, 2, batch_size=20))
    assert result == [(9, 1.5, 'unavailable', '{"wind_speed":3}')]
    query = cursor.execute.call_args_list[1]
    assert 'ORDER BY s.last_updated_ts, s.state_id' in query.args[0]
    assert query.args[1] == (12, 1, 2, 20)
    assert 'LIMIT %s' in query.args[0]
    cursor.close.assert_called_once()


def test_keyset_pages_equal_time_events_without_offset_or_full_result_buffer():
    cursor = MagicMock()
    cursor.fetchone.return_value = (12,)
    cursor.fetchall.side_effect = [
        [(9, 1.5, '1', None), (10, 1.5, '2', None)],
        [(11, 1.5, '3', None), (12, 1.8, '4', None)], []]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    result = list(recorder_events(connection, 'sensor.test', 1, 2, batch_size=2))
    assert [row[0] for row in result] == [9, 10, 11, 12]
    pages = cursor.execute.call_args_list[1:]
    assert all('LIMIT %s' in call.args[0] and 'OFFSET' not in call.args[0] for call in pages)
    assert pages[1].args[1] == (12, 1, 2, 1.5, 10, 2)
    assert pages[2].args[1] == (12, 1, 2, 1.8, 12, 2)
    cursor.fetchmany.assert_not_called()
    cursor.close.assert_called_once()


def test_generator_close_releases_cursor():
    cursor = MagicMock()
    cursor.fetchone.return_value = (12,)
    cursor.fetchall.return_value = [(9, 1.5, '1', None)]
    connection = MagicMock()
    connection.cursor.return_value = cursor
    events = recorder_events(connection, 'sensor.test', 1, 2)
    next(events)
    events.close()
    cursor.close.assert_called_once()


def test_missing_entity_does_not_claim_successful_empty_fetch():
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    connection = MagicMock()
    connection.cursor.return_value = cursor
    with pytest.raises(ValueError, match='not found'):
        list(recorder_events(connection, 'sensor.missing', 1, 2))
    cursor.close.assert_called_once()

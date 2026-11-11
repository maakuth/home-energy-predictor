"""Model calendar inputs use UTC; projection/output clocks retain caller semantics."""
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from predict_future import generate_inference_data
from process_data import process_frame


CALENDAR_FEATURES = [
    'hour', 'minute', 'quarter_hour', 'day_of_week', 'month',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
]
HELSINKI = ZoneInfo('Europe/Helsinki')


def inference_at(timestamp: datetime):
    return generate_inference_data(
        timestamp, timestamp, 15, pd.DataFrame(), pd.DataFrame(),
        {}, {'now': timestamp - timedelta(days=2)},
    )


@pytest.mark.parametrize('instant', [
    '2026-01-31T22:45:00+00:00',  # local February / UTC January
    '2026-05-31T21:15:00+00:00',  # local Monday / UTC Sunday, month boundary
    '2026-12-31T22:30:00+00:00',  # local next year
    '2026-03-29T00:45:00+00:00',  # before spring DST jump
    '2026-03-29T01:00:00+00:00',  # after spring DST jump
    '2026-10-25T00:30:00+00:00',  # first local 03:30 (fold=0)
    '2026-10-25T01:30:00+00:00',  # repeated local 03:30 (fold=1)
])
def test_inference_calendar_matches_training_for_same_instant_in_both_zones(instant):
    utc = datetime.fromisoformat(instant)
    local = utc.astimezone(HELSINKI)
    # Use the actual training feature path, not just a second inference call.
    raw = pd.DataFrame(dict(total_power=[3.], solar_actual=[0.], battery_power=[0.],
                            gshp_power=[1000.], leaf_power=[0.]), index=[local])
    training = process_frame(raw, validate_coverage=False).iloc[0]
    for timestamp in (utc, local):
        rows, outputs = inference_at(timestamp)
        np.testing.assert_allclose(
            [rows[0][name] for name in CALENDAR_FEATURES],
            training[CALENDAR_FEATURES].to_numpy(dtype=float), rtol=0, atol=1e-15,
        )
        assert outputs == [timestamp.isoformat()]


def test_repeated_autumn_hour_keeps_distinct_utc_calendar_features():
    first = datetime(2026, 10, 25, 3, 30, tzinfo=HELSINKI, fold=0)
    second = first.replace(fold=1)
    first_rows, first_outputs = inference_at(first)
    second_rows, second_outputs = inference_at(second)
    assert first_rows[0]['hour'] == 0
    assert second_rows[0]['hour'] == 1
    assert first_outputs == ['2026-10-25T03:30:00+03:00']
    assert second_outputs == ['2026-10-25T03:30:00+02:00']


def test_naive_calendar_is_utc_without_host_timezone_conversion():
    # Existing bare-datetime callers retain clock values and bare output strings.
    # Like process_frame(... utc=True), only the model interprets these as UTC.
    naive = datetime(2026, 1, 31, 22, 45)
    bare_rows, outputs = inference_at(naive)
    aware_rows, _ = inference_at(naive.replace(tzinfo=timezone.utc))
    assert [bare_rows[0][name] for name in CALENDAR_FEATURES] == [
        aware_rows[0][name] for name in CALENDAR_FEATURES
    ]
    assert outputs == [naive.isoformat()]


@pytest.mark.parametrize(('timestamp', 'sauna', 'ev'), [
    (datetime(2026, 1, 12, 8, tzinfo=HELSINKI), 0, 0),
    (datetime(2026, 1, 12, 17, tzinfo=HELSINKI), 0, 1),
    (datetime(2026, 1, 12, 18, tzinfo=HELSINKI), 1, 1),
    (datetime(2026, 1, 12, 22, tzinfo=HELSINKI), 0, 1),
    (datetime(2026, 6, 1, 18, tzinfo=HELSINKI), 0, 1),  # no summer sauna
    (datetime(2026, 2, 1, 0, tzinfo=HELSINKI), 0, 1),  # local date boundary
])
def test_helsinki_heuristic_windows_and_output_zone_are_unchanged(timestamp, sauna, ev):
    rows, outputs = inference_at(timestamp)
    assert rows[0]['is_sauna_active'] == sauna
    assert rows[0]['ev_position'] == ev
    assert rows[0]['hour'] == timestamp.astimezone(timezone.utc).hour
    assert outputs == [timestamp.isoformat()]


def test_local_projection_overrides_and_inclusive_horizon_are_unchanged():
    now = datetime(2026, 1, 12, 10, tzinfo=HELSINKI)
    rows, outputs = generate_inference_data(
        now, now + timedelta(hours=4), 60, pd.DataFrame(), pd.DataFrame(),
        {'ev_pos_val': 1}, {'now': now, 'is_sauna_detected': True},
    )
    assert [row['ev_position'] for row in rows] == [1, 1, 0, 0, 0]
    assert [row['is_sauna_active'] for row in rows] == [1, 1, 1, 1, 0]
    assert outputs == [(now + timedelta(hours=h)).isoformat() for h in range(5)]

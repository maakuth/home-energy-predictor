from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np

from optimize_plan import build_tariff_prices
from utils.price_utils import fetch_market_prices


def test_external_prediction_fills_missing_tomorrow_as_raw_export_price() -> None:
    timestamps = [
        '2026-06-01T00:00:00+00:00',
        '2026-06-01T00:15:00+00:00',
        '2026-06-01T01:00:00+00:00',
        '2026-06-01T01:15:00+00:00',
    ]
    ha_state = {
        'attributes': {
            'raw_today': [
                {'start': '2026-06-01T00:00:00+00:00', 'value': 0.10},
                {'start': '2026-06-01T00:15:00+00:00', 'value': 0.10},
            ],
            'tomorrow_valid': False,
        },
    }
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = [
        [1780275600000, 7.0],  # 2026-06-01T01:00:00Z, cents/kWh
    ]

    with patch('utils.price_utils.get_ha_state', return_value=ha_state), \
            patch('utils.price_utils.requests.get', return_value=response) as get:
        prices, is_fallback, source, is_inclusive, tomorrow_valid, export_base = fetch_market_prices(
            timestamps,
            interval_minutes=15,
        )

    assert prices is not None
    assert is_fallback is not None
    np.testing.assert_allclose(prices, [0.04, 0.04, 0.07, 0.07])
    np.testing.assert_array_equal(is_fallback, [False, False, False, False])
    assert source == 'sensor.nordpool_total + nordpool-predict-fi'
    assert is_inclusive is False
    assert tomorrow_valid is True
    assert export_base is None
    get.assert_called_once()

    import_prices, export_prices = build_tariff_prices(prices, is_inclusive=is_inclusive)
    np.testing.assert_allclose(import_prices, [0.10, 0.10, 0.13, 0.13])
    np.testing.assert_allclose(export_prices, prices)


def test_external_prediction_is_not_requested_after_official_tomorrow_is_valid() -> None:
    timestamps = ['2026-06-01T00:00:00+00:00']
    ha_state = {
        'attributes': {
            'raw_today': [{'start': timestamps[0], 'value': 0.10}],
            'raw_tomorrow': [{'start': '2026-06-02T00:00:00+00:00', 'value': 0.20}],
            'tomorrow_valid': True,
        },
    }

    with patch('utils.price_utils.get_ha_state', return_value=ha_state), \
            patch('utils.price_utils.requests.get') as get:
        _, _, _, _, tomorrow_valid, _ = fetch_market_prices(timestamps, interval_minutes=15)

    assert tomorrow_valid is True
    get.assert_not_called()

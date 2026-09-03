from __future__ import annotations

import pandas as pd

from dump_battery_data import build_measurement_records


def test_measurement_records_include_actual_battery_power():
    index = pd.DatetimeIndex(['2026-08-20T12:00:00Z'])
    records = build_measurement_records(pd.DataFrame({
        'total_power_kw': [3.0],
        'solar_actual_kw': [1.0],
        'battery_power_w': [2500.0],
    }, index=index))

    assert records == [{
        'timestamp': '2026-08-20T12:00:00+00:00',
        'total_power_kw': 3.0,
        'solar_actual_kw': 1.0,
        'battery_power_kw': 2.5,
    }]

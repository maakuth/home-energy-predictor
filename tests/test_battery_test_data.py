from __future__ import annotations

import pickle

import pytest

from utils.battery_test_data import BatteryTestData, load_battery_test_data


@pytest.fixture
def battery_data_path(tmp_path):
    path = tmp_path / "battery_test_data.pkl"
    path.write_bytes(pickle.dumps({
        "metadata": {
            "dumped_at": "2026-01-03T00:00:00Z",
            "period_start": "2026-01-01T00:00:00Z",
            "period_end": "2026-01-02T00:00:00Z",
            "model_version": "1.2.3",
        },
        "ha_states": {"sensor.soc": {"state": "42.5"}, "sensor.bad": {"state": "unknown"}},
        "predictions": [{"timestamp": "2026-01-01T00:00:00Z", "predicted_baseload": 1.5}],
        "market_prices": [{"timestamp": "2026-01-01T00:00:00Z", "import_price": 0.12}],
        "history": {"sensor.load": [{"timestamp": "2026-01-01T00:00:00Z", "value": "1.5"}]},
        "battery_config": {"enabled": True, "capacity_kwh": 50.0},
        "gshp_config": {"enabled": True, "max_power_kw": 3.0},
    }))
    return path


def test_battery_test_data_loads_and_normalizes_fixture_data(battery_data_path):
    data = load_battery_test_data(str(battery_data_path))

    assert data.model_version() == "1.2.3"
    assert data.period_start() < data.period_end() <= data.dumped_at()
    assert data.ha_state_float("sensor.soc") == 42.5
    assert data.ha_state_float("sensor.bad", default=7.0) == 7.0
    assert data.ha_state_float("sensor.missing", default=7.0) == 7.0
    assert data.predictions_df().index.tz is not None
    assert data.market_prices_df().index.tz is not None
    assert data.history_df("sensor.load")["value"].iloc[0] == 1.5
    assert "Model version: 1.2.3" in data.summary()


def test_battery_test_data_rejects_missing_or_invalid_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        BatteryTestData.load(str(tmp_path / "missing.pkl"))

    invalid = tmp_path / "invalid.pkl"
    invalid.write_bytes(pickle.dumps({"not_metadata": {}}))
    with pytest.raises(ValueError, match="Invalid battery test data format"):
        BatteryTestData.load(str(invalid))

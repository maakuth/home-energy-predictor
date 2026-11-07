"""Planned grid flows are interval kWh, whereas load forecasts are kW."""
import pandas as pd
import pytest

from analyze_performance import summarize_battery_performance


@pytest.mark.parametrize('minutes', [15, 30, 60])
@pytest.mark.parametrize('net_kw,grid_kwh,export_kwh', [
    (4.0, 0.5, 0.0),
    (4.0, 2.0, 0.0),
    (-4.0, 0.0, 0.5),
])
def test_savings_convert_only_power_to_interval_energy(minutes, net_kw, grid_kwh, export_kwh):
    hours = minutes / 60
    frame = pd.DataFrame({
        'battery_action': ['discharge_load', 'charge_grid'],
        'predicted_usage': [max(net_kw, 0)] * 2,
        'solar_forecast_kw': [max(-net_kw, 0)] * 2,
        'import_price': [0.20] * 2,
        'export_price': [0.05] * 2,
        'grid_import_kwh': [grid_kwh] * 2,
        'grid_export_kwh': [export_kwh] * 2,
    }, index=pd.date_range('2026-01-01', periods=2, freq=f'{minutes}min', tz='UTC'))
    baseline = (max(net_kw, 0) * 0.20 - max(-net_kw, 0) * 0.05) * hours
    planned_cost = grid_kwh * 0.20 - export_kwh * 0.05

    result = summarize_battery_performance(frame)

    assert result['planned_savings_eur'] == pytest.approx(2 * (baseline - planned_cost))


def test_legacy_savings_fallback_does_not_rescale_grid_energy():
    frame = pd.DataFrame({
        'battery_action': ['discharge_load'] * 2,
        'predicted_usage': [4.0] * 2,
        'import_price': [0.20] * 2,
        'grid_import_kwh': [0.5] * 2,
    }, index=pd.date_range('2026-01-01', periods=2, freq='15min', tz='UTC'))

    result = summarize_battery_performance(frame)

    assert result['planned_savings_eur'] == pytest.approx(0.20)

import pandas as pd
import pytest

from utils.heating_benchmark import analyze_heating, heating_regimes


def frame(start='2026-10-07', days=1):
    index = pd.date_range(start, periods=1440 * days, freq='min', tz='Europe/Helsinki')
    return pd.DataFrame({'gshp_power': 1000.0, 'accumulator_temp': 45.0,
                         'outside_temp': 5.0}, index=index)


def test_regimes_use_local_calendar_boundaries():
    index = pd.to_datetime(['2026-08-19T20:59Z', '2026-08-19T21:00Z',
                            '2026-10-05T21:00Z', '2026-10-06T21:00Z',
                            '2026-10-18T21:00Z', '2026-10-19T21:00Z',
                            '2026-10-20T21:00Z'])
    assert heating_regimes(index).tolist() == [
        'old_pre_failure', 'old_intermittent', 'removal_transition',
        'resistance_only', 'installation_transition', 'integration_transition', 'new_gshp']


def test_energy_preserves_short_heater_runs_and_input():
    data = frame()
    data['gshp_power'] = 0.0
    data.iloc[:5, 0] = 6000.0
    original = data.copy(deep=True)
    intervals, daily = analyze_heating(data)
    assert daily.iloc[0].heating_electricity_kwh == pytest.approx(0.5)
    assert not intervals.fireplace_candidate.any()
    pd.testing.assert_frame_equal(data, original)


def test_shared_meter_explains_warming_even_with_no_relay_history():
    data = frame()
    data['gshp_power'] = 6000.0
    data.iloc[:16, 1] = [45 + i * .1 for i in range(16)]
    intervals, _ = analyze_heating(data)
    assert not intervals.fireplace_candidate.any()


def test_unexplained_warming_and_storage_tail_are_not_clean():
    data = frame(days=2)
    data['gshp_power'] = 0.0
    data.loc[data.index[60:76], 'accumulator_temp'] = [45 + i * .1 for i in range(16)]
    data.loc[data.index[76:], 'accumulator_temp'] = 46.5
    intervals, _ = analyze_heating(data, tail_hours=24)
    assert intervals.fireplace_candidate.any()
    assert intervals.iloc[1000].fireplace_affected
    assert not intervals.iloc[1000].thermal_candidate
    assert intervals.iloc[-1].thermal_candidate


def test_missing_power_is_not_zero_and_missing_weather_is_not_filled():
    data = frame()
    data.iloc[10, 0] = float('nan')
    data.iloc[20, 2] = float('nan')
    data = data.drop(data.index[30])
    intervals, daily = analyze_heating(data)
    assert pd.isna(intervals.iloc[10].heating_electricity_kwh)
    assert not intervals.iloc[[10, 20, 30]].thermal_candidate.any()
    assert daily.iloc[0].power_coverage == pytest.approx(1438 / 1440)


def test_transition_days_and_invalid_temperatures_are_excluded_not_deleted():
    data = frame('2026-10-06', days=2)
    data.iloc[1500, 1] = 97.2
    intervals, _ = analyze_heating(data)
    assert len(intervals) == len(data)
    assert not intervals.iloc[:1440].thermal_candidate.any()
    assert not intervals.iloc[1500].thermal_candidate
    assert not intervals.iloc[1500].fireplace_candidate


def test_partial_day_reports_missing_coverage_and_no_storage_estimate():
    intervals, daily = analyze_heating(frame().iloc[:60])
    assert len(intervals) == 1440
    assert daily.iloc[0].power_coverage == pytest.approx(1 / 24)
    assert daily.iloc[0].heating_electricity_kwh == pytest.approx(1.0)


def test_duplicates_and_off_grid_timestamps_rejected():
    data = frame()
    with pytest.raises(ValueError, match='duplicate'):
        analyze_heating(pd.concat([data, data.iloc[:1]]))
    data.index = data.index + pd.Timedelta(seconds=20)
    with pytest.raises(ValueError, match='aligned'):
        analyze_heating(data)


def test_dst_day_has_actual_number_of_intervals():
    index = pd.date_range('2026-10-25', '2026-10-26', inclusive='left',
                          freq='min', tz='Europe/Helsinki')
    data = pd.DataFrame({'gshp_power': 1000.0}, index=index)
    _, daily = analyze_heating(data)
    assert daily.iloc[0].heating_electricity_kwh == pytest.approx(25.0)
    assert daily.iloc[0].power_coverage == 1.0


def test_no_observations_has_no_energy_total():
    data = frame()
    data['gshp_power'] = float('nan')
    _, daily = analyze_heating(data)
    assert pd.isna(daily.iloc[0].heating_electricity_kwh)


def test_cli_writes_only_requested_new_directory(tmp_path, monkeypatch, capsys):
    from utils.heating_benchmark import main
    source = tmp_path / 'raw.csv'
    destination = tmp_path / 'report'
    frame().to_csv(source)
    original = source.read_bytes()
    monkeypatch.setattr('sys.argv', ['benchmark', '--input', str(source),
                                    '--output-dir', str(destination)])
    main()
    assert source.read_bytes() == original
    assert (destination / 'heating_daily.csv').is_file()
    assert (destination / 'heating_intervals.csv').is_file()
    assert '48h:' in capsys.readouterr().out
    with pytest.raises(FileExistsError):
        main()


def test_cli_rejects_processed_data(tmp_path, monkeypatch):
    from utils.heating_benchmark import main
    source = tmp_path / 'processed.csv'
    data = frame()
    data['heating_power'] = data.gshp_power
    data.to_csv(source)
    monkeypatch.setattr('sys.argv', ['benchmark', '--input', str(source)])
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize('parameter,value', [('tail_hours', -1), ('tail_hours', float('nan')),
                                           ('reservoir_liters', 0), ('interval_minutes', 2)])
def test_invalid_diagnostic_options(parameter, value):
    with pytest.raises(ValueError):
        analyze_heating(frame(), **{parameter: value})

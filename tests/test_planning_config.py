import pytest

from utils.planning_config import frozen_config, planning_contains, planning_getenv


def test_frozen_config_preserves_absence_and_restores_scope(monkeypatch):
    monkeypatch.setenv('PLAN_TEST', 'live')
    values = {'PRESENT_EMPTY': '', 'PLAN_TEST': 'captured'}
    with frozen_config(values):
        values['PLAN_TEST'] = 'changed'
        assert planning_getenv('PLAN_TEST') == 'captured'
        assert planning_contains('PRESENT_EMPTY')
        assert planning_getenv('PRESENT_EMPTY') == ''
        assert not planning_contains('MISSING')
        assert planning_getenv('MISSING', 'default') == 'default'
        with pytest.raises(RuntimeError):
            with frozen_config({'PLAN_TEST': 'inner'}):
                raise RuntimeError('test')
        assert planning_getenv('PLAN_TEST') == 'captured'
    assert planning_getenv('PLAN_TEST') == 'live'

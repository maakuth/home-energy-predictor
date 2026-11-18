"""Compatibility alias for optional forecast snapshot collection."""
from utils import forecast_snapshots as _impl


def collect(*args, **kwargs):
    return _impl.collect(*args, **kwargs)


def main(argv=None):
    return _impl.main(argv)


__all__ = list(getattr(_impl, '__all__', [])) + ['collect', 'main']

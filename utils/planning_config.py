"""Scoped configuration for offline planning without changing the process environment."""
from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator, Mapping
import os
from typing import overload

_config: ContextVar[Mapping[str, str] | None] = ContextVar('planning_config', default=None)


@contextmanager
def frozen_config(values: Mapping[str, str]) -> Iterator[None]:
    snapshot = dict(values)
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in snapshot.items()):
        raise ValueError('Planning configuration keys and values must be strings')
    token = _config.set(snapshot)
    try:
        yield
    finally:
        _config.reset(token)


@overload
def planning_getenv(name: str) -> str | None: ...


@overload
def planning_getenv(name: str, default: str) -> str: ...


def planning_getenv(name: str, default: str | None = None) -> str | None:
    config = _config.get()
    return os.getenv(name, default) if config is None else config.get(name, default)


def planning_contains(name: str) -> bool:
    config = _config.get()
    return name in os.environ if config is None else name in config

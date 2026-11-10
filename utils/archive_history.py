"""Explicit immutable historical-input context shared by archive consumers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import pandas as pd

from utils.archive_features import load_archive_features, load_archive_raw
from utils.archive_quality import LEGACY_POLICIES, measurement_eligibility
from utils.measurement_reader import interval_index


@dataclass(frozen=True)
class ArchiveHistory:
    path: str | Path
    cutoff: float
    sources_path: str | Path | None = None
    allow_legacy: bool = False
    legacy_policy: str = 'verified-only'
    preferred_imports: tuple[int, ...] = ()
    policy_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.cutoff) or self.cutoff % 60:
            raise ValueError('History cutoff must be finite and aligned to a whole UTC minute')
        if self.legacy_policy not in LEGACY_POLICIES:
            raise ValueError('Unknown legacy policy')
        if (self.legacy_policy != 'verified-only' or self.preferred_imports) and not self.allow_legacy:
            raise ValueError('Legacy policy/precedence requires allow_legacy=True')
        if len(set(self.preferred_imports)) != len(self.preferred_imports):
            raise ValueError('Duplicate import precedence')

    def _options(self) -> dict[str, Any]:
        options: dict[str, Any] = {'allow_legacy': self.allow_legacy,
                                   'preferred_imports': list(self.preferred_imports) or None}
        if self.sources_path is not None:
            options['sources_path'] = self.sources_path
        if self.policy_path is not None:
            options['policy_path'] = self.policy_path
            options['require_policies'] = True
        return options

    def _bounds(self, start: float, end: float, interval_seconds: int) -> None:
        if end > self.cutoff:
            raise ValueError('Historical input extends after the fixed cutoff')
        interval_index(start, end, interval_seconds)

    def raw(self, start: float, end: float, *, measurements: list[str] | None = None,
            interval_seconds: int = 60) -> pd.DataFrame:
        self._bounds(start, end, interval_seconds)
        return load_archive_raw(self.path, start, end, measurements=measurements,
                                 interval_seconds=interval_seconds, **self._options())

    def features(self, start: float, end: float, *, interval_seconds: int = 60) -> pd.DataFrame:
        self._bounds(start, end, interval_seconds)
        return load_archive_features(self.path, start, end, interval_seconds=interval_seconds,
                                      legacy_policy=self.legacy_policy, **self._options())

    def eligible(self, raw: pd.DataFrame, measurement: str) -> pd.Series:
        return measurement_eligibility(raw, measurement, self.legacy_policy)


def aware_epoch(value: str) -> float:
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError('Timezone required')
        return result.timestamp()
    except ValueError as error:
        raise argparse.ArgumentTypeError('Expected ISO timestamp with timezone') from error


def add_history_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--archive', help='Explicit read-only historical archive; no compatibility fallback')
    parser.add_argument('--cutoff', type=aware_epoch, help='Fixed exclusive history cutoff, aligned to a UTC minute')
    parser.add_argument('--sources', help='Dated measurement source overrides')
    parser.add_argument('--measurement-policy', help='Explicit measurement units/kinds/freshness registry')
    parser.add_argument('--allow-legacy', action='store_true')
    parser.add_argument('--legacy-policy', choices=LEGACY_POLICIES, default='verified-only')
    parser.add_argument('--preferred-import', type=int, action='append')


def history_from_arguments(args: argparse.Namespace) -> ArchiveHistory | None:
    if not args.archive:
        if (args.cutoff is not None or args.sources or args.measurement_policy or args.allow_legacy
                or args.preferred_import or args.legacy_policy != 'verified-only'):
            raise ValueError('History cutoff/source/policy options require --archive')
        return None
    if args.cutoff is None:
        raise ValueError('--archive requires an explicit --cutoff')
    return ArchiveHistory(args.archive, args.cutoff, sources_path=args.sources,
        allow_legacy=args.allow_legacy, legacy_policy=args.legacy_policy,
        preferred_imports=tuple(args.preferred_import or ()), policy_path=args.measurement_policy)

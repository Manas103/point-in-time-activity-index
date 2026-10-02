"""The publish gate: refuse to publish if any required series is stale.

Staleness rule (config.STALENESS_LOOKBACK_PERIODS = 2): a series is stale
as of a release for `current_period` if it has no vintage at all, visible
as of that release's as_of date, for either `current_period` or the period
immediately before it. One period of ordinary production delay is
tolerated; two is treated as a feed outage and refused.

The pure logic lives in `evaluate`, which takes a plain dict of
"latest period known per series" so it can be exercised with synthetic
scenarios in tests without touching a real database. `check` is the
DB-backed wrapper the real pipeline calls.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field

from activityindex import config, vintage_store


@dataclass
class GateResult:
    ok: bool
    stale_series: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.ok


def evaluate(
    latest_period_known: dict[str, datetime.date | None],
    all_series_ids: list[str],
    current_period: datetime.date,
) -> GateResult:
    """Pure staleness check. `latest_period_known` maps series_id to the most
    recent reference_period it has any visible vintage for, or None/missing
    if it has none at all."""
    cutoff = config.month_add(current_period, -(config.STALENESS_LOOKBACK_PERIODS - 1))
    stale = []
    for series_id in all_series_ids:
        latest = latest_period_known.get(series_id)
        if latest is None or latest < cutoff:
            stale.append(series_id)
    return GateResult(ok=(len(stale) == 0), stale_series=sorted(stale))


def check(
    con, as_of: datetime.date, current_period: datetime.date, all_series_ids: list[str]
) -> GateResult:
    """DB-backed gate check for a real release attempt."""
    latest_df = vintage_store.latest_periods_as_of(con, as_of)
    latest_map = dict(zip(latest_df["series_id"], latest_df["latest_period"]))
    return evaluate(latest_map, all_series_ids, current_period)

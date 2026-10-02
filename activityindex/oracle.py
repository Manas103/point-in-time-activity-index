"""An independent, from-scratch reference recomputation of the index value.

Plain dicts, lists, loops, and the math module. No pandas, no duckdb
vectorized SQL for the actual computation, no import of index_builder.py
or vintage_store.py. The only thing this module shares with the fast path
is config.py (declared policy: the staleness lookback and minimum history
length), the same discipline point-in-time-gas-backtest's oracle.py uses
and explains in its own README. A bug specific to the fast path's SQL or
pandas code has a real chance of showing up as a disagreement here; it has
no chance at all of showing up if both paths run the same code.

`fetch_raw_rows` is the one place this module touches duckdb, and only to
pull plain Python tuples out of it; everything after that is loops.
"""

from __future__ import annotations

import datetime
import math

from activityindex import config


def fetch_raw_rows(con) -> list[tuple]:
    return con.execute(
        """
        SELECT series_id, category, reference_period, vintage_date, value, revision_number
        FROM source_vintages
        """
    ).fetchall()


def compute_index_value_oracle(
    raw_rows: list[tuple], as_of: datetime.date, target_period: datetime.date
) -> tuple[float | None, int]:
    """Pure-Python recomputation of the equal-weight index value.

    Returns (index_value, series_count_included), mirroring
    index_builder.compute_release_value's first two return values.
    """
    # Step 1: filter the raw, undeduped log by vintage_date <= as_of first.
    visible = [row for row in raw_rows if row[3] <= as_of]

    # Step 2: within what survives, keep only the highest revision_number
    # per (series_id, reference_period).
    best: dict[tuple[str, datetime.date], tuple[int, float, str]] = {}
    for series_id, category, reference_period, _vintage_date, value, revision_number in visible:
        key = (series_id, reference_period)
        current = best.get(key)
        if current is None or revision_number > current[0]:
            best[key] = (revision_number, value, category)

    # Step 3: group by series, restricted to periods <= target_period.
    history: dict[str, list[tuple[datetime.date, float]]] = {}
    category_of: dict[str, str] = {}
    for (series_id, reference_period), (_rev, value, category) in best.items():
        if reference_period > target_period:
            continue
        history.setdefault(series_id, []).append((reference_period, value))
        category_of[series_id] = category

    # Step 4: per-series z-score of the target period against its own
    # history mean and sample standard deviation.
    z_scores: dict[str, float] = {}
    for series_id, points in history.items():
        points.sort(key=lambda p: p[0])
        periods = [p[0] for p in points]
        if target_period not in periods:
            continue
        if len(points) < config.MIN_HISTORY_MONTHS:
            continue
        values = [p[1] for p in points]
        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
        if std == 0:
            continue
        target_value = dict(points)[target_period]
        z_scores[series_id] = (target_value - mean) / std

    if not z_scores:
        return None, 0

    included = len(z_scores)
    index_value = sum(z_scores.values()) / included
    return index_value, included

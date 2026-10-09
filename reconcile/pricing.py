"""Prices the vendor choice: at each period's own first vintage (the
release where that period is first published, lag_months = 0), what does
the published activity index look like if it is built entirely from
vendor A's panel versus entirely from vendor B's panel.

Reuses activityindex.index_builder.compute_release_value unchanged, the
same equal-weight z-score combination the original single-vendor index
already uses; this module only supplies it a different vendor's snapshot.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from activityindex import config as aconfig, index_builder
from reconcile import store


def first_vintage_index_by_vendor(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    rows = []
    for period in aconfig.all_periods():
        release_date = aconfig.index_release_date(period)
        snap_a = store.query_point_in_time(con, "A", release_date)
        snap_b = store.query_point_in_time(con, "B", release_date)

        value_a, count_a, _ = index_builder.compute_release_value(snap_a, period)
        value_b, count_b, _ = index_builder.compute_release_value(snap_b, period)
        if value_a is None or value_b is None:
            continue

        rows.append(
            {
                "reference_period": period,
                "release_date": release_date,
                "index_value_vendor_a": value_a,
                "index_value_vendor_b": value_b,
                "series_count_a": count_a,
                "series_count_b": count_b,
                "abs_diff": abs(value_a - value_b),
            }
        )
    return pd.DataFrame(rows)

"""Independent pure-Python recomputation of the point-in-time panel read.

No DuckDB, no pandas, no SQL window functions: a plain loop over the raw
row list, filtering by vintage_date first and picking the max
revision_number second, the same order panel_store.query_point_in_time
uses. Diffed exactly against the fast path in tests/test_panel_store.py.
"""

from __future__ import annotations

import datetime


def point_in_time_value(
    rows: list[dict], name_id: str, metric: str, abs_month_index: int, as_of: datetime.date
) -> float | None:
    visible = [
        r
        for r in rows
        if r["name_id"] == name_id
        and r["metric"] == metric
        and r["abs_month_index"] == abs_month_index
        and r["vintage_date"] <= as_of
    ]
    if not visible:
        return None
    best = max(visible, key=lambda r: (r["revision_number"], r["vintage_date"]))
    return best["value"]

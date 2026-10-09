"""Drives the two-vendor reconciliation: for every release date, for every
lag activityindex itself publishes at (0 to 3 months back), compare vendor
A's and vendor B's visible value for every series and classify every
disagreement.

"Release by release" is taken literally: this does not compare the two
vendors' final, settled values once; it replays activityindex's own
REVISION_LAGS schedule and checks agreement at every one of those releases,
the same granularity a vendor-health monitor running continuously would
actually see.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from activityindex import config as aconfig, simulate
from reconcile import classifier, config as rconfig, store, vendor_b as vb


def build_vendor_store(con: duckdb.DuckDBPyConnection) -> dict[tuple, str]:
    """Loads both vendors' append-only logs into vendor_vintages. Returns
    the injected cause assignment, kept only for the test suite's white-box
    check that the blind classifier actually recovers it; production code
    (measurements.py, scripts/run_reconcile.py) never looks at it."""
    series = simulate.build_series_metadata()
    periods = aconfig.all_periods()
    vendor_a_rows = simulate.generate_vintages(series, periods)
    vendor_b_rows, assignment = vb.generate_vendor_b_vintages(vendor_a_rows)
    store.load_vintages(con, "A", vendor_a_rows)
    store.load_vintages(con, "B", vendor_b_rows)
    return assignment


def _index_snapshot(snapshot: pd.DataFrame) -> dict[tuple, tuple[float, int]]:
    if snapshot.empty:
        return {}
    return {
        (sid, period): (float(val), int(rev))
        for sid, period, val, rev in zip(
            snapshot["series_id"], snapshot["reference_period"],
            snapshot["value"], snapshot["revision_number"],
        )
    }


def _value_rev(indexed: dict, series_id: str, period) -> tuple[float | None, int | None]:
    return indexed.get((series_id, period), (None, None))


def reconcile_all(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    raw_a = store.raw_log(con, "A")
    raw_b = store.raw_log(con, "B")
    lookups = classifier.build_lookups(raw_a, raw_b)

    all_series_ids = sorted(
        con.execute("SELECT DISTINCT series_id FROM vendor_vintages WHERE vendor = 'A'").fetchdf()["series_id"]
    )

    rows = []
    for current_period in aconfig.all_periods():
        release_date = aconfig.index_release_date(current_period)
        snap_a = _index_snapshot(store.query_point_in_time(con, "A", release_date))
        snap_b = _index_snapshot(store.query_point_in_time(con, "B", release_date))

        for lag in aconfig.REVISION_LAGS:
            target_period = aconfig.month_add(current_period, -lag)
            if target_period < aconfig.FIRST_PERIOD:
                continue
            prior_period = aconfig.month_add(target_period, -1)

            for series_id in all_series_ids:
                a_val, a_rev = _value_rev(snap_a, series_id, target_period)
                if a_val is None:
                    continue  # not yet known to vendor A either at this release
                b_val, b_rev = _value_rev(snap_b, series_id, target_period)

                agree = b_val is not None and abs(a_val - b_val) <= max(
                    abs(a_val), abs(b_val), 1.0
                ) * rconfig.AGREEMENT_RELATIVE_TOLERANCE

                cause = None
                if not agree:
                    cause = classifier.classify(
                        series_id, target_period, prior_period, a_val, b_val, a_rev, b_rev,
                        lookups,
                    )

                rows.append(
                    {
                        "series_id": series_id,
                        "reference_period": target_period,
                        "release_date": release_date,
                        "lag_months": lag,
                        "vendor_a_value": a_val,
                        "vendor_b_value": b_val,
                        "agree": agree,
                        "cause": cause,
                    }
                )

    return pd.DataFrame(rows)

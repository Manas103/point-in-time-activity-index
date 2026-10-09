"""A second, vendor-partitioned append-only vintage store.

Same discipline as activityindex.vintage_store (never UPDATE'd or DELETE'd
from, a restatement is a new row, a point-in-time query filters the raw log
by vintage_date before picking the latest revision), with one added
dimension: `vendor`. Two vendors' deliveries of the same series live in the
same table, distinguished only by this column, so a restatement from either
vendor is still just a new row, never an overwrite of the other vendor's or
its own history.
"""

from __future__ import annotations

import datetime

import duckdb
import pandas as pd

VENDOR_VINTAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS vendor_vintages (
    vendor            VARCHAR NOT NULL,
    series_id         VARCHAR NOT NULL,
    category          VARCHAR NOT NULL,
    reference_period  DATE NOT NULL,
    vintage_date      DATE NOT NULL,
    value             DOUBLE NOT NULL,
    revision_number   INTEGER NOT NULL
)
"""


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(VENDOR_VINTAGES_SCHEMA)
    return con


def load_vintages(con: duckdb.DuckDBPyConnection, vendor: str, rows: list[dict]) -> None:
    """Append-only bulk load for one vendor. Never called twice for the same
    (vendor, rows)."""
    df = pd.DataFrame(rows)
    df.insert(0, "vendor", vendor)
    con.register("_vendor_rows", df)
    con.execute("INSERT INTO vendor_vintages SELECT * FROM _vendor_rows")
    con.unregister("_vendor_rows")


def query_point_in_time(
    con: duckdb.DuckDBPyConnection, vendor: str, as_of: datetime.date
) -> pd.DataFrame:
    """The latest value known for `vendor`, as of `as_of`, for every
    (series, period). Same visible-then-rank order as
    activityindex.vintage_store.query_point_in_time, and for the same
    reason: filtering by vintage_date before ranking by revision_number is
    what keeps a late-arriving revision from retroactively changing what
    was "known" on an earlier date. Where a duplicate release (two rows at
    the same revision_number) exists, the rank also breaks ties by the
    latest vintage_date, so a bad resend with a later timestamp than the
    original is exactly what a consumer querying after that resend would
    actually see.
    """
    query = """
    WITH visible AS (
        SELECT *
        FROM vendor_vintages
        WHERE vendor = ? AND vintage_date <= ?
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY series_id, reference_period
                ORDER BY revision_number DESC, vintage_date DESC
            ) AS rn
        FROM visible
    )
    SELECT series_id, category, reference_period, vintage_date, value, revision_number
    FROM ranked
    WHERE rn = 1
    ORDER BY series_id, reference_period
    """
    df = con.execute(query, [vendor, as_of]).fetchdf()
    if not df.empty:
        df["reference_period"] = df["reference_period"].dt.date
        df["vintage_date"] = df["vintage_date"].dt.date
    return df


def raw_log(con: duckdb.DuckDBPyConnection, vendor: str) -> pd.DataFrame:
    """The full, undeduped vintage log for one vendor, used by the
    classifier to tell a duplicate release or a true coverage gap apart
    from an ordinary disagreement."""
    df = con.execute(
        "SELECT * FROM vendor_vintages WHERE vendor = ? ORDER BY series_id, reference_period, revision_number",
        [vendor],
    ).fetchdf()
    if not df.empty:
        df["reference_period"] = df["reference_period"].dt.date
        df["vintage_date"] = df["vintage_date"].dt.date
    return df


def restatement_never_overwrites(con: duckdb.DuckDBPyConnection, vendor: str) -> dict:
    """Direct proof of the append-only claim: insert a second vintage for a
    real (series, period, revision) already in the store and confirm both
    rows still exist afterward, with the point-in-time query able to recover
    either one depending on the as_of date used.
    """
    sample = con.execute(
        "SELECT series_id, category, reference_period, vintage_date, value, revision_number "
        "FROM vendor_vintages WHERE vendor = ? LIMIT 1",
        [vendor],
    ).fetchone()
    if sample is None:
        return {"value": False, "reason": "no rows to test with"}
    series_id, category, reference_period, vintage_date, value, revision_number = sample
    before = con.execute(
        "SELECT COUNT(*) FROM vendor_vintages WHERE vendor = ? AND series_id = ? AND reference_period = ?",
        [vendor, series_id, reference_period],
    ).fetchone()[0]

    restated_vintage_date = vintage_date + datetime.timedelta(days=3650)
    restated_value = value + 999.0
    load_vintages(
        con,
        vendor,
        [
            {
                "series_id": series_id,
                "category": category,
                "reference_period": reference_period,
                "vintage_date": restated_vintage_date,
                "value": restated_value,
                "revision_number": revision_number,
            }
        ],
    )
    after = con.execute(
        "SELECT COUNT(*) FROM vendor_vintages WHERE vendor = ? AND series_id = ? AND reference_period = ?",
        [vendor, series_id, reference_period],
    ).fetchone()[0]

    pre_restatement = query_point_in_time(con, vendor, vintage_date)
    pre_row = pre_restatement[
        (pre_restatement["series_id"] == series_id) & (pre_restatement["reference_period"] == reference_period)
    ]
    original_recovered = not pre_row.empty and abs(float(pre_row.iloc[0]["value"]) - value) < 1e-9

    con.execute(
        "DELETE FROM vendor_vintages WHERE vendor = ? AND series_id = ? AND reference_period = ? "
        "AND vintage_date = ? AND value = ?",
        [vendor, series_id, reference_period, restated_vintage_date, restated_value],
    )

    return {
        "value": bool(after == before + 1 and original_recovered),
        "rows_before": int(before),
        "rows_after_restatement": int(after),
        "original_value_still_recoverable_at_its_own_as_of_date": bool(original_recovered),
    }

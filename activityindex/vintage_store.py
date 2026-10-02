"""The DuckDB vintage store: append-only history, point-in-time queries.

Two tables, both append-only, never UPDATE'd or DELETE'd from:

- source_vintages(series_id, category, reference_period, vintage_date,
  value, revision_number): every vintage of every series, ever.
- published_index(release_date, reference_period, lag_months, index_value,
  published_at): every index value this project has ever published, kept
  exactly as it was computed, never recomputed with hindsight.

The one rule that matters most in this file: a point-in-time query always
filters the raw, undeduped vintage log by vintage_date first, and only
picks "the latest known revision per series/period" from what survives
that filter. Deduping first and filtering second silently resurrects
lookahead, because the final row's own vintage_date can be later than the
as_of date even though an earlier, visible vintage of the same row existed.
point-in-time-gas-backtest hit exactly this bug; see the README here for
how it showed up a second time while this repo was built.
"""

from __future__ import annotations

import datetime

import duckdb
import pandas as pd

SOURCE_VINTAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_vintages (
    series_id         VARCHAR NOT NULL,
    category          VARCHAR NOT NULL,
    reference_period  DATE NOT NULL,
    vintage_date      DATE NOT NULL,
    value             DOUBLE NOT NULL,
    revision_number   INTEGER NOT NULL
)
"""

PUBLISHED_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS published_index (
    release_date      DATE NOT NULL,
    reference_period  DATE NOT NULL,
    lag_months        INTEGER NOT NULL,
    index_value       DOUBLE NOT NULL,
    series_count      INTEGER NOT NULL,
    published_at      TIMESTAMP NOT NULL
)
"""

CATEGORY_CONTRIBUTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS category_contributions (
    release_date      DATE NOT NULL,
    reference_period  DATE NOT NULL,
    lag_months        INTEGER NOT NULL,
    category          VARCHAR NOT NULL,
    contribution      DOUBLE NOT NULL
)
"""


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(SOURCE_VINTAGES_SCHEMA)
    con.execute(PUBLISHED_INDEX_SCHEMA)
    con.execute(CATEGORY_CONTRIBUTIONS_SCHEMA)
    return con


def load_vintages(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    """Append-only bulk load. Never called twice for the same rows."""
    df = pd.DataFrame(rows)
    con.register("_vintage_rows", df)
    con.execute("INSERT INTO source_vintages SELECT * FROM _vintage_rows")
    con.unregister("_vintage_rows")


def query_point_in_time(
    con: duckdb.DuckDBPyConnection, as_of: datetime.date
) -> pd.DataFrame:
    """The latest value known, as of `as_of`, for every (series, period).

    Filters the raw vintage log by vintage_date <= as_of first (the
    undeduped log), then within that filtered set keeps the highest
    revision_number per (series_id, reference_period). This order is the
    whole point: deduping before filtering would let a later revision's
    own vintage_date decide whether an earlier, already-visible vintage of
    the same row counts as known.
    """
    query = """
    WITH visible AS (
        SELECT *
        FROM source_vintages
        WHERE vintage_date <= ?
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
    df = con.execute(query, [as_of]).fetchdf()
    # Normalize DuckDB's DATE columns (datetime64[us] via fetchdf) to plain
    # datetime.date, matching every date value produced elsewhere in this
    # project (config.all_periods, config.month_add). Without this, pandas
    # (3.0.x here) raises TypeError on reference_period <= target_period
    # instead of the old implicit coercion; see README Findings.
    if not df.empty:
        df["reference_period"] = df["reference_period"].dt.date
        df["vintage_date"] = df["vintage_date"].dt.date
    return df


def latest_period_known(
    con: duckdb.DuckDBPyConnection, as_of: datetime.date, series_id: str
) -> datetime.date | None:
    """The most recent reference_period this series has any vintage for,
    among vintages visible as of `as_of`. None if nothing is visible yet."""
    row = con.execute(
        """
        SELECT MAX(reference_period)
        FROM source_vintages
        WHERE series_id = ? AND vintage_date <= ?
        """,
        [series_id, as_of],
    ).fetchone()
    return row[0] if row else None


def latest_periods_as_of(
    con: duckdb.DuckDBPyConnection, as_of: datetime.date
) -> pd.DataFrame:
    """Per series, the most recent reference_period with any vintage visible
    as of `as_of`. One series per row; series with nothing visible yet are
    simply absent, left for the caller to treat as stale."""
    query = """
    SELECT series_id, MAX(reference_period) AS latest_period
    FROM source_vintages
    WHERE vintage_date <= ?
    GROUP BY series_id
    """
    df = con.execute(query, [as_of]).fetchdf()
    # DuckDB hands back DATE columns as pandas Timestamp via fetchdf(), not
    # datetime.date; normalize here so every caller can compare against
    # plain datetime.date values (config.month_add etc.) without having to
    # know that detail. See README Findings for how this surfaced.
    if not df.empty:
        df["latest_period"] = df["latest_period"].dt.date
    return df


def insert_published_index(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    con.register("_published_rows", df)
    con.execute("INSERT INTO published_index SELECT * FROM _published_rows")
    con.unregister("_published_rows")


def insert_category_contributions(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    con.register("_contrib_rows", df)
    con.execute("INSERT INTO category_contributions SELECT * FROM _contrib_rows")
    con.unregister("_contrib_rows")

"""The DuckDB panel store: append-only monthly vintages, point-in-time reads.

One table, source_panel(name_id, metric, abs_month_index, vintage_date,
value, revision_number), never UPDATE'd or DELETE'd from. The same rule
that matters most in activityindex/vintage_store.py applies here
unchanged: a point-in-time query filters the raw, undeduped log by
vintage_date first, and only picks the latest known revision per
(name_id, metric, abs_month_index) from what survives that filter.
"""

from __future__ import annotations

import datetime

import duckdb
import pandas as pd

SOURCE_PANEL_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_panel (
    name_id           VARCHAR NOT NULL,
    metric            VARCHAR NOT NULL,
    abs_month_index   INTEGER NOT NULL,
    vintage_date      DATE NOT NULL,
    value             DOUBLE NOT NULL,
    revision_number   INTEGER NOT NULL
)
"""


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(SOURCE_PANEL_SCHEMA)
    return con


def load_panel(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> None:
    """Append-only bulk load. Never called twice for the same rows."""
    df = pd.DataFrame(rows)
    con.register("_panel_rows", df)
    con.execute("INSERT INTO source_panel SELECT * FROM _panel_rows")
    con.unregister("_panel_rows")


def query_point_in_time(
    con: duckdb.DuckDBPyConnection, as_of: datetime.date
) -> pd.DataFrame:
    """The latest value known, as of `as_of`, for every (name, metric, month).

    Filters by vintage_date <= as_of first, then within that filtered set
    keeps the highest revision_number per (name_id, metric,
    abs_month_index). Deduping first and filtering second would let a
    later revision's own vintage_date decide whether an earlier,
    already-visible vintage of the same row counts as known, the same
    footgun activityindex/vintage_store.py documents.
    """
    query = """
    WITH visible AS (
        SELECT * FROM source_panel WHERE vintage_date <= ?
    ),
    ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY name_id, metric, abs_month_index
                ORDER BY revision_number DESC, vintage_date DESC
            ) AS rn
        FROM visible
    )
    SELECT name_id, metric, abs_month_index, vintage_date, value, revision_number
    FROM ranked
    WHERE rn = 1
    ORDER BY name_id, metric, abs_month_index
    """
    df = con.execute(query, [as_of]).fetchdf()
    if not df.empty:
        df["vintage_date"] = df["vintage_date"].dt.date
    return df


def query_final(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The final (highest-revision) value for every (name, metric, month),
    with no vintage_date restriction. This is the "restated, with hindsight"
    view used only to measure the point-in-time-vs-restated overstatement;
    the honest nowcast path never calls this.
    """
    query = """
    WITH ranked AS (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY name_id, metric, abs_month_index
                ORDER BY revision_number DESC
            ) AS rn
        FROM source_panel
    )
    SELECT name_id, metric, abs_month_index, value
    FROM ranked
    WHERE rn = 1
    ORDER BY name_id, metric, abs_month_index
    """
    return con.execute(query).fetchdf()

import datetime

from activityindex import vintage_store


def _row(series_id, category, reference_period, vintage_date, value, revision_number):
    return {
        "series_id": series_id,
        "category": category,
        "reference_period": reference_period,
        "vintage_date": vintage_date,
        "value": value,
        "revision_number": revision_number,
    }


def test_point_in_time_returns_latest_visible_revision(memory_con):
    rows = [
        _row("s1", "production", datetime.date(2024, 1, 1), datetime.date(2024, 2, 10), 100.0, 0),
        _row("s1", "production", datetime.date(2024, 1, 1), datetime.date(2024, 3, 10), 101.0, 1),
    ]
    vintage_store.load_vintages(memory_con, rows)

    before_revision = vintage_store.query_point_in_time(memory_con, datetime.date(2024, 2, 20))
    assert len(before_revision) == 1
    assert before_revision.iloc[0]["value"] == 100.0
    assert before_revision.iloc[0]["revision_number"] == 0

    after_revision = vintage_store.query_point_in_time(memory_con, datetime.date(2024, 3, 20))
    assert after_revision.iloc[0]["value"] == 101.0
    assert after_revision.iloc[0]["revision_number"] == 1


def test_point_in_time_excludes_vintages_not_yet_released(memory_con):
    rows = [_row("s1", "production", datetime.date(2024, 1, 1), datetime.date(2024, 2, 10), 100.0, 0)]
    vintage_store.load_vintages(memory_con, rows)

    too_early = vintage_store.query_point_in_time(memory_con, datetime.date(2024, 2, 1))
    assert len(too_early) == 0


def test_filter_before_dedup_regression():
    """Regression test for the exact footgun point-in-time-gas-backtest hit:
    deduping to 'latest revision per period' before filtering by vintage_date
    would make a later revision's own vintage_date decide whether an
    earlier, already-visible vintage counts as known. query_point_in_time
    must filter the raw log first, then dedupe, so this case stays correct.
    """
    con = vintage_store.connect(":memory:")
    rows = [
        # revision 0 visible well before as_of.
        _row("s1", "production", datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), 100.0, 0),
        # revision 1 is NOT yet visible as of the query below.
        _row("s1", "production", datetime.date(2024, 1, 1), datetime.date(2024, 4, 1), 999.0, 1),
    ]
    vintage_store.load_vintages(con, rows)
    as_of = datetime.date(2024, 3, 1)

    # The buggy shape: dedupe to "latest revision per period" FIRST (picking
    # revision 1, value 999.0, vintage_date 2024-04-01), THEN filter by
    # vintage_date <= as_of. That collapsed row's own vintage_date is after
    # as_of, so the whole period would vanish even though revision 0 was
    # visible all along.
    buggy = con.execute(
        """
        WITH deduped AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY series_id, reference_period ORDER BY revision_number DESC
            ) AS rn
            FROM source_vintages
        )
        SELECT * FROM deduped WHERE rn = 1 AND vintage_date <= ?
        """,
        [as_of],
    ).fetchdf()
    assert len(buggy) == 0, "demonstrates the bug this project deliberately avoids"

    correct = vintage_store.query_point_in_time(con, as_of)
    assert len(correct) == 1
    assert correct.iloc[0]["value"] == 100.0
    con.close()


def test_latest_periods_as_of_returns_plain_date_not_timestamp(memory_con):
    """Regression test for a real bug hit while building this project: DuckDB's
    fetchdf() hands DATE columns back as pandas Timestamp, and this pandas
    version refuses to compare a Timestamp to a plain datetime.date. Every
    date value crossing this boundary must come back as datetime.date."""
    rows = [_row("s1", "production", datetime.date(2024, 1, 1), datetime.date(2024, 2, 1), 100.0, 0)]
    vintage_store.load_vintages(memory_con, rows)
    df = vintage_store.latest_periods_as_of(memory_con, datetime.date(2024, 3, 1))
    assert len(df) == 1
    latest = df.iloc[0]["latest_period"]
    assert type(latest) is datetime.date
    # This comparison is exactly what raised TypeError before the fix.
    assert latest < datetime.date(2024, 2, 2)

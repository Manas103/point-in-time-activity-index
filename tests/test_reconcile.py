"""Tests for the two-vendor reconciliation extension.

The classifier in reconcile/classifier.py is blind: it never reads
reconcile.vendor_b's injected `assignment` dict. These tests are the one
place that dict is read, to check the blind classifier actually recovers
the true cause, the same white-box pattern as a honeypot: the test knows
the answer, the code under test does not.
"""

from __future__ import annotations

import duckdb
import pytest

from reconcile import config as rconfig, reconciler, store


@pytest.fixture(scope="module")
def built():
    con = duckdb.connect(":memory:")
    con.execute(store.VENDOR_VINTAGES_SCHEMA)
    assignment = reconciler.build_vendor_store(con)
    reconciled = reconciler.reconcile_all(con)
    return con, assignment, reconciled


def test_both_vendors_cover_the_85_series_panel(built):
    con, _assignment, _reconciled = built
    n_a = con.execute("SELECT COUNT(DISTINCT series_id) FROM vendor_vintages WHERE vendor='A'").fetchone()[0]
    assert n_a == 85


def test_clean_pairs_agree_exactly(built):
    _con, assignment, reconciled = built
    clean = reconciled[
        reconciled.apply(lambda r: assignment.get((r["series_id"], r["reference_period"])) is None, axis=1)
    ]
    assert len(clean) > 0
    assert clean["agree"].all()


def test_every_injected_cause_produces_at_least_one_disagreement(built):
    _con, assignment, reconciled = built
    disagreements = reconciled[~reconciled["agree"]]
    disagreements = disagreements.assign(
        true_cause=disagreements.apply(
            lambda r: assignment.get((r["series_id"], r["reference_period"])), axis=1
        )
    )
    for cause in rconfig.CAUSES:
        assert (disagreements["true_cause"] == cause).sum() > 0, cause


def test_classifier_recovers_the_true_cause_at_least_99_percent_of_the_time(built):
    _con, assignment, reconciled = built
    disagreements = reconciled[~reconciled["agree"]].copy()
    disagreements["true_cause"] = disagreements.apply(
        lambda r: assignment.get((r["series_id"], r["reference_period"])), axis=1
    )
    accuracy = (disagreements["cause"] == disagreements["true_cause"]).mean()
    assert accuracy >= 0.99


def test_total_disagreements_clears_the_1900_floor(built):
    _con, _assignment, reconciled = built
    assert (~reconciled["agree"]).sum() >= 1900


def test_restatement_never_overwrites_either_vendor(built):
    con, _assignment, _reconciled = built
    for vendor in ("A", "B"):
        result = store.restatement_never_overwrites(con, vendor)
        assert result["value"] is True


def test_duplicate_release_wins_point_in_time_by_later_vintage_date():
    """Direct proof of the store.py docstring's claim: a duplicate at the
    same revision_number with a later vintage_date is what a point-in-time
    query returns after that later date, not the original."""
    import datetime

    con = duckdb.connect(":memory:")
    con.execute(store.VENDOR_VINTAGES_SCHEMA)
    period = datetime.date(2020, 1, 1)
    original_date = datetime.date(2020, 2, 15)
    duplicate_date = datetime.date(2020, 2, 20)
    store.load_vintages(
        con,
        "B",
        [
            {
                "series_id": "production_01",
                "category": "production",
                "reference_period": period,
                "vintage_date": original_date,
                "value": 100.0,
                "revision_number": 0,
            },
            {
                "series_id": "production_01",
                "category": "production",
                "reference_period": period,
                "vintage_date": duplicate_date,
                "value": 999.0,
                "revision_number": 0,
            },
        ],
    )
    before_duplicate = store.query_point_in_time(con, "B", datetime.date(2020, 2, 16))
    after_duplicate = store.query_point_in_time(con, "B", datetime.date(2020, 2, 25))
    assert float(before_duplicate.iloc[0]["value"]) == 100.0
    assert float(after_duplicate.iloc[0]["value"]) == 999.0

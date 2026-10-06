import datetime
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nowcast import config, model, oracle, panel_store, simulate


@pytest.fixture(scope="module")
def built_db(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("nowcast") / "panel.duckdb")
    names = simulate.build_name_metadata()
    quarterly_growth = simulate.true_quarterly_growth(names)
    monthly_true = simulate.monthly_true_values(quarterly_growth)
    rows = simulate.generate_panel_vintages(names, monthly_true)
    con = panel_store.connect(db_path)
    panel_store.load_panel(con, rows)
    yield con, rows, names, quarterly_growth
    con.close()


def test_name_count_and_coverage_spread():
    names = simulate.build_name_metadata()
    assert len(names) == 120
    coverages = [m.coverage for m in names]
    assert min(coverages) < 0.02
    assert max(coverages) > 0.5


def test_monthly_decomposition_sums_to_quarterly_truth():
    names = simulate.build_name_metadata()
    quarterly_growth = simulate.true_quarterly_growth(names)
    monthly_true = simulate.monthly_true_values(quarterly_growth)
    for name_id, growth in quarterly_growth.items():
        for metric in ("ticket", "txn"):
            monthly = monthly_true[name_id][metric]
            for q in range(config.N_QUARTERS):
                month_sum = monthly[3 * q] + monthly[3 * q + 1] + monthly[3 * q + 2]
                assert abs(month_sum - growth[metric][q]) < 1e-9


def test_point_in_time_excludes_not_yet_visible(built_db):
    con, rows, names, _ = built_db
    name_id = names[0].name_id
    # abs_month_index 0's R2 (final, noise-free) lands 80 days after month end.
    as_of_before = config.month_end_date(0) + datetime.timedelta(days=79)
    as_of_after = config.month_end_date(0) + datetime.timedelta(days=81)
    before = panel_store.query_point_in_time(con, as_of_before)
    after = panel_store.query_point_in_time(con, as_of_after)
    before_rev = before[(before.name_id == name_id) & (before.abs_month_index == 0) & (before.metric == "ticket")]
    after_rev = after[(after.name_id == name_id) & (after.abs_month_index == 0) & (after.metric == "ticket")]
    assert int(before_rev.iloc[0].revision_number) == 1
    assert int(after_rev.iloc[0].revision_number) == 2


def test_oracle_matches_fast_path_point_in_time(built_db):
    con, rows, names, _ = built_db
    as_of = config.nowcast_as_of_date(20)
    fast = panel_store.query_point_in_time(con, as_of)
    sample = fast.sample(n=30, random_state=0)
    for row in sample.itertuples():
        oracle_value = oracle.point_in_time_value(
            rows, row.name_id, row.metric, row.abs_month_index, as_of
        )
        assert oracle_value is not None
        assert abs(oracle_value - row.value) < 1e-12


def test_quarter_table_has_two_of_three_months_only(built_db):
    con, rows, names, _ = built_db
    as_of = config.nowcast_as_of_date(20)
    m0, m1, m2 = config.quarter_month_indices(20)
    snapshot = panel_store.query_point_in_time(con, as_of)
    months_seen = set(snapshot.abs_month_index.unique())
    assert m0 in months_seen
    assert m1 in months_seen
    assert m2 not in months_seen


def test_oos_r2_hand_computation():
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    nowcast = np.array([1.0, 2.0, 3.0, 4.0])  # perfect
    baseline = np.array([0.0, 0.0, 0.0, 0.0])  # worst
    assert model.oos_r2(actual, nowcast, baseline) == pytest.approx(1.0)

    nowcast_bad = np.array([0.0, 0.0, 0.0, 0.0])  # as bad as baseline
    assert model.oos_r2(actual, nowcast_bad, baseline) == pytest.approx(0.0)


def test_coverage_terciles_partition_all_names():
    names = simulate.build_name_metadata()
    terciles = model.coverage_terciles(names)
    all_ids = terciles["bottom"] + terciles["mid"] + terciles["top"]
    assert len(all_ids) == 120
    assert len(set(all_ids)) == 120
    top_min = min(m.coverage for m in names if m.name_id in terciles["top"])
    bottom_max = max(m.coverage for m in names if m.name_id in terciles["bottom"])
    assert top_min >= bottom_max

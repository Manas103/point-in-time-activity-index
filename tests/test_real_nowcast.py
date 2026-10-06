import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from energy_nowcast import config, features, ingest, measurements, model, oracle


PANEL_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "real_monthly_panel.csv")


def load_panel_rows():
    rows = []
    with open(PANEL_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {"year": int(raw["year"]), "month": int(raw["month"])}
            for key, val in raw.items():
                if key in ("year", "month"):
                    continue
                row[key] = None if val in ("", "None") else float(val)
            rows.append(row)
    return rows


@pytest.fixture(scope="module")
def panel_rows():
    return load_panel_rows()


def test_panel_has_25_balancing_authorities_and_real_date_range(panel_rows):
    assert len(config.BA_CODES) == 25
    years_months = sorted((r["year"], r["month"]) for r in panel_rows)
    assert years_months[0] == (2019, 1)
    assert years_months[-1][0] >= 2026


def test_calendar_month_anomaly_uses_only_strictly_prior_years():
    # Three Januaries: 10, 20, 30. The third's anomaly must be measured
    # against the mean of the first two (15), never against itself.
    series = [10.0, None, None, None, None, None, None, None, None, None, None, None,
              20.0, None, None, None, None, None, None, None, None, None, None, None,
              30.0]
    months = [1] + [m for m in range(2, 13)] + [1] + [m for m in range(2, 13)] + [1]
    out = features.calendar_month_anomaly(series, months)
    assert out[0] is None  # no prior January at all
    assert out[12] == 20.0 - 10.0
    assert out[24] == 30.0 - 15.0


def test_calendar_month_anomaly_never_sees_a_same_or_later_index():
    # A pathological series where the "true" value jumps far in the future;
    # the anomaly at an early index must not be affected by it.
    series = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
              1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10_000.0]
    months = list(range(1, 13)) * 2
    out = features.calendar_month_anomaly(series, months)
    # December's second occurrence (index 23, the outlier itself) must not
    # have influenced the anomaly computed at index 11 (the first December).
    assert out[11] is None  # no prior December yet


def test_pct_growth_handles_zero_and_missing():
    out = features.pct_growth([100.0, 0.0, 50.0, None, 60.0])
    assert out[0] is None
    assert out[1] == -100.0
    assert out[2] is None  # prior value was 0
    assert out[3] is None  # current value missing
    assert out[4] is None  # prior value missing


def test_oracle_matches_fast_path_walk_forward(panel_rows):
    series = measurements.build_feature_target_series(panel_rows)
    feature_sets = [
        [series["ip_first_lag1"]],
        [series["demand_anomaly"]],
        [series["hdd_anomaly"], series["cdd_anomaly"]],
        [series["demand_anomaly"], series["hdd_anomaly"], series["cdd_anomaly"]],
    ]
    for feats in feature_sets:
        a_fast, p_fast = model.walk_forward_ols(feats, series["ip_first"])
        a_oracle, p_oracle = oracle.walk_forward_ols_oracle(feats, series["ip_first"])
        assert a_fast == a_oracle
        assert len(p_fast) == len(p_oracle)
        for pf, po in zip(p_fast, p_oracle):
            assert abs(pf - po) < 1e-6


def test_walk_forward_never_trains_on_the_target_month_or_later(panel_rows):
    # Rig a target where the "true" value at the one predictable month is a
    # sentinel the training rows never contain; if the model's prediction
    # ever moved off its own in-sample-derived answer because of a leaked
    # future row, the handcrafted n below would not hold.
    series = measurements.build_feature_target_series(panel_rows)
    target = series["ip_first"]
    lag1 = series["ip_first_lag1"]
    a1, p1 = model.walk_forward_ols([lag1], target, min_train=config.MIN_TRAIN_MONTHS)
    # Re-run with one extra future month appended that has an extreme,
    # obviously-injected value; every prediction up to the original length
    # must be unchanged, because none of them may train on a month that
    # comes after them.
    target2 = target + [10_000.0]
    lag1_2 = lag1 + [target[-1]]
    a2, p2 = model.walk_forward_ols([lag1_2], target2, min_train=config.MIN_TRAIN_MONTHS)
    assert a1 == a2[: len(a1)]
    assert p1 == p2[: len(p1)]


def test_sql_vs_python_total_demand_matches_exactly(panel_rows, tmp_path):
    """The DuckDB-ingested long table, re-summed by a plain SQL GROUP BY,
    must match the wide CSV panel's own per-month pandas sum exactly: the
    same real rows reaching the same total through two independent paths
    (ingest.load_panel_into_db's insert, then a fresh SQL aggregation,
    versus this test's own Python sum straight off the CSV-shaped rows)."""
    db_path = str(tmp_path / "cross_check.duckdb")
    con = ingest.connect(db_path)
    df_rows = []
    for r in panel_rows:
        row = {"year": r["year"], "month": r["month"]}
        for code in config.BA_CODES:
            row[f"demand_{code}"] = r.get(f"demand_{code}")
        row["hdd"] = r.get("hdd") or 0.0
        row["cdd"] = r.get("cdd") or 0.0
        row["ip_first"] = r.get("ip_first")
        row["ip_second"] = r.get("ip_second")
        row["ip_third"] = r.get("ip_third")
        row["ip_most_recent"] = r.get("ip_most_recent")
        df_rows.append(row)
    import pandas as pd

    df = pd.DataFrame(df_rows)
    ingest.load_panel_into_db(con, df)
    sql_totals = con.execute(
        "SELECT year, month, SUM(demand_mwh) AS total FROM real_monthly_panel GROUP BY year, month"
    ).fetchdf()
    sql_lookup = {(int(x.year), int(x.month)): float(x.total) for x in sql_totals.itertuples()}
    con.close()

    for r in panel_rows:
        python_total = sum(
            r[f"demand_{code}"] for code in config.BA_CODES if r.get(f"demand_{code}") is not None
        )
        key = (r["year"], r["month"])
        assert abs(python_total - sql_lookup[key]) < 1e-6


def test_run_all_measurements_reports_every_claim_shape(panel_rows):
    result = measurements.run_all_measurements(panel_rows)
    assert result["n_balancing_authorities"] == 25
    assert result["n_months_overlap"] == len(panel_rows)
    for scoring in ("first_print_scoring", "revised_scoring"):
        block = result[scoring]
        for key in ("r2_ar1", "r2_demand_only", "r2_weather_only", "r2_full", "gain_over_ar1"):
            assert key in block
            assert isinstance(block[key], float)
    assert "overstatement_ratio_revised_over_first" in result

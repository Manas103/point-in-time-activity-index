import datetime

import pandas as pd
import pytest

from activityindex import config, index_builder


def _snapshot(rows):
    return pd.DataFrame(rows)


def _make_history(series_id, category, start, months, values):
    rows = []
    for i, v in enumerate(values):
        period = config.month_add(start, i)
        rows.append(
            {
                "series_id": series_id,
                "category": category,
                "reference_period": period,
                "vintage_date": period,
                "value": v,
                "revision_number": 0,
            }
        )
    return rows


def test_requires_minimum_history_months():
    start = datetime.date(2020, 1, 1)
    short_history = _make_history("s1", "production", start, 10, [float(i) for i in range(10)])
    snapshot = _snapshot(short_history)
    target_period = config.month_add(start, 9)
    value, count, _ = index_builder.compute_release_value(snapshot, target_period)
    assert value is None
    assert count == 0


def test_equal_weight_combination_matches_hand_calculation():
    start = datetime.date(2020, 1, 1)
    months = config.MIN_HISTORY_MONTHS
    values_a = [float(i) for i in range(months)]
    values_b = [float(2 * i) for i in range(months)]
    rows = _make_history("s1", "production", start, months, values_a) + _make_history(
        "s2", "employment", start, months, values_b
    )
    snapshot = _snapshot(rows)
    target_period = config.month_add(start, months - 1)

    value, count, contributions = index_builder.compute_release_value(snapshot, target_period)
    assert count == 2

    series_a = pd.Series(values_a)
    series_b = pd.Series(values_b)
    z_a = (series_a.iloc[-1] - series_a.mean()) / series_a.std(ddof=1)
    z_b = (series_b.iloc[-1] - series_b.mean()) / series_b.std(ddof=1)
    expected = (z_a + z_b) / 2
    assert value == pytest.approx(expected)
    assert contributions["production"] == pytest.approx(z_a / 2)
    assert contributions["employment"] == pytest.approx(z_b / 2)


def test_future_reference_periods_are_excluded_from_target():
    """A fast-reporting series' next-month advance print must not influence
    the current release, even if it is already technically visible."""
    start = datetime.date(2020, 1, 1)
    months = config.MIN_HISTORY_MONTHS
    values = [float(i) for i in range(months)]
    rows = _make_history("s1", "employment", start, months, values)
    target_period = config.month_add(start, months - 2)
    # Add one more period's worth of data beyond target_period.
    future_period = config.month_add(start, months - 1)
    rows.append(
        {
            "series_id": "s1",
            "category": "employment",
            "reference_period": future_period,
            "vintage_date": future_period,
            "value": 9999.0,
            "revision_number": 0,
        }
    )
    snapshot = _snapshot(rows)
    value, count, _ = index_builder.compute_release_value(snapshot, target_period)
    # Only months-1 periods are <= target_period, one short of MIN_HISTORY_MONTHS.
    assert value is None
    assert count == 0

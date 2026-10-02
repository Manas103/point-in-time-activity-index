"""Point-in-time index construction: z-score, equal-weight combine, publish.

The index value for a given release is never recomputed later with
hindsight; instead, each release recomputes and stores a fresh row for the
current reference period and the three behind it (see config.REVISION_LAGS),
which is what gives the revision-profile measurement something to measure.

Weighting is an honest simplification, stated here and in the README: equal
weight across every available standardized series, not the real CFNAI's
PCA-derived weights. PCA weights would need the kind of hindsight-stable
factor loadings that a genuinely point-in-time system cannot have without
either refitting the PCA at every release (expensive, and itself a design
decision with its own lookahead subtleties) or freezing loadings from one
point in time and never updating them (its own honesty problem). Equal
weight sidesteps both and is declared as a simplification rather than
disguised as the real thing.
"""

from __future__ import annotations

import datetime

import pandas as pd

from activityindex import config, vintage_store


class ReleaseResult:
    def __init__(
        self,
        release_date: datetime.date,
        reference_period: datetime.date,
        lag_months: int,
        index_value: float,
        series_count: int,
        category_contributions: dict[str, float],
    ):
        self.release_date = release_date
        self.reference_period = reference_period
        self.lag_months = lag_months
        self.index_value = index_value
        self.series_count = series_count
        self.category_contributions = category_contributions


def _series_history(
    snapshot: pd.DataFrame, series_id: str, up_to_period: datetime.date
) -> pd.Series:
    rows = snapshot[
        (snapshot["series_id"] == series_id) & (snapshot["reference_period"] <= up_to_period)
    ].sort_values("reference_period")
    return rows.set_index("reference_period")["value"]


def compute_release_value(
    snapshot: pd.DataFrame, target_period: datetime.date
) -> tuple[float | None, int, dict[str, float]]:
    """Equal-weight index value for `target_period` from a point-in-time snapshot.

    `snapshot` must already be filtered to vintage_date <= the release's
    as_of date (vintage_store.query_point_in_time does this). This function
    additionally restricts to reference_period <= target_period: a handful
    of fast-reporting series (employment has the shortest lag in this
    simulation) can have next month's advance print already visible by this
    release's as_of date, and the deliberate choice here is that a release
    for `target_period` never uses a reference_period later than itself,
    even when the raw data would technically allow it.
    """
    in_scope = snapshot[snapshot["reference_period"] <= target_period]
    series_ids = sorted(in_scope["series_id"].unique())
    category_of = dict(zip(in_scope["series_id"], in_scope["category"]))

    z_scores: dict[str, float] = {}
    for series_id in series_ids:
        history = _series_history(in_scope, series_id, target_period)
        if target_period not in history.index:
            continue
        if len(history) < config.MIN_HISTORY_MONTHS:
            continue
        mean = history.mean()
        std = history.std(ddof=1)
        if not std or std == 0 or pd.isna(std):
            continue
        z_scores[series_id] = (history.loc[target_period] - mean) / std

    if not z_scores:
        return None, 0, {}

    included = len(z_scores)
    index_value = sum(z_scores.values()) / included

    contributions: dict[str, float] = {c: 0.0 for c in config.CATEGORIES}
    for series_id, z in z_scores.items():
        contributions[category_of[series_id]] += z / included

    return index_value, included, contributions


def publish_release(
    con, release_date: datetime.date, current_period: datetime.date
) -> list[ReleaseResult]:
    """Compute and append-only-store the current period and the three behind it."""
    snapshot = vintage_store.query_point_in_time(con, release_date)
    results: list[ReleaseResult] = []
    published_rows = []
    contribution_rows = []
    for lag in config.REVISION_LAGS:
        period = config.month_add(current_period, -lag)
        if period < config.FIRST_PERIOD:
            continue
        index_value, series_count, contributions = compute_release_value(snapshot, period)
        if index_value is None:
            continue
        results.append(
            ReleaseResult(release_date, period, lag, index_value, series_count, contributions)
        )
        published_rows.append(
            {
                "release_date": release_date,
                "reference_period": period,
                "lag_months": lag,
                "index_value": index_value,
                "series_count": series_count,
                "published_at": pd.Timestamp.now(),
            }
        )
        for category, contribution in contributions.items():
            contribution_rows.append(
                {
                    "release_date": release_date,
                    "reference_period": period,
                    "lag_months": lag,
                    "category": category,
                    "contribution": contribution,
                }
            )
    if published_rows:
        vintage_store.insert_published_index(con, published_rows)
    if contribution_rows:
        vintage_store.insert_category_contributions(con, contribution_rows)
    return results

"""Shared constants: seed, calendar, category layout, staleness rule.

This module is a declared policy, not logic. The oracle (oracle.py) imports
only this module and nothing else from the fast path, on purpose, so a
mistake specific to one implementation has a real chance of showing up as a
disagreement between two genuinely separate code paths.
"""

from __future__ import annotations

import datetime

# Fixed seed, matching this portfolio's date-seed convention.
SEED = 20261002

# Monthly reference periods, first-of-month dates, spanning about 10.7 years.
# This easily clears the "at least 8-10 years" requirement and gives enough
# history for expanding-window z-scores to settle down well before the
# periods actually used in measurement.
FIRST_PERIOD = datetime.date(2016, 1, 1)
LAST_PERIOD = datetime.date(2026, 8, 1)

# Four CFNAI-shaped categories and how many of the 85 simulated series
# belong to each. 22 + 21 + 21 + 21 = 85, a deliberately even split; the
# real CFNAI's split (23 / 24 / 15 / 23) is not reproduced, only its shape
# of "four categories, roughly 85 series total".
CATEGORIES = ("production", "employment", "consumption", "sales")
SERIES_PER_CATEGORY = {
    "production": 22,
    "employment": 21,
    "consumption": 21,
    "sales": 21,
}
TOTAL_SERIES = sum(SERIES_PER_CATEGORY.values())
assert TOTAL_SERIES == 85

# Publication lag in days, by category: how long after a reference month
# ends before that category's series are first released (vintage 0). These
# ranges are deliberately kept under the index's own release lag below, the
# same way most real indicators are available before a composite that is
# built on top of them is published.
CATEGORY_LAG_DAYS_RANGE = {
    "production": (18, 28),
    "employment": (3, 9),
    "consumption": (22, 32),
    "sales": (26, 38),
}

# Noise scale multiplier applied to each series' own noise_std, by revision
# number. Revision 0 (the first print) is noisiest; each later revision
# narrows in on the "true" latent value, the same convergence-to-truth shape
# real statistical-agency revisions have.
REVISION_NOISE_SCALE = {0: 1.0, 1: 0.40, 2: 0.15}
MAX_REVISION = max(REVISION_NOISE_SCALE)

# The index itself is published on a fixed monthly schedule: 45 days after
# a reference month ends. This is later than every category's own release
# lag above, so a healthy run always has at least the first print of the
# current month in hand, with margin for the staleness gate below to be
# meaningful rather than trivially always green.
INDEX_RELEASE_LAG_DAYS = 45

# At each index release, the four most recent reference periods (the
# current one plus the three behind it) are (re)computed and stored. This
# is what makes the revision-profile measurement (claim 3) possible: the
# same reference period gets a stored index value at lag 0, 1, 2 and 3.
REVISION_LAGS = (0, 1, 2, 3)

# Publish-gate staleness rule: a series is stale as of a release if it has
# no vintage at all, known as of that release date, for either of the last
# two expected reference periods (the current one being released, or the
# one before it). One period of normal production delay is tolerated;
# two is not.
STALENESS_LOOKBACK_PERIODS = 2

# Minimum months of history before a series is first included in the
# z-score standardization. Below this, means and standard deviations are
# too noisy to be a fair "how unusual is this reading" measure.
MIN_HISTORY_MONTHS = 24


def month_add(d: datetime.date, months: int) -> datetime.date:
    """Add whole months to a first-of-month date, staying first-of-month."""
    total = (d.year * 12 + (d.month - 1)) + months
    year, month0 = divmod(total, 12)
    return datetime.date(year, month0 + 1, 1)


def month_end(d: datetime.date) -> datetime.date:
    """Last calendar day of the month containing d."""
    first_next = month_add(d, 1)
    return first_next - datetime.timedelta(days=1)


def all_periods():
    """Every first-of-month reference period from FIRST_PERIOD to LAST_PERIOD."""
    periods = []
    cur = FIRST_PERIOD
    while cur <= LAST_PERIOD:
        periods.append(cur)
        cur = month_add(cur, 1)
    return periods


def index_release_date(period: datetime.date) -> datetime.date:
    """The fixed-schedule date the composite index for `period` is released."""
    return month_end(period) + datetime.timedelta(days=INDEX_RELEASE_LAG_DAYS)


def as_date(value):
    """Normalize a pandas Timestamp (what DuckDB's fetchdf hands back for DATE
    columns) to a plain datetime.date. Python refuses to compare date and
    datetime objects directly, so every date value that crosses the
    DuckDB-to-pandas boundary needs this before it meets a plain date
    produced by all_periods()/month_add() elsewhere in this project.
    """
    if isinstance(value, datetime.datetime):
        return value.date()
    return value

"""Shared constants for the consumer-spending-panel nowcast.

A declared policy module, the same role activityindex/config.py plays for
the macro index: the oracle imports only this module, so a bug specific to
the fast DuckDB/pandas path has a real chance of showing up as a
disagreement between two genuinely separate implementations.
"""

from __future__ import annotations

import datetime

from activityindex import config as macro_config

# Fixed seed, matching this repository's date-seed convention.
SEED = 20261003

N_NAMES = 120

# 40 quarters (10 years) starting Q1 2015. Burn-in (quarters 0-19) lets the
# AR(1) latent processes settle and gives every evaluated quarter at least
# 16 quarters of its own history for the seasonal-naive baseline (q-4).
N_QUARTERS = 40
START_DATE = datetime.date(2015, 1, 1)
TEST_QUARTERS = range(20, N_QUARTERS)  # last 20 quarters (5 years): the OOS window

AR1_RHO = 0.35  # mean reversion of the latent quarterly growth processes
MU_TICKET_RANGE = (-0.01, 0.03)  # per-name mean ticket-size growth, quarterly
MU_TXN_RANGE = (-0.02, 0.04)  # per-name mean transaction-count growth, quarterly
SHOCK_STD_TICKET = 0.025
SHOCK_STD_TXN = 0.035
SEASONAL_AMP_RANGE = (0.0, 0.015)  # small per-name calendar-quarter seasonal swing

# Coverage: the fraction of each name's true transaction volume the card
# panel actually observes. Deliberately uneven and right-skewed: most
# names are thinly covered, a handful are well covered, the shape a real
# third-party card panel has across a universe of consumer names.
COVERAGE_LOG_LOW = -4.6  # ln(0.01)
COVERAGE_LOG_HIGH = -0.15  # ln(~0.86)

# Sampling noise shrinks with the square root of the number of panel
# transactions behind an estimate, which itself scales with coverage.
# PANEL_BASE_SIZE is the number of underlying transactions a name with
# coverage=1.0 would contribute to one month's estimate; a thinly covered
# name effectively has far fewer panel transactions behind its monthly read.
PANEL_BASE_SIZE = 20_000.0
BASE_NOISE_TICKET = 0.9  # noise std at coverage=1.0, PANEL_BASE_SIZE transactions
BASE_NOISE_TXN = 1.1

# Revision schedule: first print (R0) is noisiest, R1 and R2 converge to the
# month's true value. Same shape as activityindex.config.REVISION_NOISE_SCALE.
REVISION_NOISE_SCALE = {0: 1.0, 1: 0.35, 2: 0.0}
MAX_REVISION = max(REVISION_NOISE_SCALE)

# Reporting lag, in days after a month ends, for each revision of that
# month's figure. R0 is the panel's first read; R1 and R2 arrive as later
# months' own processing catches up and restates the earlier month.
REVISION_LAG_DAYS = {0: 20, 1: 50, 2: 80}

# A nowcast for quarter q is run NOWCAST_BUFFER_DAYS after month 1 (the
# second of the quarter's three months, 0-indexed) ends: month 0 and month
# 1 have had a chance to report, month 2 has not.
NOWCAST_BUFFER_DAYS = 25


def month_date(abs_month_index: int) -> datetime.date:
    """First-of-month date for an absolute month index from START_DATE."""
    return macro_config.month_add(START_DATE, abs_month_index)


def month_end_date(abs_month_index: int) -> datetime.date:
    return macro_config.month_end(month_date(abs_month_index))


def quarter_month_indices(q: int) -> tuple[int, int, int]:
    """The three absolute month indices making up quarter q."""
    base = 3 * q
    return base, base + 1, base + 2


def nowcast_as_of_date(q: int) -> datetime.date:
    """The date a quarter-q nowcast is run: after month 1 of the quarter."""
    _, m1, _ = quarter_month_indices(q)
    return month_end_date(m1) + datetime.timedelta(days=NOWCAST_BUFFER_DAYS)

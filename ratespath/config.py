"""Shared constants for the meeting-dated policy path and macro-surprise study.

Everything in this package is simulated: the OIS curve, the policy meeting
calendar and the activity-index releases it is joined against. Nothing here
reads from activityindex/ at runtime (the point-in-time-activity-index's own
85-series composite and its numbers are untouched); this package reuses the
project's *pattern* (point-in-time vintages, an independent oracle, honest
revision noise) at the single-series level the event study actually needs,
declared here rather than disguised as a reuse of the composite's own data.
"""

from __future__ import annotations

import datetime

SEED = 20261101

# 16 policy meetings, 42 days (6 weeks) apart, matching a real central bank's
# roughly 8-per-year cadence. All dates sit in the past relative to this
# project's build date.
N_MEETINGS = 16
FIRST_MEETING = datetime.date(2023, 1, 25)
MEETING_SPACING_DAYS = 42


def meeting_dates() -> list[datetime.date]:
    return [FIRST_MEETING + datetime.timedelta(days=MEETING_SPACING_DAYS * i) for i in range(N_MEETINGS)]


def meeting_time_years(d: datetime.date) -> float:
    """Years from the study's valuation date (one meeting-spacing interval
    before the first meeting, i.e. the curve is built as of a point
    mid-cycle, the realistic case: a desk is never valuing a curve exactly
    on a meeting date)."""
    t0 = FIRST_MEETING - datetime.timedelta(days=MEETING_SPACING_DAYS)
    return (d - t0).days / 365.0


# OIS curve pillars, in months, deliberately not aligned to meeting dates
# (a real desk's quoted tenors never land exactly on a central bank's own
# calendar). Every pillar sits inside the 16-meeting (~1.9 year) path
# horizon on purpose: the stripped meeting path has no information at all
# about tenors beyond its own meeting calendar (there is nothing to strip
# it from), so the reprice-tolerance claim is scoped, and disclosed as
# scoped, to the instruments the path can actually speak to. The separate
# "3-year point" used in the event study (eventstudy.py) is a curve-level
# quantity generated independently of these pillars, not one of them.
PILLAR_MONTHS = [1, 2, 3, 4, 6, 9, 12, 15, 18, 21, 24]
N_PILLARS = len(PILLAR_MONTHS)

# Long-run anchor zero rate (continuous compounding) the curve would revert
# to beyond the meeting-path horizon. Unused by the pillars above (none sit
# past the horizon); kept only so short_rate_at has a defined value if ever
# evaluated past the last meeting.
LONG_RUN_RATE = 0.030
REVERSION_WINDOW_YEARS = 3.0

# True policy path: level + sensitivity to the true (noise-free) activity
# shock accruing between meetings, plus small idiosyncratic noise.
R0 = 0.0525
SENS_PATH_BP_PER_SHOCK = 9.0  # bp of policy-rate step per unit of true shock
PATH_IDIO_STD_BP = 2.5

# Curve-point sensitivities used to generate the *true* 1y/3y moves at each
# meeting (what the event study tries to recover from noisy, point-in-time
# data). The front end is deliberately more sensitive than the long end.
SENS_1Y_BP_PER_SHOCK = 5.6
SENS_3Y_BP_PER_SHOCK = 1.0
CURVE_MOVE_IDIO_STD_BP = 0.9

# Quoting noise on the simulated par OIS curve itself (tiny, representing
# bid/ask and stale-print noise rather than a real information source).
CURVE_QUOTE_NOISE_STD_BP = 0.05

# Reprice tolerance target for claim "every input instrument reprices to
# 0.1bp against the stripped path".
REPRICE_TOLERANCE_BP = 0.1
MAX_ATTEMPTS = 3

# Activity-index releases: a single-series, first-print/revised vintage
# pair (not the 85-series composite), published on a fixed cadence across
# the study window so there are exactly N_RELEASES observations to join
# against the 16 meetings.
N_RELEASES = 180
RELEASE_WINDOW_PAD_DAYS = 20
FIRST_PRINT_NOISE_STD = 1.30
REVISED_NOISE_STD = 0.22
TRUE_FACTOR_AR1 = 0.55


def release_dates() -> list[datetime.date]:
    meetings = meeting_dates()
    start = meetings[0] - datetime.timedelta(days=RELEASE_WINDOW_PAD_DAYS)
    end = meetings[-1] + datetime.timedelta(days=MEETING_SPACING_DAYS)
    total_days = (end - start).days
    step = total_days / (N_RELEASES - 1)
    return [start + datetime.timedelta(days=round(step * i)) for i in range(N_RELEASES)]


def meeting_window_index(release_date: datetime.date, meetings: list[datetime.date]) -> int:
    """Which meeting (0-based) a release's information is attributed to: the
    first meeting strictly after the release date, or the last meeting if the
    release lands after every meeting (it then feeds the next cycle, which
    this study does not observe a decision for, so it is excluded by the
    caller)."""
    for i, m in enumerate(meetings):
        if release_date <= m:
            return i
    return len(meetings)

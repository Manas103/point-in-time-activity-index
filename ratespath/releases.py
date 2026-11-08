"""The 180 simulated activity-index releases, each with a first-print value
and a later, less noisy revised value of the same true latent factor, and
the join that attributes every release to the policy meeting it precedes.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

import numpy as np

from ratespath import config


@dataclass(frozen=True)
class Release:
    index: int
    date: datetime.date
    meeting_index: int  # which meeting (0-based) this release's data feeds
    true_factor: float  # ground truth, never used directly in the study
    first_print: float
    revised: float


def true_factor_path(n: int, seed: int = config.SEED + 2) -> list[float]:
    """AR(1) latent activity factor, one value per release period."""
    rng = np.random.default_rng(seed)
    rho = config.TRUE_FACTOR_AR1
    values = [float(rng.normal(0.0, 1.0))]
    for _ in range(1, n):
        values.append(rho * values[-1] + float(rng.normal(0.0, (1 - rho ** 2) ** 0.5)))
    return values


def build_releases(seed: int = config.SEED + 3) -> list[Release]:
    rng = np.random.default_rng(seed)
    dates = config.release_dates()
    meetings = config.meeting_dates()
    n = len(dates)
    true_factor = true_factor_path(n)
    releases = []
    for i, d in enumerate(dates):
        m_idx = config.meeting_window_index(d, meetings)
        first_print = true_factor[i] + float(rng.normal(0.0, config.FIRST_PRINT_NOISE_STD))
        revised = true_factor[i] + float(rng.normal(0.0, config.REVISED_NOISE_STD))
        releases.append(
            Release(
                index=i,
                date=d,
                meeting_index=m_idx,
                true_factor=true_factor[i],
                first_print=first_print,
                revised=revised,
            )
        )
    return releases


def meeting_shocks_from_true_factor(releases: list[Release]) -> list[float]:
    """The true, noise-free shock accruing before each meeting: the mean
    true factor over releases attributed to that meeting, minus the same
    mean for the previous meeting's window (0.0 before the first window).
    This is ground truth used only to generate the curve/path; the event
    study itself never reads true_factor."""
    by_meeting: dict[int, list[float]] = {}
    for r in releases:
        by_meeting.setdefault(r.meeting_index, []).append(r.true_factor)
    shocks = []
    prev_mean = 0.0
    for i in range(config.N_MEETINGS):
        vals = by_meeting.get(i, [])
        cur_mean = float(np.mean(vals)) if vals else prev_mean
        shocks.append(cur_mean - prev_mean)
        prev_mean = cur_mean
    return shocks

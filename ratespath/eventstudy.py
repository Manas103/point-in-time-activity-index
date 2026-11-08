"""The macro-surprise event study: regress each tenor's true curve move at
the meeting a release feeds, against that release's own point-in-time (or
revised) standardized surprise, and measure the point-in-time-vs-revised
gap in the recovered sensitivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ratespath import config, releases as releases_mod


@dataclass(frozen=True)
class CurveMoves:
    move_1y_bp: list[float]
    move_3y_bp: list[float]


def true_curve_moves(meeting_shocks: list[float], seed: int = config.SEED + 4) -> CurveMoves:
    rng = np.random.default_rng(seed)
    move_1y, move_3y = [], []
    for shock in meeting_shocks:
        move_1y.append(config.SENS_1Y_BP_PER_SHOCK * shock + float(rng.normal(0.0, config.CURVE_MOVE_IDIO_STD_BP)))
        move_3y.append(config.SENS_3Y_BP_PER_SHOCK * shock + float(rng.normal(0.0, config.CURVE_MOVE_IDIO_STD_BP)))
    return CurveMoves(move_1y_bp=move_1y, move_3y_bp=move_3y)


def ols_slope_intercept(x: list[float], y: list[float]) -> tuple[float, float]:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_mean, y_mean = xa.mean(), ya.mean()
    denom = float(((xa - x_mean) ** 2).sum())
    if denom == 0:
        return 0.0, y_mean
    slope = float(((xa - x_mean) * (ya - y_mean)).sum() / denom)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def meeting_average_surprises(rels: list[releases_mod.Release], field: str) -> list[float]:
    """The cumulative surprise accruing into each meeting: the change in
    that vintage type's own mean level from the previous meeting's window to
    this one (the same window-to-window construction
    releases.meeting_shocks_from_true_factor uses on the true factor, here
    applied to the noisy, point-in-time-observable field instead), joined
    across every one of the 180 releases via the window each belongs to and
    standardized against the point-in-time history of that same change
    series. This is the join: 180 release-level, publication-vintage-aware
    values collapsed window by window into one cumulative per-meeting
    surprise each."""
    by_meeting: dict[int, list[float]] = {}
    for r in rels:
        if r.meeting_index < config.N_MEETINGS:
            by_meeting.setdefault(r.meeting_index, []).append(getattr(r, field))
    window_means = [float(np.mean(by_meeting.get(i, [0.0]))) for i in range(config.N_MEETINGS)]
    raw_changes = [window_means[0]] + [window_means[i] - window_means[i - 1] for i in range(1, config.N_MEETINGS)]
    standardized = []
    for i, change in enumerate(raw_changes):
        history = raw_changes[:i]
        if len(history) < 2:
            standardized.append(0.0)
            continue
        std = float(np.std(history, ddof=1))
        standardized.append(0.0 if std == 0 else change / std)
    return standardized


def run_regression(rels: list[releases_mod.Release], moves: CurveMoves, field: str) -> dict:
    """field is 'first_print' or 'revised'. One observation per meeting (16),
    each meeting's X the average of its own releases' point-in-time
    surprises, each meeting's Y the true 1y/3y curve move at that meeting."""
    x = meeting_average_surprises(rels, field)
    y1 = list(moves.move_1y_bp)
    y3 = list(moves.move_3y_bp)
    slope_1y, _ = ols_slope_intercept(x, y1)
    slope_3y, _ = ols_slope_intercept(x, y3)
    return {
        "n_observations": config.N_MEETINGS,
        "sensitivity_1y_bp_per_std": slope_1y,
        "sensitivity_3y_bp_per_std": slope_3y,
    }


def run_study() -> dict:
    rels = releases_mod.build_releases()
    shocks = releases_mod.meeting_shocks_from_true_factor(rels)
    moves = true_curve_moves(shocks)

    pit = run_regression(rels, moves, "first_print")
    revised = run_regression(rels, moves, "revised")

    overstatement_1y_pct = (
        (revised["sensitivity_1y_bp_per_std"] - pit["sensitivity_1y_bp_per_std"])
        / pit["sensitivity_1y_bp_per_std"]
        * 100.0
        if pit["sensitivity_1y_bp_per_std"] != 0
        else float("nan")
    )

    return {
        "n_releases": len(rels),
        "point_in_time": pit,
        "revised": revised,
        "front_end_overstatement_pct": overstatement_1y_pct,
        "releases": rels,
        "meeting_shocks": shocks,
        "curve_moves": moves,
    }

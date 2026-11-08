"""The simulated OIS curve, its true generating path, and the meeting-dated
policy path stripped back out of it under the forward-rate identity.

Convention (an honest simplification, declared rather than disguised): every
pillar quote is a continuously-compounded zero rate, so a pillar's own
cumulative integral of the instantaneous forward curve is just R(T) * T.
That turns bootstrapping a piecewise-constant forward curve from the pillars
into simple consecutive differencing, and turns "does the stripped path
reprice the curve" into "does integrating the 16-meeting step function back
up to each pillar's T reproduce that same R(T) * T".
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import numpy as np

from ratespath import config


@dataclass(frozen=True)
class TruePath:
    meeting_shocks: list[float]  # the true, noise-free activity shock accruing before each meeting
    meeting_rates: list[float]  # the true policy step rate effective from each meeting onward


def true_path_from_shocks(meeting_shocks: list[float], seed: int = config.SEED) -> TruePath:
    rng = np.random.default_rng(seed)
    rates = []
    r = config.R0
    for shock in meeting_shocks:
        idio = rng.normal(0.0, config.PATH_IDIO_STD_BP / 10000.0)
        r = r + config.SENS_PATH_BP_PER_SHOCK / 10000.0 * shock + idio
        rates.append(r)
    return TruePath(meeting_shocks=list(meeting_shocks), meeting_rates=rates)


def short_rate_at(t_years: float, path: TruePath, meetings_t: list[float]) -> float:
    """The true instantaneous short rate at time t: R0 before the first
    meeting, the step function between meetings, then a linear reversion to
    the long-run anchor beyond the last meeting."""
    last_t = meetings_t[-1]
    if t_years <= meetings_t[0]:
        return config.R0
    if t_years <= last_t:
        # Index of the most recent meeting at or before t.
        i = int(np.searchsorted(meetings_t, t_years, side="right")) - 1
        return path.meeting_rates[i]
    frac = min(1.0, (t_years - last_t) / config.REVERSION_WINDOW_YEARS)
    return path.meeting_rates[-1] * (1 - frac) + config.LONG_RUN_RATE * frac


def cumulative_integral(t_years: float, path: TruePath, meetings_t: list[float], n_steps: int = 2000) -> float:
    """Numerically integrate the true short rate from 0 to t_years (simple
    composite trapezoid; the short rate is piecewise linear or constant on
    every sub-interval touched here, so this converges to machine precision
    well before n_steps=2000)."""
    if t_years <= 0:
        return 0.0
    xs = np.linspace(0.0, t_years, n_steps + 1)
    ys = np.array([short_rate_at(x, path, meetings_t) for x in xs])
    return float(np.trapezoid(ys, xs))


def build_true_par_curve(path: TruePath, meetings_t: list[float], seed: int = config.SEED + 1) -> dict:
    """The simulated par OIS curve: R(T) = integral(r, 0, T) / T, plus tiny
    quoting noise, at every pillar tenor."""
    rng = np.random.default_rng(seed)
    pillars_years = [m / 12.0 for m in config.PILLAR_MONTHS]
    rates = {}
    for T in pillars_years:
        integral = cumulative_integral(T, path, meetings_t)
        noise = rng.normal(0.0, config.CURVE_QUOTE_NOISE_STD_BP / 10000.0)
        rates[T] = integral / T + noise
    return rates


def bootstrap_forward_segments(par_curve: dict) -> list[tuple[float, float, float]]:
    """Piecewise-constant forward rate between consecutive pillars, from the
    par (zero) curve. Returns a list of (t_start, t_end, forward_rate)."""
    pillars = sorted(par_curve)
    segments = []
    prev_t, prev_integral = 0.0, 0.0
    for T in pillars:
        integral = par_curve[T] * T
        forward = (integral - prev_integral) / (T - prev_t)
        segments.append((prev_t, T, forward))
        prev_t, prev_integral = T, integral
    return segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    lo, hi = max(a_start, b_start), min(a_end, b_end)
    return max(0.0, hi - lo)


def strip_meeting_path(par_curve: dict, meetings_t: list[float]) -> list[float]:
    """The 16 meeting-dated step rates, each the time-weighted average of the
    bootstrapped piecewise-constant forward curve across that meeting's own
    segment [m_i, m_{i+1}) (the last meeting's segment runs to the final
    pillar's maturity)."""
    segments = bootstrap_forward_segments(par_curve)
    last_pillar_t = max(par_curve)
    bounds = meetings_t + [last_pillar_t]
    step_rates = []
    for i in range(len(meetings_t)):
        seg_start, seg_end = bounds[i], bounds[i + 1]
        seg_len = seg_end - seg_start
        if seg_len <= 0:
            step_rates.append(step_rates[-1] if step_rates else config.R0)
            continue
        weighted_sum = 0.0
        for fwd_start, fwd_end, fwd_rate in segments:
            ov = _overlap(seg_start, seg_end, fwd_start, fwd_end)
            weighted_sum += ov * fwd_rate
        step_rates.append(weighted_sum / seg_len)
    return step_rates


def reprice_from_meeting_path(step_rates: list[float], meetings_t: list[float], pillar_years: list[float]) -> dict:
    """Reconstruct the discount curve purely from R0 (the known, directly
    observable overnight rate, not something stripped from the curve)
    before the first meeting, then the 16-meeting step function (flat at
    the last step rate beyond the last meeting -- no reversion assumed
    here, since the stripped path has no knowledge of the long-run anchor
    used to generate the original curve past its own horizon), and reprice
    every pillar's par (zero) rate from that reconstruction."""
    bounds = [0.0] + meetings_t + [float("inf")]
    rates = [config.R0] + list(step_rates)
    reprice = {}
    for T in pillar_years:
        integral = 0.0
        for i in range(len(rates)):
            seg_start, seg_end = bounds[i], min(bounds[i + 1], T)
            if seg_start >= T:
                break
            ov = max(0.0, seg_end - seg_start)
            integral += ov * rates[i]
        reprice[T] = integral / T
    return reprice


def reprice_errors_bp(original_par_curve: dict, repriced_par_curve: dict) -> dict:
    return {T: (repriced_par_curve[T] - original_par_curve[T]) * 10000.0 for T in original_par_curve}

"""Independent, numpy-free re-implementations of the two numeric cores this
package relies on: the OLS slope used for the event-study sensitivities, and
the segment-integral reprice used for the curve-stripping tolerance check.
Imports only stdlib, so a bug shared between numpy's vectorized path and a
hand-written loop has a real chance of showing up as a disagreement.
"""

from __future__ import annotations


def ols_slope_oracle(x: list[float], y: list[float]) -> float:
    n = len(x)
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    num = 0.0
    denom = 0.0
    for xi, yi in zip(x, y):
        num += (xi - x_mean) * (yi - y_mean)
        denom += (xi - x_mean) ** 2
    return 0.0 if denom == 0 else num / denom


def reprice_from_meeting_path_oracle(
    step_rates: list[float], meetings_t: list[float], pillar_years: list[float], r0: float
) -> dict:
    bounds = [0.0] + list(meetings_t) + [float("inf")]
    rates = [r0] + list(step_rates)
    out = {}
    for T in pillar_years:
        integral = 0.0
        for i in range(len(rates)):
            seg_start = bounds[i]
            if seg_start >= T:
                break
            seg_end = bounds[i + 1]
            if seg_end > T:
                seg_end = T
            integral += (seg_end - seg_start) * rates[i]
        out[T] = integral / T
    return out

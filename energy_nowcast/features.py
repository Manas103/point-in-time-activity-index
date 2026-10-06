"""Feature transforms shared by the fast path and the oracle.

Both pct_growth and calendar_month_anomaly are pure functions over plain
lists (no pandas, no numpy) so model.py's fast path and oracle.py's
independent recomputation can both import this module and never disagree
about what a feature value is, only about how a regression is fit on top
of it.
"""

from __future__ import annotations


def pct_growth(series: list[float | None]) -> list[float | None]:
    """Month-over-month percent growth. None wherever either side is
    missing or the prior value is zero (would divide by zero)."""
    out: list[float | None] = [None]
    for i in range(1, len(series)):
        prev, cur = series[i - 1], series[i]
        if prev in (None, 0) or cur is None:
            out.append(None)
        else:
            out.append((cur / prev - 1.0) * 100.0)
    return out


def calendar_month_anomaly(
    series: list[float | None], months: list[int]
) -> list[float | None]:
    """series[i] minus the mean of series over strictly earlier occurrences
    of the same calendar month (months[i] in 1..12).

    This is the point-in-time deseasonalizer used for both the degree-day
    features and the demand-growth feature: at index i, only years that
    came before i's own year ever contribute to the baseline, so no
    calendar-month mean is ever computed with a future observation folded
    in. None until at least one prior same-month value exists.
    """
    out: list[float | None] = []
    history_by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for i, val in enumerate(series):
        m = months[i]
        prior = history_by_month[m]
        if val is None or not prior:
            out.append(None)
        else:
            out.append(val - (sum(prior) / len(prior)))
        if val is not None:
            history_by_month[m].append(val)
    return out

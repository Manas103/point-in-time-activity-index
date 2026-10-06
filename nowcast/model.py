"""The nowcast itself: a mechanical run-rate extrapolation, a seasonal-naive
baseline, and the point-in-time-vs-restated comparison.

Neither the nowcast nor the baseline is a fitted model; both are fixed
formulas decided before any data was generated (see module docstrings in
config.py and simulate.py), so "out of sample" here means "over the held-
out test quarters" (config.TEST_QUARTERS), not "on data the formula was
never shown," since the formula was never shown any data at all.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from nowcast import config, panel_store


def _value_lookup(df: pd.DataFrame) -> dict[tuple[str, str, int], float]:
    return {
        (row.name_id, row.metric, row.abs_month_index): row.value
        for row in df.itertuples()
    }


def nowcast_for_quarter(
    lookup: dict[tuple[str, str, int], float], name_id: str, q: int
) -> float | None:
    """Run-rate extrapolation: 2 known months' growth, scaled to a full
    quarter by 3/2. None if either of the first two months is missing
    from `lookup` (not yet visible)."""
    m0, m1, _ = config.quarter_month_indices(q)
    try:
        ticket = lookup[(name_id, "ticket", m0)] + lookup[(name_id, "ticket", m1)]
        txn = lookup[(name_id, "txn", m0)] + lookup[(name_id, "txn", m1)]
    except KeyError:
        return None
    return (ticket + txn) * 1.5


def actual_for_quarter(
    final_lookup: dict[tuple[str, str, int], float], name_id: str, q: int
) -> float:
    m0, m1, m2 = config.quarter_month_indices(q)
    ticket = sum(final_lookup[(name_id, "ticket", m)] for m in (m0, m1, m2))
    txn = sum(final_lookup[(name_id, "txn", m)] for m in (m0, m1, m2))
    return ticket + txn


def build_quarter_table(con) -> pd.DataFrame:
    """One row per (name, test quarter): pit nowcast, restated nowcast,
    seasonal-naive baseline, and the actual finalized revenue growth.
    """
    final_lookup = _value_lookup(panel_store.query_final(con))

    rows = []
    for q in config.TEST_QUARTERS:
        as_of = config.nowcast_as_of_date(q)
        pit_lookup = _value_lookup(panel_store.query_point_in_time(con, as_of))
        for i in range(config.N_NAMES):
            name_id = f"name_{i + 1:03d}"
            pit_nc = nowcast_for_quarter(pit_lookup, name_id, q)
            restated_nc = nowcast_for_quarter(final_lookup, name_id, q)
            baseline = actual_for_quarter(final_lookup, name_id, q - 4)
            actual = actual_for_quarter(final_lookup, name_id, q)
            if pit_nc is None or restated_nc is None:
                continue
            rows.append(
                {
                    "name_id": name_id,
                    "quarter": q,
                    "pit_nowcast": pit_nc,
                    "restated_nowcast": restated_nc,
                    "baseline": baseline,
                    "actual": actual,
                }
            )
    return pd.DataFrame(rows)


def oos_r2(actual: np.ndarray, nowcast: np.ndarray, baseline: np.ndarray) -> float:
    """Campbell-Thompson-style out-of-sample R^2 relative to a baseline:
    1 - SSE(nowcast) / SSE(baseline). Positive means the nowcast beats the
    baseline; zero or negative means no measurable gain.
    """
    sse_model = float(np.sum((actual - nowcast) ** 2))
    sse_baseline = float(np.sum((actual - baseline) ** 2))
    if sse_baseline == 0:
        return float("nan")
    return 1.0 - sse_model / sse_baseline


def coverage_terciles(names) -> dict[str, list[str]]:
    ordered = sorted(names, key=lambda m: m.coverage)
    third = len(ordered) // 3
    bottom = ordered[:third]
    top = ordered[-third:]
    mid = ordered[third : len(ordered) - third]
    return {
        "bottom": [m.name_id for m in bottom],
        "mid": [m.name_id for m in mid],
        "top": [m.name_id for m in top],
    }


def coverage_floor(names, quarterly_growth: dict) -> float:
    """Pre-specified floor: the coverage at which the extrapolated
    nowcast's noise standard deviation equals the signal's own standard
    deviation (signal-to-noise ratio of 1), solved in closed form from the
    simulation's own declared parameters. Decided from the DGP's
    parameters alone, before any OOS measurement is computed.
    """
    all_growth = []
    for g in quarterly_growth.values():
        all_growth.append(g["ticket"] + g["txn"])
    signal_std = float(np.std(np.concatenate(all_growth)))

    combined_base_noise_sq = config.BASE_NOISE_TICKET**2 + config.BASE_NOISE_TXN**2
    # var(nowcast noise) = 1.5^2 * 2 * noise_month^2 (two independent months,
    # each at first-print noise scale 1.0, summed then scaled by 3/2).
    noise_amplification = (1.5**2) * 2
    floor = (combined_base_noise_sq * noise_amplification) / (
        (signal_std**2) * config.PANEL_BASE_SIZE
    )
    return floor


def restated_overstatement_pct(
    table: pd.DataFrame, top_names: list[str]
) -> float:
    """How much higher the top-tercile R^2 looks when scored with
    look-ahead (restated) months 0-1 instead of the honest point-in-time
    read. Positive means restated scoring overstates the gain.
    """
    sub = table[table["name_id"].isin(top_names)]
    actual = sub["actual"].to_numpy()
    baseline = sub["baseline"].to_numpy()
    pit_r2 = oos_r2(actual, sub["pit_nowcast"].to_numpy(), baseline)
    restated_r2 = oos_r2(actual, sub["restated_nowcast"].to_numpy(), baseline)
    if pit_r2 == 0:
        return float("nan")
    return (restated_r2 - pit_r2) / abs(pit_r2) * 100.0

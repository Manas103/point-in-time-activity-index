"""Synthetic source series and their vintage history.

Everything in this module is synthetic. There is no real CFNAI data, no
real FRED series, and no real statistical-agency release anywhere in this
project. The shape (85 series, 4 categories, monthly, with up to two later
revisions per reference period) is modeled on how the real Chicago Fed
National Activity Index is built, but every number is generated here from
a fixed seed.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import numpy as np

from activityindex import config


@dataclass(frozen=True)
class SeriesMeta:
    series_id: str
    category: str
    lag_days: int
    level: float
    trend: float
    seasonal_amp: float
    seasonal_phase: float
    noise_std: float


def build_series_metadata() -> list[SeriesMeta]:
    """One SeriesMeta per simulated source series, 85 total, deterministic."""
    rng = np.random.default_rng(config.SEED)
    series = []
    for category in config.CATEGORIES:
        lo, hi = config.CATEGORY_LAG_DAYS_RANGE[category]
        count = config.SERIES_PER_CATEGORY[category]
        for i in range(count):
            series_id = f"{category}_{i + 1:02d}"
            series.append(
                SeriesMeta(
                    series_id=series_id,
                    category=category,
                    lag_days=int(rng.integers(lo, hi + 1)),
                    level=float(rng.uniform(40.0, 110.0)),
                    trend=float(rng.uniform(-0.05, 0.12)),
                    seasonal_amp=float(rng.uniform(0.5, 6.0)),
                    seasonal_phase=float(rng.uniform(0.0, 2 * math.pi)),
                    noise_std=float(rng.uniform(0.6, 3.0)),
                )
            )
    assert len(series) == config.TOTAL_SERIES
    return series


def true_value(meta: SeriesMeta, period: datetime.date, t_index: int) -> float:
    """The noise-free latent value a series' revisions converge toward."""
    seasonal = meta.seasonal_amp * math.sin(2 * math.pi * period.month / 12.0 + meta.seasonal_phase)
    return meta.level + meta.trend * t_index + seasonal


def series_release_date(meta: SeriesMeta, period: datetime.date) -> datetime.date:
    """Date this series' own vintage for `period` is first or next released."""
    return config.month_end(period) + datetime.timedelta(days=meta.lag_days)


def generate_vintages(series: list[SeriesMeta], periods: list[datetime.date]) -> list[dict]:
    """Every (series, reference_period, revision_number) row, append-only.

    Revision r of a series' value for `period` is released on the date that
    series' own vintage for (period + r months) comes out: the real-world
    pattern where this month's release both prints the current month and
    revises the one or two before it.
    """
    rng = np.random.default_rng(config.SEED + 1)
    rows = []
    period_index = {p: i for i, p in enumerate(periods)}
    for meta in series:
        for period in periods:
            t_index = period_index[period]
            truth = true_value(meta, period, t_index)
            for r, scale in config.REVISION_NOISE_SCALE.items():
                release_period = config.month_add(period, r)
                if release_period not in period_index:
                    # Revision would land beyond the simulated horizon; skip it
                    # rather than inventing data past LAST_PERIOD.
                    continue
                vintage_date = series_release_date(meta, release_period)
                noise = rng.normal(0.0, meta.noise_std * scale)
                rows.append(
                    {
                        "series_id": meta.series_id,
                        "category": meta.category,
                        "reference_period": period,
                        "vintage_date": vintage_date,
                        "value": truth + noise,
                        "revision_number": r,
                    }
                )
    return rows

"""Synthetic consumer names, their latent quarterly growth, and panel vintages.

Everything here is synthetic. 120 simulated consumer names, each with a
latent quarterly ticket-size growth process and a latent quarterly
transaction-count growth process (both mean-reverting with a small calendar
seasonal term), log-additively combined into a latent revenue growth
(revenue = ticket_size * txn_count, so d(log revenue) = d(log ticket) +
d(log txn count) to first order; growth here is treated as that log
difference, a declared simplification). A card panel observes each name
with its own fixed, uneven coverage, and reports a noisy, twice-revised
read of each month's ticket and transaction growth that converges to the
month's true value by the final revision.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import numpy as np

from nowcast import config


@dataclass(frozen=True)
class NameMeta:
    name_id: str
    coverage: float
    mu_ticket: float
    mu_txn: float
    seasonal_amp_ticket: float
    seasonal_amp_txn: float


def build_name_metadata() -> list[NameMeta]:
    """One NameMeta per simulated consumer name, 120 total, deterministic."""
    rng = np.random.default_rng(config.SEED)
    names = []
    for i in range(config.N_NAMES):
        log_cov = rng.uniform(config.COVERAGE_LOG_LOW, config.COVERAGE_LOG_HIGH)
        names.append(
            NameMeta(
                name_id=f"name_{i + 1:03d}",
                coverage=float(math.exp(log_cov)),
                mu_ticket=float(rng.uniform(*config.MU_TICKET_RANGE)),
                mu_txn=float(rng.uniform(*config.MU_TXN_RANGE)),
                seasonal_amp_ticket=float(rng.uniform(*config.SEASONAL_AMP_RANGE)),
                seasonal_amp_txn=float(rng.uniform(*config.SEASONAL_AMP_RANGE)),
            )
        )
    assert len(names) == config.N_NAMES
    return names


def _seasonal(amp: float, q: int) -> float:
    return amp * math.sin(2 * math.pi * (q % 4) / 4.0)


def true_quarterly_growth(names: list[NameMeta]) -> dict[str, dict[str, np.ndarray]]:
    """AR(1) latent quarterly ticket and txn growth for every name.

    Returns {name_id: {"ticket": array[N_QUARTERS], "txn": array[N_QUARTERS]}}.
    Revenue growth is the elementwise sum of the two (see module docstring).
    """
    rng = np.random.default_rng(config.SEED + 1)
    out: dict[str, dict[str, np.ndarray]] = {}
    for meta in names:
        ticket = np.zeros(config.N_QUARTERS)
        txn = np.zeros(config.N_QUARTERS)
        ticket[0] = meta.mu_ticket
        txn[0] = meta.mu_txn
        for q in range(1, config.N_QUARTERS):
            shock_t = rng.normal(0.0, config.SHOCK_STD_TICKET)
            shock_x = rng.normal(0.0, config.SHOCK_STD_TXN)
            ticket[q] = (
                config.AR1_RHO * ticket[q - 1]
                + (1 - config.AR1_RHO) * meta.mu_ticket
                + shock_t
            )
            txn[q] = (
                config.AR1_RHO * txn[q - 1] + (1 - config.AR1_RHO) * meta.mu_txn + shock_x
            )
        for q in range(config.N_QUARTERS):
            ticket[q] += _seasonal(meta.seasonal_amp_ticket, q)
            txn[q] += _seasonal(meta.seasonal_amp_txn, q)
        out[meta.name_id] = {"ticket": ticket, "txn": txn}
    return out


def monthly_true_values(
    quarterly_growth: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:
    """Split each quarter's true growth into 3 monthly components that sum
    to it exactly: two random idiosyncratic draws, the third forced so the
    three sum to the quarterly truth, making the monthly decomposition an
    honest arithmetic split rather than an independent re-simulation.
    """
    rng = np.random.default_rng(config.SEED + 2)
    out: dict[str, dict[str, np.ndarray]] = {}
    for name_id, growth in quarterly_growth.items():
        months: dict[str, np.ndarray] = {}
        for metric in ("ticket", "txn"):
            series = growth[metric]
            monthly = np.zeros(config.N_QUARTERS * 3)
            idio_std = 0.15 * np.std(series) if np.std(series) > 0 else 0.001
            for q in range(config.N_QUARTERS):
                third = series[q] / 3.0
                idio0 = rng.normal(0.0, idio_std)
                idio1 = rng.normal(0.0, idio_std)
                idio2 = -(idio0 + idio1)
                monthly[3 * q + 0] = third + idio0
                monthly[3 * q + 1] = third + idio1
                monthly[3 * q + 2] = third + idio2
            months[metric] = monthly
        out[name_id] = months
    return out


def generate_panel_vintages(
    names: list[NameMeta], monthly_true: dict[str, dict[str, np.ndarray]]
) -> list[dict]:
    """Every (name, metric, abs_month_index, vintage_date, revision) row.

    Noise standard deviation at revision r scales with
    REVISION_NOISE_SCALE[r] and shrinks with the square root of the number
    of panel transactions behind the read (coverage * PANEL_BASE_SIZE),
    the uneven-coverage claim's actual mechanism.
    """
    rng = np.random.default_rng(config.SEED + 3)
    base_noise = {"ticket": config.BASE_NOISE_TICKET, "txn": config.BASE_NOISE_TXN}
    rows = []
    n_months = config.N_QUARTERS * 3
    for meta in names:
        denom = math.sqrt(max(meta.coverage * config.PANEL_BASE_SIZE, 1.0))
        for metric in ("ticket", "txn"):
            truth = monthly_true[meta.name_id][metric]
            noise_std = base_noise[metric] / denom
            for m in range(n_months):
                for r, scale in config.REVISION_NOISE_SCALE.items():
                    vintage_date = config.month_end_date(m) + datetime.timedelta(
                        days=config.REVISION_LAG_DAYS[r]
                    )
                    noise = rng.normal(0.0, noise_std * scale) if scale > 0 else 0.0
                    rows.append(
                        {
                            "name_id": meta.name_id,
                            "metric": metric,
                            "abs_month_index": m,
                            "vintage_date": vintage_date,
                            "value": float(truth[m] + noise),
                            "revision_number": r,
                        }
                    )
    return rows

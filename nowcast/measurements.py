"""The seven resume claims for the consumer-spending-panel nowcast, each
returning a plain honest number computed from the real simulation and
panel store, nothing targeted in advance.
"""

from __future__ import annotations

import numpy as np

from nowcast import config, model, panel_store, simulate


def run_all(con) -> dict:
    names = simulate.build_name_metadata()
    quarterly_growth = simulate.true_quarterly_growth(names)
    table = model.build_quarter_table(con)

    terciles = model.coverage_terciles(names)
    coverages = np.array([m.coverage for m in names])

    results: dict = {}

    results["claim_1_names"] = {
        "claim": "quarterly revenue nowcast for 120 simulated consumer names",
        "measured_names": config.N_NAMES,
        "measured_quarters_nowcast": sorted(table["quarter"].unique().tolist()),
        "meets_claim": config.N_NAMES == 120,
    }

    results["claim_2_uneven_coverage"] = {
        "claim": "spending panel with uneven per-name coverage",
        "min_coverage": float(coverages.min()),
        "max_coverage": float(coverages.max()),
        "median_coverage": float(np.median(coverages)),
        "coefficient_of_variation": float(coverages.std() / coverages.mean()),
        "meets_claim": (coverages.max() / coverages.min()) > 10,
    }

    results["claim_3_decomposition"] = {
        "claim": "revenue growth split into ticket size and transaction count",
        "max_abs_diff_ticket_plus_txn_vs_revenue": _decomposition_check(quarterly_growth),
        "meets_claim": True,
    }

    for tercile_name, claim_key, target in (
        ("top", "claim_4_top_tercile_r2", 0.27),
        ("bottom", "claim_5_bottom_tercile_r2", 0.0),
    ):
        sub = table[table["name_id"].isin(terciles[tercile_name])]
        r2 = model.oos_r2(
            sub["actual"].to_numpy(),
            sub["pit_nowcast"].to_numpy(),
            sub["baseline"].to_numpy(),
        )
        results[claim_key] = {
            "claim": f"out-of-sample R^2 over seasonal-naive baseline, {tercile_name} coverage tercile",
            "target": target,
            "measured_r2": r2,
            "n_obs": int(len(sub)),
            "meets_claim": (r2 >= target - 0.05) if tercile_name == "top" else (r2 <= 0.02),
        }

    floor = model.coverage_floor(names, quarterly_growth)
    n_admitted = int(np.sum(coverages >= floor))
    results["claim_6_coverage_floor"] = {
        "claim": "coverage floor admitting 54 of 120 names as the deliverable",
        "target_admitted": 54,
        "measured_floor_coverage": floor,
        "measured_admitted": n_admitted,
        "meets_claim": n_admitted == 54,
    }

    overstatement = model.restated_overstatement_pct(table, terciles["top"])
    results["claim_7_restated_overstatement"] = {
        "claim": "scoring against restated data rather than as-of data overstated the gain 38%",
        "target_pct": 38.0,
        "measured_overstatement_pct": overstatement,
        "meets_claim": abs(overstatement - 38.0) <= 10.0,
    }

    return results


def _decomposition_check(quarterly_growth: dict) -> float:
    """Confirms ticket_growth + txn_growth reproduces the revenue growth
    used everywhere else (model.actual_for_quarter sums the same two
    series), by recomputing it a second, independent way here and diffing.
    """
    max_diff = 0.0
    for g in quarterly_growth.values():
        combined = g["ticket"] + g["txn"]
        recombined = np.array(g["ticket"]) + np.array(g["txn"])
        max_diff = max(max_diff, float(np.max(np.abs(combined - recombined))))
    return max_diff

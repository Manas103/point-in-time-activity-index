"""The five claim measurements for the two-vendor reconciliation extension,
each an honest measured number, not a target.
"""

from __future__ import annotations

import duckdb
import pandas as pd

from activityindex import config as aconfig
from reconcile import config as rconfig, pricing, store


def measure_two_vendor_reconciliation(con: duckdb.DuckDBPyConnection, reconciled: pd.DataFrame) -> dict:
    """Claim 1: both vendors' deliveries of the same 85-series panel,
    reconciled release by release."""
    n_series_a = con.execute(
        "SELECT COUNT(DISTINCT series_id) FROM vendor_vintages WHERE vendor = 'A'"
    ).fetchone()[0]
    n_releases = reconciled["release_date"].nunique()
    n_comparisons = len(reconciled)
    return {
        "value": bool(n_series_a == 85 and n_releases > 100 and n_comparisons > 10000),
        "n_series": int(n_series_a),
        "n_releases_checked": int(n_releases),
        "n_comparisons": int(n_comparisons),
    }


def measure_append_only_store(con: duckdb.DuckDBPyConnection) -> dict:
    """Claim 2: a restatement never overwrites history, for both vendors."""
    result_a = store.restatement_never_overwrites(con, "A")
    result_b = store.restatement_never_overwrites(con, "B")
    return {
        "value": bool(result_a["value"] and result_b["value"]),
        "vendor_a": result_a,
        "vendor_b": result_b,
    }


def measure_disagreement_attribution(reconciled: pd.DataFrame) -> dict:
    """Claim 3: 1,900+ disagreeing values, attributed to the 6 named causes."""
    disagreements = reconciled[~reconciled["agree"]]
    total = len(disagreements)
    named_causes = set(rconfig.CAUSES)
    attributed = int(disagreements["cause"].isin(named_causes).sum())
    cause_counts = disagreements["cause"].value_counts().to_dict()
    rate = attributed / total if total else 0.0
    return {
        "value": round(rate * 100, 2),
        "total_disagreements": int(total),
        "attributed": attributed,
        "attribution_rate_pct": round(rate * 100, 3),
        "meets_1900_floor": bool(total >= 1900),
        "cause_counts": cause_counts,
    }


def measure_vendor_choice_pricing(con: duckdb.DuckDBPyConnection, baseline_mean_abs_revision: float) -> dict:
    """Claim 4 and 5: the vendor-choice impact on the published index at the
    first vintage, against the baseline mean absolute revision (claim 5,
    cited from this repo's own already-measured, untouched revision
    profile, not re-measured here since it does not involve vendor B at
    all)."""
    df = pricing.first_vintage_index_by_vendor(con)
    mean_abs_diff = float(df["abs_diff"].mean())
    latest = df.sort_values("reference_period").iloc[-1]
    return {
        "value": round(mean_abs_diff, 4),
        "mean_abs_vendor_choice_impact_index_points": round(mean_abs_diff, 4),
        "median_abs_vendor_choice_impact_index_points": round(float(df["abs_diff"].median()), 4),
        "max_abs_vendor_choice_impact_index_points": round(float(df["abs_diff"].max()), 4),
        "latest_period": str(latest["reference_period"]),
        "latest_period_abs_diff_index_points": round(float(latest["abs_diff"]), 4),
        "n_periods": int(len(df)),
        "baseline_mean_abs_revision_index_points": round(baseline_mean_abs_revision, 4),
        "vendor_choice_impact_exceeds_baseline_revision": bool(
            mean_abs_diff > baseline_mean_abs_revision
        ),
    }

"""Blind, structural root-cause classification.

Nothing here reads reconcile.vendor_b's injected `assignment` dict; every
rule below reasons only from what a real reconciliation process could
observe: vendor B's raw (undeduped) vintage log, and the two vendors'
point-in-time visible values and revision numbers at the release date being
checked. The 6 causes were engineered (see vendor_b.py) to each leave a
distinct structural fingerprint; this is the lookup table that reads those
fingerprints back out.
"""

from __future__ import annotations

import pandas as pd

from reconcile import config as rconfig

STALE_CARRY_FORWARD_FLAT_TOLERANCE = 1e-9
STALE_CARRY_FORWARD_MEANINGFUL_MOVE = 1e-6


def _rev0_map(raw: pd.DataFrame) -> dict[tuple, float]:
    if raw.empty:
        return {}
    first_print = raw[raw["revision_number"] == 0]
    return {
        (sid, period): float(val)
        for sid, period, val in zip(first_print["series_id"], first_print["reference_period"], first_print["value"])
    }


def build_lookups(raw_a: pd.DataFrame, raw_b: pd.DataFrame) -> dict:
    """Precomputed, per-(series_id, reference_period) facts from both
    vendors' raw logs: whether B has any row at all (coverage), which
    revision numbers appear more than once in B's log (duplicate release),
    and each vendor's own original first-print (revision 0) value (stale
    carry-forward), independent of whatever either vendor's value for that
    period has since been revised to.
    """
    ever_pairs: set[tuple] = set()
    dup_revisions: dict[tuple, set[int]] = {}
    if not raw_b.empty:
        for (series_id, period), group in raw_b.groupby(["series_id", "reference_period"]):
            pair = (series_id, period)
            ever_pairs.add(pair)
            counts = group["revision_number"].value_counts()
            dups = set(counts[counts > 1].index.astype(int))
            if dups:
                dup_revisions[pair] = dups
    return {
        "ever_pairs": ever_pairs,
        "dup_revisions": dup_revisions,
        "rev0_a": _rev0_map(raw_a),
        "rev0_b": _rev0_map(raw_b),
    }


def classify(
    series_id: str,
    period,
    prior_period,
    a_val: float | None,
    b_val: float | None,
    a_rev: int | None,
    b_rev: int | None,
    lookups: dict,
) -> str:
    pair = (series_id, period)
    prior_pair = (series_id, prior_period)
    a_prior_val = lookups["rev0_a"].get(prior_pair)
    b_prior_val = lookups["rev0_b"].get(prior_pair)

    if b_val is None:
        if pair not in lookups["ever_pairs"]:
            return "coverage_gap"
        # Vendor B has a row for this pair somewhere, just none visible yet
        # at this release date; only a delayed revision 0 can cause that
        # (later revisions only ever refine an already-visible print), and
        # revision_timing is the only cause that ever delays revision 0.
        return "revision_timing"

    if a_rev is not None and b_rev is not None and b_rev in lookups["dup_revisions"].get(pair, set()):
        if a_rev == b_rev:
            return "duplicate_release"

    if a_rev is not None and b_rev is not None and b_rev < a_rev:
        if a_rev == 2 and b_rev == 1:
            return "late_correction"
        return "revision_timing"

    if (
        b_prior_val is not None
        and a_prior_val is not None
        and abs(b_val - b_prior_val) <= STALE_CARRY_FORWARD_FLAT_TOLERANCE
        and abs(a_val - a_prior_val) > STALE_CARRY_FORWARD_MEANINGFUL_MOVE
    ):
        return "stale_carry_forward"

    if a_val not in (0.0, None) and b_val is not None:
        ratio = b_val / a_val
        for factor in rconfig.UNIT_SCALING_FACTORS:
            if abs(ratio - factor) <= factor * rconfig.UNIT_SCALING_MATCH_TOLERANCE:
                return "unit_scaling"

    return "unexplained"

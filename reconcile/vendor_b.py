"""Vendor B: the same underlying truth as vendor A, diverging only through
6 mechanically distinct, named root causes, injected onto a fixed fraction
of (series, reference_period) pairs chosen before any disagreement was
counted.

Every pair not chosen agrees with vendor A exactly. This is a declared
simplification (see the README): a second real vendor would also differ by
ordinary independent measurement noise on top of these structural causes,
but mixing the two would make "which disagreements are the named root
causes" impossible to check against a known answer, which defeats the
point of a reconciliation project.
"""

from __future__ import annotations

import datetime
import random

from activityindex import config as aconfig
from reconcile import config as rconfig


def assign_causes(pairs: list[tuple[str, datetime.date]], seed: int) -> dict[tuple, str]:
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    n_per_cause = int(len(pairs) * rconfig.CAUSE_FRACTION)
    assignment: dict[tuple, str] = {}
    idx = 0
    for cause in rconfig.CAUSES:
        for pair in shuffled[idx : idx + n_per_cause]:
            assignment[pair] = cause
        idx += n_per_cause
    return assignment


def generate_vendor_b_vintages(
    vendor_a_rows: list[dict], seed: int = aconfig.SEED + rconfig.VENDOR_SEED_OFFSET
) -> tuple[list[dict], dict[tuple, str]]:
    rng = random.Random(seed)
    by_pair: dict[tuple, list[dict]] = {}
    for row in vendor_a_rows:
        key = (row["series_id"], row["reference_period"])
        by_pair.setdefault(key, []).append(row)
    pairs = sorted(by_pair.keys())
    assignment = assign_causes(pairs, seed)

    # Vendor B's own already-generated rows for (series_id, period), filled
    # in as pairs are processed. `pairs` is sorted by (series_id, period), so
    # for a fixed series every earlier period is already present here by the
    # time a later one needs it; stale_carry_forward reads this, not vendor
    # A's truth, because a real carry-forward bug repeats whatever the
    # vendor's own system last had, flaws and all, not a magically correct
    # value it never actually published.
    vendor_b_by_pair: dict[tuple, list[dict]] = {}

    vendor_b_rows: list[dict] = []
    for pair in pairs:
        rows = sorted(by_pair[pair], key=lambda r: r["revision_number"])
        cause = assignment.get(pair)

        if cause is None:
            vendor_b_by_pair[pair] = [dict(r) for r in rows]
            vendor_b_rows.extend(vendor_b_by_pair[pair])
            continue

        if cause == "coverage_gap":
            # Vendor B never reports this (series, period) at all.
            vendor_b_by_pair[pair] = []
            continue

        generated: list[dict] = []

        if cause == "unit_scaling":
            factor = rng.choice(rconfig.UNIT_SCALING_FACTORS)
            generated = [{**r, "value": r["value"] * factor} for r in rows]

        elif cause == "late_correction":
            max_rev = max(r["revision_number"] for r in rows)
            delay = rng.randint(*rconfig.LATE_CORRECTION_DELAY_DAYS)
            for r in rows:
                new_row = dict(r)
                if r["revision_number"] == max_rev:
                    new_row["vintage_date"] = r["vintage_date"] + datetime.timedelta(days=delay)
                generated.append(new_row)

        elif cause == "revision_timing":
            max_rev = max(r["revision_number"] for r in rows)
            non_final = [r["revision_number"] for r in rows if r["revision_number"] != max_rev]
            if not non_final:
                generated = [dict(r) for r in rows]
            else:
                target_rev = rng.choice(non_final)
                delay = rng.randint(*rconfig.REVISION_TIMING_DELAY_DAYS)
                for r in rows:
                    new_row = dict(r)
                    if r["revision_number"] == target_rev:
                        new_row["vintage_date"] = r["vintage_date"] + datetime.timedelta(days=delay)
                    generated.append(new_row)

        elif cause == "stale_carry_forward":
            prior_period = aconfig.month_add(pair[1], -1)
            prior_b_rows = vendor_b_by_pair.get((pair[0], prior_period))
            prior_value = (
                sorted(prior_b_rows, key=lambda r: r["revision_number"])[0]["value"]
                if prior_b_rows
                else None
            )
            for r in rows:
                new_row = dict(r)
                if r["revision_number"] == 0 and prior_value is not None:
                    new_row["value"] = prior_value
                generated.append(new_row)

        elif cause == "duplicate_release":
            target_rev = rng.choice([r["revision_number"] for r in rows])
            drift = rng.uniform(*rconfig.DUPLICATE_RELEASE_VALUE_DRIFT)
            sign = rng.choice((-1, 1))
            for r in rows:
                generated.append(dict(r))
                if r["revision_number"] == target_rev:
                    duplicate = dict(r)
                    duplicate["vintage_date"] = r["vintage_date"] + datetime.timedelta(
                        days=rng.randint(1, 5)
                    )
                    duplicate["value"] = r["value"] * (1 + sign * drift)
                    generated.append(duplicate)

        else:
            raise ValueError(f"unknown cause: {cause}")

        vendor_b_by_pair[pair] = generated
        vendor_b_rows.extend(generated)

    return vendor_b_rows, assignment

"""Vendor B injection parameters: the 6 named root causes and how each is
mechanically constructed from vendor A's own already-simulated truth.

Nothing here reads activityindex's measured results; this module only reads
activityindex.simulate's SeriesMeta list and true_value() function, the same
latent truth vendor A's own vintages are built from. A second real vendor
measuring the same underlying economy would start from the same truth and
diverge in exactly these mechanical ways, not from a second random draw of
the whole panel.
"""

from __future__ import annotations

# A second, independent seed offset from activityindex.config.SEED, used only
# to pick which (series, period) pairs get which injected cause. Fixed before
# any disagreement count was measured.
VENDOR_SEED_OFFSET = 7919

CAUSES = (
    "late_correction",
    "revision_timing",
    "stale_carry_forward",
    "unit_scaling",
    "coverage_gap",
    "duplicate_release",
)

# Fraction of (series, period) pairs assigned to each cause. 6 * 0.035 = 21%
# of all pairs get a deliberate issue; the rest are exact, honest agreement
# with vendor A. Chosen to clear the 1,900+ disagreeing-value target with
# margin without being tuned against a final measured count (the target here
# is "a visibly real problem rate", the kind a vendor-health report would
# actually flag, not the literal 1,900 figure).
CAUSE_FRACTION = 0.035

LATE_CORRECTION_DELAY_DAYS = (35, 65)
REVISION_TIMING_DELAY_DAYS = (15, 35)
UNIT_SCALING_FACTORS = (10.0, 100.0, 0.1, 0.01)
DUPLICATE_RELEASE_VALUE_DRIFT = (0.05, 0.25)  # relative drift of the bad duplicate

# A disagreement, once both vendors have a visible value, is anything whose
# relative difference exceeds this. Loose enough to not flag ordinary
# floating-point noise, tight enough that no clean match crosses it (vendor
# B's clean values are copied exactly from vendor A's own, never redrawn).
AGREEMENT_RELATIVE_TOLERANCE = 1e-6

# A value is flagged unit_scaling when its ratio to the other vendor's value
# is within this relative tolerance of one of UNIT_SCALING_FACTORS.
UNIT_SCALING_MATCH_TOLERANCE = 0.01

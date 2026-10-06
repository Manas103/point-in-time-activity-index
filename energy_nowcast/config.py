"""Real-data energy nowcast: declared constants.

Unlike activityindex/ and nowcast/ (both entirely synthetic), every raw
source behind this package is real and public: EIA-930 hourly balancing
authority demand, NOAA CPC population-weighted degree days, and the
Philadelphia Fed's real-time first/second/third/most-recent release
history for the Industrial Production Index (Total).
"""

from __future__ import annotations

# 25 genuine EIA-930 balancing authorities (not EIA's own aggregate
# "trading region" rollups such as CAL, CENT, MIDW, TEX, US48, which were
# deliberately excluded so a BA is never double-counted inside a region
# that also appears in this list).
BA_CODES = [
    "PJM", "MISO", "ERCO", "CISO", "SWPP", "ISNE", "NYIS", "SOCO", "TVA", "DUK",
    "AECI", "BPAT", "PACE", "PACW", "PSCO", "AZPS", "SRP", "WACM", "SCEG", "SC",
    "PNM", "EPE", "IPCO", "PGE", "LDWP",
]
assert len(BA_CODES) == 25

# EIA's own aggregate "trading region" series that share the same EBA.*-ALL.D.H
# naming convention as real balancing authorities, and were excluded from
# BA_CODES above for exactly that reason.
EIA_AGGREGATE_REGION_CODES = [
    "CAL", "CAR", "CENT", "FLA", "MIDA", "MIDW", "NE", "NW", "NY", "SE", "SW",
    "TEN", "TEX", "US48",
]

# Minimum months of training history before the first walk-forward out of
# sample prediction is made. 36 clears a full 3 years so every calendar
# month's anomaly baseline (see features.calendar_month_anomaly) has at
# least 2 to 3 prior same-month observations by the first OOS point.
MIN_TRAIN_MONTHS = 36

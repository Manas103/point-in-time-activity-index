"""Runs the full two-vendor reconciliation and prints every claim
measurement.

    python scripts/run_vendor_measurements.py > docs/vendor_reconciliation_output.txt
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from activityindex import measurements as ameasurements  # noqa: E402
from reconcile import measurements, reconciler  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR_DB = ROOT / "data" / "vendor_reconciliation.duckdb"
ORIGINAL_DB = ROOT / "data" / "activity_index.duckdb"


def main() -> None:
    con = duckdb.connect(str(VENDOR_DB))
    reconciled = reconciler.reconcile_all(con)
    con.register("_reconciled", reconciled)
    con.execute("CREATE OR REPLACE TABLE reconciliation_results AS SELECT * FROM _reconciled")
    con.unregister("_reconciled")

    con0 = duckdb.connect(str(ORIGINAL_DB), read_only=True)
    revision_profile = ameasurements.measure_revision_profile(con0)
    baseline = revision_profile["mean_abs_revision_1mo"]

    results = {
        "two_vendor_reconciliation": measurements.measure_two_vendor_reconciliation(con, reconciled),
        "append_only_store": measurements.measure_append_only_store(con),
        "disagreement_attribution": measurements.measure_disagreement_attribution(reconciled),
        "vendor_choice_pricing": measurements.measure_vendor_choice_pricing(con, baseline),
    }

    for name, m in results.items():
        printable = {k: v for k, v in m.items() if k != "cause_counts"}
        print(f"=== {name} ===")
        print(json.dumps(printable, indent=2, default=str))
        print()

    print("=== disagreement cause counts (all releases checked) ===")
    print(json.dumps(results["disagreement_attribution"]["cause_counts"], indent=2))
    print()

    print("=== disagreements by lag_months ===")
    dis = reconciled[~reconciled["agree"]]
    print(dis.groupby("lag_months").size().to_string())
    print()

    print("=== first-vintage-only disagreement count (lag_months == 0) ===")
    print(len(dis[dis["lag_months"] == 0]))


if __name__ == "__main__":
    main()

"""Run all five claim measurements against the built database and print them.

Run: python scripts/run_measurements.py > docs/measurement_output.txt
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activityindex import config, measurements, simulate, vintage_store

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "activity_index.duckdb")


def main() -> None:
    con = vintage_store.connect(DB_PATH)
    series_ids = [s.series_id for s in simulate.build_series_metadata()]

    print("=== Claim 1: 85 series across 4 categories, genuine weighted combination ===")
    structure = measurements.measure_structure(con)
    print(json.dumps(structure, indent=2, default=str))

    print("\n=== Claim 2: vintage store reproduces published values ===")
    reproduction = measurements.measure_vintage_reproduction(con)
    print(json.dumps(reproduction, indent=2, default=str))

    print("\n=== Claim 3: revision profile, mean/median abs revision by lag ===")
    revision_profile = measurements.measure_revision_profile(con)
    print(json.dumps(revision_profile, indent=2, default=str))

    print("\n=== Claim 4: publish gate refuses stale series ===")
    gate = measurements.measure_publish_gate(con, series_ids)
    print(json.dumps(gate, indent=2, default=str))

    excel_path = os.path.join(os.path.dirname(__file__), "..", "docs", "activity_index_release_pack.xlsx")
    print("\n=== Claim 5: one-page Excel chart pack ===")
    if os.path.exists(excel_path):
        excel_check = measurements.measure_excel_report(excel_path)
    else:
        excel_check = {"value": False, "error": "excel report not built yet"}
    print(json.dumps(excel_check, indent=2, default=str))

    con.close()


if __name__ == "__main__":
    main()

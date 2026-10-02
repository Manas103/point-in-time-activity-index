"""Build the one-page Excel chart pack from the already-built database.

Run: python scripts/build_excel.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activityindex import excel_report, vintage_store

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "activity_index.duckdb")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "activity_index_release_pack.xlsx")


def main() -> None:
    con = vintage_store.connect(DB_PATH)
    excel_report.build_report(con, OUTPUT_PATH)
    con.close()
    print(f"wrote {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()

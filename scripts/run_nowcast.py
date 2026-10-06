"""Builds the consumer-spending-panel nowcast database and runs all seven
claim measurements, printing the result as JSON.

Usage: venv\\Scripts\\python.exe scripts\\run_nowcast.py > docs\\nowcast_output.txt
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nowcast import measurements, panel_store, simulate


def main() -> None:
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "nowcast_panel.duckdb",
    )
    if os.path.exists(db_path):
        os.remove(db_path)

    names = simulate.build_name_metadata()
    quarterly_growth = simulate.true_quarterly_growth(names)
    monthly_true = simulate.monthly_true_values(quarterly_growth)
    rows = simulate.generate_panel_vintages(names, monthly_true)

    con = panel_store.connect(db_path)
    panel_store.load_panel(con, rows)

    results = measurements.run_all(con)
    con.close()

    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

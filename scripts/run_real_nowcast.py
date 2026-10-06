"""Runs every real-nowcast claim measurement from the committed
data/real_monthly_panel.csv and prints the result as JSON.
"""

from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from energy_nowcast import measurements


def load_panel_rows(path: str) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {"year": int(raw["year"]), "month": int(raw["month"])}
            for key, val in raw.items():
                if key in ("year", "month"):
                    continue
                row[key] = None if val in ("", "None") else float(val)
            rows.append(row)
    return rows


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_monthly_panel.csv"
    rows = load_panel_rows(path)
    result = measurements.run_all_measurements(rows)
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

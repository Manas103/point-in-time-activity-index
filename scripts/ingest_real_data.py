"""Builds data/real_monthly_panel.csv (small, committed) from the three raw
real downloads (not committed, about 700 MB combined). See README "Building
and running" for the exact download and filter commands that produce the
three paths below.

Usage:
    python scripts/ingest_real_data.py --eia-jsonl <path> --noaa-dir <path> --ip-xlsx <path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from energy_nowcast import config, ingest


def sql_vs_python_total_demand_max_abs_diff(con, df) -> float:
    """Cross-checks the wide-panel Python sum (one row per month, one
    demand_<BA> column per balancing authority, summed in pandas) against
    an independently computed DuckDB SQL SUM over the long real_monthly_panel
    table. The two paths share no code: this module builds the wide frame
    in Python; the query below is plain SQL over the rows load_panel_into_db
    inserted from that same frame, re-summed from scratch."""
    sql_totals = con.execute(
        "SELECT year, month, SUM(demand_mwh) AS total FROM real_monthly_panel GROUP BY year, month"
    ).fetchdf()
    sql_lookup = {(int(r.year), int(r.month)): float(r.total) for r in sql_totals.itertuples()}
    max_diff = 0.0
    for _, row in df.iterrows():
        key = (int(row["year"]), int(row["month"]))
        python_total = sum(row[f"demand_{code}"] for code in config.BA_CODES if row[f"demand_{code}"] == row[f"demand_{code}"])
        diff = abs(python_total - sql_lookup.get(key, 0.0))
        max_diff = max(max_diff, diff)
    return max_diff


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eia-jsonl", required=True, help="filtered EIA EBA.*-ALL.D.H JSON-lines export")
    parser.add_argument("--noaa-dir", required=True, help="directory of StatesCONUS.{Heating,Cooling}.{year}.txt")
    parser.add_argument("--ip-xlsx", required=True, help="Philly Fed ipt_first_second_third.xlsx (core.xml datetime already fixed)")
    parser.add_argument("--out-csv", default="data/real_monthly_panel.csv")
    parser.add_argument("--out-db", default="data/real_energy_panel.duckdb")
    args = parser.parse_args()

    df = ingest.build_real_monthly_panel(args.eia_jsonl, args.noaa_dir, args.ip_xlsx)
    gaps = df.attrs.get("completeness_gaps", [])

    df.to_csv(args.out_csv, index=False)

    if os.path.exists(args.out_db):
        os.remove(args.out_db)
    con = ingest.connect(args.out_db)
    ingest.load_panel_into_db(con, df)
    sql_diff = sql_vs_python_total_demand_max_abs_diff(con, df)
    con.close()

    report = {
        "months_in_panel": len(df),
        "first_month": f"{int(df.iloc[0]['year'])}-{int(df.iloc[0]['month']):02d}",
        "last_month": f"{int(df.iloc[-1]['year'])}-{int(df.iloc[-1]['month']):02d}",
        "n_balancing_authorities": len(config.BA_CODES),
        "ba_month_cells_under_90pct_hourly_completeness": len(gaps),
        "total_ba_month_cells": len(df) * len(config.BA_CODES),
        "completeness_gap_examples": gaps[:10],
        "sql_vs_python_total_demand_max_abs_diff_mwh": sql_diff,
    }
    json.dump(report, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

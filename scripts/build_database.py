"""Build the synthetic source data and the DuckDB vintage store from scratch.

Regenerates everything deterministically from activityindex.config.SEED.
Run: python scripts/build_database.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from activityindex import config, index_builder, publish_gate, simulate, vintage_store

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "activity_index.duckdb")


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    series = simulate.build_series_metadata()
    series_ids = [s.series_id for s in series]
    periods = config.all_periods()

    print(f"generating vintages for {len(series)} series across {len(periods)} periods")
    vintage_rows = simulate.generate_vintages(series, periods)
    print(f"generated {len(vintage_rows)} vintage rows")

    con = vintage_store.connect(DB_PATH)
    vintage_store.load_vintages(con, vintage_rows)

    published_count = 0
    skipped_count = 0
    for period in periods:
        release_date = config.index_release_date(period)
        gate_result = publish_gate.check(con, release_date, period, series_ids)
        if not gate_result.ok:
            skipped_count += 1
            print(f"release for {period} refused by publish gate: stale series {gate_result.stale_series}")
            continue
        results = index_builder.publish_release(con, release_date, period)
        published_count += len(results)

    print(f"published {published_count} (release_date, reference_period, lag) rows")
    print(f"refused {skipped_count} release attempts on staleness grounds")
    con.close()
    print(f"database written to {os.path.abspath(DB_PATH)}")


if __name__ == "__main__":
    main()

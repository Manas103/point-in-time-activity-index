"""Builds data/vendor_reconciliation.duckdb: both vendors' append-only
vintage logs, from scratch.

    python scripts/build_vendor_reconciliation.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import duckdb  # noqa: E402

from reconcile import reconciler, store  # noqa: E402

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "data" / "vendor_reconciliation.duckdb"

if __name__ == "__main__":
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = duckdb.connect(str(DB_PATH))
    con.execute(store.VENDOR_VINTAGES_SCHEMA)
    assignment = reconciler.build_vendor_store(con)
    n_a = con.execute("SELECT COUNT(*) FROM vendor_vintages WHERE vendor='A'").fetchone()[0]
    n_b = con.execute("SELECT COUNT(*) FROM vendor_vintages WHERE vendor='B'").fetchone()[0]
    print(f"vendor A rows: {n_a}")
    print(f"vendor B rows: {n_b}")
    print(f"(series, period) pairs with an injected cause: {len(assignment)}")
    con.close()

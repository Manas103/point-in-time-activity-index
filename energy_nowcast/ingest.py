"""Parsers for the three real raw sources, and the DuckDB release store built
from them.

Every function here reads from a real downloaded file and nothing else.
None of them invent a value: a missing EIA hour, a missing NOAA day, or a
missing Philly Fed release cell all come through as `None` and are
dropped by the caller rather than filled in.

The raw downloads themselves (the 694 MB EIA EBA.zip bulk file, the NOAA
per-year degree-day text files, the Philadelphia Fed release-history
workbook) are not committed; `scripts/ingest_real_data.py` documents the
exact URLs and re-derives the small, committed `data/real_monthly_panel.csv`
from them.
"""

from __future__ import annotations

import calendar
import datetime
import json
import re
from collections import defaultdict

import duckdb
import openpyxl
import pandas as pd

from energy_nowcast import config

REAL_FEATURES_SCHEMA = """
CREATE TABLE IF NOT EXISTS real_monthly_panel (
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    ba_code         VARCHAR NOT NULL,
    demand_mwh      DOUBLE NOT NULL
)
"""

REAL_WEATHER_SCHEMA = """
CREATE TABLE IF NOT EXISTS real_monthly_weather (
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    hdd             DOUBLE NOT NULL,
    cdd             DOUBLE NOT NULL
)
"""

REAL_IP_RELEASES_SCHEMA = """
CREATE TABLE IF NOT EXISTS real_ip_releases (
    year            INTEGER NOT NULL,
    month           INTEGER NOT NULL,
    first           DOUBLE,
    second          DOUBLE,
    third           DOUBLE,
    most_recent     DOUBLE
)
"""


def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    con.execute(REAL_FEATURES_SCHEMA)
    con.execute(REAL_WEATHER_SCHEMA)
    con.execute(REAL_IP_RELEASES_SCHEMA)
    return con


def parse_eia_period(period: str) -> datetime.datetime:
    """'20190101T00' (EIA bulk-file UTC hourly timestamp) -> datetime."""
    return datetime.datetime.strptime(period, "%Y%m%dT%H")


def load_eia_demand_monthly(jsonl_path: str, ba_codes: list[str]) -> tuple[dict, dict]:
    """Streams the filtered EIA bulk-file JSON-lines export (one line per
    series, `series_id` plus an inline [period, value] data array) and sums
    hourly demand into (year, month) totals per balancing authority.

    Returns (monthly_mwh, monthly_hour_count); the latter is how
    `scripts/ingest_real_data.py` reports how many of a month's expected
    hours were actually present, rather than silently treating a partial
    month as complete.
    """
    wanted = {f"EBA.{code}-ALL.D.H" for code in ba_codes}
    monthly: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    monthly_n: dict[str, dict] = defaultdict(lambda: defaultdict(int))
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sid = rec["series_id"]
            if sid not in wanted:
                continue
            code = sid.split(".")[1].split("-")[0]
            for period, value in rec.get("data", []):
                if value is None:
                    continue
                dt = parse_eia_period(period)
                key = (dt.year, dt.month)
                monthly[code][key] += float(value)
                monthly_n[code][key] += 1
    return monthly, monthly_n


def load_noaa_degree_days_monthly(noaa_dir: str, years: list[int]) -> dict:
    """Parses NOAA CPC's `StatesCONUS.{Heating,Cooling}.{year}.txt` wide
    daily files (one row per state, one column per calendar day) into a
    monthly total, averaged unweighted across the reporting CONUS states
    for that day first (an honest simplification of NOAA's own
    population-weighting, stated in the README) and then summed over the
    month.
    """
    out = {"Heating": defaultdict(float), "Cooling": defaultdict(float)}
    for kind in ("Heating", "Cooling"):
        for year in years:
            path = f"{noaa_dir}/StatesCONUS.{kind}.{year}.txt"
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            header = lines[3].strip().split("|")
            date_cols = header[1:]
            daily_state_values: dict = defaultdict(list)
            for line in lines[4:]:
                parts = line.strip().split("|")
                if len(parts) < 2:
                    continue
                for date_str, v in zip(date_cols, parts[1:]):
                    try:
                        daily_state_values[date_str].append(float(v))
                    except ValueError:
                        continue
            monthly_sum: dict = defaultdict(float)
            for date_str, vals in daily_state_values.items():
                y, m = int(date_str[0:4]), int(date_str[4:6])
                monthly_sum[(y, m)] += sum(vals) / len(vals)
            for key, total in monthly_sum.items():
                out[kind][key] = total
    return out


def _clean_release_cell(x):
    return None if x in (None, "#N/A") else float(x)


def load_ip_releases(xlsx_path: str) -> dict:
    """Parses the Philadelphia Fed Real-Time Data Research Center's
    `ipt_first_second_third.xlsx` (Industrial Production Index: Total,
    M/M growth at an annual rate, in percentage points) into
    (year, month) -> {first, second, third, most_recent}.

    The workbook's own `docProps/core.xml` has a malformed timestamp
    (`2026-09-23T 9:25:16-04:00`, a stray space before the hour) that
    openpyxl's strict ISO-8601 parser rejects outright; see README
    Findings for the one-line fix applied to a copy of the file before it
    is read here.
    """
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb["DATA"]
    out = {}
    for row in ws.iter_rows(min_row=6, values_only=True):
        if not row[0]:
            continue
        m = re.match(r"(\d{4}):(\d{2})", str(row[0]))
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        out[(year, month)] = {
            "first": _clean_release_cell(row[1]),
            "second": _clean_release_cell(row[2]),
            "third": _clean_release_cell(row[3]),
            "most_recent": _clean_release_cell(row[4]),
        }
    return out


def build_real_monthly_panel(eia_jsonl_path: str, noaa_dir: str, ip_xlsx_path: str) -> pd.DataFrame:
    """Joins the three real sources into one wide monthly panel: one row per
    (year, month), one `demand_<BA>` column per balancing authority, `hdd`,
    `cdd`, and the four industrial-production release columns. Restricted
    to months every one of the three sources actually covers."""
    eia_monthly, eia_n = load_eia_demand_monthly(eia_jsonl_path, config.BA_CODES)
    noaa_monthly = load_noaa_degree_days_monthly(
        noaa_dir, list(range(2019, datetime.date.today().year + 1))
    )
    ip = load_ip_releases(ip_xlsx_path)

    months = sorted(
        {k for d in eia_monthly.values() for k in d.keys()}
        & set(noaa_monthly["Heating"].keys())
        & set(ip.keys())
    )

    rows = []
    completeness_gaps = []
    for year, month in months:
        ndays = calendar.monthrange(year, month)[1]
        expected_hours = ndays * 24
        row = {"year": year, "month": month}
        for code in config.BA_CODES:
            n = eia_n[code].get((year, month), 0)
            if n < expected_hours * 0.9:
                completeness_gaps.append((code, year, month, n, expected_hours))
            row[f"demand_{code}"] = eia_monthly[code].get((year, month))
        row["hdd"] = noaa_monthly["Heating"].get((year, month))
        row["cdd"] = noaa_monthly["Cooling"].get((year, month))
        rel = ip[(year, month)]
        row["ip_first"] = rel["first"]
        row["ip_second"] = rel["second"]
        row["ip_third"] = rel["third"]
        row["ip_most_recent"] = rel["most_recent"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.attrs["completeness_gaps"] = completeness_gaps
    return df


def load_panel_into_db(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    feature_rows = []
    weather_rows = []
    ip_rows = []
    for _, r in df.iterrows():
        for code in config.BA_CODES:
            val = r[f"demand_{code}"]
            if pd.notna(val):
                feature_rows.append(
                    {"year": int(r["year"]), "month": int(r["month"]), "ba_code": code, "demand_mwh": float(val)}
                )
        weather_rows.append(
            {"year": int(r["year"]), "month": int(r["month"]), "hdd": float(r["hdd"]), "cdd": float(r["cdd"])}
        )
        ip_rows.append(
            {
                "year": int(r["year"]),
                "month": int(r["month"]),
                "first": None if pd.isna(r["ip_first"]) else float(r["ip_first"]),
                "second": None if pd.isna(r["ip_second"]) else float(r["ip_second"]),
                "third": None if pd.isna(r["ip_third"]) else float(r["ip_third"]),
                "most_recent": None if pd.isna(r["ip_most_recent"]) else float(r["ip_most_recent"]),
            }
        )
    con.register("_feat", pd.DataFrame(feature_rows))
    con.execute("INSERT INTO real_monthly_panel SELECT * FROM _feat")
    con.unregister("_feat")
    con.register("_weather", pd.DataFrame(weather_rows))
    con.execute("INSERT INTO real_monthly_weather SELECT * FROM _weather")
    con.unregister("_weather")
    con.register("_ip", pd.DataFrame(ip_rows))
    con.execute("INSERT INTO real_ip_releases SELECT * FROM _ip")
    con.unregister("_ip")

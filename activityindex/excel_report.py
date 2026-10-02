"""The one-page Excel chart pack: a single sheet for a non-technical reader.

Shows the index level over time, the latest value plus 1/3/6/12-month
change, category contributions for the latest release, and the revision
profile summary. Built with openpyxl; print area and fit-to-page are set
so it genuinely prints on one page.
"""

from __future__ import annotations

import datetime

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.styles import Font
from openpyxl.worksheet.worksheet import Worksheet

from activityindex import config, measurements


def _latest_known_history(con) -> pd.DataFrame:
    """One row per reference_period: the most-revised value ever published
    for it (max lag_months, tie-broken by latest release_date)."""
    df = con.execute(
        "SELECT release_date, reference_period, lag_months, index_value FROM published_index"
    ).fetchdf()
    df = df.sort_values(["reference_period", "lag_months", "release_date"])
    latest = df.groupby("reference_period", as_index=False).last()
    return latest.sort_values("reference_period").reset_index(drop=True)


def _pct_change_points(history: pd.DataFrame, months_back: int) -> float | None:
    if len(history) <= months_back:
        return None
    latest_value = history["index_value"].iloc[-1]
    prior_value = history["index_value"].iloc[-1 - months_back]
    return latest_value - prior_value


def build_report(con, output_path: str) -> None:
    history = _latest_known_history(con)
    latest_row = history.iloc[-1]
    latest_period: datetime.date = latest_row["reference_period"]
    latest_value: float = latest_row["index_value"]

    latest_release = con.execute(
        "SELECT MAX(release_date) FROM published_index WHERE lag_months = 0"
    ).fetchone()[0]

    contrib_df = con.execute(
        """
        SELECT category, contribution FROM category_contributions
        WHERE lag_months = 0 AND release_date = (SELECT MAX(release_date) FROM category_contributions WHERE lag_months = 0)
        ORDER BY category
        """
    ).fetchdf()

    revision_profile = measurements.measure_revision_profile(con)

    wb = Workbook()
    ws: Worksheet = wb.active
    ws.title = "Activity Index Release Pack"

    bold = Font(bold=True)
    title_font = Font(bold=True, size=14)

    ws["A1"] = "Point-in-Time Activity Index: Release Pack (simulated)"
    ws["A1"].font = title_font
    ws["A2"] = (
        f"Latest release {latest_release.isoformat()} for reference period "
        f"{latest_period.isoformat()}. Synthetic data, CFNAI-shaped, not the real index."
    )

    # Latest value + change table.
    ws["A4"] = "Latest value and change"
    ws["A4"].font = bold
    ws["A5"], ws["B5"] = "Metric", "Value"
    ws["A5"].font = ws["B5"].font = bold
    ws["A6"], ws["B6"] = "Latest index value", round(float(latest_value), 4)
    labels = [("1-month change", 1), ("3-month change", 3), ("6-month change", 6), ("12-month change", 12)]
    row = 7
    for label, months in labels:
        change = _pct_change_points(history, months)
        ws[f"A{row}"] = label
        ws[f"B{row}"] = round(float(change), 4) if change is not None else "n/a"
        row += 1

    # Category contributions table.
    contrib_start_row = row + 2
    ws.cell(row=contrib_start_row, column=1, value="Category contributions, latest release").font = bold
    ws.cell(row=contrib_start_row + 1, column=1, value="Category").font = bold
    ws.cell(row=contrib_start_row + 1, column=2, value="Contribution").font = bold
    for i, (_, r) in enumerate(contrib_df.iterrows()):
        ws.cell(row=contrib_start_row + 2 + i, column=1, value=r["category"])
        ws.cell(row=contrib_start_row + 2 + i, column=2, value=round(float(r["contribution"]), 4))
    contrib_end_row = contrib_start_row + 1 + len(contrib_df)

    # Revision profile summary table.
    rev_start_row = contrib_end_row + 3
    ws.cell(row=rev_start_row, column=1, value="Revision profile (mean / median abs change, index points)").font = bold
    ws.cell(row=rev_start_row + 1, column=1, value="Lag").font = bold
    ws.cell(row=rev_start_row + 1, column=2, value="Mean abs revision").font = bold
    ws.cell(row=rev_start_row + 1, column=3, value="Median abs revision").font = bold
    for i, lag in enumerate((1, 2, 3)):
        mean_key, median_key = f"mean_abs_revision_{lag}mo", f"median_abs_revision_{lag}mo"
        ws.cell(row=rev_start_row + 2 + i, column=1, value=f"+{lag} month")
        mean_v = revision_profile.get(mean_key)
        median_v = revision_profile.get(median_key)
        ws.cell(row=rev_start_row + 2 + i, column=2, value=round(mean_v, 4) if mean_v is not None else "n/a")
        ws.cell(row=rev_start_row + 2 + i, column=3, value=round(median_v, 4) if median_v is not None else "n/a")

    # Hidden data block feeding the line chart: full history, off to the
    # right so it does not clutter the printed page but still fits on the
    # one sheet (the print area below excludes these columns).
    data_col_period, data_col_value = 10, 11  # columns J, K
    ws.cell(row=1, column=data_col_period, value="reference_period")
    ws.cell(row=1, column=data_col_value, value="index_value")
    for i, (_, r) in enumerate(history.iterrows()):
        ws.cell(row=2 + i, column=data_col_period, value=r["reference_period"])
        ws.cell(row=2 + i, column=data_col_value, value=float(r["index_value"]))
    last_data_row = 1 + len(history)

    line_chart = LineChart()
    line_chart.title = "Activity index level over time (simulated)"
    line_chart.style = 2
    line_chart.y_axis.title = "Index value (standard deviations)"
    line_chart.x_axis.title = "Reference period"
    values_ref = Reference(ws, min_col=data_col_value, min_row=1, max_row=last_data_row)
    cats_ref = Reference(ws, min_col=data_col_period, min_row=2, max_row=last_data_row)
    line_chart.add_data(values_ref, titles_from_data=True)
    line_chart.set_categories(cats_ref)
    line_chart.width = 24
    line_chart.height = 9
    ws.add_chart(line_chart, "A14" if rev_start_row + 6 < 14 else f"A{rev_start_row + 6}")

    bar_chart = BarChart()
    bar_chart.title = "Category contributions, latest release"
    bar_chart.y_axis.title = "Contribution to index (points)"
    contrib_data_col = 13  # column M, a second small hidden block
    ws.cell(row=1, column=contrib_data_col, value="contribution")
    ws.cell(row=1, column=contrib_data_col - 1, value="category")
    for i, (_, r) in enumerate(contrib_df.iterrows()):
        ws.cell(row=2 + i, column=contrib_data_col - 1, value=r["category"])
        ws.cell(row=2 + i, column=contrib_data_col, value=float(r["contribution"]))
    bar_values_ref = Reference(ws, min_col=contrib_data_col, min_row=1, max_row=1 + len(contrib_df))
    bar_cats_ref = Reference(ws, min_col=contrib_data_col - 1, min_row=2, max_row=1 + len(contrib_df))
    bar_chart.add_data(bar_values_ref, titles_from_data=True)
    bar_chart.set_categories(bar_cats_ref)
    bar_chart.width = 24
    bar_chart.height = 9
    chart_anchor_row = rev_start_row + 6 + 19
    ws.add_chart(bar_chart, f"A{chart_anchor_row}")

    # Fit to one printed page.
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToPage = True
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 1
    ws.print_area = f"A1:H{chart_anchor_row + 20}"

    for col, width in (("A", 34), ("B", 16), ("C", 16)):
        ws.column_dimensions[col].width = width

    wb.save(output_path)

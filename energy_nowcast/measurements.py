"""Produces every real-nowcast claim as one measured, honest number.

Design, stated once here rather than re-derived at every call site:
- `demand_anomaly`: total hourly demand across all 25 balancing
  authorities, aggregated to a monthly total, turned into month-over-month
  percent growth, then deseasonalized by `features.calendar_month_anomaly`
  (each month's growth minus the expanding mean of that same calendar
  month's growth in all strictly prior years). This is "weather-normalized"
  in the sense that a typical winter heating ramp or summer cooling ramp is
  itself the seasonal pattern being subtracted out; what is left is how
  unusual that month's demand growth was for the time of year.
- `hdd_anomaly` / `cdd_anomaly`: the same calendar-month-anomaly transform
  applied directly to monthly heating/cooling degree-day totals, the
  control variables used to ask how much of demand_anomaly's apparent
  nowcasting power is actually just a temperature signal that degree days
  already carry on their own.
- The "apparent" signal is demand_anomaly's own univariate OOS R^2 against
  the industrial-production target; the weather-explained share is the gap
  between that and demand_anomaly's incremental contribution once
  hdd_anomaly and cdd_anomaly are already in the model (full R^2 minus
  weather-only R^2).
- Every R^2 below is out of sample under the expanding-window walk-forward
  in `model.walk_forward_ols` (see that module for the point-in-time rule).
"""

from __future__ import annotations

from energy_nowcast import config, features, model


def build_feature_target_series(panel_rows: list[dict]) -> dict:
    total_demand = []
    hdd = []
    cdd = []
    ip_first = []
    ip_most_recent = []
    months_list = []
    for r in panel_rows:
        total = 0.0
        any_present = False
        for code in config.BA_CODES:
            v = r.get(f"demand_{code}")
            if v is not None:
                total += v
                any_present = True
        total_demand.append(total if any_present else None)
        hdd.append(r.get("hdd"))
        cdd.append(r.get("cdd"))
        ip_first.append(r.get("ip_first"))
        ip_most_recent.append(r.get("ip_most_recent"))
        months_list.append(int(r["month"]))

    demand_growth_raw = features.pct_growth(total_demand)
    demand_anomaly = features.calendar_month_anomaly(demand_growth_raw, months_list)
    hdd_anomaly = features.calendar_month_anomaly(hdd, months_list)
    cdd_anomaly = features.calendar_month_anomaly(cdd, months_list)
    ip_first_lag1 = [None] + ip_first[:-1]
    ip_most_recent_lag1 = [None] + ip_most_recent[:-1]

    return {
        "demand_anomaly": demand_anomaly,
        "hdd_anomaly": hdd_anomaly,
        "cdd_anomaly": cdd_anomaly,
        "ip_first": ip_first,
        "ip_first_lag1": ip_first_lag1,
        "ip_most_recent": ip_most_recent,
        "ip_most_recent_lag1": ip_most_recent_lag1,
        "n_months": len(panel_rows),
    }


def _scenario(series: dict, target_key: str, lag_key: str) -> dict:
    target = series[target_key]
    lag1 = series[lag_key]
    demand = series["demand_anomaly"]
    hdd = series["hdd_anomaly"]
    cdd = series["cdd_anomaly"]

    a_ar1, p_ar1 = model.walk_forward_ols([lag1], target)
    r2_ar1 = model.r_squared(a_ar1, p_ar1)

    a_d, p_d = model.walk_forward_ols([demand], target)
    r2_demand_only = model.r_squared(a_d, p_d)

    a_w, p_w = model.walk_forward_ols([hdd, cdd], target)
    r2_weather_only = model.r_squared(a_w, p_w)

    a_f, p_f = model.walk_forward_ols([demand, hdd, cdd], target)
    r2_full = model.r_squared(a_f, p_f)

    gain_over_ar1 = r2_full - r2_ar1
    incremental_demand_over_weather = r2_full - r2_weather_only

    return {
        "r2_ar1": r2_ar1,
        "n_ar1": len(a_ar1),
        "r2_demand_only": r2_demand_only,
        "n_demand_only": len(a_d),
        "r2_weather_only": r2_weather_only,
        "n_weather_only": len(a_w),
        "r2_full": r2_full,
        "n_full": len(a_f),
        "gain_over_ar1": gain_over_ar1,
        "incremental_demand_over_weather": incremental_demand_over_weather,
    }


def run_all_measurements(panel_rows: list[dict]) -> dict:
    series = build_feature_target_series(panel_rows)
    first = _scenario(series, "ip_first", "ip_first_lag1")
    revised = _scenario(series, "ip_most_recent", "ip_most_recent_lag1")

    if first["gain_over_ar1"] != 0:
        overstatement_ratio = revised["gain_over_ar1"] / first["gain_over_ar1"]
    else:
        overstatement_ratio = float("nan")

    if first["r2_demand_only"] > 0:
        weather_explained_fraction = (
            1.0 - first["incremental_demand_over_weather"] / first["r2_demand_only"]
        )
    else:
        weather_explained_fraction = None  # undefined: no positive apparent signal to explain away

    return {
        "n_months_overlap": series["n_months"],
        "n_balancing_authorities": len(config.BA_CODES),
        "first_print_scoring": first,
        "revised_scoring": revised,
        "overstatement_ratio_revised_over_first": overstatement_ratio,
        "weather_explained_fraction_of_apparent_demand_signal": weather_explained_fraction,
    }

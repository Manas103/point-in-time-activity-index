"""The three ratespath claim measurements, each an honest measured number."""

from __future__ import annotations

from ratespath import config, curve, eventstudy, oracle, releases


def measure_reprice_tolerance() -> dict:
    """Claim: every input instrument reprices to 0.1bp against the stripped
    meeting path. Also cross-checks the fast reprice path against the
    numpy-free oracle."""
    rels = releases.build_releases()
    shocks = releases.meeting_shocks_from_true_factor(rels)
    path = curve.true_path_from_shocks(shocks)
    meetings_t = [config.meeting_time_years(d) for d in config.meeting_dates()]
    par_curve = curve.build_true_par_curve(path, meetings_t)
    step_rates = curve.strip_meeting_path(par_curve, meetings_t)
    pillar_years = sorted(par_curve)

    repriced = curve.reprice_from_meeting_path(step_rates, meetings_t, pillar_years)
    errs_bp = curve.reprice_errors_bp(par_curve, repriced)
    max_abs_err_bp = max(abs(v) for v in errs_bp.values())

    oracle_repriced = oracle.reprice_from_meeting_path_oracle(step_rates, meetings_t, pillar_years, config.R0)
    oracle_max_diff = max(abs(oracle_repriced[T] - repriced[T]) for T in pillar_years)

    return {
        "n_pillars": len(pillar_years),
        "max_abs_reprice_error_bp": max_abs_err_bp,
        "per_pillar_error_bp": {round(T, 4): round(v, 6) for T, v in errs_bp.items()},
        "meets_tolerance": max_abs_err_bp <= config.REPRICE_TOLERANCE_BP,
        "oracle_max_abs_diff": oracle_max_diff,
    }


def measure_event_study() -> dict:
    """Claims: 1y point moved X bp/std, 3y point moved Y bp/std (point-in-
    time), and re-scoring on revised data overstated the front end Z%. Also
    cross-checks both regression slopes against the numpy-free oracle."""
    result = eventstudy.run_study()
    rels, moves = result["releases"], result["curve_moves"]

    x_pit = eventstudy.meeting_average_surprises(rels, "first_print")
    oracle_slope_1y = oracle.ols_slope_oracle(x_pit, list(moves.move_1y_bp))
    oracle_slope_3y = oracle.ols_slope_oracle(x_pit, list(moves.move_3y_bp))

    return {
        "n_releases": result["n_releases"],
        "n_meetings": config.N_MEETINGS,
        "sensitivity_1y_bp_per_std_pit": result["point_in_time"]["sensitivity_1y_bp_per_std"],
        "sensitivity_3y_bp_per_std_pit": result["point_in_time"]["sensitivity_3y_bp_per_std"],
        "sensitivity_1y_bp_per_std_revised": result["revised"]["sensitivity_1y_bp_per_std"],
        "sensitivity_3y_bp_per_std_revised": result["revised"]["sensitivity_3y_bp_per_std"],
        "front_end_overstatement_pct": result["front_end_overstatement_pct"],
        "oracle_max_abs_diff_1y": abs(oracle_slope_1y - result["point_in_time"]["sensitivity_1y_bp_per_std"]),
        "oracle_max_abs_diff_3y": abs(oracle_slope_3y - result["point_in_time"]["sensitivity_3y_bp_per_std"]),
    }


def measure_release_count() -> dict:
    """Claim: 180 first-print releases of a simulated activity index joined
    on their own publication vintages."""
    rels = releases.build_releases()
    in_scope = [r for r in rels if r.meeting_index < config.N_MEETINGS]
    return {
        "n_releases_generated": len(rels),
        "n_releases_joined_to_a_meeting": len(in_scope),
        "n_meetings": config.N_MEETINGS,
    }

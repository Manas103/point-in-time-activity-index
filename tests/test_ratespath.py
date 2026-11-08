import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from ratespath import config, curve, eventstudy, measurements, oracle, releases


def test_meeting_calendar_shape():
    meetings = config.meeting_dates()
    assert len(meetings) == 16
    for a, b in zip(meetings, meetings[1:]):
        assert (b - a).days == config.MEETING_SPACING_DAYS


def test_release_count_and_join():
    result = measurements.measure_release_count()
    assert result["n_releases_generated"] == 180
    assert result["n_releases_joined_to_a_meeting"] <= 180
    assert result["n_releases_joined_to_a_meeting"] > 0


def test_reprice_oracle_matches_fast_path():
    result = measurements.measure_reprice_tolerance()
    assert result["n_pillars"] == 11
    assert result["oracle_max_abs_diff"] < 1e-9


def test_reprice_error_reported_honestly_either_way():
    result = measurements.measure_reprice_tolerance()
    # Never asserted to meet the 0.1bp target; just that the measurement
    # ran and produced a real, finite number for every pillar.
    assert result["max_abs_reprice_error_bp"] >= 0.0
    assert len(result["per_pillar_error_bp"]) == 11


def test_event_study_oracle_matches_fast_path():
    result = measurements.measure_event_study()
    assert result["oracle_max_abs_diff_1y"] < 1e-9
    assert result["oracle_max_abs_diff_3y"] < 1e-9
    assert result["n_meetings"] == 16


def test_revised_data_has_less_noise_than_first_print_by_construction():
    assert config.REVISED_NOISE_STD < config.FIRST_PRINT_NOISE_STD


def test_front_end_more_sensitive_than_long_end_in_true_generator():
    assert config.SENS_1Y_BP_PER_SHOCK > config.SENS_3Y_BP_PER_SHOCK


def test_ols_slope_zero_variance_x_returns_zero_not_a_crash():
    assert eventstudy.ols_slope_intercept([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) == (0.0, 3.0)


def test_meeting_window_index_assigns_release_before_first_meeting_to_meeting_zero():
    meetings = config.meeting_dates()
    before = meetings[0] - __import__("datetime").timedelta(days=1)
    assert config.meeting_window_index(before, meetings) == 0


def test_meeting_window_index_excludes_release_after_last_meeting():
    meetings = config.meeting_dates()
    after = meetings[-1] + __import__("datetime").timedelta(days=1)
    assert config.meeting_window_index(after, meetings) == len(meetings)


def test_reprice_oracle_rebuild_from_scratch_matches_curve_module():
    rels = releases.build_releases()
    shocks = releases.meeting_shocks_from_true_factor(rels)
    path = curve.true_path_from_shocks(shocks)
    meetings_t = [config.meeting_time_years(d) for d in config.meeting_dates()]
    par_curve = curve.build_true_par_curve(path, meetings_t)
    step_rates = curve.strip_meeting_path(par_curve, meetings_t)
    pillar_years = sorted(par_curve)
    fast = curve.reprice_from_meeting_path(step_rates, meetings_t, pillar_years)
    slow = oracle.reprice_from_meeting_path_oracle(step_rates, meetings_t, pillar_years, config.R0)
    for T in pillar_years:
        assert abs(fast[T] - slow[T]) < 1e-12

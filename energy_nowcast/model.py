"""The fast path: statsmodels OLS inside a walk-forward, expanding-window,
point-in-time loop.

Every prediction for month t is made by a model fit only on months
strictly before t (and only on rows where every feature and the target are
non-missing), never on t itself or anything after it. `oracle.py`
re-implements the regression with hand-rolled normal equations and no
statsmodels/numpy, diffed against this module in
`tests/test_real_nowcast.py`.
"""

from __future__ import annotations

import statistics

import numpy as np
import statsmodels.api as sm

from energy_nowcast import config


def r_squared(actuals: list[float], preds: list[float]) -> float:
    mean_y = statistics.mean(actuals)
    ss_tot = sum((a - mean_y) ** 2 for a in actuals)
    ss_res = sum((a - p) ** 2 for a, p in zip(actuals, preds))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def walk_forward_ols(
    feature_lists: list[list[float | None]],
    target: list[float | None],
    min_train: int = config.MIN_TRAIN_MONTHS,
) -> tuple[list[float], list[float]]:
    """Expanding-window OLS: at each t >= min_train, fit on every i < t with
    a complete target and feature row, predict at t if t's own row is
    complete. `feature_lists` is a list of equal-length feature columns
    (e.g. [demand_anomaly, hdd_anomaly, cdd_anomaly]); pass a single column
    for a univariate model.

    Returns (actuals, predictions) only for the months a prediction was
    actually possible.
    """
    n = len(target)
    actuals: list[float] = []
    preds: list[float] = []
    for t in range(min_train, n):
        if target[t] is None or any(f[t] is None for f in feature_lists):
            continue
        train_idx = [
            i
            for i in range(t)
            if target[i] is not None and all(f[i] is not None for f in feature_lists)
        ]
        if len(train_idx) < min_train // 2:
            continue
        y_train = np.array([target[i] for i in train_idx], dtype=float)
        X_train = np.array([[f[i] for f in feature_lists] for i in train_idx], dtype=float)
        X_train = sm.add_constant(X_train)
        fit = sm.OLS(y_train, X_train).fit()
        x_test = np.array([1.0] + [f[t] for f in feature_lists], dtype=float)
        pred = float(fit.predict(x_test.reshape(1, -1))[0])
        actuals.append(target[t])
        preds.append(pred)
    return actuals, preds

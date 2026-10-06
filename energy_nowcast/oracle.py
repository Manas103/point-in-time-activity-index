"""Independent recomputation of the walk-forward OLS, with no numpy,
pandas, or statsmodels: plain Python lists and Gaussian elimination on the
normal equations (X^T X) beta = X^T y.

Imports nothing from model.py. A bug specific to statsmodels' fit path
(a wrong design-matrix column order, a dropped row) has a real chance of
showing up as a disagreement between this and model.walk_forward_ols; it
has no chance of showing up at all if both paths shared the same code.
"""

from __future__ import annotations

from energy_nowcast import config


def _ols_normal_equations(X_train: list[list[float]], y_train: list[float], x_test: list[float]) -> float:
    n = len(X_train)
    k = len(X_train[0]) + 1  # +1 for the intercept column

    rows = [[1.0] + list(r) for r in X_train]
    XtX = [[sum(rows[i][a] * rows[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(rows[i][a] * y_train[i] for i in range(n)) for a in range(k)]

    augmented = [XtX[i][:] + [Xty[i]] for i in range(k)]
    for col in range(k):
        pivot_row = max(range(col, k), key=lambda r: abs(augmented[r][col]))
        augmented[col], augmented[pivot_row] = augmented[pivot_row], augmented[col]
        pivot = augmented[col][col]
        if abs(pivot) < 1e-12:
            continue
        for r in range(k):
            if r == col:
                continue
            factor = augmented[r][col] / pivot
            for c in range(col, k + 1):
                augmented[r][c] -= factor * augmented[col][c]
    beta = [
        augmented[i][k] / augmented[i][i] if abs(augmented[i][i]) > 1e-12 else 0.0
        for i in range(k)
    ]
    x_full = [1.0] + list(x_test)
    return sum(b * x for b, x in zip(beta, x_full))


def walk_forward_ols_oracle(
    feature_lists: list[list[float | None]],
    target: list[float | None],
    min_train: int = config.MIN_TRAIN_MONTHS,
) -> tuple[list[float], list[float]]:
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
        y_train = [target[i] for i in train_idx]
        X_train = [[f[i] for f in feature_lists] for i in train_idx]
        x_test = [f[t] for f in feature_lists]
        pred = _ols_normal_equations(X_train, y_train, x_test)
        actuals.append(target[t])
        preds.append(pred)
    return actuals, preds

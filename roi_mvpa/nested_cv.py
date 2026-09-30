"""Nested leave-one-out cross-validation for Ridge and PCA-Ridge.

Outer LOOCV: each participant is held out once and receives one genuinely
out-of-sample prediction.

Inner LOOCV: run on the outer-training participants only, it selects the
hyperparameters (alpha, and for PCA-Ridge the number of components) by
mean squared error.

The held-out participant never influences scaling, PCA, hyperparameter
selection or coefficient fitting. tests/test_pipeline.py checks this
directly: changing a held-out participant's outcome leaves that
participant's prediction unchanged.
"""

import numpy as np

from .pca import fit_scaler_pca, transform_pca, valid_component_grid
from .ridge import fit_ridge, predict_ridge, solve_dual_ridge


def alpha_grid(log10_start=-4, log10_stop=10, n_values=15):
    """Log-spaced Ridge penalties, one per order of magnitude by default.

    The upper end is deliberately very large so that the grid includes
    effectively intercept-only models: if the data carry no usable signal,
    inner CV is free to choose maximal shrinkage.
    """
    return np.logspace(log10_start, log10_stop, n_values)


def _leave_one_out(n):
    for test in range(n):
        train = np.ones(n, dtype=bool)
        train[test] = False
        yield test, train


# ---------------------------------------------------------------------------
# Primary model: voxel-space Ridge
# ---------------------------------------------------------------------------

def inner_loocv_mse(X, y, alphas):
    """Inner-LOOCV mean squared error for every alpha."""
    n = len(y)
    predictions = np.empty((len(alphas), n))
    for test, train in _leave_one_out(n):
        for a, alpha in enumerate(alphas):
            model = fit_ridge(X[train], y[train], alpha)
            predictions[a, test] = predict_ridge(model, X[test:test + 1])[0]
    return np.mean((predictions - y[np.newaxis, :]) ** 2, axis=1)


def select_alpha(X, y, alphas):
    """Alpha with the lowest inner-LOOCV MSE (first minimum on exact ties)."""
    index = int(np.argmin(inner_loocv_mse(X, y, alphas)))
    return float(alphas[index]), index


def nested_loocv_ridge(X, y, alphas):
    """Run nested-LOOCV Ridge and a fold-specific training-mean baseline.

    The baseline predicts each held-out participant with the mean outcome of
    the corresponding training participants. It uses no imaging data and is
    the reference any useful model must beat.
    """
    n = len(y)
    predictions = np.full(n, np.nan)
    baseline = np.full(n, np.nan)
    selected_alpha = np.full(n, np.nan)
    selected_index = np.zeros(n, dtype=int)

    for test, train in _leave_one_out(n):
        alpha, index = select_alpha(X[train], y[train], alphas)
        model = fit_ridge(X[train], y[train], alpha)
        predictions[test] = predict_ridge(model, X[test:test + 1])[0]
        baseline[test] = float(np.mean(y[train]))
        selected_alpha[test] = alpha
        selected_index[test] = index

    for name, values in (("predictions", predictions),
                         ("baseline", baseline),
                         ("selected alphas", selected_alpha)):
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite " + name)

    return {"predictions": predictions, "baseline": baseline,
            "selected_alpha": selected_alpha,
            "selected_alpha_index": selected_index}


# ---------------------------------------------------------------------------
# Sensitivity model: scaling -> PCA -> Ridge
# ---------------------------------------------------------------------------

def select_pca_ridge(X, y, component_grid, alphas):
    """Jointly select component count and alpha by inner-LOOCV MSE.

    Each inner fold fits scaling + PCA once at the largest component count.
    Smaller counts reuse its leading components, which is valid because PCA
    components are ordered and nested.
    """
    n = len(y)
    candidates = valid_component_grid(component_grid, n - 1, X.shape[1])
    k_max = max(candidates)
    squared_errors = np.zeros((len(candidates), len(alphas)))

    for test, train in _leave_one_out(n):
        pca = fit_scaler_pca(X[train], k_max)
        train_scores = pca["X_scores"]
        test_scores = transform_pca(pca, X[test:test + 1])
        for c, k in enumerate(candidates):
            for a, alpha in enumerate(alphas):
                y_mean, beta = solve_dual_ridge(train_scores[:, :k],
                                                y[train], alpha)
                prediction = y_mean + float(np.dot(test_scores[0, :k], beta))
                squared_errors[c, a] += (y[test] - prediction) ** 2

    mse = squared_errors / float(n)
    # Row-major argmin: on exact ties the smaller component count wins,
    # then the smaller alpha.
    c, a = np.unravel_index(int(np.argmin(mse)), mse.shape)
    return {"n_components": int(candidates[c]), "alpha": float(alphas[a]),
            "inner_mse": float(mse[c, a])}


def nested_loocv_pca_ridge(X, y, component_grid, alphas):
    """Nested-LOOCV PCA-Ridge, refitting the full pipeline per outer fold."""
    n = len(y)
    predictions = np.full(n, np.nan)
    baseline = np.full(n, np.nan)
    components = np.zeros(n, dtype=int)
    selected_alpha = np.full(n, np.nan)
    inner_mse = np.full(n, np.nan)

    for test, train in _leave_one_out(n):
        choice = select_pca_ridge(X[train], y[train], component_grid, alphas)
        pca = fit_scaler_pca(X[train], choice["n_components"])
        y_mean, beta = solve_dual_ridge(pca["X_scores"], y[train],
                                        choice["alpha"])
        test_scores = transform_pca(pca, X[test:test + 1])
        predictions[test] = y_mean + float(np.dot(test_scores[0], beta))
        baseline[test] = float(np.mean(y[train]))
        components[test] = choice["n_components"]
        selected_alpha[test] = choice["alpha"]
        inner_mse[test] = choice["inner_mse"]

    if not np.isfinite(predictions).all():
        raise RuntimeError("Non-finite PCA-Ridge predictions")

    return {"predictions": predictions, "baseline": baseline,
            "selected_n_components": components,
            "selected_alpha": selected_alpha, "inner_mse": inner_mse}

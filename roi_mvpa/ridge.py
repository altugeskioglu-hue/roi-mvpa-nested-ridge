"""Standardised Ridge regression solved in dual form.

With far more voxels than participants (p >> n), the dual solution

    beta = Z.T @ (Z @ Z.T + alpha * I)^-1 @ (y - mean(y))

replaces a p x p system with an n x n one: with a dozen training
participants and thousands of voxels it is a 12 x 12 solve.

The intercept is not penalised: features are centred on the training mean,
so the intercept equals the training-set mean of y.
"""

import numpy as np


def fit_standardiser(X_train):
    """Return voxel-wise mean and SD estimated from training rows only.

    Constant features keep an effective scale of 1, matching scikit-learn's
    StandardScaler. Population SD (ddof=0) is used for the same reason.
    """
    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0, ddof=0).copy()
    scale[scale == 0] = 1.0
    return mean, scale


def solve_dual_ridge(Z, y, alpha):
    """Fit Ridge on already centred/scaled features.

    Returns the training mean of y (the unpenalised intercept) and the
    primal coefficient vector.
    """
    y_mean = float(np.mean(y))
    kernel = np.dot(Z, Z.T)
    try:
        dual = np.linalg.solve(kernel + float(alpha) * np.eye(Z.shape[0]),
                               y - y_mean)
    except np.linalg.LinAlgError as error:
        raise RuntimeError(
            "Ridge solve failed for alpha {0}: {1}".format(alpha, error))
    beta = np.dot(Z.T, dual)
    if not np.isfinite(beta).all():
        raise RuntimeError("Ridge produced non-finite coefficients")
    return y_mean, beta


def fit_ridge(X_train, y_train, alpha):
    """Standardise X on the training rows, then fit dual Ridge."""
    X_mean, X_scale = fit_standardiser(X_train)
    Z = (X_train - X_mean) / X_scale
    y_mean, beta = solve_dual_ridge(Z, y_train, alpha)
    return {"X_mean": X_mean, "X_scale": X_scale,
            "y_mean": y_mean, "beta": beta}


def predict_ridge(model, X):
    """Apply the training-fold standardisation and coefficients to new rows."""
    Z = (X - model["X_mean"]) / model["X_scale"]
    predictions = model["y_mean"] + np.dot(Z, model["beta"])
    predictions = np.asarray(predictions, dtype=float)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Ridge produced non-finite predictions")
    return predictions

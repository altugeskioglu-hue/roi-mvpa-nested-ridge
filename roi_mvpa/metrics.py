"""Held-out prediction metrics."""

import json

import numpy as np


def cv_r2(y_true, y_pred):
    """Cross-validated R^2 = 1 - SSE / SST.

    Unlike squared Pearson r, this penalises biased or badly scaled
    predictions, and it is negative when the model is worse than predicting
    the sample mean.
    """
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst == 0:
        return float("nan")
    return 1.0 - float(np.sum((y_true - y_pred) ** 2)) / sst


def calculate_metrics(y_true, y_pred):
    """CV R^2 (primary) plus Pearson r, MAE and RMSE (complementary).

    r measures association only, MAE gives the typical absolute error and
    RMSE weights large errors more heavily.
    """
    residuals = y_true - y_pred
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_r = float("nan")
    else:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {
        "pearson_r": pearson_r,
        "MAE": float(np.mean(np.abs(residuals))),
        "RMSE": float(np.sqrt(np.mean(residuals ** 2))),
        "CV_R2": cv_r2(y_true, y_pred),
    }


def selection_counts(values, integer=False):
    """JSON string of how often each hyperparameter value was selected."""
    unique, counts = np.unique(values, return_counts=True)
    label = (lambda v: str(int(v))) if integer else \
        (lambda v: "{0:.10g}".format(float(v)))
    return json.dumps({label(v): int(c) for v, c in zip(unique, counts)},
                      sort_keys=True)

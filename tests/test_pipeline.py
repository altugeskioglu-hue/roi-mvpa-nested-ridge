"""Correctness checks for the modelling code.

Run with:  pytest -q
"""

import numpy as np
import pytest

from roi_mvpa.metrics import cv_r2
from roi_mvpa.nested_cv import (alpha_grid, nested_loocv_pca_ridge,
                                nested_loocv_ridge)
from roi_mvpa.pca import fit_scaler_pca
from roi_mvpa.permutation import (NestedRidgeEngine, check_against_direct,
                                  permutation_test)
from roi_mvpa.ridge import fit_ridge, fit_standardiser, predict_ridge

ALPHAS = alpha_grid(-4, 10, 15)


def make_data(n=12, p=300, signal=1.0, seed=0):
    rng = np.random.RandomState(seed)
    latent = rng.normal(size=(n, 3))
    X = latent.dot(rng.normal(size=(3, p))) + rng.normal(scale=2.0,
                                                         size=(n, p))
    y = signal * latent[:, 0] + rng.normal(size=n)
    return X, y


def test_dual_ridge_matches_primal_solution():
    """Dual solve must equal the textbook primal Ridge solution."""
    X, y = make_data(n=10, p=25)
    for alpha in (0.1, 10.0, 1e4):
        model = fit_ridge(X, y, alpha)
        mean, scale = fit_standardiser(X)
        Z = (X - mean) / scale
        primal = np.linalg.solve(Z.T.dot(Z) + alpha * np.eye(Z.shape[1]),
                                 Z.T.dot(y - y.mean()))
        np.testing.assert_allclose(model["beta"], primal, atol=1e-8)


def test_huge_alpha_predicts_training_mean():
    X, y = make_data()
    model = fit_ridge(X[:-1], y[:-1], 1e12)
    assert predict_ridge(model, X[-1:])[0] == pytest.approx(y[:-1].mean(),
                                                            abs=1e-6)


@pytest.mark.parametrize("held_out", [0, 5, 11])
def test_held_out_outcome_never_used_ridge(held_out):
    """Leakage check: changing a held-out participant's y must not change
    that participant's own prediction."""
    X, y = make_data()
    before = nested_loocv_ridge(X, y, ALPHAS)["predictions"][held_out]
    y_changed = y.copy()
    y_changed[held_out] += 100.0
    after = nested_loocv_ridge(X, y_changed, ALPHAS)["predictions"][held_out]
    assert after == pytest.approx(before, abs=1e-10)


def test_held_out_outcome_never_used_pca_ridge():
    X, y = make_data()
    grid = [1, 2, 3, 5]
    before = nested_loocv_pca_ridge(X, y, grid, ALPHAS)["predictions"][3]
    y_changed = y.copy()
    y_changed[3] -= 50.0
    after = nested_loocv_pca_ridge(X, y_changed, grid,
                                   ALPHAS)["predictions"][3]
    assert after == pytest.approx(before, abs=1e-10)


def test_pca_leading_components_are_nested():
    """Reusing the first k components of a larger fit must equal a k fit."""
    X, _ = make_data(n=12, p=200)
    full = fit_scaler_pca(X, 8)
    small = fit_scaler_pca(X, 3)
    np.testing.assert_allclose(np.abs(full["X_scores"][:, :3]),
                               np.abs(small["X_scores"]), atol=1e-8)


def test_fast_permutation_engine_matches_direct_code():
    """The cached engine must reproduce the direct nested CV exactly,
    for the observed outcome and for permuted outcomes."""
    X, y = make_data(n=10, p=200)
    engine = NestedRidgeEngine(X, ALPHAS, verbose=False)
    rng = np.random.RandomState(1)
    for y_test in (y, y[rng.permutation(len(y))], y[rng.permutation(len(y))]):
        assert check_against_direct(X, y_test, engine) < 1e-8


def test_permutation_p_values_are_valid():
    X, y = make_data(n=10, p=150, signal=0.0)
    engines = {"a": NestedRidgeEngine(X, ALPHAS, verbose=False),
               "b": NestedRidgeEngine(X[:, ::-1] * 2.0, ALPHAS,
                                      verbose=False)}
    result = permutation_test(engines, y, n_permutations=30, seed=0)
    for row in result["summary"]:
        assert 1.0 / 31 <= row["p_nominal"] <= 1.0
        # FWER-corrected p can never be smaller than the nominal p.
        assert row["p_FWER_max_stat"] >= row["p_nominal"]
    assert np.all(result["max_null"] >= result["null"].max(axis=1) - 1e-12)


def test_pipeline_recovers_real_signal():
    """Positive control: with enough participants and a real signal the
    model should clearly beat the training-mean baseline."""
    X, y = make_data(n=40, p=300, signal=2.0, seed=3)
    result = nested_loocv_ridge(X, y, ALPHAS)
    assert cv_r2(y, result["predictions"]) > 0.3
    assert cv_r2(y, result["predictions"]) > cv_r2(y, result["baseline"])

"""Exact, fast full-pipeline permutation test for nested-LOOCV Ridge.

Why this is fast but still exact
--------------------------------
For a fixed X, fold and alpha, a Ridge prediction is a linear function of the
training outcomes: prediction = w @ y_train. The weights w depend only on X
and the CV split, so they can be computed once. Each permutation then needs
only matrix-vector products, while alpha is still re-selected by inner LOOCV
from the permuted outcomes in every outer fold. Nothing y-dependent is frozen.

Inference
---------
* Statistic: cross-validated R^2 (upper tail; larger is better).
* Nominal p = (1 + #{null >= observed}) / (1 + B).
* FWER across models: compare each observed value with the per-permutation
  maximum over models (max-statistic correction). The same permutation is
  applied to every model so that their dependence is preserved.
* Simple label permutation assumes exchangeable, independent participants and
  no nuisance covariates. Adding covariates would need a different scheme.
"""

import time

import numpy as np

from .metrics import cv_r2
from .nested_cv import nested_loocv_ridge
from .ridge import fit_standardiser


def prediction_weights(X_train, x_test, alphas):
    """Linear weights mapping y_train to the test prediction, per alpha.

    Row a satisfies prediction(alpha_a) = weights[a] @ y_train, including
    the unpenalised intercept (the 1/n term) and the centring of y.
    """
    mean, scale = fit_standardiser(X_train)
    Z = (X_train - mean) / scale
    z_test = (x_test - mean) / scale
    kernel = np.dot(Z, Z.T)
    test_kernel = np.dot(Z, z_test.ravel())
    n = Z.shape[0]
    weights = np.empty((len(alphas), n))
    for a, alpha in enumerate(alphas):
        solved = np.linalg.solve(kernel + float(alpha) * np.eye(n),
                                 test_kernel)
        weights[a] = np.ones(n) / n + solved - solved.mean()
    return weights


class NestedRidgeEngine(object):
    """Precomputed nested-LOOCV operators for one feature matrix."""

    def __init__(self, X, alphas, name="model", verbose=True):
        self.alphas = np.asarray(alphas, dtype=float)
        n = X.shape[0]
        self.train_indices = []
        self.outer_weights = np.zeros((n, len(alphas), n))
        self.inner_operators = []

        for outer in range(n):
            train = np.array([i for i in range(n) if i != outer])
            self.train_indices.append(train)
            X_outer = X[train]
            self.outer_weights[outer][:, train] = prediction_weights(
                X_outer, X[outer:outer + 1], self.alphas)

            n_inner = len(train)
            H = np.zeros((len(alphas), n_inner, n_inner))
            for inner in range(n_inner):
                inner_train = np.array(
                    [i for i in range(n_inner) if i != inner])
                H[:, inner, inner_train] = prediction_weights(
                    X_outer[inner_train], X_outer[inner:inner + 1],
                    self.alphas)
            self.inner_operators.append(H)
            if verbose:
                print("  {0}: fold {1}/{2} precomputed".format(
                    name, outer + 1, n), flush=True)

    def run(self, y):
        """Full nested alpha selection and outer predictions for one y."""
        n = len(y)
        predictions = np.empty(n)
        selected = np.empty(n, dtype=int)
        for outer in range(n):
            y_train = y[self.train_indices[outer]]
            inner_pred = np.einsum("aij,j->ai", self.inner_operators[outer],
                                   y_train)
            mse = np.mean((inner_pred - y_train[np.newaxis, :]) ** 2, axis=1)
            best = int(np.argmin(mse))
            selected[outer] = best
            predictions[outer] = np.dot(self.outer_weights[outer, best], y)
        return predictions, selected


def check_against_direct(X, y, engine, atol=1e-8):
    """Require the fast engine to reproduce the direct nested-CV code."""
    fast_pred, fast_idx = engine.run(y)
    direct = nested_loocv_ridge(X, y, engine.alphas)
    difference = float(np.max(np.abs(fast_pred - direct["predictions"])))
    same_alpha = np.array_equal(fast_idx, direct["selected_alpha_index"])
    if difference > atol or not same_alpha:
        raise RuntimeError(
            "Fast engine does not match direct nested CV "
            "(max diff {0:.2e}, same alphas: {1})".format(difference,
                                                         same_alpha))
    return difference


def permutation_test(engines, y, n_permutations, seed):
    """Run the permutation test for one or more models sharing y.

    Parameters
    ----------
    engines : dict of name -> NestedRidgeEngine
    y : outcome vector (observed order)
    """
    names = list(engines)
    observed = np.array([cv_r2(y, engines[m].run(y)[0]) for m in names])

    rng = np.random.RandomState(seed)
    n = len(y)
    indices = np.empty((n_permutations, n), dtype=np.int32)
    null = np.empty((n_permutations, len(names)))
    alpha_idx = np.empty((n_permutations, len(names), n), dtype=np.uint8)

    start = time.time()
    report_every = max(1, n_permutations // 10)
    for b in range(n_permutations):
        order = rng.permutation(n)  # same permutation for every model
        indices[b] = order
        y_perm = y[order]
        for m, name in enumerate(names):
            pred, sel = engines[name].run(y_perm)
            null[b, m] = cv_r2(y_perm, pred)
            alpha_idx[b, m] = sel
        if (b + 1) % report_every == 0 or b + 1 == n_permutations:
            print("Completed {0}/{1}".format(b + 1, n_permutations),
                  flush=True)

    max_null = null.max(axis=1)
    summary = []
    for m, name in enumerate(names):
        summary.append({
            "model": name,
            "observed_CV_R2": float(observed[m]),
            "n_permutations": int(n_permutations),
            "p_nominal": float((1 + np.sum(null[:, m] >= observed[m]))
                               / (n_permutations + 1.0)),
            "p_FWER_max_stat": float((1 + np.sum(max_null >= observed[m]))
                                     / (n_permutations + 1.0)),
            "null_mean_CV_R2": float(null[:, m].mean()),
            "null_SD_CV_R2": float(null[:, m].std()),
            "null_95th_percentile_CV_R2": float(np.percentile(null[:, m],
                                                              95)),
        })

    return {"names": names, "observed": observed, "null": null,
            "max_null": max_null, "permutation_indices": indices,
            "selected_alpha_indices": alpha_idx, "summary": summary,
            "seconds": time.time() - start}

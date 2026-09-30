"""Training-fold scaling followed by non-whitened PCA (via SVD).

Scaling before PCA stops high-variance voxels dominating purely because of
numerical scale. Both steps are fitted on training rows only.

The PCA is non-whitened: component scores keep their variance.
"""

import numpy as np

from .ridge import fit_standardiser


def valid_component_grid(component_grid, n_train, n_features):
    """Drop component counts a training fold cannot support.

    After centring, n training rows have rank at most n - 1.
    """
    maximum = min(int(n_train) - 1, int(n_features))
    if maximum < 1:
        raise ValueError("Training fold cannot support any PCA components")
    valid = sorted(set(k for k in component_grid if k <= maximum))
    return valid if valid else [maximum]


def fit_scaler_pca(X_train, n_components):
    """Fit scaling + PCA on X_train and return the model and training scores.

    SVD of the (n x p) scaled matrix finds the principal directions without
    building a p x p voxel covariance matrix.
    """
    X_mean, X_scale = fit_standardiser(X_train)
    X_scaled = (X_train - X_mean) / X_scale

    maximum = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
    if not 1 <= n_components <= maximum:
        raise ValueError(
            "Requested {0} PCA components; this fold supports 1 to {1}"
            .format(n_components, maximum))

    _, singular_values, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    components = Vt[:n_components, :]
    scores = np.dot(X_scaled, components.T)
    if not np.isfinite(scores).all():
        raise RuntimeError("PCA produced non-finite training scores")

    return {"X_mean": X_mean, "X_scale": X_scale, "components": components,
            "singular_values": singular_values[:n_components],
            "X_scores": scores}


def transform_pca(pca_model, X):
    """Project new rows using the training-fold scaling and components."""
    X_scaled = (X - pca_model["X_mean"]) / pca_model["X_scale"]
    scores = np.asarray(np.dot(X_scaled, pca_model["components"].T),
                        dtype=float)
    if not np.isfinite(scores).all():
        raise RuntimeError("PCA transformation produced non-finite scores")
    return scores

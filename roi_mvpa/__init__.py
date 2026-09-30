"""Leakage-controlled ROI-based multivoxel prediction for small-sample fMRI.

The package predicts a continuous participant-level outcome from voxel-wise
task-fMRI features using Ridge regression inside nested leave-one-out
cross-validation (LOOCV), with optional PCA-Ridge sensitivity analysis and an
exact full-pipeline permutation test with max-statistic FWER control.

Every data-dependent step (feature standardisation, PCA, hyperparameter
selection and model fitting) is learned from training participants only.
"""

__version__ = "1.0.0"

DEFAULT_ALPHA_GRID = (-4, 10, 15)  # log10 start, log10 stop, number of values

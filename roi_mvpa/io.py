"""Loading and validating feature matrices and outcomes.

Expected layout (produced by scripts/02_extract_features.py):

    <feature_dir>/<name>_X.npy          participants x voxels
    <feature_dir>/<name>_subjects.txt   one participant ID per line, row order
    outcome.csv                         columns: subject_id (or subject), y

Row order is checked against the outcome file so that X and y can never be
silently misaligned.
"""

from pathlib import Path

import numpy as np
import pandas as pd

X_SUFFIX = "_X.npy"


def load_outcome(path, y_column="y"):
    """Load participant IDs and a continuous outcome with strict checks."""
    table = pd.read_csv(str(path))
    if "subject_id" in table.columns:
        id_column = "subject_id"
    elif "subject" in table.columns:
        id_column = "subject"
    else:
        raise ValueError("Outcome file needs a 'subject_id' or 'subject' "
                         "column")
    if y_column not in table.columns:
        raise ValueError("Outcome file is missing the '{0}' column"
                         .format(y_column))
    if table[id_column].isna().any() or table[y_column].isna().any():
        raise ValueError("Outcome file contains missing values")

    subjects = [str(v).strip() for v in table[id_column]]
    if any(s == "" for s in subjects) or len(set(subjects)) != len(subjects):
        raise ValueError("Outcome file has empty or duplicate subject IDs")

    y = pd.to_numeric(table[y_column], errors="raise").values.astype(float)
    if not np.isfinite(y).all():
        raise ValueError("Outcome contains NaN or infinite values")
    if len(y) < 4:
        raise ValueError("At least four participants are required")
    if np.std(y) == 0:
        raise ValueError("Outcome has zero variance")
    return subjects, y


def read_subjects(path):
    subjects = [line.strip() for line in Path(path).read_text().splitlines()
                if line.strip()]
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Empty or duplicate subject IDs in {0}".format(path))
    return subjects


def discover_feature_sets(feature_dir, pattern="*_X.npy"):
    """Return {name: path} for every feature matrix matching the pattern."""
    paths = sorted(Path(feature_dir).glob(pattern), key=lambda p: p.name)
    if not paths:
        raise FileNotFoundError("No feature matrices matching '{0}' in {1}"
                                .format(pattern, feature_dir))
    return {p.name[:-len(X_SUFFIX)]: p for p in paths
            if p.name.endswith(X_SUFFIX)}


def load_feature_set(x_path, outcome_subjects):
    """Load one X matrix and require its row order to match the outcome."""
    x_path = Path(x_path)
    name = x_path.name[:-len(X_SUFFIX)]
    X = np.asarray(np.load(str(x_path), allow_pickle=False), dtype=np.float64)
    subjects = read_subjects(x_path.with_name(name + "_subjects.txt"))

    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("{0}: expected a non-empty 2D matrix, got {1}"
                         .format(name, X.shape))
    if X.shape[0] != len(subjects) or X.shape[0] != len(outcome_subjects):
        raise ValueError("{0}: row count does not match subject/outcome "
                         "files".format(name))
    if subjects != outcome_subjects:
        mismatches = [(i + 1, a, b) for i, (a, b)
                      in enumerate(zip(subjects, outcome_subjects)) if a != b]
        raise ValueError("{0}: subject order mismatch, first: {1}"
                         .format(name, mismatches[:5]))
    if not np.isfinite(X).all():
        raise ValueError("{0}: X contains NaN or infinite values"
                         .format(name))
    return name, X

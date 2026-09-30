#!/usr/bin/env python3
"""Generate synthetic ROI features and an outcome to try the pipeline.

The features mimic fMRI in a simple way: voxels are driven by a few shared
latent sources plus noise, so neighbouring features are correlated and
p >> n. The outcome can depend on one latent source (--signal > 0) or be
pure noise (--signal 0).

Two runs are useful:
  * a null scenario (small n, no signal): the pipeline should NOT find
    anything, i.e. CV R^2 around or below 0 and non-significant p-values;
  * a positive control (larger n, real signal): the pipeline should recover
    it, showing that a null result is not simply a broken pipeline.

Usage
-----
    python examples/make_synthetic_data.py --output-dir data/null \
        --n-subjects 14 --signal 0
    python examples/make_synthetic_data.py --output-dir data/signal \
        --n-subjects 60 --signal 1.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_features(rng, latent, n_voxels, noise_sd):
    loadings = rng.normal(size=(latent.shape[1], n_voxels))
    return latent.dot(loadings) + rng.normal(scale=noise_sd,
                                             size=(latent.shape[0], n_voxels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-subjects", type=int, default=14)
    parser.add_argument("--n-voxels", type=int, default=2000)
    parser.add_argument("--n-latent", type=int, default=5)
    parser.add_argument("--signal", type=float, default=0.0,
                        help="Outcome dependence on latent source 1 "
                             "(0 = no signal)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    n = args.n_subjects
    latent = rng.normal(size=(n, args.n_latent))
    subjects = ["sub-{0:02d}".format(i + 1) for i in range(n)]

    out = Path(args.output_dir)
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Two "task contrasts" sharing the same latent sources.
    for name in ("contrast_A", "contrast_B"):
        X = make_features(rng, latent, args.n_voxels, noise_sd=3.0)
        np.save(str(feature_dir / (name + "_X.npy")), X)
        (feature_dir / (name + "_subjects.txt")).write_text(
            "\n".join(subjects) + "\n")

    y = args.signal * latent[:, 0] + rng.normal(size=n)
    pd.DataFrame({"subject_id": subjects, "y": y}).to_csv(
        str(out / "outcome.csv"), index=False)

    print("Wrote {0} participants x {1} voxels (x2 contrasts), signal={2} "
          "to {3}".format(n, args.n_voxels, args.signal, out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Extract one participant x voxel matrix per task contrast.

Usage
-----
    python scripts/02_extract_features.py \
        --mask masks/union_roi_2mm.nii.gz \
        --manifest manifests/encoding.csv \
        --name encoding \
        --output-dir features/

The manifest is a CSV with columns `subject_id,image_path`, one row per
participant, in the same order as the outcome file. image_path points to a
standard-space first-level contrast image (e.g. an FSL cope*_standard.nii.gz).

Run once per contrast. Contrasts are kept as separate models rather than
concatenated.

Outputs (refuses to overwrite existing files):
    <name>_X.npy, <name>_subjects.txt, <name>_extraction_QC.txt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from roi_mvpa.features import extract_features


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mask", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-voxels", type=int, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if not {"subject_id", "image_path"} <= set(manifest.columns):
        raise ValueError("Manifest needs subject_id and image_path columns")
    subjects = [str(s).strip() for s in manifest["subject_id"]]
    if len(set(subjects)) != len(subjects):
        raise ValueError("Duplicate subject IDs in manifest")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    targets = [out / (args.name + suffix) for suffix in
               ("_X.npy", "_subjects.txt", "_extraction_QC.txt")]
    existing = [str(p) for p in targets if p.exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite: " + ", ".join(existing))

    X, n_voxels = extract_features(args.mask, manifest["image_path"],
                                   args.expected_voxels)

    np.save(str(targets[0]), X)
    targets[1].write_text("\n".join(subjects) + "\n")
    targets[2].write_text("\n".join([
        "Feature extraction QC",
        "Mask: {0}".format(args.mask),
        "Mask voxels: {0}".format(n_voxels),
        "Participants: {0}".format(X.shape[0]),
        "Matrix shape: {0}".format(X.shape),
        "Min / max: {0:.6g} / {1:.6g}".format(X.min(), X.max()),
        "All shapes and affines matched the mask; all values finite.",
    ]) + "\n")
    print("Saved {0} with shape {1}".format(targets[0], X.shape))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Full-pipeline permutation test with max-statistic FWER correction.

Every permutation reruns the complete nested-LOOCV Ridge pipeline, including
inner-CV alpha selection, on permuted outcomes. The same permutation is used
for all models so the max-statistic correction respects their dependence.

Before permuting, the fast engine is checked against the direct nested-CV
implementation on the observed data; the run stops if they disagree.

Usage
-----
    python scripts/05_run_permutation_test.py \
        --feature-dir features/ --outcome outcome.csv \
        --output-dir results/permutation --n-permutations 5000 --seed 12345

Use a small --n-permutations (e.g. 100) first to benchmark run time.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from roi_mvpa.io import discover_feature_sets, load_feature_set, load_outcome
from roi_mvpa.nested_cv import alpha_grid
from roi_mvpa.permutation import (NestedRidgeEngine, check_against_direct,
                                  permutation_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--pattern", default="*_X.npy")
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--alpha-log10", nargs=3, type=float,
                        default=[-4, 10, 15],
                        metavar=("START", "STOP", "N"))
    parser.add_argument("--skip-direct-check", action="store_true",
                        help="Skip the fast-vs-direct equivalence check")
    args = parser.parse_args()

    if args.n_permutations < 1:
        raise ValueError("--n-permutations must be positive")
    out = Path(args.output_dir)
    if out.exists():
        raise RuntimeError("Output directory already exists: {0}".format(out))

    alphas = alpha_grid(args.alpha_log10[0], args.alpha_log10[1],
                        int(args.alpha_log10[2]))
    subjects, y = load_outcome(args.outcome)

    engines, matrices = {}, {}
    for name, path in discover_feature_sets(args.feature_dir,
                                            args.pattern).items():
        name, X = load_feature_set(path, subjects)
        print("Precomputing " + name)
        engines[name] = NestedRidgeEngine(X, alphas, name)
        matrices[name] = X

    if not args.skip_direct_check:
        print("\nFast-vs-direct equivalence checks:")
        for name in engines:
            diff = check_against_direct(matrices[name], y, engines[name])
            print("  {0}: PASSED (max diff {1:.2e})".format(name, diff))

    result = permutation_test(engines, y, args.n_permutations, args.seed)

    out.mkdir(parents=True)
    null_table = pd.DataFrame(result["null"], columns=[
        n + "_CV_R2" for n in result["names"]])
    null_table.insert(0, "permutation",
                      np.arange(1, args.n_permutations + 1))
    null_table["maximum_CV_R2"] = result["max_null"]
    null_table.to_csv(str(out / "permutation_null_CV_R2.csv"), index=False)

    summary = pd.DataFrame(result["summary"])
    summary.to_csv(str(out / "permutation_summary.csv"), index=False)

    np.savez_compressed(
        str(out / "permutation_arrays.npz"),
        null_cv_r2=result["null"],
        permutation_indices=result["permutation_indices"],
        selected_alpha_indices=result["selected_alpha_indices"],
        alpha_grid=alphas, observed_cv_r2=result["observed"])

    (out / "permutation_config.json").write_text(json.dumps({
        "n_permutations": args.n_permutations, "seed": args.seed,
        "models": result["names"], "subjects": subjects,
        "alpha_grid": alphas.tolist(),
        "statistic": "cross-validated R2 (upper tail)",
        "correction": "max-statistic FWER across models",
        "direct_check_skipped": bool(args.skip_direct_check),
        "permutation_loop_seconds": result["seconds"],
    }, indent=2) + "\n")

    print("\n" + summary.to_string(index=False))
    print("\nSaved to {0}".format(out.resolve()))


if __name__ == "__main__":
    main()

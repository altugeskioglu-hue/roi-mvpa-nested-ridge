#!/usr/bin/env python3
"""Primary analysis: nested-LOOCV Ridge for every feature matrix.

Usage
-----
    python scripts/03_run_nested_ridge.py \
        --feature-dir features/ --outcome outcome.csv \
        --output-dir results/ridge

Outputs
-------
    <model>_predictions.csv   held-out prediction, baseline and chosen alpha
                              for every participant
    ridge_summary.csv         CV R^2, r, MAE, RMSE vs the training-mean baseline
"""

import argparse
from pathlib import Path

import pandas as pd

from roi_mvpa.io import discover_feature_sets, load_feature_set, load_outcome
from roi_mvpa.metrics import calculate_metrics, selection_counts
from roi_mvpa.nested_cv import alpha_grid, nested_loocv_ridge


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--pattern", default="*_X.npy")
    parser.add_argument("--outcome", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-log10", nargs=3, type=float,
                        default=[-4, 10, 15],
                        metavar=("START", "STOP", "N"))
    args = parser.parse_args()

    alphas = alpha_grid(args.alpha_log10[0], args.alpha_log10[1],
                        int(args.alpha_log10[2]))
    subjects, y = load_outcome(args.outcome)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Participants: {0}".format(len(y)))
    print("Alpha grid: {0}".format(alphas.tolist()))

    rows = []
    for name, path in discover_feature_sets(args.feature_dir,
                                            args.pattern).items():
        name, X = load_feature_set(path, subjects)
        print("\nRunning {0} ({1} features) ...".format(name, X.shape[1]),
              flush=True)
        result = nested_loocv_ridge(X, y, alphas)

        pd.DataFrame({
            "subject": subjects, "observed_y": y,
            "predicted_y": result["predictions"],
            "baseline_prediction": result["baseline"],
            "selected_alpha": result["selected_alpha"],
        }).to_csv(str(out / (name + "_predictions.csv")), index=False)

        model = calculate_metrics(y, result["predictions"])
        base = calculate_metrics(y, result["baseline"])
        rows.append({
            "model": name, "n": len(y), "n_features": X.shape[1],
            **model,
            "baseline_MAE": base["MAE"], "baseline_RMSE": base["RMSE"],
            "baseline_CV_R2": base["CV_R2"],
            "MAE_improvement_vs_baseline": base["MAE"] - model["MAE"],
            "RMSE_improvement_vs_baseline": base["RMSE"] - model["RMSE"],
            "selected_alpha_counts":
                selection_counts(result["selected_alpha"]),
        })
        print("  CV R2={CV_R2:.4f}, r={pearson_r:.4f}, MAE={MAE:.4f}, "
              "RMSE={RMSE:.4f}".format(**model))

    summary = pd.DataFrame(rows).sort_values("model")
    summary.to_csv(str(out / "ridge_summary.csv"), index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()

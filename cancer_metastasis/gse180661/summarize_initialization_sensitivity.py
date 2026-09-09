"""Aggregate per-pair ConfidenceOT initialization-sensitivity results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.input_root.glob("pairs/*/initialization_runs.csv"))
    if not paths:
        raise FileNotFoundError("No initialization_runs.csv files found")
    runs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    runs.to_csv(args.output_root / "all_initialization_runs.csv.gz", index=False, compression="gzip")
    nondefault = runs[runs["strategy"].ne("all_one")].copy()

    pair_summary = nondefault.groupby(
        ["pair_id", "patient_id", "source_sample", "target_sample"], as_index=False
    ).agg(
        run_n=("strategy", "size"),
        minimum_source_agreement=("source_agreement_with_default", "min"),
        median_source_agreement=("source_agreement_with_default", "median"),
        minimum_target_agreement=("target_agreement_with_default", "min"),
        median_target_agreement=("target_agreement_with_default", "median"),
        minimum_source_retained_jaccard=("source_retained_jaccard_with_default", "min"),
        median_source_retained_jaccard=("source_retained_jaccard_with_default", "median"),
        minimum_target_retained_jaccard=("target_retained_jaccard_with_default", "min"),
        median_target_retained_jaccard=("target_retained_jaccard_with_default", "median"),
        minimum_source_score_correlation=("source_score_correlation_with_default", "min"),
        minimum_target_score_correlation=("target_score_correlation_with_default", "min"),
        maximum_absolute_objective_delta=("objective_delta_from_default", lambda x: float(np.max(np.abs(x)))),
        outer_converged_fraction=("outer_converged", "mean"),
        cycle_detected_fraction=("cycle_detected", "mean"),
    )
    pair_summary.to_csv(args.output_root / "pair_initialization_sensitivity.csv", index=False)

    metrics = [
        ("source_retained_jaccard_with_default", "Primary retained-set Jaccard"),
        ("target_retained_jaccard_with_default", "Metastasis retained-set Jaccard"),
        ("source_agreement_with_default", "Primary gate agreement"),
        ("target_agreement_with_default", "Metastasis gate agreement"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for axis, (column, title) in zip(axes.flat, metrics):
        strategies = ("all_zero_projected_to_budget_floor", "random_50_percent")
        values = [
            nondefault.loc[nondefault["strategy"].eq(strategy), column].dropna().to_numpy()
            for strategy in strategies
        ]
        axis.boxplot(values, tick_labels=["Projected zero", "Random 50%"], showfliers=False)
        axis.axhline(0.9 if "jaccard" in column else 0.95, color="#C44E52", linestyle="--")
        axis.set(title=title, ylim=(0, 1.02))
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("M4-E initialization sensitivity across exact primary-metastasis pairs")
    figure.tight_layout()
    figure.savefig(args.output_root / "initialization_sensitivity_summary.png", dpi=300)
    figure.savefig(args.output_root / "initialization_sensitivity_summary.pdf")
    plt.close(figure)

    source_cell_paths = sorted(args.input_root.glob("pairs/*/source_cell_initialization_stability.csv.gz"))
    target_cell_paths = sorted(args.input_root.glob("pairs/*/target_cell_initialization_stability.csv.gz"))
    source_cells = pd.concat([pd.read_csv(path) for path in source_cell_paths], ignore_index=True)
    target_cells = pd.concat([pd.read_csv(path) for path in target_cell_paths], ignore_index=True)
    report = {
        "pair_n": int(pair_summary["pair_id"].nunique()),
        "patient_n": int(pair_summary["patient_id"].nunique()),
        "nondefault_fit_n": len(nondefault),
        "all_zero_status": (
            "infeasible under the rejection-budget constraint; replaced by the "
            "closest feasible deterministic budget-floor gate"
        ),
        "stored_baseline_reconstruction": {
            "median_source_agreement": float(runs["source_default_matches_stored_baseline"].median()),
            "median_target_agreement": float(runs["target_default_matches_stored_baseline"].median()),
        },
        "random_start_robustness": {
            "median_source_gate_agreement": float(nondefault["source_agreement_with_default"].median()),
            "median_target_gate_agreement": float(nondefault["target_agreement_with_default"].median()),
            "median_source_retained_jaccard": float(nondefault["source_retained_jaccard_with_default"].median()),
            "median_target_retained_jaccard": float(nondefault["target_retained_jaccard_with_default"].median()),
            "pair_fraction_minimum_source_jaccard_at_least_0p90": float(pair_summary["minimum_source_retained_jaccard"].ge(0.90).mean()),
            "pair_fraction_minimum_target_jaccard_at_least_0p90": float(pair_summary["minimum_target_retained_jaccard"].ge(0.90).mean()),
            "source_cell_occurrence_consensus_stable_fraction": float(source_cells["consensus_stable"].mean()),
            "target_cell_occurrence_consensus_stable_fraction": float(target_cells["consensus_stable"].mean()),
            "outer_converged_fraction": float(nondefault["outer_converged"].mean()),
            "cycle_detected_fraction": float(nondefault["cycle_detected"].mean()),
        },
        "interpretation": (
            "retained=True is initialization-robust only when retained-set Jaccard, "
            "whole-gate agreement, and score correlation remain high across feasible starts."
        ),
    }
    (args.output_root / "initialization_sensitivity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

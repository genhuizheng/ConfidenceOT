"""Aggregate pair outputs without treating multiple pairs from one patient as independent."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(args.result_root.glob("*/scope_*/budget_*/pair_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No pair_metrics.csv under {args.result_root}")
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    metrics.to_csv(args.output_dir / "all_pair_metrics.csv", index=False)
    numeric = [
        "source_raw_rejection_rate", "target_raw_rejection_rate",
        "source_final_rejection_rate", "target_final_rejection_rate",
        "source_budget_override_rate", "target_budget_override_rate",
        "rejection_cost", "transported_mass", "fit_seconds",
    ]
    patients = metrics.groupby(
        ["dataset_id", "patient_id", "analysis_scope", "method", "rejection_budget_cap"],
        as_index=False
    )[numeric].mean()
    patients["pair_n"] = metrics.groupby(
        ["dataset_id", "patient_id", "analysis_scope", "method", "rejection_budget_cap"]
    ).size().to_numpy()
    patients.to_csv(args.output_dir / "patient_level_metrics.csv", index=False)
    summary = patients.groupby(
        ["dataset_id", "analysis_scope", "method", "rejection_budget_cap"], as_index=False
    ).agg(
        patient_n=("patient_id", "nunique"), pair_n=("pair_n", "sum"),
        source_raw_rejection_rate=("source_raw_rejection_rate", "mean"),
        target_raw_rejection_rate=("target_raw_rejection_rate", "mean"),
        source_final_rejection_rate=("source_final_rejection_rate", "mean"),
        target_final_rejection_rate=("target_final_rejection_rate", "mean"),
        source_budget_override_rate=("source_budget_override_rate", "mean"),
        target_budget_override_rate=("target_budget_override_rate", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"), total_fit_seconds=("fit_seconds", "sum"),
    )
    summary.to_csv(args.output_dir / "dataset_budget_summary_patient_weighted.csv", index=False)
    run_keys = [
        "pair_id", "dataset_id", "patient_id", "analysis_scope", "rejection_budget_cap",
    ]
    timing = metrics.drop_duplicates(run_keys)[
        run_keys + ["calibration_seconds_shared", "pipeline_seconds_shared"]
    ].copy()
    timing.to_csv(args.output_dir / "run_level_timing.csv", index=False)
    timing.groupby(
        ["dataset_id", "analysis_scope", "rejection_budget_cap"], as_index=False
    ).agg(
        run_n=("pair_id", "size"),
        mean_calibration_seconds=("calibration_seconds_shared", "mean"),
        total_calibration_seconds=("calibration_seconds_shared", "sum"),
        mean_pipeline_seconds=("pipeline_seconds_shared", "mean"),
        total_pipeline_seconds=("pipeline_seconds_shared", "sum"),
    ).to_csv(args.output_dir / "dataset_run_timing.csv", index=False)
    populations = []
    for path in sorted(args.result_root.glob("*/scope_*/budget_*/population_rejection.csv")):
        table = pd.read_csv(path)
        table["pair_id"] = path.parents[2].name
        table["analysis_scope"] = path.parents[1].name.removeprefix("scope_")
        table["rejection_budget_cap"] = float(path.parent.name.removeprefix("budget_"))
        populations.append(table)
    if populations:
        pd.concat(populations, ignore_index=True).to_csv(args.output_dir / "all_population_rejection.csv", index=False)
    print(f"Aggregated {len(metrics)} runs from {metrics.patient_id.nunique()} patients")


if __name__ == "__main__":
    main()

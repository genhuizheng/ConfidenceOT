"""Aggregate pair outputs without treating multiple pairs from one patient as independent."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


BUDGET_COLUMNS = ["source_rejection_budget_cap", "target_rejection_budget_cap"]


def ensure_side_budgets(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for column in BUDGET_COLUMNS:
        if column not in metrics:
            metrics[column] = metrics["rejection_budget_cap"]
    return metrics


def aggregate_results(result_root: Path, output_dir: Path) -> None:
    """Aggregate completed pair runs while preserving method-specific results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(result_root.glob("*/scope_*/budget_*/pair_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No pair_metrics.csv under {result_root}")
    metrics = ensure_side_budgets(
        pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    )
    metrics.to_csv(output_dir / "all_pair_metrics.csv", index=False)
    numeric = [
        "source_raw_rejection_rate", "target_raw_rejection_rate",
        "source_final_rejection_rate", "target_final_rejection_rate",
        "source_budget_override_rate", "target_budget_override_rate",
        "rejection_cost", "transported_mass", "fit_seconds",
    ]
    patients = metrics.groupby(
        ["dataset_id", "patient_id", "analysis_scope", "method", *BUDGET_COLUMNS],
        as_index=False
    )[numeric].mean()
    patients["pair_n"] = metrics.groupby(
        ["dataset_id", "patient_id", "analysis_scope", "method", *BUDGET_COLUMNS]
    ).size().to_numpy()
    patients.to_csv(output_dir / "patient_level_metrics.csv", index=False)
    summary = patients.groupby(
        ["dataset_id", "analysis_scope", "method", *BUDGET_COLUMNS], as_index=False
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
    summary.to_csv(output_dir / "dataset_budget_summary_patient_weighted.csv", index=False)

    diagnostic_columns = {
        "calibration_valid_for_m4r",
        "inner_converged",
        "outer_converged",
        "cycle_detected",
    }
    if diagnostic_columns.issubset(metrics.columns):
        diagnostics = metrics.groupby(
            ["dataset_id", "analysis_scope", "method", *BUDGET_COLUMNS],
            as_index=False,
        ).agg(
            run_n=("pair_id", "size"),
            calibration_valid_n=("calibration_valid_for_m4r", "sum"),
            calibration_valid_rate=("calibration_valid_for_m4r", "mean"),
            inner_converged_n=("inner_converged", "sum"),
            inner_converged_rate=("inner_converged", "mean"),
            outer_converged_n=("outer_converged", "sum"),
            outer_converged_rate=("outer_converged", "mean"),
            cycle_detected_n=("cycle_detected", "sum"),
            cycle_detected_rate=("cycle_detected", "mean"),
        )
        diagnostics.to_csv(output_dir / "method_terminal_diagnostics.csv", index=False)
    run_keys = [
        "pair_id", "dataset_id", "patient_id", "analysis_scope", *BUDGET_COLUMNS,
    ]
    timing = metrics.drop_duplicates(run_keys)[
        run_keys + ["calibration_seconds_shared", "pipeline_seconds_shared"]
    ].copy()
    timing.to_csv(output_dir / "run_level_timing.csv", index=False)
    timing.groupby(
        ["dataset_id", "analysis_scope", *BUDGET_COLUMNS], as_index=False
    ).agg(
        run_n=("pair_id", "size"),
        mean_calibration_seconds=("calibration_seconds_shared", "mean"),
        total_calibration_seconds=("calibration_seconds_shared", "sum"),
        mean_pipeline_seconds=("pipeline_seconds_shared", "mean"),
        total_pipeline_seconds=("pipeline_seconds_shared", "sum"),
    ).to_csv(output_dir / "dataset_run_timing.csv", index=False)
    populations = []
    for path in sorted(result_root.glob("*/scope_*/budget_*/population_rejection.csv")):
        table = pd.read_csv(path)
        table["pair_id"] = path.parents[2].name
        table["analysis_scope"] = path.parents[1].name.removeprefix("scope_")
        pair_metric = ensure_side_budgets(pd.read_csv(path.parent / "pair_metrics.csv"))
        table["source_rejection_budget_cap"] = pair_metric.iloc[0][
            "source_rejection_budget_cap"
        ]
        table["target_rejection_budget_cap"] = pair_metric.iloc[0][
            "target_rejection_budget_cap"
        ]
        populations.append(table)
    if populations:
        pd.concat(populations, ignore_index=True).to_csv(output_dir / "all_population_rejection.csv", index=False)
    print(f"Aggregated {len(metrics)} runs from {metrics.patient_id.nunique()} patients")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    aggregate_results(args.result_root, args.output_dir)


if __name__ == "__main__":
    main()

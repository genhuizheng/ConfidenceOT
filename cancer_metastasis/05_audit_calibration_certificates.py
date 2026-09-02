"""Separate exact cost-selection evidence from reversible deployment validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def audit_calibrations(result_root: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for path in sorted(result_root.glob("*/scope_*/budget_*/calibration.json")):
        with path.open(encoding="utf-8") as handle:
            calibration = json.load(handle)
        metric_path = path.parent / "pair_metrics.csv"
        if not metric_path.is_file():
            continue
        metric = pd.read_csv(metric_path).iloc[0]
        source_budget = metric.get(
            "source_rejection_budget_cap", metric.get("rejection_budget_cap")
        )
        target_budget = metric.get(
            "target_rejection_budget_cap", metric.get("rejection_budget_cap")
        )
        warnings = [str(value) for value in calibration.get("warning_messages", [])]
        validation = calibration.get("validation", [])
        m4e_terminal_warning = any("M4-E" in value for value in warnings)
        no_feasible_warning = any("No jointly feasible" in value for value in warnings)
        nonmonotone_warning = any("nonmonotone" in value for value in warnings)
        m4r_terminal_valid = bool(validation) and all(
            bool(record.get("inner_converged", False))
            and bool(record.get("outer_converged", False))
            and not bool(record.get("cycle_detected", False))
            for record in validation
        )
        m4e_cost_selection_valid = bool(
            calibration.get("selection_status") == "largest_jointly_feasible"
            and calibration.get("source_monotone", False)
            and calibration.get("target_monotone", False)
            and not m4e_terminal_warning
            and not no_feasible_warning
            and not nonmonotone_warning
        )
        rows.append({
            "pair_id": metric["pair_id"],
            "dataset_id": metric["dataset_id"],
            "patient_id": metric["patient_id"],
            "source_sample": metric["source_sample"],
            "target_sample": metric["target_sample"],
            "analysis_scope": metric["analysis_scope"],
            "rejection_budget_cap": metric["rejection_budget_cap"],
            "source_rejection_budget_cap": source_budget,
            "target_rejection_budget_cap": target_budget,
            "rejection_cost": calibration.get("rejection_cost"),
            "selection_status": calibration.get("selection_status"),
            "source_monotone": calibration.get("source_monotone"),
            "target_monotone": calibration.get("target_monotone"),
            "m4e_cost_selection_valid": m4e_cost_selection_valid,
            "m4e_terminal_warning": m4e_terminal_warning,
            "m4r_rate_validation_valid": bool(
                calibration.get("validation_aggregate_valid", False)
            ),
            "m4r_terminal_validation_valid": m4r_terminal_valid,
            "m4r_deployment_valid": bool(calibration.get("calibration_valid", False)),
            "validation_source_raw_acceptance": calibration.get(
                "validation_source_raw_acceptance"
            ),
            "validation_target_raw_acceptance": calibration.get(
                "validation_target_raw_acceptance"
            ),
            "validation_run_n": len(validation),
            "validation_cycle_n": sum(
                bool(record.get("cycle_detected", False)) for record in validation
            ),
            "warning_messages": " | ".join(warnings),
        })
    if not rows:
        raise FileNotFoundError(f"No completed calibration outputs under {result_root}")
    certificates = pd.DataFrame(rows)
    certificates.to_csv(output_dir / "pair_calibration_certificates.csv", index=False)
    summary = certificates.groupby(
        [
            "dataset_id", "analysis_scope", "source_rejection_budget_cap",
            "target_rejection_budget_cap",
        ], as_index=False
    ).agg(
        pair_n=("pair_id", "size"),
        m4e_cost_selection_valid_n=("m4e_cost_selection_valid", "sum"),
        m4e_cost_selection_valid_rate=("m4e_cost_selection_valid", "mean"),
        m4r_rate_validation_valid_n=("m4r_rate_validation_valid", "sum"),
        m4r_rate_validation_valid_rate=("m4r_rate_validation_valid", "mean"),
        m4r_terminal_validation_valid_n=("m4r_terminal_validation_valid", "sum"),
        m4r_terminal_validation_valid_rate=("m4r_terminal_validation_valid", "mean"),
        m4r_deployment_valid_n=("m4r_deployment_valid", "sum"),
        m4r_deployment_valid_rate=("m4r_deployment_valid", "mean"),
    )
    summary.to_csv(output_dir / "dataset_calibration_certificate_summary.csv", index=False)
    print(f"Audited calibration certificates for {len(certificates)} pairs")
    return certificates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    audit_calibrations(args.result_root, args.output_dir)


if __name__ == "__main__":
    main()

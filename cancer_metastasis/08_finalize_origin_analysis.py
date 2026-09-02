"""Finalize malignant-cell origin analysis with explicit validity certificates."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
GROUP = [
    "dataset_id",
    "patient_id",
    "target_sample",
    "analysis_scope",
    "rejection_budget_cap",
]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AGGREGATE = load_script("cancer_aggregate", "04_aggregate_results.py")
AUDIT = load_script("cancer_calibration_audit", "05_audit_calibration_certificates.py")
RANK = load_script("cancer_origin_rank", "06_rank_primary_origins.py")


def completion_table(
    manifest_csv: Path,
    result_root: Path,
    *,
    analysis_scope: str,
    budget: float,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_csv)
    if "pair_id" not in manifest:
        raise ValueError("Manifest is missing pair_id")
    if manifest["pair_id"].duplicated().any():
        raise ValueError("Manifest contains duplicate pair_id values")
    rows: list[dict] = []
    budget_dir = f"budget_{budget:.2f}"
    for pair_id in manifest["pair_id"].astype(str):
        run = result_root / pair_id / f"scope_{analysis_scope}" / budget_dir
        metric_path = run / "pair_metrics.csv"
        methods: set[str] = set()
        metric_readable = False
        if metric_path.is_file():
            try:
                metric = pd.read_csv(metric_path)
                methods = set(metric["method"].astype(str))
                metric_readable = True
            except Exception:
                metric_readable = False
        required_files = [
            "SUCCESS",
            "run.json",
            "calibration.json",
            "pair_metrics.csv",
            "population_rejection.csv",
            "population_transitions.csv",
            "cell_confidence.csv",
        ]
        missing = [name for name in required_files if not (run / name).is_file()]
        rows.append({
            "pair_id": pair_id,
            "run_directory": str(run),
            "success_marker": (run / "SUCCESS").is_file(),
            "metric_readable": metric_readable,
            "m4e_present": "M4-E" in methods,
            "m4r_present": "M4-R" in methods,
            "missing_files": "|".join(missing),
            "complete": not missing and metric_readable and methods.issuperset({"M4-E", "M4-R"}),
        })
    return pd.DataFrame(rows)


def finalize_origin_analysis(
    manifest_csv: Path,
    result_root: Path,
    output_dir: Path,
    *,
    analysis_scope: str = "malignant",
    budget: float = 0.95,
    allow_incomplete: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    completion = completion_table(
        manifest_csv, result_root, analysis_scope=analysis_scope, budget=budget
    )
    completion.to_csv(output_dir / "pair_completion_audit.csv", index=False)
    incomplete = completion[~completion["complete"]]
    if not incomplete.empty and not allow_incomplete:
        raise RuntimeError(
            f"{len(incomplete)} of {len(completion)} expected pairs are incomplete; "
            f"see {output_dir / 'pair_completion_audit.csv'}"
        )

    aggregate_dir = output_dir / "aggregate"
    certificate_dir = output_dir / "certificates"
    ranking_dir = output_dir / "origin_ranking"
    AGGREGATE.aggregate_results(result_root, aggregate_dir)
    certificates = AUDIT.audit_calibrations(result_root, certificate_dir)
    ranking = RANK.rank_primary_origins(result_root, ranking_dir)

    certificate_columns = [
        "pair_id",
        "m4e_cost_selection_valid",
        "m4r_rate_validation_valid",
        "m4r_terminal_validation_valid",
        "m4r_deployment_valid",
        "warning_messages",
    ]
    ranking = ranking.merge(
        certificates[certificate_columns], on="pair_id", how="left", validate="many_to_one"
    )

    m4e = ranking[ranking["method"].eq("M4-E")].copy()
    if not m4e.empty:
        m4e["pair_inference_valid"] = m4e["m4e_cost_selection_valid"].fillna(False)
        m4e["group_all_candidates_certified"] = m4e.groupby(GROUP)[
            "pair_inference_valid"
        ].transform("all")
        m4e["biological_interpretation"] = (
            "malignant-cell expression-state compatibility; not lineage proof"
        )
        m4e.to_csv(output_dir / "m4e_origin_candidates.csv", index=False)
        m4e_winners = m4e[m4e["primary_rank"].eq(1)].copy()
        m4e_winners["inference_valid"] = m4e_winners[
            "group_all_candidates_certified"
        ]
        m4e_winners.to_csv(output_dir / "m4e_origin_group_winners.csv", index=False)
    else:
        m4e_winners = m4e
        m4e.to_csv(output_dir / "m4e_origin_candidates.csv", index=False)
        m4e.to_csv(output_dir / "m4e_origin_group_winners.csv", index=False)

    m4r = ranking[ranking["method"].eq("M4-R")].copy()
    if not m4r.empty:
        m4r["observed_terminal_valid"] = (
            m4r["inner_converged"].fillna(False)
            & m4r["outer_converged"].fillna(False)
            & ~m4r["cycle_detected"].fillna(True)
        )
        m4r["sensitivity_usable"] = (
            m4r["m4r_deployment_valid"].fillna(False)
            & m4r["m4r_terminal_validation_valid"].fillna(False)
            & m4r["observed_terminal_valid"]
        )
        m4r["interpretation"] = "diagnostic sensitivity analysis only"
        m4r.to_csv(output_dir / "m4r_sensitivity_diagnostics.csv", index=False)
    else:
        m4r.to_csv(output_dir / "m4r_sensitivity_diagnostics.csv", index=False)

    summary = {
        "expected_pair_n": int(len(completion)),
        "complete_pair_n": int(completion["complete"].sum()),
        "incomplete_pair_n": int((~completion["complete"]).sum()),
        "analysis_scope": analysis_scope,
        "rejection_budget_cap": budget,
        "multi_primary_group_n": int(m4e_winners[GROUP].drop_duplicates().shape[0]),
        "m4e_valid_origin_group_n": int(
            m4e_winners.get("inference_valid", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "m4r_usable_candidate_n": int(
            m4r.get("sensitivity_usable", pd.Series(dtype=bool)).fillna(False).sum()
        ),
        "primary_inference_method": "M4-E",
        "m4r_role": "diagnostic sensitivity only; excluded unless calibration and observed terminals are valid",
        "biological_claim_limit": "Ranks malignant expression-state compatibility; does not establish clonal lineage without orthogonal evidence such as shared CNV clones.",
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--analysis-scope", choices=("all", "malignant"), default="malignant")
    parser.add_argument("--rejection-budget", type=float, default=0.95)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    finalize_origin_analysis(
        args.manifest_csv,
        args.result_root,
        args.output_dir,
        analysis_scope=args.analysis_scope,
        budget=args.rejection_budget,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":
    main()

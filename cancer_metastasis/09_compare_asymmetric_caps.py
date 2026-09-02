"""Compare source-cap sensitivity while keeping the target cap fixed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = ["dataset_id", "patient_id", "target_sample", "analysis_scope"]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=/absolute/result/root")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--run must contain a non-empty label and path")
    return label, Path(path)


def load_metrics(label: str, root: Path) -> pd.DataFrame:
    files = sorted(root.glob("*/scope_*/budget_*/pair_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No pair metrics under {root}")
    tables = []
    for path in files:
        metric = pd.read_csv(path)
        population = pd.read_csv(path.parent / "population_rejection.csv")
        target = population[
            population["method"].eq("M4-E") & population["side"].eq("target")
        ]
        metric["target_mean_rejection_score"] = (
            np.average(target["mean_confidence_score"], weights=target["n"])
            if not target.empty else np.nan
        )
        tables.append(metric)
    table = pd.concat(tables, ignore_index=True)
    table = table[table["method"].eq("M4-E")].copy()
    for column in ("source_rejection_budget_cap", "target_rejection_budget_cap"):
        if column not in table:
            table[column] = table["rejection_budget_cap"]
    table["run_label"] = label
    table["result_root"] = str(root)
    return table


def weighted_mean(table: pd.DataFrame, value: str, weight: str) -> float:
    return float(np.average(table[value], weights=table[weight]))


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, table in metrics.groupby("run_label", sort=False):
        rows.append({
            "run_label": label,
            "pair_n": int(table["pair_id"].nunique()),
            "source_rejection_budget_cap": table["source_rejection_budget_cap"].iloc[0],
            "target_rejection_budget_cap": table["target_rejection_budget_cap"].iloc[0],
            "source_cell_occurrence_n": int(table["source_analyzed_n"].sum()),
            "target_cell_occurrence_n": int(table["target_analyzed_n"].sum()),
            "source_weighted_rejection_rate": weighted_mean(
                table, "source_final_rejection_rate", "source_analyzed_n"
            ),
            "target_weighted_rejection_rate": weighted_mean(
                table, "target_final_rejection_rate", "target_analyzed_n"
            ),
            "source_weighted_override_rate": weighted_mean(
                table, "source_budget_override_rate", "source_analyzed_n"
            ),
            "target_weighted_override_rate": weighted_mean(
                table, "target_budget_override_rate", "target_analyzed_n"
            ),
            "mean_rejection_cost": float(table["rejection_cost"].mean()),
            "outer_converged_rate": float(table["outer_converged"].mean()),
            "cycle_detected_rate": float(table["cycle_detected"].mean()),
        })
    return pd.DataFrame(rows)


def winner_stability(metrics: pd.DataFrame, baseline_label: str) -> pd.DataFrame:
    candidate_n = metrics.groupby(["run_label", *GROUP])["source_sample"].transform("nunique")
    ranked = metrics[candidate_n >= 2].copy()
    ranked = ranked.sort_values(
        ["run_label", *GROUP, "target_final_rejection_rate",
         "target_mean_rejection_score", "source_final_rejection_rate",
         "transported_mass", "source_sample"],
        ascending=[True] * (len(GROUP) + 1) + [True, True, True, False, True],
        kind="stable",
    )
    ranked["rank"] = ranked.groupby(["run_label", *GROUP]).cumcount() + 1
    top = ranked[ranked["rank"].eq(1)][
        ["run_label", *GROUP, "source_sample", "target_final_rejection_rate"]
    ]
    baseline = top[top["run_label"].eq(baseline_label)].drop(columns="run_label").rename(
        columns={
            "source_sample": "baseline_source_sample",
            "target_final_rejection_rate": "baseline_target_rejection_rate",
        }
    )
    compared = top.merge(baseline, on=GROUP, how="left")
    compared["winner_matches_baseline"] = (
        compared["source_sample"] == compared["baseline_source_sample"]
    )
    compared["target_rejection_delta_from_baseline"] = (
        compared["target_final_rejection_rate"]
        - compared["baseline_target_rejection_rate"]
    )
    return compared


def cell_gate_stability(
    runs: list[tuple[str, Path]], baseline_label: str
) -> pd.DataFrame:
    roots = dict(runs)
    baseline_files = {
        path.parents[2].name: path
        for path in roots[baseline_label].glob("*/scope_*/budget_*/cell_confidence.csv")
    }
    rows = []
    for label, root in runs:
        if label == baseline_label:
            continue
        comparison_files = {
            path.parents[2].name: path
            for path in root.glob("*/scope_*/budget_*/cell_confidence.csv")
        }
        if set(comparison_files) != set(baseline_files):
            missing = sorted(set(baseline_files) - set(comparison_files))
            extra = sorted(set(comparison_files) - set(baseline_files))
            raise RuntimeError(
                f"Pair mismatch for {label}: missing={missing[:5]} extra={extra[:5]}"
            )
        for pair_id, baseline_path in baseline_files.items():
            use = ["method", "side", "observation_id", "rejected", "normalized_rejection_score"]
            base = pd.read_csv(baseline_path, usecols=use)
            comp = pd.read_csv(comparison_files[pair_id], usecols=use)
            base = base[base["method"].eq("M4-E")].drop(columns="method").rename(columns={
                "rejected": "baseline_rejected",
                "normalized_rejection_score": "baseline_score",
            })
            comp = comp[comp["method"].eq("M4-E")].drop(columns="method").rename(columns={
                "rejected": "comparison_rejected",
                "normalized_rejection_score": "comparison_score",
            })
            joined = base.merge(
                comp, on=["side", "observation_id"], how="inner", validate="one_to_one"
            )
            for side, side_table in joined.groupby("side"):
                changed = side_table["baseline_rejected"] != side_table["comparison_rejected"]
                added_retained = (
                    side_table["baseline_rejected"] & ~side_table["comparison_rejected"]
                )
                lost_retained = (
                    ~side_table["baseline_rejected"] & side_table["comparison_rejected"]
                )
                rows.append({
                    "run_label": label,
                    "pair_id": pair_id,
                    "side": side,
                    "cell_n": len(side_table),
                    "gate_changed_fraction": float(changed.mean()),
                    "newly_retained_fraction": float(added_retained.mean()),
                    "newly_rejected_fraction": float(lost_retained.mean()),
                    "mean_score_delta": float(
                        (side_table["comparison_score"] - side_table["baseline_score"]).mean()
                    ),
                })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    args = parser.parse_args()
    if len(args.run) < 2:
        raise ValueError("Provide the baseline and at least one asymmetric-cap run")
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        raise ValueError("Run labels must be unique")
    baseline_label = labels[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.concat(
        [load_metrics(label, root) for label, root in args.run], ignore_index=True
    )
    pair_sets = metrics.groupby("run_label")["pair_id"].apply(set)
    if any(value != pair_sets.iloc[0] for value in pair_sets.iloc[1:]):
        raise RuntimeError("Every run must contain the same completed pair set")
    summary = summarize(metrics)
    winners = winner_stability(metrics, baseline_label)
    gates = cell_gate_stability(args.run, baseline_label)
    metrics.to_csv(args.output_dir / "asymmetric_cap_pair_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "asymmetric_cap_summary.csv", index=False)
    winners.to_csv(args.output_dir / "asymmetric_cap_winner_stability.csv", index=False)
    gates.to_csv(args.output_dir / "asymmetric_cap_cell_gate_stability.csv", index=False)
    report = {
        "baseline_label": baseline_label,
        "run_labels": labels,
        "pair_n": int(metrics["pair_id"].nunique()),
        "multi_primary_group_n": int(
            winners[winners["run_label"].eq(baseline_label)][GROUP].drop_duplicates().shape[0]
        ),
        "outputs": [
            "asymmetric_cap_pair_metrics.csv",
            "asymmetric_cap_summary.csv",
            "asymmetric_cap_winner_stability.csv",
            "asymmetric_cap_cell_gate_stability.csv",
        ],
    }
    (args.output_dir / "asymmetric_cap_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nWinner stability:")
    print(
        winners.groupby("run_label", as_index=False).agg(
            group_n=("patient_id", "size"),
            winner_match_rate=("winner_matches_baseline", "mean"),
            mean_target_rejection_delta=("target_rejection_delta_from_baseline", "mean"),
        ).to_string(index=False)
    )


if __name__ == "__main__":
    main()

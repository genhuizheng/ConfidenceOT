"""Rank candidate primary sites within patient and metastatic target."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = [
    "dataset_id", "patient_id", "target_sample", "analysis_scope",
    "rejection_budget_cap",
]


def _load_metrics(result_root: Path) -> pd.DataFrame:
    files = sorted(result_root.glob("*/scope_*/budget_*/pair_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No pair metrics under {result_root}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def _load_mean_rejection_scores(result_root: Path) -> pd.DataFrame:
    tables = []
    for path in sorted(result_root.glob("*/scope_*/budget_*/population_rejection.csv")):
        table = pd.read_csv(path)
        table["pair_id"] = path.parents[2].name
        tables.append(table)
    if not tables:
        return pd.DataFrame(columns=["pair_id", "method", "side", "mean_rejection_score"])
    population = pd.concat(tables, ignore_index=True)
    population["weighted_score"] = population["n"] * population["mean_confidence_score"]
    score = population.groupby(["pair_id", "method", "side"], as_index=False).agg(
        cell_n=("n", "sum"), weighted_score=("weighted_score", "sum")
    )
    score["mean_rejection_score"] = score["weighted_score"] / score["cell_n"]
    return score[["pair_id", "method", "side", "mean_rejection_score"]]


def rank_primary_origins(result_root: Path, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics(result_root)
    scores = _load_mean_rejection_scores(result_root)
    for side in ("source", "target"):
        side_score = scores[scores["side"].eq(side)].drop(columns="side").rename(
            columns={"mean_rejection_score": f"{side}_mean_rejection_score"}
        )
        metrics = metrics.merge(side_score, on=["pair_id", "method"], how="left")

    candidate_n = metrics.groupby(GROUP)["source_sample"].transform("nunique")
    metrics["candidate_primary_n"] = candidate_n
    comparable = metrics[candidate_n >= 2].copy()
    single = metrics[candidate_n < 2].copy()
    single["interpretation"] = "pairwise_compatibility_only_not_origin_selection"
    single.to_csv(output_dir / "single_primary_compatibility.csv", index=False)

    comparable = comparable.sort_values(
        GROUP + [
            "method", "target_final_rejection_rate", "target_mean_rejection_score",
            "source_final_rejection_rate", "transported_mass", "source_sample",
        ],
        ascending=[True] * (len(GROUP) + 1) + [True, True, True, False, True],
        kind="stable",
    )
    comparable["primary_rank"] = comparable.groupby(GROUP + ["method"]).cumcount() + 1
    comparable["ranking_basis"] = (
        "target rejection rate ascending; target rejection score ascending; "
        "source rejection rate ascending; transported mass descending"
    )
    comparable["interpretation"] = np.where(
        comparable["analysis_scope"].eq("malignant"),
        "candidate malignant-cell origin compatibility",
        "whole-sample compatibility including microenvironment",
    )
    comparable.to_csv(output_dir / "origin_candidate_ranking.csv", index=False)

    top = comparable[comparable["primary_rank"].eq(1)].copy()
    second = comparable[comparable["primary_rank"].eq(2)][
        GROUP + ["method", "target_final_rejection_rate", "target_mean_rejection_score"]
    ].rename(columns={
        "target_final_rejection_rate": "second_target_rejection_rate",
        "target_mean_rejection_score": "second_target_rejection_score",
    })
    top = top.merge(second, on=GROUP + ["method"], how="left")
    top["margin_to_second_target_rejection_rate"] = (
        top["second_target_rejection_rate"] - top["target_final_rejection_rate"]
    )
    top["margin_to_second_target_rejection_score"] = (
        top["second_target_rejection_score"] - top["target_mean_rejection_score"]
    )
    top.to_csv(output_dir / "origin_group_winners_by_method.csv", index=False)

    exact = top[top["method"].eq("M4-E")][GROUP + ["source_sample", "pair_id"]].rename(
        columns={"source_sample": "m4e_source_sample", "pair_id": "m4e_pair_id"}
    )
    reversible = top[top["method"].eq("M4-R")][
        GROUP + ["source_sample", "pair_id", "outer_converged", "cycle_detected"]
    ].rename(columns={
        "source_sample": "m4r_source_sample", "pair_id": "m4r_pair_id",
        "outer_converged": "m4r_outer_converged",
        "cycle_detected": "m4r_cycle_detected",
    })
    agreement = exact.merge(reversible, on=GROUP, how="left")
    agreement["m4e_m4r_top_source_agree"] = (
        agreement["m4e_source_sample"] == agreement["m4r_source_sample"]
    )
    agreement["m4r_usable_as_sensitivity"] = (
        agreement["m4r_outer_converged"].fillna(False)
        & ~agreement["m4r_cycle_detected"].fillna(True)
    )
    agreement.to_csv(output_dir / "origin_method_agreement.csv", index=False)
    print(
        f"Ranked {comparable[GROUP].drop_duplicates().shape[0]} multi-primary "
        f"patient-target groups; {single[GROUP].drop_duplicates().shape[0]} groups "
        "are compatibility-only"
    )
    return comparable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rank_primary_origins(args.result_root, args.output_dir)


if __name__ == "__main__":
    main()

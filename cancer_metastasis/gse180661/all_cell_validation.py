"""Validate all-cell OT using pair-resolved cell-type transition matrices."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def result_directory(root: Path, pair_id: str, budget_tag: str | None) -> Path:
    scope = root / pair_id / "scope_all"
    candidates = [scope / budget_tag] if budget_tag else sorted(
        path.parent for path in scope.glob("*/SUCCESS")
    )
    candidates = [path for path in candidates if (path / "SUCCESS").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one completed all-cell result for {pair_id}; found {len(candidates)}")
    return candidates[0]


def enrich_pair(table: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    table = table.copy()
    total = float(table["transported_mass"].sum())
    source = table.groupby("source_annotation")["transported_mass"].transform("sum")
    target = table.groupby("target_annotation")["transported_mass"].transform("sum")
    table["abundance_null_mass"] = source * target / max(total, np.finfo(float).eps)
    pseudocount = max(total, 1.0) * 1e-12
    table["log2_enrichment_over_abundance_null"] = np.log2(
        (table["transported_mass"] + pseudocount)
        / (table["abundance_null_mass"] + pseudocount)
    )
    diagonal = table["source_annotation"].eq(table["target_annotation"])
    observed = float(table.loc[diagonal, "transported_mass"].sum() / max(total, 1e-300))
    expected = float(table.loc[diagonal, "abundance_null_mass"].sum() / max(total, 1e-300))
    return table, {
        "transported_mass": total,
        "same_annotation_mass_fraction": observed,
        "abundance_null_same_annotation_fraction": expected,
        "same_annotation_enrichment_ratio": observed / max(expected, 1e-300),
    }


def heatmap(table: pd.DataFrame, output: Path, title: str):
    matrix = table.pivot(index="source_annotation", columns="target_annotation",
                         values="source_conditional_probability").fillna(0)
    figure, axis = plt.subplots(figsize=(max(7, 0.42 * matrix.shape[1] + 3),
                                         max(6, 0.36 * matrix.shape[0] + 2)))
    image = axis.imshow(matrix.to_numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=45, ha="right", fontsize=7)
    axis.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=7)
    axis.set_xlabel("Metastatic sample cell type")
    axis.set_ylabel("Primary sample cell type")
    axis.set_title(title)
    plt.colorbar(image, ax=axis, label="Source-conditional transported mass")
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("all_cell_ot_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dataset", default="GSE180661")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--budget-tag")
    parser.add_argument("--plot-each-pair", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pair_root = args.output_root / "pair_transition_matrices"
    pair_root.mkdir(exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    if "dataset_id" in manifest.columns:
        manifest = manifest[manifest["dataset_id"].eq(args.dataset)].copy()
    if manifest.empty:
        raise RuntimeError(f"No manifest rows found for {args.dataset}")
    tables = []
    summaries = []
    for _, row in manifest.iterrows():
        pair = str(row["pair_id"])
        directory = result_directory(args.all_cell_ot_root, pair, args.budget_tag)
        table = pd.read_csv(directory / "population_transitions.csv")
        table = table[table["method"].eq(args.method)].copy()
        if table.empty:
            raise RuntimeError(f"No {args.method} transition rows for {pair}")
        table, summary = enrich_pair(table)
        metadata = {
            "pair_id": pair, "patient_id": str(row["patient_id"]),
            "primary_sample": str(row["source_sample"]),
            "metastatic_sample": str(row["target_sample"]),
            "source_cell_n": int(row.get("source_n", 0)),
            "target_cell_n": int(row.get("target_n", 0)),
        }
        for key, value in metadata.items():
            table[key] = value
        summary.update(metadata)
        summaries.append(summary)
        tables.append(table)
        destination = pair_root / safe_name(pair)
        destination.mkdir(exist_ok=True)
        table.to_csv(destination / "cell_type_transition_matrix.csv", index=False)
        if args.plot_each_pair:
            heatmap(table, destination / "cell_type_transition_matrix",
                    f"{row['patient_id']}: {row['source_sample']} → {row['target_sample']}")
    transitions = pd.concat(tables, ignore_index=True)
    summary = pd.DataFrame(summaries)
    transitions.to_csv(args.output_root / "all_pair_cell_type_transitions.csv.gz",
                       index=False, compression="gzip")
    summary.to_csv(args.output_root / "all_pair_transition_validation.csv", index=False)

    patient = transitions.groupby(
        ["patient_id", "source_annotation", "target_annotation"], as_index=False
    )["source_conditional_probability"].mean()
    cohort = patient.groupby(
        ["source_annotation", "target_annotation"], as_index=False
    )["source_conditional_probability"].mean()
    cohort.to_csv(args.output_root / "patient_balanced_transition_matrix.csv", index=False)
    heatmap(cohort, args.output_root / "patient_balanced_transition_matrix",
            "Patient-balanced all-cell OT validation")
    report = {
        "dataset_id": args.dataset, "method": args.method, "pair_n": len(summary),
        "patient_n": int(summary["patient_id"].nunique()),
        "median_same_annotation_mass_fraction": float(
            summary["same_annotation_mass_fraction"].median()
        ),
        "median_abundance_null_same_annotation_fraction": float(
            summary["abundance_null_same_annotation_fraction"].median()
        ),
        "role": "cell-type correspondence validation only",
    }
    (args.output_root / "transition_validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Summarize annotation composition of pan-cancer all-cell M4-E rejection results."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_audit_module():
    path = Path(__file__).with_name("18_audit_pancancer_malignant_labels.py")
    spec = importlib.util.spec_from_file_location("pancancer_malignant_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


AUDIT = load_audit_module()
COMPOSITION_CLASSES = [
    "explicit_malignant",
    "explicit_non_malignant",
    "name_supported_malignant_candidate",
    "ambiguous_epithelial",
    "named_background",
    "other_annotation",
    "missing_or_unannotated",
]


def classify_cells(table: pd.DataFrame) -> pd.Series:
    annotation_class = table["annotation"].astype(str).map(AUDIT.classify_annotation)
    if "malignant" not in table:
        return annotation_class
    normalized = table["malignant"].astype(str).str.strip().str.lower()
    explicit = AUDIT.truthy_malignant(table["malignant"])
    explicit_negative = normalized.isin(
        {"0", "false", "no", "non-malignant", "nonmalignant", "benign"}
    ).to_numpy()
    return pd.Series(
        np.where(
            explicit,
            "explicit_malignant",
            np.where(explicit_negative, "explicit_non_malignant", annotation_class),
        ),
        index=table.index,
    )


def safe_fraction(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator > 0))


def summarize(manifest_csv: Path, result_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_csv)
    manifest_lookup = manifest.set_index("pair_id")
    annotation_records = []
    pair_records = []

    for path in sorted(result_root.glob("*/scope_all/budget_source_0.85_target_0.95/cell_confidence.csv")):
        pair_id = path.parents[2].name
        if pair_id not in manifest_lookup.index:
            raise KeyError(f"Pair absent from manifest: {pair_id}")
        pair = manifest_lookup.loc[pair_id]
        cells = pd.read_csv(path)
        cells = cells[cells["method"].eq("M4-E")].copy()
        if cells.empty:
            raise RuntimeError(f"No M4-E cells in {path}")
        cells["composition_class"] = classify_cells(cells)
        if cells["rejected"].dtype != bool:
            cells["rejected"] = cells["rejected"].astype(str).str.lower().eq("true")

        for (side, annotation, composition_class), group in cells.groupby(
            ["side", "annotation", "composition_class"], dropna=False, sort=True
        ):
            annotation_records.append({
                "dataset_id": pair["dataset_id"],
                "patient_id": pair["patient_id"],
                "pair_id": pair_id,
                "side": side,
                "annotation": annotation,
                "composition_class": composition_class,
                "cell_occurrence_n": len(group),
                "rejected_occurrence_n": int(group["rejected"].sum()),
                "retained_occurrence_n": int((~group["rejected"]).sum()),
                "mean_rejection_score": float(group["normalized_rejection_score"].mean()),
            })

        for side, group in cells.groupby("side", sort=True):
            counts = group["composition_class"].value_counts()
            rejected_counts = group.loc[group["rejected"], "composition_class"].value_counts()
            record = {
                "dataset_id": pair["dataset_id"],
                "patient_id": pair["patient_id"],
                "pair_id": pair_id,
                "side": side,
                "cell_occurrence_n": len(group),
                "rejected_occurrence_n": int(group["rejected"].sum()),
            }
            for category in COMPOSITION_CLASSES:
                record[f"{category}_n"] = int(counts.get(category, 0))
                record[f"rejected_{category}_n"] = int(rejected_counts.get(category, 0))
            pair_records.append(record)

    if not pair_records:
        raise RuntimeError(f"No completed all-cell results found under {result_root}")

    annotations = pd.DataFrame(annotation_records)
    annotations.to_csv(output_root / "pair_annotation_rejection_long.csv.gz", index=False)
    pairs = pd.DataFrame(pair_records)
    pairs.to_csv(output_root / "pair_rejection_composition.csv", index=False)

    count_columns = [
        column for column in pairs.columns
        if column.endswith("_n") and column not in {"dataset_id", "patient_id", "pair_id"}
    ]
    datasets = pairs.groupby(["dataset_id", "side"], as_index=False)[count_columns].sum()
    datasets["annotated_occurrence_n"] = (
        datasets["cell_occurrence_n"] - datasets["missing_or_unannotated_n"]
    )
    datasets["rejected_annotated_occurrence_n"] = (
        datasets["rejected_occurrence_n"]
        - datasets["rejected_missing_or_unannotated_n"]
    )
    datasets["malignant_candidate_n"] = (
        datasets["explicit_malignant_n"]
        + datasets["name_supported_malignant_candidate_n"]
    )
    datasets["rejected_malignant_candidate_n"] = (
        datasets["rejected_explicit_malignant_n"]
        + datasets["rejected_name_supported_malignant_candidate_n"]
    )
    datasets["annotation_coverage"] = safe_fraction(
        datasets["annotated_occurrence_n"], datasets["cell_occurrence_n"]
    )
    datasets["m4e_rejection_rate"] = safe_fraction(
        datasets["rejected_occurrence_n"], datasets["cell_occurrence_n"]
    )
    datasets["malignant_candidate_fraction_among_annotated"] = safe_fraction(
        datasets["malignant_candidate_n"], datasets["annotated_occurrence_n"]
    )
    datasets["malignant_candidate_fraction_among_rejected_annotated"] = safe_fraction(
        datasets["rejected_malignant_candidate_n"],
        datasets["rejected_annotated_occurrence_n"],
    )
    datasets["ambiguous_epithelial_fraction_among_rejected_annotated"] = safe_fraction(
        datasets["rejected_ambiguous_epithelial_n"],
        datasets["rejected_annotated_occurrence_n"],
    )
    datasets["background_fraction_among_rejected_annotated"] = safe_fraction(
        datasets["rejected_named_background_n"],
        datasets["rejected_annotated_occurrence_n"],
    )
    datasets["majority_rejected_are_name_supported_malignant"] = (
        datasets["malignant_candidate_fraction_among_rejected_annotated"] > 0.5
    )
    datasets.to_csv(output_root / "dataset_rejection_composition.csv", index=False)

    overall = pairs.groupby("side", as_index=False)[count_columns].sum()
    overall["annotated_occurrence_n"] = (
        overall["cell_occurrence_n"] - overall["missing_or_unannotated_n"]
    )
    overall["rejected_annotated_occurrence_n"] = (
        overall["rejected_occurrence_n"] - overall["rejected_missing_or_unannotated_n"]
    )
    overall["rejected_malignant_candidate_n"] = (
        overall["rejected_explicit_malignant_n"]
        + overall["rejected_name_supported_malignant_candidate_n"]
    )
    overall["annotation_coverage"] = safe_fraction(
        overall["annotated_occurrence_n"], overall["cell_occurrence_n"]
    )
    overall["m4e_rejection_rate"] = safe_fraction(
        overall["rejected_occurrence_n"], overall["cell_occurrence_n"]
    )
    overall["malignant_candidate_fraction_among_rejected_annotated"] = safe_fraction(
        overall["rejected_malignant_candidate_n"],
        overall["rejected_annotated_occurrence_n"],
    )
    overall.to_csv(output_root / "overall_rejection_composition.csv", index=False)

    report = {
        "method": "M4-E",
        "pair_n": int(pairs["pair_id"].nunique()),
        "dataset_n": int(pairs["dataset_id"].nunique()),
        "counting_unit": "pair-wise cell occurrence",
        "malignant_definition": "explicit malignant=True when available; otherwise annotation names containing tumor, tumour, cancer, malignant, carcinoma, or neoplastic",
        "important_limit": "Name-supported malignant status is an annotation audit aid. Unannotated and generic epithelial cells are not declared malignant.",
    }
    (output_root / "all_cell_rejection_composition_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(datasets[[
        "dataset_id", "side", "cell_occurrence_n", "annotation_coverage",
        "m4e_rejection_rate", "malignant_candidate_fraction_among_annotated",
        "malignant_candidate_fraction_among_rejected_annotated",
        "ambiguous_epithelial_fraction_among_rejected_annotated",
        "background_fraction_among_rejected_annotated",
        "majority_rejected_are_name_supported_malignant",
    ]].to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    summarize(args.manifest_csv, args.result_root, args.output_root)


if __name__ == "__main__":
    main()

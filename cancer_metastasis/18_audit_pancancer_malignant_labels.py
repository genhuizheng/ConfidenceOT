"""Audit malignant-cell evidence and pair eligibility across cancer datasets."""

from __future__ import annotations

import argparse
import gc
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ANNOTATION_COLUMNS = ("cell_type", "annotation", "celltype", "cell_type_final")
MALIGNANT_NAME_PATTERN = re.compile(
    r"(^|[^a-z])(tumou?r|cancer|malignant|carcinoma|neoplastic)([^a-z]|$)",
    flags=re.IGNORECASE,
)
EPITHELIAL_PATTERN = re.compile(r"epithel", flags=re.IGNORECASE)
AUTHOR_TUMOR_CLUSTER_PATTERN = re.compile(r"^Tu\d+(?:_|$)", flags=re.IGNORECASE)
AUTHOR_BACKGROUND_CLUSTER_PATTERN = re.compile(
    r"^(?:T|B|M|F|E)\d+(?:_|$)", flags=re.IGNORECASE
)
BACKGROUND_PATTERN = re.compile(
    r"(^|[^a-z])(t[ ._-]?cells?|b[ ._-]?cells?|cd4\+?[ ._-]*t|cd8\+?[ ._-]*t|"
    r"treg|gdt|nkt?|ilc\d*|myeloid|macrophages?|monocytes?|dendritic|dcs?|pdc|"
    r"mast|plasma|endothelial|fibroblasts?|fibrblast|stromal|neutrophils?|"
    r"lymphocytes?|cafs?|tams?|myofib|myocytes?|tecs?)([^a-z]|$)",
    flags=re.IGNORECASE,
)


def classify_annotation(label: str) -> str:
    value = str(label).strip()
    if not value or value.lower() in {"nan", "none", "unknown", "unannotated"}:
        return "missing_or_unannotated"
    if AUTHOR_BACKGROUND_CLUSTER_PATTERN.search(value) or BACKGROUND_PATTERN.search(value):
        return "named_background"
    if AUTHOR_TUMOR_CLUSTER_PATTERN.search(value) or MALIGNANT_NAME_PATTERN.search(value):
        return "name_supported_malignant_candidate"
    if EPITHELIAL_PATTERN.search(value):
        return "ambiguous_epithelial"
    return "other_annotation"


def truthy_malignant(values: pd.Series) -> np.ndarray:
    return values.astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "malignant", "tumor", "tumour", "cancer"}
    ).to_numpy()


def paths_for(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def audit_sample(paths: list[str], sample: str) -> dict:
    import anndata as ad

    annotation_counts: Counter = Counter()
    annotation_columns: set[str] = set()
    total_n = 0
    explicit_malignant_n = 0
    explicit_malignant_available = False
    provenance_sources: set[str] = set()

    for value in paths:
        data = ad.read_h5ad(value, backed="r")
        try:
            obs = data.obs
            if "sample_id" in obs:
                mask = obs["sample_id"].astype(str).eq(sample)
                obs = obs.loc[mask]
            elif sample and len(data.uns.get("pairing_metadata", {}).get("samples", [])) != 1:
                raise ValueError(f"Cannot isolate sample_id={sample!r} from {value}")
            total_n += len(obs)
            annotation_column = next((column for column in ANNOTATION_COLUMNS if column in obs), None)
            if annotation_column is not None:
                annotation_columns.add(annotation_column)
                annotation_counts.update(obs[annotation_column].astype(str).tolist())
            if "malignant" in obs:
                explicit_malignant_available = True
                explicit_malignant_n += int(truthy_malignant(obs["malignant"]).sum())
            provenance = data.uns.get("provenance", {})
            if hasattr(provenance, "get"):
                source = provenance.get("cell_annotation_source", "")
                if str(source).strip():
                    provenance_sources.add(str(source).strip())
        finally:
            data.file.close()

    heuristic_labels = sorted(
        label for label in annotation_counts
        if classify_annotation(label) == "name_supported_malignant_candidate"
    )
    heuristic_n = int(sum(annotation_counts[label] for label in heuristic_labels))
    if explicit_malignant_available:
        candidate_mode = "explicit_malignant_column"
        candidate_n = explicit_malignant_n
    elif annotation_counts:
        candidate_mode = "annotation_name_heuristic"
        candidate_n = heuristic_n
    else:
        candidate_mode = "no_cell_level_malignant_evidence"
        candidate_n = 0
    return {
        "total_n": total_n,
        "annotation_counts": dict(annotation_counts),
        "annotation_columns": sorted(annotation_columns),
        "explicit_malignant_available": explicit_malignant_available,
        "explicit_malignant_n": explicit_malignant_n,
        "heuristic_malignant_labels": heuristic_labels,
        "heuristic_malignant_n": heuristic_n,
        "candidate_mode": candidate_mode,
        "candidate_malignant_n": candidate_n,
        "cell_annotation_sources": sorted(provenance_sources),
    }


def audit_manifest(manifest_csv: Path, output_root: Path, minimum_cells: int) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_csv)
    if "eligible" in manifest:
        manifest = manifest[manifest["eligible"].astype(bool)].copy()
    cache: dict[tuple[tuple[str, ...], str], dict] = {}
    vocabulary_rows = []
    pair_rows = []

    def cached_sample(row: pd.Series, side: str) -> dict:
        paths = paths_for(row, side)
        sample = str(row[f"{side}_sample"])
        key = (tuple(paths), sample)
        if key not in cache:
            cache[key] = audit_sample(paths, sample)
            record = cache[key]
            for label, count in record["annotation_counts"].items():
                vocabulary_rows.append({
                    "dataset_id": str(row["dataset_id"]),
                    "patient_id": str(row["patient_id"]),
                    "sample": sample,
                    "side": side,
                    "annotation_columns": "|".join(record["annotation_columns"]),
                    "annotation": label,
                    "cell_n": count,
                    "heuristic_class": classify_annotation(label),
                    "cell_annotation_sources": "|".join(record["cell_annotation_sources"]),
                })
        return cache[key]

    for _, row in manifest.iterrows():
        source = cached_sample(row, "source")
        target = cached_sample(row, "target")
        pair_rows.append({
            **row.to_dict(),
            "source_candidate_mode": source["candidate_mode"],
            "target_candidate_mode": target["candidate_mode"],
            "source_candidate_malignant_n": source["candidate_malignant_n"],
            "target_candidate_malignant_n": target["candidate_malignant_n"],
            "source_heuristic_malignant_annotations": "|".join(source["heuristic_malignant_labels"]),
            "target_heuristic_malignant_annotations": "|".join(target["heuristic_malignant_labels"]),
            "candidate_evaluable": (
                source["candidate_malignant_n"] >= minimum_cells
                and target["candidate_malignant_n"] >= minimum_cells
            ),
        })
        gc.collect()

    vocabulary_columns = [
        "dataset_id", "patient_id", "sample", "side", "annotation_columns",
        "annotation", "cell_n", "heuristic_class", "cell_annotation_sources",
    ]
    vocabulary = pd.DataFrame(vocabulary_rows, columns=vocabulary_columns).drop_duplicates()
    if len(vocabulary):
        vocabulary = vocabulary.sort_values(
            ["dataset_id", "heuristic_class", "annotation", "patient_id", "sample"]
        )
    vocabulary.to_csv(output_root / "annotation_vocabulary_by_sample.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(output_root / "pair_candidate_malignant_audit.csv", index=False)

    dataset_rows = []
    for dataset, group in pairs.groupby("dataset_id", sort=True):
        vocab = vocabulary[vocabulary["dataset_id"].astype(str).eq(str(dataset))]
        candidate_labels = sorted(
            vocab.loc[
                vocab["heuristic_class"].eq("name_supported_malignant_candidate"),
                "annotation",
            ].unique()
        )
        ambiguous_labels = sorted(
            vocab.loc[vocab["heuristic_class"].eq("ambiguous_epithelial"), "annotation"].unique()
        )
        mode_values = set(group["source_candidate_mode"]) | set(group["target_candidate_mode"])
        if "explicit_malignant_column" in mode_values:
            review_status = "explicit_malignant_column_available"
        elif candidate_labels:
            review_status = "review_name_supported_candidates"
        elif ambiguous_labels:
            review_status = "epithelial_only_requires_malignancy_validation"
        else:
            review_status = "requires_cnv_or_external_malignant_annotation"
        dataset_rows.append({
            "dataset_id": dataset,
            "pair_n": len(group),
            "patient_n": int(group["patient_id"].nunique()),
            "candidate_evaluable_pair_n": int(group["candidate_evaluable"].sum()),
            "name_supported_candidate_annotations": "|".join(candidate_labels),
            "ambiguous_epithelial_annotations": "|".join(ambiguous_labels),
            "review_status": review_status,
        })
    datasets = pd.DataFrame(dataset_rows)
    datasets.to_csv(output_root / "dataset_malignant_readiness.csv", index=False)
    report = {
        "manifest": str(manifest_csv),
        "minimum_candidate_cells_per_side": minimum_cells,
        "dataset_n": len(datasets),
        "pair_n": len(pairs),
        "patient_n": int(pairs["patient_id"].nunique()),
        "candidate_evaluable_pair_n": int(pairs["candidate_evaluable"].sum()),
        "review_status_counts": {
            str(key): int(value)
            for key, value in datasets["review_status"].value_counts().items()
        },
        "important_limit": "Name-based candidates are an audit aid, not malignant-cell calls. Epithelial labels are never automatically declared malignant.",
    }
    (output_root / "pancancer_malignant_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(datasets.to_string(index=False), flush=True)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-cells-per-side", type=int, default=20)
    args = parser.parse_args()
    audit_manifest(args.manifest_csv, args.output_root, args.minimum_cells_per_side)


if __name__ == "__main__":
    main()

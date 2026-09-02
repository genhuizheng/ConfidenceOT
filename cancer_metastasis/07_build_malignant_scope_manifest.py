"""Build an exact-cell-count manifest for a declared malignant annotation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_exact_side


def observation_labels(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    return np.repeat("unannotated", data.n_obs)


def paths_for(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def build_scope_manifest(
    manifest_csv: Path,
    output_dir: Path,
    *,
    dataset_id: str,
    annotations: list[str],
    minimum_cells: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(manifest_csv)
    selected = manifest[manifest["dataset_id"].astype(str).eq(dataset_id)].copy()
    if selected.empty:
        raise ValueError(f"dataset_id={dataset_id!r} has no eligible pairs")
    cache: dict[tuple[tuple[str, ...], str], tuple[int, str]] = {}

    def count_side(row: pd.Series, side: str) -> tuple[int, str]:
        paths = paths_for(row, side)
        sample = str(row[f"{side}_sample"])
        key = (tuple(paths), sample)
        if key not in cache:
            data = load_exact_side(paths, sample)
            values = observation_labels(data)
            count = int(np.isin(values, annotations).sum())
            available = "|".join(sorted(np.unique(values)))
            cache[key] = (count, available)
            del data
            gc.collect()
        return cache[key]

    source_counts = []
    target_counts = []
    source_labels = []
    target_labels = []
    for _, row in selected.iterrows():
        source_n, source_available = count_side(row, "source")
        target_n, target_available = count_side(row, "target")
        source_counts.append(source_n)
        target_counts.append(target_n)
        source_labels.append(source_available)
        target_labels.append(target_available)
    selected["malignant_annotations"] = "|".join(annotations)
    selected["source_malignant_n"] = source_counts
    selected["target_malignant_n"] = target_counts
    selected["source_available_annotations"] = source_labels
    selected["target_available_annotations"] = target_labels
    selected["malignant_evaluable"] = (
        selected["source_malignant_n"].ge(minimum_cells)
        & selected["target_malignant_n"].ge(minimum_cells)
    )
    selected["malignant_skip_reason"] = np.where(
        selected["malignant_evaluable"],
        "",
        "source_or_target_malignant_cells_below_minimum",
    )
    selected.to_csv(output_dir / "pair_manifest_malignant_audit.csv", index=False)
    eligible = selected[selected["malignant_evaluable"]].copy().reset_index(drop=True)
    eligible["scope_index"] = np.arange(len(eligible), dtype=int)
    eligible.to_csv(output_dir / "pair_manifest_malignant_eligible.csv", index=False)
    summary = {
        "dataset_id": dataset_id,
        "annotations": annotations,
        "minimum_cells_per_side": minimum_cells,
        "audited_pair_n": len(selected),
        "eligible_pair_n": len(eligible),
        "eligible_patient_n": int(eligible["patient_id"].nunique()),
        "eligible_source_sample_n": int(eligible["source_sample"].nunique()),
        "eligible_target_sample_n": int(eligible["target_sample"].nunique()),
    }
    (output_dir / "malignant_manifest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--include-annotation", action="append", required=True)
    parser.add_argument("--minimum-cells-per-side", type=int, default=20)
    args = parser.parse_args()
    build_scope_manifest(
        args.manifest_csv,
        args.output_dir,
        dataset_id=args.dataset_id,
        annotations=args.include_annotation,
        minimum_cells=args.minimum_cells_per_side,
    )


if __name__ == "__main__":
    main()

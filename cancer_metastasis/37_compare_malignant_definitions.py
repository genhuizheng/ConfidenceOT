"""What changes if the malignant scope moves from cell_type to the uniform call.

ConfidenceOT currently selects the malignant compartment with a per-dataset
list of the deposit's own `cell_type` strings -- `Ovarian.cancer.cell` for one
deposit, three labels for another, eleven for a third. The collection now
carries `malignant`, one inferCNV rule applied to every deposit, which is the
comparison the collection exists to make possible and which this project has
been approximating with four different rules.

Switching means re-running the optimal transport, so this measures what the
switch would buy before anything is re-run. Per sample it reports the cells
each definition selects, the overlap, and -- separately, because the handoff
calls it the trap -- how many cells are `undetermined`: assessed by the method
and declined, neither malignant nor not. There are 233,314 of them across the
collection and folding them either way moves a quarter of a million cells.

It reads only, and opens each file backed.

Usage:

    python cancer_metastasis/37_compare_malignant_definitions.py MANIFEST_CSV OUT.csv \
        --annotation Ovarian.cancer.cell
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--annotation", action="append", default=None,
                        dest="annotations", metavar="LABEL",
                        help="A cell_type value the current runs include; repeatable")
    parser.add_argument("--malignant-column", default="malignant")
    parser.add_argument("--malignant-value", default="malignant")
    parser.add_argument("--undetermined-value", default="undetermined")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after this many rows, for a quick look")
    return parser.parse_args()


def paths_for(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row.index and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def main() -> None:
    args = parse_args()
    import anndata as ad

    manifest = pd.read_csv(args.manifest_csv)
    annotations = set(args.annotations or ())
    seen: set[tuple[str, str]] = set()
    rows = []
    for _, row in manifest.iterrows():
        if args.limit and len(rows) >= args.limit:
            break
        for side in ("source", "target"):
            sample = str(row[f"{side}_sample"])
            for path in paths_for(row, side):
                if (path, sample) in seen:
                    continue
                seen.add((path, sample))
                data = ad.read_h5ad(path, backed="r")
                try:
                    obs = data.obs
                    keep = obs["sample_id"].astype(str) == sample if "sample_id" in obs else slice(None)
                    block = obs[keep] if "sample_id" in obs else obs
                    record = {
                        "dataset_id": str(row.get("dataset_id", "")),
                        "patient_id": str(row.get("patient_id", "")),
                        "pair_id": str(row.get("pair_id", "")),
                        "side": side, "sample": sample,
                        "cells": int(len(block)),
                        "has_cell_type": "cell_type" in block.columns,
                        "has_malignant_column": args.malignant_column in block.columns,
                    }
                    by_annotation = None
                    if record["has_cell_type"] and annotations:
                        by_annotation = block["cell_type"].astype(str).isin(annotations)
                        record["by_annotation"] = int(by_annotation.sum())
                    by_call = None
                    if record["has_malignant_column"]:
                        column = block[args.malignant_column].astype(str)
                        by_call = column == args.malignant_value
                        record["by_call"] = int(by_call.sum())
                        record["undetermined"] = int(
                            (column == args.undetermined_value).sum())
                    if by_annotation is not None and by_call is not None:
                        record["both"] = int((by_annotation & by_call).sum())
                        record["annotation_only"] = int(
                            (by_annotation & ~by_call).sum())
                        record["call_only"] = int((~by_annotation & by_call).sum())
                    rows.append(record)
                finally:
                    data.file.close()

    table = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)

    numeric = [c for c in ("cells", "by_annotation", "by_call", "both",
                           "annotation_only", "call_only", "undetermined")
               if c in table.columns]
    print(table.groupby("dataset_id")[numeric].sum().to_string())
    print()
    if {"both", "annotation_only", "call_only"} <= set(table.columns):
        for dataset, block in table.groupby("dataset_id"):
            both = block["both"].sum()
            union = both + block["annotation_only"].sum() + block["call_only"].sum()
            print(f"{dataset}: jaccard {both / union:.4f} "
                  f"| annotation-only {block['annotation_only'].sum()} "
                  f"| call-only {block['call_only'].sum()} "
                  f"| undetermined {block['undetermined'].sum()}")
    samples_without = table[~table["has_malignant_column"]]
    if len(samples_without):
        print()
        print(f"{len(samples_without)} samples lack {args.malignant_column!r}: "
              f"{samples_without['sample'].head(5).tolist()}")


if __name__ == "__main__":
    main()

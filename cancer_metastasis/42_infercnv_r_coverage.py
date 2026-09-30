"""How much of a pair manifest can the R inferCNV caller run on at all?

PROJECT.md and ``infercnv_r/README.md`` both settle on the same plan: run the
OT twice, once per CNV caller, and compare the retained and rejected subsets.
The block on disk is the infercnvpy half. Before spending a night of compute
on the R half, the question is how many pairs it can produce a call for.

``export_for_infercnv.py`` refuses a file on two gates, and both are
properties of the annotation rather than of the run:

* fewer than ``MIN_REFERENCE`` reference cells, because a baseline built on
  that few is not trustworthy;
* no query cell at all -- every cell a reference lineage -- because inferCNV
  then has no observation group and dies inside
  ``seq_len(max(obs_annotations_groups))``, an error that names no file.

A third condition is not a refusal but empties the comparison: a file with no
Epithelial cell runs, and nothing it calls can be malignant under this
collection's origin-lineage rule. Counted separately for that reason.

The lineage lists are copied from ``export_for_infercnv.py`` rather than
imported, which is the choice that file already made and states in a comment:
the inferCNV package is meant to be deletable without touching the pipeline.
If the two ever disagree, this script measures the disagreement rather than
the data, so they are printed at the top of every run.

Read from ``obs`` alone, in backed mode, for the reason
``39_count_malignant_cells.py`` gives: the expression matrix decides nothing
about how many cells carry a label, and loading it for both sides of every
pair would cost hours to learn nothing.

    python cancer_metastasis/42_infercnv_r_coverage.py MANIFEST.csv OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Must match infer_malignant.py and infercnv_r/export_for_infercnv.py.
REFERENCE_LINEAGES = ["T_cell", "B_cell", "NK_cell", "Myeloid",
                      "Dendritic_cell", "Mast_cell", "Endothelial",
                      "Fibroblast"]
NEVER_REFERENCE = ["Plasma_cell"]
MIN_REFERENCE = 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--lineage-column", default="cell_lineage",
        help="The obs column export_for_infercnv.py reads")
    parser.add_argument(
        "--minimum-reference", type=int, default=MIN_REFERENCE,
        help="The exporter's --min-reference, so refused here means refused "
             "there")
    return parser.parse_args()


def side_paths(row: pd.Series, side: str) -> list[str]:
    """The same resolution 02_run_pair.py makes, json list first."""
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def side_counts(paths: list[str], sample: str, column: str) -> dict:
    """Lineage composition of one side, read from obs without touching X."""
    import anndata as ad

    total = 0
    reference = 0
    epithelial = 0
    missing_column = False
    reference_set = set(REFERENCE_LINEAGES) - set(NEVER_REFERENCE)
    for path in paths:
        backed = ad.read_h5ad(path, backed="r")
        try:
            obs = backed.obs
        finally:
            backed.file.close()
        if "sample_id" in obs:
            obs = obs[obs["sample_id"].astype(str).to_numpy() == sample]
        if obs.shape[0] == 0:
            continue
        total += int(obs.shape[0])
        if column not in obs:
            # A file that predates apply_cell_type_map.py. Recorded rather
            # than raised: one such deposit must not stop the count for the
            # rest.
            missing_column = True
            continue
        values = obs[column].astype(str)
        reference += int(values.isin(reference_set).sum())
        epithelial += int((values == "Epithelial").sum())
    return {"total_n": total, "reference_n": reference,
            "query_n": total - reference, "epithelial_n": epithelial,
            "lineage_column_missing": missing_column}


def verdict(counts: dict, minimum_reference: int) -> dict:
    """The exporter's two gates, plus the condition that empties the result."""
    enough = counts["reference_n"] >= minimum_reference
    has_query = counts["query_n"] > 0
    return {"enough_reference": bool(enough),
            "has_query": bool(has_query),
            "has_epithelial": bool(counts["epithelial_n"] > 0),
            "runnable": bool(enough and has_query),
            "informative": bool(enough and has_query
                                and counts["epithelial_n"] > 0)}


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(f"reference lineages   {', '.join(REFERENCE_LINEAGES)}")
    print(f"never reference      {', '.join(NEVER_REFERENCE)}")
    print(f"minimum reference    {args.minimum_reference}")
    print(f"pairs                {len(manifest)}")

    records = []
    for position, (_, row) in enumerate(manifest.iterrows()):
        record = {"index": position, "pair_id": row.get("pair_id")}
        for key in ("dataset_id", "patient_id", "source_sample",
                    "target_sample"):
            if key in manifest.columns:
                record[key] = row.get(key)
        for side in ("source", "target"):
            try:
                counts = side_counts(side_paths(row, side),
                                     str(row[f"{side}_sample"]),
                                     args.lineage_column)
            except Exception as error:  # noqa: BLE001 - reported, not raised
                # One unreadable file must not cost the other pairs.
                counts = {"total_n": -1, "reference_n": -1, "query_n": -1,
                          "epithelial_n": -1, "lineage_column_missing": False}
                record[f"{side}_error"] = f"{type(error).__name__}: {error}"
            for key, value in counts.items():
                record[f"{side}_{key}"] = value
            for key, value in verdict(counts, args.minimum_reference).items():
                record[f"{side}_{key}"] = value
        # A pair needs both sides. One runnable side produces no comparison.
        record["pair_runnable"] = bool(record["source_runnable"]
                                       and record["target_runnable"])
        record["pair_informative"] = bool(record["source_informative"]
                                          and record["target_informative"])
        records.append(record)
        if (position + 1) % 20 == 0:
            print(f"  {position + 1} of {len(manifest)}")

    table = pd.DataFrame.from_records(records)
    destination = args.output_root / "infercnv_r_coverage.csv"
    table.to_csv(destination, index=False)

    runnable = int(table["pair_runnable"].sum())
    informative = int(table["pair_informative"].sum())
    print()
    print(f"pairs both sides runnable     {runnable} of {len(table)}")
    print(f"pairs both sides informative  {informative} of {len(table)}")
    if "dataset_id" in table.columns:
        print()
        print("by deposit:")
        grouped = table.groupby("dataset_id").agg(
            pairs=("pair_id", "size"),
            runnable=("pair_runnable", "sum"),
            informative=("pair_informative", "sum"))
        print(grouped.to_string())
    for side in ("source", "target"):
        refused_reference = int((~table[f"{side}_enough_reference"]).sum())
        refused_query = int((~table[f"{side}_has_query"]).sum())
        no_epithelium = int((~table[f"{side}_has_epithelial"]).sum())
        print()
        print(f"{side}: {refused_reference} below the reference minimum, "
              f"{refused_query} with no query cell, "
              f"{no_epithelium} with no epithelium")
    print()
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()

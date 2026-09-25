"""Which samples inferCNV abstained on, ranked by what re-annotating recovers.

A sample whose diploid control collapsed has every origin-lineage cell written
``undetermined``: no CNV cluster separated from the reference, so no threshold
could be set. That is a calibration failure, not a finding that the cells are
not tumour. The evidence is that the abstaining files are short of neither
epithelium nor reference cells -- in this collection the files with zero
malignant calls carry a median ``diploid_control_pct`` of 0.0 against 0.2 to
0.8 for the files that produced calls, while ``thin_reference``,
``no_epithelium`` and ``diploid_fail`` are almost all False and the epithelium
counts run to thousands.

A strict ``malignant == "malignant"`` filter therefore drops whole pairs for a
technical reason. This says which files to look at first, measured by what
recovering each would give back. A pair needs both sides, so a file whose
partner also abstained recovers nothing on its own; the two are counted
together rather than ranked as independent candidates, which would count the
same pair twice.

Re-run after re-annotating to see what came back.

Usage:

    python cancer_metastasis/38_abstained_samples_report.py \\
        SUMMARY_BY_FILE_TSV PANCANCER_MANIFEST_CSV OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summary_tsv", type=Path,
                        help="annotated/summary_by_file.tsv")
    parser.add_argument("manifest_csv", type=Path,
                        help="The pan-cancer pair manifest")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-cells", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_tsv, sep="\t")
    manifest = pd.read_csv(args.manifest_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = summary[summary["site_class"].isin(("primary", "metastasis"))].copy()
    for column in ("malignant", "undetermined", "epithelium", "reference_cells"):
        if column in summary.columns:
            summary[column] = summary[column].fillna(0)
    summary["abstained"] = (
        summary["malignant"].lt(args.minimum_cells)
        & summary["undetermined"].ge(args.minimum_cells))

    flags = [c for c in ("no_epithelium", "thin_reference", "diploid_fail",
                         "excluded_deposit") if c in summary.columns]
    show = ["dataset_id", "site_class", "cells", "epithelium",
            "reference_cells", "diploid_control_pct", "malignant",
            "undetermined"]
    keep = [c for c in show + flags + ["file"] if c in summary.columns]
    abstained = summary.loc[summary["abstained"], keep].copy()
    abstained = abstained.sort_values("epithelium", ascending=False)
    abstained.to_csv(args.output_dir / "abstained_files.csv", index=False)

    # Pair status. The manifest names samples; the summary is per file, so the
    # join is on whichever column the summary carries the sample id in.
    key = "sample_id" if "sample_id" in summary.columns else "file"
    lookup = {str(value): row for value, row in
              zip(summary[key], summary.to_dict("records"))}

    rows = []
    for _, pair in manifest.iterrows():
        record = {"dataset_id": pair.get("dataset_id"),
                  "patient_id": pair.get("patient_id"),
                  "pair_id": pair.get("pair_id")}
        blocked, abstained_sides = [], []
        for side in ("source", "target"):
            sample = str(pair.get(f"{side}_sample"))
            row = lookup.get(sample)
            if row is None:
                record[f"{side}_state"] = "absent from summary"
                blocked.append(side)
                continue
            record[f"{side}_malignant"] = int(row.get("malignant", 0))
            record[f"{side}_undetermined"] = int(row.get("undetermined", 0))
            record[f"{side}_epithelium"] = int(row.get("epithelium", 0))
            if row.get("malignant", 0) < args.minimum_cells:
                blocked.append(side)
                if row.get("abstained"):
                    abstained_sides.append(side)
                    record[f"{side}_state"] = "abstained"
                else:
                    record[f"{side}_state"] = "no malignant, not abstained"
            else:
                record[f"{side}_state"] = "usable"
        record["blocked_sides"] = "+".join(blocked)
        record["usable"] = not blocked
        # Recoverable only when every blocked side is an abstention: a side
        # with no epithelium at all is not going to come back from a re-run.
        record["recoverable"] = bool(blocked) and blocked == abstained_sides
        rows.append(record)
    pairs = pd.DataFrame(rows)
    pairs.to_csv(args.output_dir / "pair_status.csv", index=False)

    recovery = []
    for dataset, block in pairs.groupby("dataset_id"):
        usable = block[block["usable"]]
        recoverable = block[block["recoverable"]]
        after = pd.concat([usable, recoverable])
        recovery.append({
            "dataset_id": dataset,
            "pairs": int(len(block)),
            "usable_pairs_now": int(len(usable)),
            "patients_now": int(usable["patient_id"].nunique()),
            "recoverable_pairs": int(len(recoverable)),
            "patients_after": int(after["patient_id"].nunique()),
            "patients_gained": int(after["patient_id"].nunique()
                                   - usable["patient_id"].nunique()),
            "sides_to_reannotate": int(sum(
                len([s for s in str(value).split("+") if s])
                for value in recoverable["blocked_sides"])),
        })
    table = pd.DataFrame(recovery).sort_values(
        ["patients_gained", "patients_after"], ascending=False)
    table.to_csv(args.output_dir / "recovery_by_dataset.csv", index=False)

    print(table.to_string(index=False))
    print()
    print("Abstained files, largest epithelium first:")
    columns = [c for c in ("dataset_id", "site_class", "file", "epithelium",
                           "reference_cells", "diploid_control_pct",
                           "undetermined") if c in abstained.columns]
    print(abstained[columns].head(25).to_string(index=False))

    (args.output_dir / "diagnostics.json").write_text(json.dumps({
        "abstained_file_n": int(len(abstained)),
        "abstained_epithelium_total": int(abstained["epithelium"].sum()),
        "patients_now": int(table["patients_now"].sum()),
        "patients_after_recovery": int(table["patients_after"].sum()),
        "criterion": (
            "a file counts as abstained when it has fewer than "
            f"{args.minimum_cells} malignant calls and at least that many "
            "undetermined; diploid_control_pct is the column that separates "
            "these from the files that produced calls"),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Keep the pairs that can actually run, and record what was dropped.

``02_run_pair.py`` stops on a pair whose scoped compartment is smaller than
``--minimum-scope-cells``, which on the pan-cancer manifest is most of it: 93
of 252 pairs survive the uniform inferCNV filter. Running the factorial on the
untrimmed list attempts each blocked pair once per arm -- twelve times -- and
every worker exits non-zero for reasons that were knowable before submission.

So trim first. The pairs are the same in every arm, because the scoping does
not depend on the preprocessing, so this neither costs a comparison nor makes
one: it removes attempts that were going to fail identically twelve times.

Dropping is recorded, not silent. ``trim_report.json`` carries the count by
reason and by deposit, and the dropped rows are written out in full beside the
kept ones. Three deposits cannot produce a malignant call at all and are named
in the handoff -- GSE184362 withdrawn with its whole epithelium undetermined,
GSE164522 CD45-sorted so there are no tumour cells to call, and the GEO
conversion of GSE271675 never annotated because its paper used NUMBAT and that
needs BAMs behind EGA access. Those are a Methods sentence, not a surprise,
and the prostate data runs from Faming Zhao's annotated object instead.

    python cancer_metastasis/40_trim_manifest_to_evaluable.py \\
        MANIFEST.csv malignant_cell_counts.csv TRIMMED.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument(
        "counts_csv", type=Path,
        help="malignant_cell_counts.csv from 39_count_malignant_cells.py, "
             "counted under the same compartment call this run will use")
    parser.add_argument("out_csv", type=Path)
    parser.add_argument("--minimum-cells-per-side", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    counts = pd.read_csv(args.counts_csv)

    if "pair_id" not in manifest or "pair_id" not in counts:
        raise SystemExit("both files must carry pair_id")
    overlap = set(manifest["pair_id"].astype(str)) & set(
        counts["pair_id"].astype(str))
    if not overlap:
        raise SystemExit(
            "no pair_id is in both files; the counts were taken from a "
            "different manifest than the one being trimmed")

    minimum = args.minimum_cells_per_side
    keep = counts.loc[
        (counts["source_malignant_n"] >= minimum)
        & (counts["target_malignant_n"] >= minimum), "pair_id"].astype(str)
    kept = manifest[manifest["pair_id"].astype(str).isin(set(keep))].copy()
    dropped = manifest[~manifest["pair_id"].astype(str).isin(set(keep))].copy()
    if kept.empty:
        raise SystemExit(
            f"no pair has at least {minimum} cells on both sides; check that "
            f"the counts were taken under the compartment call this run uses")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Reset, because --index is positional: the trimmed file is what every
    # later stage reads, and a gap in the numbering would be a silent skip.
    kept = kept.reset_index(drop=True)
    kept.to_csv(args.out_csv, index=False)
    dropped_path = args.out_csv.with_name(args.out_csv.stem + "_dropped.csv")
    dropped.to_csv(dropped_path, index=False)

    reason = counts.set_index(counts["pair_id"].astype(str))
    why = {}
    for pair_id in dropped["pair_id"].astype(str):
        if pair_id not in reason.index:
            why[pair_id] = "not counted"
            continue
        row = reason.loc[pair_id]
        if bool(row.get("source_malignant_column_missing", False)) or \
                bool(row.get("target_malignant_column_missing", False)):
            why[pair_id] = "no compartment call in this file"
        elif int(row["source_malignant_n"]) < minimum and \
                int(row["target_malignant_n"]) < minimum:
            why[pair_id] = "both sides below minimum"
        elif int(row["source_malignant_n"]) < minimum:
            why[pair_id] = "source below minimum"
        else:
            why[pair_id] = "target below minimum"

    dropped["reason"] = dropped["pair_id"].astype(str).map(why)
    by_reason = dropped["reason"].value_counts().to_dict()
    by_deposit = {}
    if "dataset_id" in dropped:
        by_deposit = (dropped.groupby("dataset_id")["reason"]
                      .value_counts().unstack(fill_value=0).to_dict("index"))
    report = {
        "manifest": str(args.manifest_csv),
        "counts": str(args.counts_csv),
        "minimum_cells_per_side": minimum,
        "pairs_in": int(len(manifest)),
        "pairs_kept": int(len(kept)),
        "pairs_dropped": int(len(dropped)),
        "dropped_by_reason": by_reason,
        "dropped_by_deposit": by_deposit,
        "kept_by_deposit": (kept["dataset_id"].value_counts().to_dict()
                            if "dataset_id" in kept else {}),
    }
    report_path = args.out_csv.with_name("trim_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    dropped.to_csv(dropped_path, index=False)

    print(f"kept    {len(kept)} of {len(manifest)} pairs -> {args.out_csv}")
    print(f"dropped {len(dropped)} -> {dropped_path}")
    print()
    for label, count in sorted(by_reason.items(), key=lambda item: -item[1]):
        print(f"  {count:4d}  {label}")
    if "dataset_id" in kept:
        print()
        print("kept per deposit:")
        print(kept["dataset_id"].value_counts().to_string())
    print()
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

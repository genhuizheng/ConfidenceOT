"""Build auditable manifests for independent GSE180661 sensitivity analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cancer_metastasis.gse180661.primary_pseudobulk import one_result_directory


def select_largest_balanced_pair_per_patient(manifest: pd.DataFrame) -> pd.DataFrame:
    """Select one pair per patient by the smaller malignant-side cell count."""
    required = {"patient_id", "pair_id", "source_malignant_n", "target_malignant_n"}
    missing = required.difference(manifest.columns)
    if missing:
        raise KeyError(f"Manifest is missing columns: {sorted(missing)}")
    ranked = manifest.copy()
    ranked["minimum_malignant_cell_n"] = np.minimum(
        pd.to_numeric(ranked["source_malignant_n"]),
        pd.to_numeric(ranked["target_malignant_n"]),
    ).astype(int)
    ranked["total_malignant_cell_n"] = (
        pd.to_numeric(ranked["source_malignant_n"])
        + pd.to_numeric(ranked["target_malignant_n"])
    ).astype(int)
    ranked = ranked.sort_values(
        ["patient_id", "minimum_malignant_cell_n", "total_malignant_cell_n", "pair_id"],
        ascending=[True, False, False, True],
        kind="stable",
    )
    ranked["within_patient_rank"] = ranked.groupby("patient_id").cumcount() + 1
    ranked["selected_for_one_pair_analysis"] = ranked["within_patient_rank"].eq(1)
    return ranked


def audit_prior_rejected_pairs(
    manifest: pd.DataFrame,
    prior_root: Path,
    *,
    budget_tag: str,
    method: str = "M4-E",
    minimum_cells: int = 20,
) -> pd.DataFrame:
    """Count prior rejected source/target cells and mark evaluable pairs."""
    records: list[dict] = []
    for row in manifest.itertuples(index=False):
        pair_id = str(row.pair_id)
        result_dir = one_result_directory(prior_root, pair_id, budget_tag)
        confidence = pd.read_csv(result_dir / "cell_confidence.csv")
        confidence = confidence[confidence["method"].eq(method)]
        counts = {}
        for side in ("source", "target"):
            side_table = confidence[confidence["side"].eq(side)]
            if side_table.empty or side_table["observation_id"].duplicated().any():
                raise RuntimeError(f"Invalid prior confidence table: pair={pair_id}, side={side}")
            counts[side] = int(side_table["rejected"].astype(bool).sum())
        records.append({
            "pair_id": pair_id,
            "prior_source_rejected_n": counts["source"],
            "prior_target_rejected_n": counts["target"],
            "prior_rejected_evaluable": (
                counts["source"] >= minimum_cells and counts["target"] >= minimum_cells
            ),
        })
    return manifest.merge(pd.DataFrame(records), on="pair_id", how="left", validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--mode", required=True, choices=("largest-balanced-pair", "prior-rejected")
    )
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--budget-tag", default="budget_source_0.85_target_0.95")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--minimum-cells", type=int, default=20)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    if args.mode == "largest-balanced-pair":
        audit = select_largest_balanced_pair_per_patient(manifest)
        eligible = audit[audit["selected_for_one_pair_analysis"]].copy()
        skipped = audit[~audit["selected_for_one_pair_analysis"]].copy()
        stem = "one_pair_per_patient"
        criterion = "maximum min(source_malignant_n, target_malignant_n) per patient"
    else:
        if args.prior_root is None:
            parser.error("--prior-root is required for --mode prior-rejected")
        audit = audit_prior_rejected_pairs(
            manifest, args.prior_root, budget_tag=args.budget_tag,
            method=args.method, minimum_cells=args.minimum_cells,
        )
        eligible = audit[audit["prior_rejected_evaluable"]].copy()
        skipped = audit[~audit["prior_rejected_evaluable"]].copy()
        stem = "prior_rejected"
        criterion = (
            f"at least {args.minimum_cells} prior {args.method} rejected cells on both sides"
        )

    audit.to_csv(args.output_directory / f"{stem}_audit.csv", index=False)
    eligible.to_csv(args.output_directory / f"{stem}_eligible.csv", index=False)
    skipped.to_csv(args.output_directory / f"{stem}_skipped.csv", index=False)
    report = {
        "mode": args.mode,
        "criterion": criterion,
        "input_pair_n": int(len(manifest)),
        "eligible_pair_n": int(len(eligible)),
        "skipped_pair_n": int(len(skipped)),
        "eligible_patient_n": int(eligible["patient_id"].nunique()),
        "prior_root": str(args.prior_root) if args.prior_root else None,
        "budget_tag": args.budget_tag if args.prior_root else None,
        "method": args.method if args.prior_root else None,
    }
    (args.output_directory / f"{stem}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

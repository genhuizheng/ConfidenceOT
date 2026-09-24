"""Restrict a pair manifest to the pairs a completed gate root actually holds.

The expression stages read the **original** count matrices while the gate was
computed on the equalised ones, so the manifest handed to
``21_prepare_four_state_malignant_pseudobulk.py`` is the original one. The two
do not cover the same pairs: equalisation and the optimal-transport array both
drop pairs, and GSE180661 finishes 92 gates against a 94-row manifest.

That difference is not a missing-data nuisance, it is a hard failure with a
wide blast radius. ``21_`` raises for the **whole patient** when any one of that
patient's pairs has no directory under the gate root, so two absent pairs can
remove two entire patients from the differential expression -- and the run
that loses them exits non-zero, which through ``afterok`` leaves every later
stage PENDING. Trimming first turns that into a reported count.

What counts as present is the pair having a ``cell_confidence.csv`` under the
gate root, which is the file ``21_`` reads. A directory alone is not enough: a
pair whose task died part-way leaves one behind.

Usage:

    python cancer_metastasis/35_trim_manifest_to_gate.py \\
        ORIGINAL_MANIFEST_CSV GATE_ROOT TRIMMED_MANIFEST_CSV
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
    parser.add_argument("gate_root", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--scope", default="scope_malignant")
    return parser.parse_args()


def gated_pairs(gate_root: Path, scope: str) -> set[str]:
    """Pair ids the gate root can actually supply a gate for."""
    found = set()
    for path in gate_root.glob(f"*/{scope}/*/cell_confidence.csv"):
        found.add(path.parents[2].name)
    return found


def main() -> None:
    args = parse_args()
    if not args.gate_root.is_dir():
        raise FileNotFoundError(f"Gate root does not exist: {args.gate_root}")
    manifest = pd.read_csv(args.manifest_csv)
    if "pair_id" not in manifest.columns:
        raise RuntimeError(f"{args.manifest_csv} has no pair_id column")

    present = gated_pairs(args.gate_root, args.scope)
    if not present:
        raise RuntimeError(
            f"No */{args.scope}/*/cell_confidence.csv under {args.gate_root}. "
            f"Either the scope is wrong or this is not a completed gate root."
        )
    keep = manifest["pair_id"].astype(str).isin(present)
    dropped = manifest.loc[~keep, "pair_id"].astype(str).tolist()
    trimmed = manifest[keep].copy()
    if trimmed.empty:
        # The only refusal. There is deliberately no threshold on *how much*
        # is dropped: a manifest built over the whole converted root spans
        # datasets, so a gate root for one accession legitimately matches a
        # small minority of its rows, and any threshold would have to be tuned
        # per dataset to avoid refusing correct runs. The report below says
        # exactly what was dropped and which patients lost pairs, which is a
        # better guard than a number nobody can set right.
        raise RuntimeError(
            f"No manifest row matched a gated pair under {args.gate_root}. "
            f"The manifest and the gate root describe different data; check "
            f"the manifest's dataset_id column against the accession in the "
            f"gate root's name."
        )
    # pair_index is positional and the array worker strides over rows, so it is
    # renumbered rather than left with gaps that would no longer match.
    if "pair_index" in trimmed.columns:
        trimmed["pair_index"] = range(len(trimmed))
    if "eligible_index" in trimmed.columns:
        trimmed["eligible_index"] = range(len(trimmed))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(args.output_csv, index=False)

    # Which patients lost a pair, and which lost every pair. The second group
    # is what 21_ would have failed on; the first still runs but on fewer
    # lesions than the manifest promised, which changes the largest-lesion
    # selection and must not be discovered afterwards.
    patients_affected: dict[str, dict[str, int]] = {}
    if "patient_id" in manifest.columns:
        before = manifest.groupby(manifest["patient_id"].astype(str)).size()
        after = trimmed.groupby(trimmed["patient_id"].astype(str)).size()
        for patient, count in before.items():
            kept = int(after.get(patient, 0))
            if kept != int(count):
                patients_affected[patient] = {"manifest_pairs": int(count),
                                              "gated_pairs": kept}
    fraction = len(trimmed) / len(manifest)
    report = {
        "manifest_csv": str(args.manifest_csv),
        "gate_root": str(args.gate_root),
        "output_csv": str(args.output_csv),
        "manifest_rows": int(len(manifest)),
        "gated_pairs_found": int(len(present)),
        "rows_kept": int(len(trimmed)),
        "rows_dropped": int(len(dropped)),
        "retained_fraction": round(fraction, 4),
        "dropped_pair_ids": dropped,
        "patients_losing_pairs": patients_affected,
        "patients_losing_every_pair": sorted(
            patient for patient, counts in patients_affected.items()
            if counts["gated_pairs"] == 0),
        "patient_n_before": int(manifest["patient_id"].nunique())
        if "patient_id" in manifest.columns else None,
        "patient_n_after": int(trimmed["patient_id"].nunique())
        if "patient_id" in trimmed.columns else None,
    }
    args.output_csv.with_suffix(".trim.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

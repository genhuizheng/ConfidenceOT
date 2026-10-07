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

``01_build_pair_manifest.py`` scans a whole converted root and emits one
manifest covering every deposit in it, so the pan-cancer manifest holds 84
patients across ten cancer types. A single dataset's chain needs its own rows
out of that, and before the transport has run there is no gate to select them
by -- hence ``--dataset-id``, which restricts by the manifest's own column and
needs no gate at all.

The two restrictions compose. Before the transport, pass ``--dataset-id``
alone. After it, pass the gate root as well, or only the gate root: a gate
root belongs to one dataset, so selecting on it selects the dataset too.

Several gate roots that are to be compared -- arms of one factorial -- want one
cohort between them. ``--gate-root`` repeated keeps the pairs every root
supplies, and ``--require-m4e-calibration`` also drops a pair whose M4-E
calibration ``21_`` would refuse under any root. Without both, a pair missing
from one arm, or refused in it, can change which lesion ``21_`` names for that
patient, and the arms would differ in tissue as well as in gate.

Usage:

    # before the transport: one dataset out of the pan-cancer manifest
    python cancer_metastasis/35_trim_manifest_to_gate.py \\
        PANCANCER_MANIFEST_CSV OUT.csv --dataset-id GSE315534

    # after it: the pairs the gate root can actually supply
    python cancer_metastasis/35_trim_manifest_to_gate.py \\
        ORIGINAL_MANIFEST_CSV OUT.csv --gate-root GATE_ROOT

    # several arms: the pairs every one supplies with a usable calibration
    python cancer_metastasis/35_trim_manifest_to_gate.py \\
        ORIGINAL_MANIFEST_CSV OUT.csv --gate-root ARM_A --gate-root ARM_B \\
        --require-m4e-calibration
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
    parser.add_argument("--gate-root", type=Path, action="append", default=None,
                        help="Keep only pairs this completed gate root can "
                             "supply a gate for; repeat to keep the pairs "
                             "every root supplies")
    parser.add_argument("--require-m4e-calibration", action="store_true",
                        help="Also drop a pair whose M4-E calibration 21_ "
                             "would refuse under any of the gate roots, so "
                             "that every root gives the pseudobulk the same "
                             "pairs")
    parser.add_argument("--dataset-id", action="append", default=None,
                        dest="dataset_ids", metavar="GSE",
                        help="Keep only these dataset_id values; repeatable. "
                             "Needs no gate, so it works before the transport.")
    parser.add_argument("--scope", default="scope_malignant")
    args = parser.parse_args()
    if args.gate_root is None and not args.dataset_ids:
        parser.error("pass --gate-root, --dataset-id, or both")
    if args.require_m4e_calibration and not args.gate_root:
        parser.error("--require-m4e-calibration reads the gates, so it needs --gate-root")
    return args


def gated_pairs(gate_root: Path, scope: str) -> set[str]:
    """Pair ids the gate root can actually supply a gate for."""
    found = set()
    for path in gate_root.glob(f"*/{scope}/*/cell_confidence.csv"):
        found.add(path.parents[2].name)
    return found


def pseudobulk_calibration_rule():
    """21_'s own test of a pair's M4-E calibration, imported rather than
    copied, so a pair kept here is one the pseudobulk will not refuse."""
    import importlib.util

    path = Path(__file__).with_name("21_prepare_four_state_malignant_pseudobulk.py")
    spec = importlib.util.spec_from_file_location("pseudobulk_rules", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calibration_refusal


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest_csv)
    if "pair_id" not in manifest.columns:
        raise RuntimeError(f"{args.manifest_csv} has no pair_id column")

    keep = pd.Series(True, index=manifest.index)
    present: set[str] = set()
    if args.dataset_ids:
        if "dataset_id" not in manifest.columns:
            raise RuntimeError(
                f"{args.manifest_csv} has no dataset_id column, so it cannot "
                f"be restricted to {args.dataset_ids}")
        known = sorted(set(manifest["dataset_id"].astype(str)))
        unknown = [value for value in args.dataset_ids if value not in known]
        if unknown:
            # Named and absent is a typo, not an empty dataset, and an empty
            # result here would look like a dataset with no eligible pairs.
            raise RuntimeError(
                f"dataset_id {unknown} not in {args.manifest_csv}; it holds "
                f"{known}")
        keep &= manifest["dataset_id"].astype(str).isin(args.dataset_ids)
    for position, root in enumerate(args.gate_root or []):
        if not root.is_dir():
            raise FileNotFoundError(f"Gate root does not exist: {root}")
        found = gated_pairs(root, args.scope)
        if not found:
            raise RuntimeError(
                f"No */{args.scope}/*/cell_confidence.csv under "
                f"{root}. Either the scope is wrong or this is not a "
                f"completed gate root."
            )
        present = found if position == 0 else present & found
    if args.gate_root:
        keep &= manifest["pair_id"].astype(str).isin(present)
    # A pair the pseudobulk refuses under one root and accepts under another
    # gives the roots different patients, or a patient different pairs, and the
    # comparison between them would then be partly a comparison of cohorts.
    refused: dict[str, list[str]] = {}
    if args.require_m4e_calibration:
        refusal_of = pseudobulk_calibration_rule()
        for pair in sorted(set(manifest.loc[keep, "pair_id"].astype(str))):
            for root in args.gate_root:
                runs = sorted((root / pair).glob(f"{args.scope}/*/run.json"))
                if len(runs) != 1:
                    raise RuntimeError(
                        f"Expected one run.json for {pair} under {root}; found {len(runs)}")
                reason = refusal_of(json.loads(runs[0].read_text(encoding="utf-8")))
                if reason is not None:
                    refused.setdefault(pair, []).append(f"{root}: {reason}")
        keep &= ~manifest["pair_id"].astype(str).isin(refused)
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
            f"No manifest row survived. dataset_id filter "
            f"{args.dataset_ids or 'none'}, gate root "
            f"{args.gate_root or 'none'}. If a gate root was given, the "
            f"manifest and the gate root describe different data; check the "
            f"manifest's dataset_id column against the accession in the gate "
            f"root's name."
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
        "gate_root": (str(args.gate_root[0]) if len(args.gate_root) == 1
                      else [str(root) for root in args.gate_root])
        if args.gate_root else None,
        "dataset_ids": args.dataset_ids,
        "output_csv": str(args.output_csv),
        "manifest_rows": int(len(manifest)),
        "gated_pairs_found": int(len(present)) if args.gate_root else None,
        "m4e_calibration_refused": refused if args.require_m4e_calibration else None,
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

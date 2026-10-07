"""Which pairs and patients each arm of a downstream run actually used.

    python cancer_metastasis/46_report_downstream_eligibility.py DOWN_ROOT OT_ROOT [--out DIR]

A report and nothing else: it reads what tacc_factorial_downstream.slurm and
the OT array wrote, decides nothing and changes nothing. For every arm under
DOWN_ROOT/<rep>/arm_<X>/<block>/<accession>/<label>/ it lists

* the pairs of that accession the OT was asked to run (the block's
  MANIFEST_PATH under OT_ROOT) that the arm did not gate -- not evaluable, or
  failed -- which the job's trim left out;
* the pairs 21_ excluded for their M4-E calibration, with 21_'s reason;
* the patients 21_ skipped, with its reason;
* the lesion 21_ named for each patient, so a patient whose lesion differs
  between arms is visible;
* how many patients entered the paired DEG.

Written to DIR (default DOWN_ROOT/eligibility_report): arms.tsv, one row per
arm; events.tsv, one row per pair or patient left out; lesions.tsv, patient by
arm. The arms are printed with the left-out pairs named.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

CONTRAST = "primary_rejected_vs_primary_retained"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("down_root", type=Path)
    parser.add_argument("ot_root", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def requested_pairs(ot_root: Path, rep: str, arm: str, block: str, accession: str) -> set[str]:
    """The accession's pairs in the manifest the OT array ran this block on."""
    record = ot_root / rep / arm / block / "MANIFEST_PATH"
    manifest = pd.read_csv(record.read_text(encoding="utf-8").strip())
    rows = manifest[manifest["dataset_id"].astype(str) == accession]
    return set(rows["pair_id"].astype(str))


def main() -> None:
    args = parse_args()
    out = args.out or args.down_root / "eligibility_report"
    arms, events, lesions = [], [], []
    for arm_dir in sorted(args.down_root.glob("*/arm_*/*/*/*")):
        if not (arm_dir / "logs").is_dir():
            continue
        rep, arm, block, accession, label = arm_dir.parts[-5:]
        name = f"{rep}/{arm}/{label}"
        row = {"accession": accession, "block": block, "rep": rep, "arm": arm, "label": label}
        if (arm_dir / "FAILED").is_file():
            row["status"] = "FAILED: " + (arm_dir / "FAILED").read_text(encoding="utf-8").strip().splitlines()[-1]
        elif (arm_dir / "DONE").is_file():
            row["status"] = "done"
        else:
            row["status"] = "not finished"
        asked = requested_pairs(args.ot_root, rep, arm, block, accession)
        trimmed = arm_dir / "manifest_trimmed.csv"
        gated = (set(pd.read_csv(trimmed)["pair_id"].astype(str)) if trimmed.is_file() else set())
        row["pairs_requested"] = len(asked)
        row["pairs_gated"] = len(gated & asked)
        for pair in sorted(asked - gated):
            events.append({"accession": accession, "arm": name, "kind": "pair not gated in this arm",
                           "id": pair, "reason": "no cell_confidence.csv under the arm's gate root"})
        excluded = skipped = 0
        for diagnostics in sorted((arm_dir / "four_state" / "patients").glob("*/diagnostics.json")):
            report = json.loads(diagnostics.read_text(encoding="utf-8"))
            patient = str(report.get("patient_id"))
            for item in report.get("excluded_pairs", []):
                excluded += 1
                events.append({"accession": accession, "arm": name,
                               "kind": "pair excluded by 21_ (calibration)",
                               "id": item.get("pair_id"), "reason": item.get("reason")})
            if "reason" in report:
                skipped += 1
                events.append({"accession": accession, "arm": name, "kind": "patient skipped by 21_",
                               "id": patient, "reason": report["reason"]})
            lesions.append({"accession": accession, "patient_id": patient, "arm": name,
                            "designated_metastasis": report.get("designated_metastasis")})
        row["pairs_excluded_by_21"] = excluded
        row["patients_skipped_by_21"] = skipped
        deg = arm_dir / "deg" / "pydeseq2_report.json"
        patients = None
        if deg.is_file():
            for contrast in json.loads(deg.read_text(encoding="utf-8")).get("contrasts", []):
                if contrast.get("contrast") == CONTRAST:
                    patients = contrast.get("patient_n")
        row["patients_in_deg"] = patients
        arms.append(row)
    if not arms:
        raise SystemExit(f"no arm directory under {args.down_root}")

    out.mkdir(parents=True, exist_ok=True)
    arms = pd.DataFrame(arms)
    events = pd.DataFrame(events, columns=["accession", "arm", "kind", "id", "reason"])
    lesions = pd.DataFrame(lesions, columns=["accession", "patient_id", "arm", "designated_metastasis"])
    arms.to_csv(out / "arms.tsv", sep="\t", index=False)
    events.to_csv(out / "events.tsv", sep="\t", index=False)
    table = (lesions.pivot_table(index=["accession", "patient_id"], columns="arm",
                                 values="designated_metastasis", aggfunc="first")
             if len(lesions) else pd.DataFrame())
    table.to_csv(out / "lesions.tsv", sep="\t")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 80)
    print(arms.to_string(index=False))
    if len(events):
        print("\nleft out, by arm:")
        for (accession, arm), part in events.groupby(["accession", "arm"], sort=True):
            print(f"  {accession} {arm}:")
            for item in part.itertuples(index=False):
                print(f"    {item.kind}: {item.id}  ({item.reason})")
    else:
        print("\nnothing left out in any arm")
    varying = table[table.nunique(axis=1, dropna=False) > 1] if len(table) else table
    print(f"\npatients whose named lesion differs between arms: {len(varying)}")
    if len(varying):
        print(varying.to_string())
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()

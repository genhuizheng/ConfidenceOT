"""Build a cross-patient mismatched manifest as a negative control.

The matched analysis gates a patient's primary malignant cells against that
same patient's metastatic lesion and reads the retained set as evidence about
metastatic compatibility.  Nothing in the pipeline tests whether the split
carries patient-specific information at all: a gate driven by the shared
geometry of two malignant clouds would produce a similar retained fraction,
and a similar differential expression signature, no matter whose metastasis
sat on the other side.

This script replaces each metastatic side with a *different* patient's lesion
and changes nothing else.  Matched and mismatched runs must then separate.  If
they do not, the matched result carries no patient-specific information, and
the keratin/SPRR signature that appears in the rejected group of all three
datasets is a property of the method rather than of metastasis.

**The control must differ in patient identity alone.**  Two consequences shape
the assignment:

*Size.*  A lesion is chosen to be as close in cell count as possible to the one
it replaces, because the retained fraction can depend on the size ratio of the
two sides.  A control whose metastatic sides were systematically larger or
smaller would differ from the matched run for a reason that has nothing to do
with whose metastasis it is.

*Distinctness.*  A source sample never receives the same donor lesion twice.
An earlier draft deranged the patient list and cycled through the donor's
lesions, which collapsed whenever a patient had more rows than the donor had
lesions: SPECTRUM-OV-003 has two primary samples across four metastatic sites,
but its donor had two, so combinations repeated.  Drawing from the pooled
lesions of every other patient removes the coupling between a donor's lesion
count and a recipient's row count.

The source side is untouched: the same primary cells are gated in both runs,
which is what makes the two retained fractions comparable pair by pair rather
than only in distribution.

The emitted manifest has the same columns and the same row count as its input,
and ``02_run_pair.py`` indexes it positionally, so the existing array runs it
unmodified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = (
    "target_h5ad",
    "target_h5ads_json",
    "target_sample",
    "target_n",
    "target_file_n",
    "target_file_kind",
)


def pooled_lesions(manifest: pd.DataFrame) -> list[tuple[tuple[str, str], dict]]:
    """Every distinct metastatic side in the manifest, keyed by patient and sample."""
    lesions: list[tuple[tuple[str, str], dict]] = []
    seen: set[tuple[str, str]] = set()
    for row in manifest.to_dict("records"):
        key = (str(row["patient_id"]), str(row["target_sample"]))
        if key in seen:
            continue
        seen.add(key)
        lesions.append((key, {name: row[name] for name in TARGET_COLUMNS}))
    return lesions


def choose_donor(
    lesions: list[tuple[tuple[str, str], dict]],
    order: list[int],
    source_patient: str,
    matched_n: float,
    taken: set[tuple[str, str]],
) -> tuple[tuple[str, str], dict]:
    """Pick another patient's lesion of the most similar size.

    Similarity is the absolute log ratio of cell counts, so a lesion twice as
    large and one half as large are equally far from the target.  Ties break on
    a seeded shuffle of the pool, which keeps the assignment deterministic
    without letting manifest order decide it.
    """
    best = None
    for rank, index in enumerate(order):
        key, record = lesions[index]
        if key[0] == source_patient or key in taken:
            continue
        donor_n = max(float(record["target_n"]), 1.0)
        score = (abs(math.log(donor_n / max(matched_n, 1.0))), rank)
        if best is None or score < best[0]:
            best = (score, key, record)
    if best is None:
        raise RuntimeError(
            f"No unused lesion from another patient remains for {source_patient!r}; "
            "the pooled lesions are exhausted"
        )
    return best[1], best[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    missing = [name for name in TARGET_COLUMNS if name not in manifest.columns]
    if missing:
        raise RuntimeError(f"{args.manifest_csv}: missing columns {missing}")
    manifest["patient_id"] = manifest["patient_id"].astype(str)

    lesions = pooled_lesions(manifest)
    patients = sorted({key[0] for key, _ in lesions})
    if len(patients) < 2:
        raise RuntimeError(
            f"A mismatched control needs at least two patients, found {len(patients)}"
        )
    rng = np.random.default_rng(args.seed)
    order = [int(index) for index in rng.permutation(len(lesions))]

    rows = []
    # Keyed by primary sample, not by patient: a patient with two primaries may
    # legitimately send each of them to the same donor lesion, and only a repeat
    # within one primary would be a duplicated computation.
    taken: dict[tuple[str, str], set[tuple[str, str]]] = {}
    for row in manifest.to_dict("records"):
        source_patient = str(row["patient_id"])
        source_key = (source_patient, str(row["source_sample"]))
        used = taken.setdefault(source_key, set())
        donor_key, replacement = choose_donor(
            lesions, order, source_patient, float(row["target_n"]), used
        )
        used.add(donor_key)

        new_row = dict(row)
        new_row.update(replacement)
        new_row["source_patient_id"] = source_patient
        new_row["target_patient_id"] = donor_key[0]
        new_row["matched_pair_id"] = str(row["pair_id"])
        new_row["matched_target_sample"] = str(row["target_sample"])
        new_row["matched_target_n"] = int(row["target_n"])
        new_row["pair_id"] = (
            f"{source_patient}__{row['source_sample']}"
            f"__VS__{donor_key[0]}__{donor_key[1]}"
        )
        rows.append(new_row)

    mismatched = pd.DataFrame(rows)
    if mismatched["pair_id"].duplicated().any():
        duplicated = sorted(
            mismatched.loc[mismatched["pair_id"].duplicated(), "pair_id"].unique()
        )
        raise RuntimeError(f"Mismatched pair_id is not unique: {duplicated[:5]}")
    if len(mismatched) != len(manifest):
        raise RuntimeError("Row count changed; the array indexes positionally")
    same = mismatched["source_patient_id"] == mismatched["target_patient_id"]
    if same.any():
        raise RuntimeError(f"{int(same.sum())} rows kept their own patient")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    mismatched.to_csv(args.output_csv, index=False)

    size_gap = np.abs(
        np.log2(
            mismatched["target_n"].clip(lower=1)
            / mismatched["matched_target_n"].clip(lower=1)
        )
    )
    report = {
        "input_manifest": str(args.manifest_csv),
        "output_manifest": str(args.output_csv),
        "pairs": int(len(mismatched)),
        "patients": len(patients),
        "pooled_lesions": len(lesions),
        "seed": args.seed,
        "distinct_donor_patients_used": int(mismatched["target_patient_id"].nunique()),
        "most_used_donor_patient_rows": int(
            mismatched["target_patient_id"].value_counts().max()
        ),
        # The source side is identical between the runs, so any shift on the
        # target side is the whole difference in pair geometry.  A median
        # absolute log2 size gap near zero is what makes patient identity the
        # only thing the control varies.
        "target_n_matched_median": float(manifest["target_n"].median()),
        "target_n_mismatched_median": float(mismatched["target_n"].median()),
        "target_n_abs_log2_gap_median": float(np.median(size_gap)),
        "target_n_abs_log2_gap_q90": float(np.quantile(size_gap, 0.90)),
        "source_n_median": float(manifest["source_n"].median()),
    }
    print(json.dumps(report, indent=2), flush=True)
    print(
        mismatched[
            ["source_patient_id", "source_sample", "target_patient_id",
             "target_sample", "matched_target_sample", "source_n",
             "target_n", "matched_target_n"]
        ].head(12).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

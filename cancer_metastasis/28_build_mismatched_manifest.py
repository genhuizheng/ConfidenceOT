"""Build a cross-patient mismatched manifest as a negative control.

The matched analysis gates a patient's primary malignant cells against that
same patient's metastatic lesion and reads the retained set as evidence about
metastatic compatibility.  Nothing in the pipeline tests whether the split
carries patient-specific information at all: a gate driven by the shared
geometry of two malignant clouds would produce a similar retained fraction,
and a similar differential expression signature, no matter whose metastasis
sat on the other side.

This script pairs each primary with a *different* patient's metastasis and
changes nothing else.  Matched and mismatched runs must then separate.  If
they do not, the matched result carries no patient-specific information, and
the keratin/SPRR signature that appears in the rejected group of all three
datasets is a property of the method rather than of metastasis.

Donors are assigned by a derangement of the patient list, so no patient can
receive their own metastasis, and every patient donates.  The source side is
untouched: the same primary cells are gated in both runs, which is what makes
the two retained fractions directly comparable.

The emitted manifest has the same columns and the same row count as its
input, and ``02_run_pair.py`` indexes it positionally, so the existing array
runs it unmodified.
"""

from __future__ import annotations

import argparse
import json
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


def derange(patients: list[str], rng: np.random.Generator) -> dict[str, str]:
    """Map every patient to a different patient, with no fixed point.

    A shuffle followed by a single-position rotation is a derangement by
    construction: rotating a permutation moves every element, so no patient
    can be mapped to themselves and every patient donates exactly once.
    """
    if len(patients) < 2:
        raise RuntimeError(
            f"A mismatched control needs at least two patients, found {len(patients)}"
        )
    order = list(rng.permutation(np.asarray(patients, dtype=object)))
    rotated = order[1:] + order[:1]
    mapping = dict(zip(order, rotated))
    fixed = [name for name, donor in mapping.items() if name == donor]
    if fixed:
        raise RuntimeError(f"Derangement left fixed points: {fixed}")
    return mapping


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

    # Every distinct metastatic side available, kept per patient. A patient
    # contributing several lesions donates them in a fixed order, so a source
    # patient with many pairs receives several distinct lesions rather than
    # the same one repeated.
    donors: dict[str, list[dict[str, object]]] = {}
    for row in manifest.to_dict("records"):
        record = {name: row[name] for name in TARGET_COLUMNS}
        bucket = donors.setdefault(str(row["patient_id"]), [])
        if record not in bucket:
            bucket.append(record)

    rng = np.random.default_rng(args.seed)
    patients = sorted(donors)
    mapping = derange(patients, rng)

    rows = []
    used_within_patient: dict[str, int] = {}
    for row in manifest.to_dict("records"):
        source_patient = str(row["patient_id"])
        donor_patient = mapping[source_patient]
        available = donors[donor_patient]
        position = used_within_patient.get(source_patient, 0)
        used_within_patient[source_patient] = position + 1
        replacement = available[position % len(available)]

        new_row = dict(row)
        new_row.update(replacement)
        new_row["source_patient_id"] = source_patient
        new_row["target_patient_id"] = donor_patient
        new_row["matched_pair_id"] = str(row["pair_id"])
        new_row["pair_id"] = (
            f"{source_patient}__{row['source_sample']}"
            f"__VS__{donor_patient}__{replacement['target_sample']}"
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

    report = {
        "input_manifest": str(args.manifest_csv),
        "output_manifest": str(args.output_csv),
        "pairs": int(len(mismatched)),
        "patients": len(patients),
        "seed": args.seed,
        "distinct_target_lesions_used": int(
            mismatched[["target_patient_id", "target_sample"]]
            .drop_duplicates().shape[0]
        ),
        "source_n": {
            "median": float(manifest["source_n"].median()),
            "total": int(manifest["source_n"].sum()),
        },
        # The source side is identical, so any shift here is the whole
        # difference in pair size between the two runs.
        "target_n_matched": {
            "median": float(manifest["target_n"].median()),
            "total": int(manifest["target_n"].sum()),
        },
        "target_n_mismatched": {
            "median": float(mismatched["target_n"].median()),
            "total": int(mismatched["target_n"].sum()),
        },
    }
    print(json.dumps(report, indent=2), flush=True)
    print(
        mismatched[
            ["pair_id", "source_patient_id", "source_sample",
             "target_patient_id", "target_sample", "source_n", "target_n"]
        ].head(12).to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

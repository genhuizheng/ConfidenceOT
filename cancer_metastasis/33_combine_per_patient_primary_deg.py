"""Combine the per-patient primary DE of method B three ways.

``32_per_patient_primary_deg.py`` writes one effect table per patient. This
reads them together and reports the three things a per-patient design can say
that a pooled fit cannot:

**Gene overlap.** In how many patients does this gene clear the thresholds on
its own, and in which direction. A gene called in 24 of 29 patients and a gene
called in 3 are different findings even at the same combined FDR, and only the
per-patient design can tell them apart.

**Direction consistency.** The fraction of patients whose effect agrees with
the sign of the median. This is reported beside the FDR rather than folded
into it, because a large effect in a few patients and a small effect in nearly
all of them are both real and are not the same claim.

**Patient-level meta-analysis.** A Wilcoxon over the per-patient effect sizes,
with the patient as the inference unit rather than the cell. Cell-level
p-values within one patient are inflated by within-patient correlation; asking
instead whether the *patients* agree does not inherit that.

Together these are what the differential-expression prespecification's
Disqualifier 3 asks for -- whether a result rests on particular patients --
turned into columns of the result rather than an audit performed afterwards.

**Scope.** This needs patients. The prespecification runs it in full on
GSE180661 and treats it as exploratory on GSE271675; the two small datasets do
not support it, and the script says so rather than producing a table that
looks like the others.

Usage:

    python cancer_metastasis/33_combine_per_patient_primary_deg.py \\
        PER_PATIENT_ROOT OUTPUT_ROOT [--minimum-patients 6]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import bh_adjust, meta_table  # noqa: F401  (bh_adjust re-exported)

INTERPRETATION = "primary_rejected_minus_primary_retained"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("per_patient_root", type=Path,
                        help="Output root of 32_per_patient_primary_deg.py")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-patients", type=int, default=6,
                        help="Patients a gene needs before the meta-analysis "
                             "is attempted for it; also the floor below which "
                             "the whole combination is refused")
    parser.add_argument("--maximum-fdr", type=float, default=0.05,
                        help="Per-patient significance, for the overlap count")
    parser.add_argument("--absolute-log2-fold-change", type=float, default=1.0,
                        help="Per-patient effect floor, for the overlap count")
    return parser.parse_args()


def read_patients(root: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Every per-patient table, plus one status row per patient directory.

    A patient that 32_ skipped is carried in the status rows rather than being
    absent, so the combination reports how many patients it did not have and
    why. A directory with neither a table nor a skip marker is an incomplete
    run and is called that, because it is the one case that would otherwise
    shrink the denominator silently.
    """
    frames, status = [], []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        report_path = directory / "diagnostics.json"
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.is_file() else {})
        patient = str(report.get("patient_id", directory.name.split("_", 1)[-1]))
        table_path = directory / "per_patient_primary_deg.csv.gz"
        row = {
            "patient_id": patient,
            "directory": directory.name,
            "retained_n": (report.get("status_n") or {}).get("retained"),
            "rejected_n": (report.get("status_n") or {}).get("rejected"),
        }
        if table_path.is_file():
            frame = pd.read_csv(table_path)
            frame["patient_id"] = patient
            frames.append(frame)
            row["state"] = "included"
            row["gene_n"] = int(frame["gene"].nunique())
        elif (directory / "PER_PATIENT_DEG_SKIPPED").is_file():
            row["state"] = "skipped"
            row["reason"] = report.get("skip_reason", "")
        else:
            row["state"] = "incomplete"
            row["reason"] = "neither a result table nor a skip marker"
        status.append(row)
    if not frames:
        raise RuntimeError(
            f"No per-patient tables under {root}. Run "
            f"32_per_patient_primary_deg.py first."
        )
    return pd.concat(frames, ignore_index=True), status


def overlap_columns(effects: pd.DataFrame, maximum_fdr: float,
                    minimum_abs_log2: float) -> pd.DataFrame:
    """Per-gene counts of the patients that call it on their own."""
    called = effects[effects["fdr"].lt(maximum_fdr)
                     & effects["log2_fold_change"].abs().ge(minimum_abs_log2)]
    up = called[called["log2_fold_change"] > 0].groupby("gene").size()
    down = called[called["log2_fold_change"] < 0].groupby("gene").size()
    tested = effects.groupby("gene").size()
    table = pd.DataFrame({"gene": tested.index})
    table["patients_tested"] = tested.to_numpy()
    table["patients_significant_up"] = (
        up.reindex(tested.index).fillna(0).astype(int).to_numpy())
    table["patients_significant_down"] = (
        down.reindex(tested.index).fillna(0).astype(int).to_numpy())
    table["patients_significant"] = (table["patients_significant_up"]
                                     + table["patients_significant_down"])
    table["overlap_fraction"] = (table["patients_significant"]
                                 / table["patients_tested"].clip(lower=1))
    return table


def attainable_fdr(patient_n: int, gene_n: int) -> float:
    """The smallest FDR any gene could reach, given the design alone.

    A two-sided exact signed-rank test over ``patient_n`` non-zero effects
    cannot return a p-value below ``2 / 2 ** patient_n``, whatever the data.
    Benjamini-Hochberg multiplies the smallest p by the number of genes, so the
    best attainable adjusted value is ``gene_n * 2 ** (1 - patient_n)``.

    With 20,000 genes at 0.05 that needs about 20 patients. Below it **no gene
    can be significant**, and the output is a clean-looking null rather than an
    error -- which is the failure this function exists to make impossible to
    miss. It is a property of the design, computable before any data is read.
    """
    if patient_n <= 0:
        return float("inf")
    return float(gene_n) * float(2.0 ** (1 - patient_n))


def main() -> None:
    args = parse_args()
    effects, status = read_patients(args.per_patient_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    status_frame = pd.DataFrame(status)
    status_frame.to_csv(args.output_root / "patient_status.csv", index=False)

    included = int((status_frame["state"] == "included").sum())
    report: dict[str, object] = {
        "per_patient_root": str(args.per_patient_root),
        "patient_directories": int(len(status_frame)),
        "patients_included": included,
        "patients_skipped": int((status_frame["state"] == "skipped").sum()),
        "patients_incomplete": int((status_frame["state"] == "incomplete").sum()),
        "minimum_patients": args.minimum_patients,
        "per_patient_significance": {
            "maximum_fdr": args.maximum_fdr,
            "absolute_log2_fold_change": args.absolute_log2_fold_change,
        },
        "interpretation": INTERPRETATION,
    }

    if included < args.minimum_patients:
        # Refused rather than produced. A meta-analysis over four patients
        # returns numbers with the same column names as one over twenty-nine,
        # and nothing downstream distinguishes them; the prespecification
        # scopes this method to the datasets that can carry it, so a dataset
        # that cannot must fail to produce a table rather than produce a thin
        # one.
        report["combined"] = False
        report["refusal"] = (
            f"{included} patients carry both states, below the floor of "
            f"{args.minimum_patients}. This dataset does not support the "
            f"per-patient method; run the paired pseudobulk alone."
        )
        (args.output_root / "diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    meta = meta_table(effects, args.minimum_patients, INTERPRETATION)
    combined = meta.merge(
        overlap_columns(effects, args.maximum_fdr,
                        args.absolute_log2_fold_change),
        on="gene", how="left")
    # Sorted by the three readings together: a gene near the top clears the
    # meta-analysis, points the same way in most patients, and is called on its
    # own by many of them. Any one of the three alone is weaker evidence.
    combined = combined.sort_values(
        ["fdr", "direction_consistency", "patients_significant"],
        ascending=[True, False, False], kind="stable")
    combined.to_csv(args.output_root / "per_patient_meta.csv.gz",
                    index=False, compression="gzip")

    significant = combined[combined["fdr"].lt(args.maximum_fdr)]
    report["combined"] = True
    report["gene_n"] = int(len(combined))
    # Reported whether or not anything was significant, because the number is
    # only interesting when it is bad and a reader will not think to ask.
    floor = attainable_fdr(int(combined["patient_n"].max()), len(combined))
    report["minimum_attainable_meta_fdr"] = floor
    report["meta_analysis_can_reach_threshold"] = bool(floor < args.maximum_fdr)
    if floor >= args.maximum_fdr:
        report["meta_analysis_warning"] = (
            f"With {int(combined['patient_n'].max())} patients and "
            f"{len(combined)} genes the smallest FDR any gene could reach is "
            f"{floor:.3g}, above the {args.maximum_fdr} threshold. No gene can "
            f"be significant by the meta-analysis in this dataset whatever the "
            f"effects are; read the overlap and direction-consistency columns "
            f"instead, and do not report the absence as a null result."
        )
    report["meta_fdr_below_threshold_n"] = int(len(significant))
    report["meta_and_consistent_n"] = int(
        (significant["direction_consistency"] >= 0.70).sum())
    report["called_in_at_least_half_the_patients_n"] = int(
        (combined["overlap_fraction"] >= 0.5).sum())
    report["called_in_no_patient_but_meta_significant_n"] = int(
        (significant["patients_significant"] == 0).sum())
    (args.output_root / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

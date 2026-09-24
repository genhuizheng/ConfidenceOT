"""Leave-one-patient-out refit of the paired pseudobulk, method A's robustness.

The differential-expression prespecification's third disqualifier is written
as: *if removing any single patient changes the significant set by more than
half, the result rests on one patient rather than on all of them.* Nothing in
the repository performed that. ``31_audit_deg_disqualifiers.py`` computes a
per-patient sign-agreement proxy from pseudobulk CPM and says in its own
docstring that it cannot replace the refit. This is the refit.

It runs ``13_run_paired_pydeseq2.py`` once on every patient and then once per
patient with that patient excluded, and reports two things:

**The disqualifier, as written.** For each held-out patient, the Jaccard index
between the full significant set and the one fitted without them, and whether
any single patient moves it by more than half.

**A per-gene survival profile, which is the more useful output.** In how many
of the leave-one-out fits each gene stayed significant. A gene surviving every
fit and a gene surviving eighteen of twenty-nine are both "not disqualified",
and they are not the same evidence. The profile is what method B's overlap
column is for the per-patient design, computed here for the pooled one, so the
two methods can be compared on the same question.

**Cost.** One PyDESeq2 fit per patient plus one, each a minutes job at ~29
patients. The fits are independent and this runs them in sequence; the SLURM
wrapper gives it one node and hours rather than an array, because the
bookkeeping of collecting P output directories is the part that goes wrong.

Usage:

    python cancer_metastasis/34_leave_one_patient_out.py \\
        FOUR_STATE_ROOT OUTPUT_ROOT [--minimum-cells-per-patient-status 10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

CONTRAST = "primary_rejected_vs_primary_retained"
# 13_ writes its filtered tables under this name for each fold-change floor.
RESULT_TEMPLATE = "pydeseq2_all_gene_fdr_005_abs_log2fc_{tag}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("four_state_root", type=Path,
                        help="Output root of 21_prepare_four_state_malignant_pseudobulk.py")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--minimum-cells-per-patient-status", type=int, default=10)
    parser.add_argument("--minimum-total-count", type=int, default=10)
    parser.add_argument("--n-cpus", type=int, default=16)
    parser.add_argument("--maximum-fdr", type=float, default=0.05)
    parser.add_argument("--absolute-log2-fold-change", type=float, default=1.0,
                        help="Which of 13_'s filtered tables to read; it must "
                             "be one 13_ was asked to write")
    parser.add_argument("--contrast", default=CONTRAST)
    parser.add_argument("--skip-completed", action="store_true")
    return parser.parse_args()


def patients(four_state_root: Path) -> list[str]:
    """Patient ids that method A actually produced a pseudobulk for.

    Read from the metadata rather than the directory names, because a patient
    21_ skipped still leaves a directory and would otherwise be held out of a
    fit it was never in -- producing a leave-one-out result identical to the
    full fit and quietly inflating the count of patients that change nothing.
    """
    found: set[str] = set()
    for ready in sorted(four_state_root.glob("patients/*/PSEUDOBULK_READY")):
        metadata = pd.read_csv(ready.parent / "pseudobulk_sample_metadata.csv")
        found.update(metadata["patient_id"].astype(str))
    return sorted(found)


def run_fit(four_state_root: Path, output: Path, args: argparse.Namespace,
            exclude: str | None) -> bool:
    """One PyDESeq2 fit. Returns whether it produced the table we need."""
    if args.skip_completed and (output / "pydeseq2_report.json").is_file():
        return True
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "cancer_metastasis/13_run_paired_pydeseq2.py",
        str(four_state_root), str(output),
        "--minimum-cells-per-patient-status", str(args.minimum_cells_per_patient_status),
        "--minimum-total-count", str(args.minimum_total_count),
        "--maximum-fdr", str(args.maximum_fdr),
        "--absolute-log2-fold-change-thresholds", str(args.absolute_log2_fold_change),
        "--n-cpus", str(args.n_cpus),
    ]
    if exclude is not None:
        command += ["--exclude-patient", exclude]
    result = subprocess.run(command, capture_output=True, text=True)
    (output / "pydeseq2_stdout.log").write_text(
        result.stdout + result.stderr, encoding="utf-8")
    return result.returncode == 0


def significant_genes(root: Path, contrast: str, tag: str) -> set[str] | None:
    path = root / "contrasts" / contrast / RESULT_TEMPLATE.format(tag=tag)
    if not path.is_file():
        candidates = sorted(
            str(value.relative_to(root))
            for value in root.rglob("pydeseq2_all_gene_fdr_*.csv"))
        raise FileNotFoundError(
            f"{path} is absent. Tables present: {candidates or 'none'}. The "
            f"--absolute-log2-fold-change must be one 13_ was asked to write."
        )
    table = pd.read_csv(path)
    return set(table["gene"].astype(str))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def main() -> None:
    args = parse_args()
    tag = f"{args.absolute_log2_fold_change:g}".replace(".", "p")
    people = patients(args.four_state_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "four_state_root": str(args.four_state_root),
        "contrast": args.contrast,
        "patient_n": len(people),
        "patients": people,
        "minimum_cells_per_patient_status": args.minimum_cells_per_patient_status,
        "maximum_fdr": args.maximum_fdr,
        "absolute_log2_fold_change": args.absolute_log2_fold_change,
    }
    if len(people) < 3:
        report["completed"] = False
        report["refusal"] = (
            f"{len(people)} patients produced a pseudobulk; a leave-one-out "
            f"profile needs at least three to say anything."
        )
        (args.output_root / "diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return

    print(f"full fit over {len(people)} patients", flush=True)
    if not run_fit(args.four_state_root, args.output_root / "full", args, None):
        raise RuntimeError(
            f"The full fit failed; see "
            f"{args.output_root / 'full' / 'pydeseq2_stdout.log'}")
    full = significant_genes(args.output_root / "full", args.contrast, tag)

    rows, survival = [], {gene: 0 for gene in full}
    failures = []
    for index, patient in enumerate(people, start=1):
        print(f"[{index}/{len(people)}] without {patient}", flush=True)
        destination = args.output_root / "without" / patient
        if not run_fit(args.four_state_root, destination, args, patient):
            # Recorded rather than raised: one patient whose removal makes the
            # design unfittable is itself a finding about that patient, and
            # stopping here would discard the folds already computed.
            failures.append(patient)
            continue
        held_out = significant_genes(destination, args.contrast, tag)
        for gene in held_out & full:
            survival[gene] += 1
        rows.append({
            "held_out_patient": patient,
            "significant_n": len(held_out),
            "shared_with_full_n": len(held_out & full),
            "lost_n": len(full - held_out),
            "gained_n": len(held_out - full),
            "jaccard_with_full": jaccard(full, held_out),
        })

    folds = len(rows)
    per_patient = pd.DataFrame(rows)
    per_patient.to_csv(args.output_root / "leave_one_out_per_patient.csv",
                       index=False)
    profile = pd.DataFrame({
        "gene": sorted(full),
        "folds_survived": [survival[gene] for gene in sorted(full)],
    })
    profile["folds"] = folds
    profile["survival_fraction"] = (profile["folds_survived"]
                                    / max(folds, 1))
    profile.sort_values(["folds_survived", "gene"],
                        ascending=[False, True]).to_csv(
        args.output_root / "leave_one_out_gene_survival.csv", index=False)

    report["completed"] = True
    report["folds_run"] = folds
    report["fold_failures"] = failures
    report["full_significant_n"] = len(full)
    if folds:
        worst = per_patient.loc[per_patient["jaccard_with_full"].idxmin()]
        report["worst_patient"] = str(worst["held_out_patient"])
        report["worst_jaccard"] = float(worst["jaccard_with_full"])
        # The disqualifier as the prespecification words it. Jaccard below 0.5
        # is the operational reading of "changes the significant set by more
        # than half"; it is recorded alongside the raw counts so a reader can
        # apply a different reading without re-running anything.
        report["disqualifier_3_fires"] = bool(worst["jaccard_with_full"] < 0.5)
        report["genes_surviving_every_fold_n"] = int(
            (profile["folds_survived"] == folds).sum())
        report["genes_surviving_no_fold_n"] = int(
            (profile["folds_survived"] == 0).sum())
    (args.output_root / "diagnostics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

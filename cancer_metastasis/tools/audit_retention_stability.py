"""Is the retained fraction a property of the primary sample, or of its partner?

The gate's output is read as "which primary cells have metastatic potential".
That reading requires the retained set to be a property of the primary sample:
if at most ~15% of a tumour's malignant cells carry metastatic potential, then
comparing that same tumour against a different metastatic site of the same
patient should give roughly the same answer.

It does not. In the first pairs of the 2026-09-21 run, SPECTRUM-OV-003's
`right_adnexa` retained 5%, 23%, 28% and 41% against four metastatic sites of
the same patient -- an eightfold swing from changing only the partner. This
measures that properly, across every source sample that appears in more than
one pair, rather than from six lines of a log.

Three questions, from files the run already wrote:

1. **Within-source spread.** One primary sample against several metastases. A
   large spread means the retained fraction is reporting the partner.
2. **Against distance.** Section 5.8.4 of `CURRENT_WORK_SUMMARY_2026-09-13.md`
   records the suspicion that the retained fraction is close to inversely
   proportional to how far apart the two sides are, which would mean the method
   returns least exactly where there is most to study. `rejection_cost` is the
   calibrated threshold in units of the median cross-side pair distance, so it
   is the available per-pair measure of that distance.
3. **Against the partner's size.** A retained fraction that tracks how many
   cells the metastasis happens to contain is reporting sequencing effort.

None of these is a verdict on the method. They are three ways for the retained
fraction to be about something other than the primary cells, and each is
either present in the data or it is not.

Usage:
  python cancer_metastasis/tools/audit_retention_stability.py
      --dataset GSE180661=.../ot_GSE180661_rank256_ds_cos_20260921
      --out retention_stability.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=OT_ROOT")
    label, path = value.split("=", 1)
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=dataset_argument, action="append",
                        required=True, metavar="LABEL=OT_ROOT")
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for label, root in args.dataset:
        for path in sorted(root.glob(f"*/{args.scope}/*/pair_metrics.csv")):
            metrics = pd.read_csv(path)
            metrics = metrics.loc[metrics["method"].eq(args.method)]
            if metrics.empty:
                continue
            record = metrics.iloc[0]
            rows.append({
                "dataset": label,
                "pair_id": str(record.get("pair_id",
                                          path.parent.parent.parent.name)),
                "patient": str(record.get("patient_id", "")),
                "source_sample": str(record.get("source_sample", "")),
                "target_sample": str(record.get("target_sample", "")),
                "source_n": float(record.get("source_analyzed_n", np.nan)),
                "target_n": float(record.get("target_analyzed_n", np.nan)),
                "retained_fraction": 1.0 - float(
                    record.get("source_final_rejection_rate", np.nan)),
                "rejection_cost": float(record.get("rejection_cost", np.nan)),
                "transported_mass": float(record.get("transported_mass",
                                                     np.nan)),
            })
    if not rows:
        raise SystemExit("no pair_metrics.csv found under the given roots")
    pairs = pd.DataFrame(rows)
    pairs["source_key"] = pairs.patient + "__" + pairs.source_sample
    pd.set_option("display.width", 220)

    print("1. One primary sample against several metastases of the same "
          "patient\n")
    grouped = pairs.groupby(["dataset", "source_key"]).retained_fraction.agg(
        ["count", "min", "median", "max"])
    repeated = grouped[grouped["count"] > 1].copy()
    if repeated.empty:
        print("   no source sample appears in more than one pair; "
              "this question is not answerable on these roots\n")
    else:
        # Additive rather than multiplicative: a source retaining 0.005 in one
        # pair and 0.05 in another is a tenfold ratio and a 4.5-point
        # difference, and the ratio would make it the headline when it is not.
        repeated["spread"] = repeated["max"] - repeated["min"]
        repeated["fold"] = repeated["max"] / repeated["min"].clip(lower=1e-6)
        print(repeated.sort_values("spread", ascending=False).round(3)
              .to_string())
        print(f"\n   {len(repeated)} source samples appear in more than one "
              f"pair.")
        print(f"   median spread between their smallest and largest retained "
              f"fraction: {repeated.spread.median():.3f}")
        print(f"   spread exceeds 0.10 for "
              f"{repeated.spread.gt(0.10).mean():.0%} of them, and 0.25 for "
              f"{repeated.spread.gt(0.25).mean():.0%}")
        print("\n   A spread comparable to the retained fraction itself means "
              "the gate is\n   reporting the partner rather than the primary "
              "cells, and 'which primary\n   cells have metastatic potential' "
              "is not what the number answers.")

    print("\n2. Retained fraction against distance and against partner size\n")
    print(f"{'dataset':12s} {'pairs':>5s} {'rho(kept, cost)':>16s} {'p':>9s} "
          f"{'rho(kept, target n)':>20s} {'p':>9s}")
    for dataset, block in pairs.groupby("dataset", sort=True):
        usable = block.dropna(subset=["retained_fraction", "rejection_cost"])
        if len(usable) < 4:
            print(f"{dataset:12s} {len(usable):5d} {'too few':>16s}")
            continue
        first = spearmanr(usable.retained_fraction, usable.rejection_cost)
        second_frame = usable.dropna(subset=["target_n"])
        second = (spearmanr(second_frame.retained_fraction,
                            second_frame.target_n)
                  if len(second_frame) >= 4 else None)
        print(f"{dataset:12s} {len(usable):5d} {first.statistic:16.3f} "
              f"{first.pvalue:9.2g} "
              + (f"{second.statistic:20.3f} {second.pvalue:9.2g}"
                 if second is not None else f"{'too few':>20s}"))
    print("\n   `rejection_cost` is the calibrated threshold in units of the "
          "median\n   cross-side pair distance, so it is this run's measure of "
          "how far apart the\n   two sides are. A strong negative correlation "
          "is the section 5.8.4 pattern:\n   the method returns least where "
          "there is most divergence to study.")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.out, index=False)
        if not repeated.empty:
            repeated.to_csv(
                args.out.with_name(args.out.stem + "_by_source.csv"))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

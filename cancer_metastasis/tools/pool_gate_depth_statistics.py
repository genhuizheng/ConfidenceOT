"""Evaluate the depth acceptance test on cells pooled across pairs.

``25_diagnose_gate_covariates.py`` answers the question per pair and then
summarises those answers by their median, with a Wilcoxon test that needs at
least six of them. On a dataset with four pairs that estimator has nothing to
work with, and the per-pair answers themselves are not estimable either: the
depth AUC is an AUC of depth against the retained/rejected label, so its
standard error runs about ``sqrt(1/(12k))`` in ``k`` rejected cells, and
GSE181919's pairs hold 89 to 427 cells across both sides. At a 20% rejection
rate the distance between the null and the 0.60 acceptance threshold is one to
two standard errors on three of its four pairs.

Pooling the cells fixes the arithmetic: 810 cells give about 81 rejected and a
standard error near 0.032, which does resolve 0.10. It is the right estimator
for a small dataset and it is also the one that hides heterogeneity between
pairs, so the per-pair values are printed beside the pooled one -- not to be
interpreted, but so the spread a reader is being asked to set aside is visible.

The definitions are the ones in `25_diagnose_gate_covariates.py`: the same
Mann-Whitney AUC with ties averaged, the same Spearman correlation, the same
join of `cell_confidence.csv` onto `predownsample_depth.csv.gz` by sample and
observation. Depth is each cell's **pre-equalisation** depth, because after
equalisation the stored depth is nearly constant and an AUC against it is
trivially 0.5.

Usage:
  python cancer_metastasis/tools/pool_gate_depth_statistics.py
      --dataset GSE181919=.../ot_GSE181919_rank256_ds_cos_20260921
      --predownsample-depth .../downsampled_GSE181919_20260921/predownsample_depth.csv.gz
      --out pooled.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

REQUIRED = ("method", "side", "observation_id", "retained", "decision_cost")
# The thresholds of SEQUENCING_DEPTH_RESOLUTION.md, with one correction.
# The document wrote the AUC criterion one-sided, as `auc <= 0.60`, against observed
# values of 0.67 and 0.79. An AUC of 0.35 would clear that while tracking
# depth exactly as hard, in the other direction, so the criterion has to be
# the distance from the null. The bound is unchanged; only its two-sidedness
# is, and no measured value moves.
ACCEPTANCE = {"abs_auc_deviation": 0.10, "abs_spearman": 0.20}


def dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=OT_ROOT")
    label, path = value.split("=", 1)
    return label, Path(path)


def rank_auc(values: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUC with ties averaged, as 25_diagnose defines it."""
    numeric = np.asarray(values, dtype=np.float64)
    label = np.asarray(positive, dtype=bool)
    finite = np.isfinite(numeric)
    numeric, label = numeric[finite], label[finite]
    n_positive = int(label.sum())
    n_negative = int(numeric.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(numeric)
    return (float(ranks[label].sum())
            - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def load_depths(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = ["sample_id", "observation_id", "predownsample_total_counts"]
    missing = [column for column in required if column not in table]
    if missing:
        raise SystemExit(f"{path} is missing {missing}")
    table["sample_id"] = table["sample_id"].astype(str)
    table["observation_id"] = table["observation_id"].astype(str)
    # Same collision check as the per-pair script: a repeated barcode within
    # one site label would silently join the wrong depth.
    duplicated = table.duplicated(["sample_id", "observation_id"], keep=False)
    if duplicated.any():
        conflicts = (table[duplicated]
                     .groupby(["sample_id", "observation_id"])
                     ["predownsample_total_counts"].nunique())
        if (conflicts > 1).any():
            raise SystemExit(
                f"{path}: {int((conflicts > 1).sum())} keys carry conflicting "
                "depths; the same barcode appears in more than one patient's "
                "sample, so the join needs the patient too."
            )
    return table.drop_duplicates(["sample_id", "observation_id"])[required]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=dataset_argument, action="append",
                        required=True, metavar="LABEL=OT_ROOT")
    parser.add_argument("--predownsample-depth", type=Path, action="append",
                        required=True,
                        help="one per --dataset, in the same order")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--side", default="source",
                        choices=("source", "target"))
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if len(args.dataset) != len(args.predownsample_depth):
        raise SystemExit("--dataset and --predownsample-depth must pair up")

    rows, per_pair_rows = [], []
    for (label, root), depth_path in zip(args.dataset,
                                         args.predownsample_depth):
        depths = load_depths(depth_path)
        files = sorted(root.glob(f"*/{args.scope}/*/cell_confidence.csv"))
        if not files:
            print(f"{label}: no cell_confidence.csv under {root}")
            continue
        cells = []
        for path in files:
            table = pd.read_csv(path)
            missing = [c for c in REQUIRED if c not in table]
            if missing:
                raise SystemExit(f"{path} is missing {missing}")
            table = table.loc[table["method"].eq(args.method)
                              & table["side"].eq(args.side)]
            if table.empty:
                continue
            table = table.copy()
            table["pair_id"] = path.parent.parent.parent.name
            cells.append(table)
        if not cells:
            print(f"{label}: no {args.method}/{args.side} rows")
            continue
        joined = pd.concat(cells, ignore_index=True)
        joined["sample_id"] = joined["sample_id"].astype(str)
        joined["observation_id"] = joined["observation_id"].astype(str)
        joined = joined.merge(depths, on=["sample_id", "observation_id"],
                              how="left", validate="many_to_one")
        unmatched = int(joined["predownsample_total_counts"].isna().sum())
        if unmatched:
            print(f"{label}: {unmatched} of {len(joined)} cells found no "
                  f"pre-equalisation depth; they are excluded")
        joined = joined[joined["predownsample_total_counts"].notna()]

        # `retained` is the column the run writes; fall back to the gate flag.
        if "retained" in joined:
            retained = joined["retained"].astype(bool).to_numpy()
        elif "final_rejected" in joined:
            retained = ~joined["final_rejected"].astype(bool).to_numpy()
        else:
            raise SystemExit(
                f"{label}: cell_confidence.csv carries neither 'retained' nor "
                f"'final_rejected'; columns are {list(joined.columns)}"
            )
        depth = joined["predownsample_total_counts"].to_numpy(dtype=np.float64)
        rejected = ~retained
        k = int(rejected.sum())
        # `retained` is the positive class, matching
        # 25_diagnose_gate_covariates.py: AUC > 0.5 means retained cells carry
        # higher depth. Using `rejected` instead would mirror the value about
        # 0.5 and apply the bound to the wrong side.
        auc = rank_auc(depth, retained)
        rho = (float(spearmanr(joined["decision_cost"], depth).statistic)
               if len(joined) > 2 else float("nan"))
        standard_error = np.sqrt(1.0 / (12.0 * k)) if k else float("nan")
        verdict = "unread"
        if np.isfinite(auc) and np.isfinite(standard_error):
            deviation = abs(auc - 0.5)
            if standard_error > 0 and (ACCEPTANCE["abs_auc_deviation"]
                                       / standard_error) < 2.0:
                # The threshold sits inside two standard errors of the null, so
                # neither outcome would carry information.
                verdict = "not estimable"
            elif (deviation <= ACCEPTANCE["abs_auc_deviation"]
                  and abs(rho) <= ACCEPTANCE["abs_spearman"]):
                verdict = "pass"
            else:
                verdict = f"fail (deviation {deviation:.3f})"
        rows.append({
            "dataset": label,
            "pairs": joined["pair_id"].nunique(),
            "cells": len(joined),
            "rejected": k,
            "rejection_rate": float(rejected.mean()),
            "auc_predownsample_total_counts": auc,
            "auc_standard_error": standard_error,
            "auc_in_standard_errors": ((auc - 0.5) / standard_error
                                       if standard_error else float("nan")),
            "spearman_decision_cost_depth": rho,
            "median_depth_retained": float(np.median(depth[retained]))
            if retained.any() else float("nan"),
            "median_depth_rejected": float(np.median(depth[rejected]))
            if rejected.any() else float("nan"),
            "verdict": verdict,
        })
        for pair_id, block in joined.groupby("pair_id", sort=True):
            block_retained = (block["retained"].astype(bool).to_numpy()
                              if "retained" in block
                              else ~block["final_rejected"].astype(bool).to_numpy())
            block_depth = block["predownsample_total_counts"].to_numpy(
                dtype=np.float64)
            block_k = int((~block_retained).sum())
            per_pair_rows.append({
                "dataset": label, "pair_id": pair_id, "cells": len(block),
                "rejected": block_k,
                "auc": rank_auc(block_depth, block_retained),
                "standard_error": (np.sqrt(1.0 / (12.0 * block_k))
                                   if block_k else float("nan")),
            })

    if not rows:
        raise SystemExit("no dataset produced a pooled statistic")
    pooled = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print("Pooled across pairs, source side, depth from before equalisation\n")
    print(pooled.round(4).to_string(index=False))
    print(f"\nacceptance: |auc - 0.5| <= {ACCEPTANCE['abs_auc_deviation']} AND "
          f"|spearman| <= {ACCEPTANCE['abs_spearman']}. AUC > 0.5 means "
          f"retained cells are deeper, the direction every dataset showed. "
          f"'not estimable' means the bound sits within two standard errors of "
          f"0.5 even pooled, so neither outcome would carry information.")
    if per_pair_rows:
        # ASCII only: a section sign prints as mojibake on a console that is
        # not UTF-8, and a garbled caveat is a caveat nobody reads.
        print("\nPer pair, for the spread only. Where the standard error is "
              "large these are not verdicts, by the amendment in "
              "SEQUENCING_DEPTH_RESOLUTION.md:\n")
        print(pd.DataFrame(per_pair_rows).round(4).to_string(index=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pooled.to_csv(args.out, index=False)
        pd.DataFrame(per_pair_rows).to_csv(
            args.out.with_name(args.out.stem + "_per_pair.csv"), index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

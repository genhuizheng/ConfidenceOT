"""How much of the representation's leading axis is sequencing depth?

The depth spread is a property of the data. Whether it lands in the leading
principal component is a property of the representation, and that is the part
this project chose. Before the fix, `25_diagnose_gate_covariates.py` measured
`max_abs_spearman_pc_depth` = 0.60 on the old configuration: some principal
component tracked each cell's depth at rho 0.6, so the cost was partly a
distance along a depth axis.

Nobody has measured it under `rank256_ds_cos`, which is the configuration
chosen precisely to remove that. The individual UMAPs suggest it survived --
on SPECTRUM-OV-071 the axis separating the primary from the metastasis is
visibly the depth gradient, with the 6% retained cells sitting on the bridge
and 3.69x deeper than the rejected ones -- but a figure is not a number.

Measured against three covariates, because they fail differently:

* `predownsample_total_counts` -- the depth before equalisation. This is the
  covariate the gate must not track, since the equalised depth is nearly
  constant by construction.
* detected genes after equalisation -- **detection breadth**, which
  equalisation does not equalise. A cell subsampled from 20,000 counts to
  3,119 still has more non-zero genes than one that arrived at 3,500 and was
  left alone, so two cells at identical totals can differ here. On GSE180661
  the gate tracked original depth at AUC 0.557 among cells all sitting at
  exactly 3,119 counts, and this is the quantity that explains it.
* the equalised total, as a control. It should be near zero; if it is not, the
  equalisation did not do what it claims.

Reported per pair and per PC, from `joint_pca_coordinates.csv.gz` -- the run's
own coordinates, so this cannot disagree with the run about what the
representation was.

Usage:
  python cancer_metastasis/tools/audit_depth_axis.py
      --dataset GSE180661=.../ot_GSE180661_rank256_ds_cos_20260921
      --predownsample-depth .../downsampled_GSE180661_20260914/predownsample_depth.csv.gz
      --out depth_axis.csv
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


def safe_spearman(left, right) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left) & np.isfinite(right)
    if finite.sum() < 3:
        return float("nan")
    if np.ptp(left[finite]) == 0 or np.ptp(right[finite]) == 0:
        return float("nan")
    return float(spearmanr(left[finite], right[finite]).statistic)


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
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if len(args.dataset) != len(args.predownsample_depth):
        raise SystemExit("--dataset and --predownsample-depth must pair up")

    rows = []
    for (label, root), depth_path in zip(args.dataset,
                                         args.predownsample_depth):
        depths = pd.read_csv(depth_path)
        needed = {"sample_id", "observation_id", "predownsample_total_counts"}
        if not needed <= set(depths.columns):
            raise SystemExit(f"{depth_path} is missing "
                             f"{needed - set(depths.columns)}")
        depths["sample_id"] = depths["sample_id"].astype(str)
        depths["observation_id"] = depths["observation_id"].astype(str)
        depths = depths.drop_duplicates(["sample_id", "observation_id"])[
            ["sample_id", "observation_id", "predownsample_total_counts"]]

        for path in sorted(root.glob(
                f"*/{args.scope}/*/joint_pca_coordinates.csv.gz")):
            pair_id = path.parent.parent.parent.name
            frame = pd.read_csv(path)
            pcs = [c for c in frame.columns if c.startswith("PC")]
            if not pcs:
                continue
            frame["sample_id"] = frame["sample_id"].astype(str)
            frame["observation_id"] = frame["observation_id"].astype(str)
            frame = frame.merge(depths, on=["sample_id", "observation_id"],
                                how="left")
            # The QC table carries the equalised total and the detected-gene
            # count for the same cells, which is where detection breadth comes
            # from. Absent, that column is reported as missing rather than
            # silently skipped.
            qc_path = path.parent / "cell_qc.csv.gz"
            detected = equalised = None
            if qc_path.is_file():
                qc = pd.read_csv(qc_path)
                if "observation_id" in qc:
                    qc["observation_id"] = qc["observation_id"].astype(str)
                    keep = ["observation_id"]
                    for column in ("n_genes_by_counts", "total_counts"):
                        if column in qc:
                            keep.append(column)
                    qc = qc.drop_duplicates("observation_id")[keep]
                    frame = frame.merge(qc, on="observation_id", how="left")
                    detected = frame.get("n_genes_by_counts")
                    equalised = frame.get("total_counts")

            record = {"dataset": label, "pair_id": pair_id,
                      "cells": len(frame), "pcs": len(pcs)}
            for name, values in (("predownsample_depth",
                                  frame["predownsample_total_counts"]),
                                 ("detected_genes", detected),
                                 ("equalised_depth", equalised)):
                if values is None:
                    record[f"max_abs_rho_{name}"] = float("nan")
                    record[f"argmax_pc_{name}"] = ""
                    record[f"rho_pc1_{name}"] = float("nan")
                    continue
                correlations = {pc: safe_spearman(frame[pc], values)
                                for pc in pcs}
                finite = {pc: value for pc, value in correlations.items()
                          if np.isfinite(value)}
                if finite:
                    strongest = max(finite, key=lambda pc: abs(finite[pc]))
                    record[f"max_abs_rho_{name}"] = abs(finite[strongest])
                    record[f"argmax_pc_{name}"] = strongest
                else:
                    record[f"max_abs_rho_{name}"] = float("nan")
                    record[f"argmax_pc_{name}"] = ""
                record[f"rho_pc1_{name}"] = correlations.get("PC1",
                                                             float("nan"))
            rows.append(record)
            del frame

    if not rows:
        raise SystemExit("no joint_pca_coordinates.csv.gz found; the run needs "
                         "--save-pairing-edges for these to exist")
    pairs = pd.DataFrame(rows)
    pd.set_option("display.width", 240)

    print("Spearman between each principal component and each covariate, "
          "summarised per dataset.\n")
    print(f"{'dataset':12s} {'pairs':>5s} "
          f"{'max|rho| pre-eq depth':>22s} {'PC1':>7s} "
          f"{'max|rho| detected':>18s} {'max|rho| eq depth':>18s}")
    for dataset, block in pairs.groupby("dataset", sort=True):
        # An all-NaN control means the equalised total has no variance to
        # correlate against, which is the control passing rather than the
        # control missing. Printing NaN would read as the second.
        control = block.max_abs_rho_equalised_depth
        control_text = (f"{control.median():18.3f}" if control.notna().any()
                        else f"{'constant':>18s}")
        print(f"{dataset:12s} {len(block):5d} "
              f"{block.max_abs_rho_predownsample_depth.median():22.3f} "
              f"{block.rho_pc1_predownsample_depth.abs().median():7.3f} "
              f"{block.max_abs_rho_detected_genes.median():18.3f} "
              f"{control_text}")
    print("\n(medians over pairs)")
    print("\nThe number to compare against is 0.60, which is what "
          "25_diagnose_gate_covariates.py\nmeasured for "
          "max_abs_spearman_pc_depth on the *old* configuration. A value near "
          "0.6\nhere means rank + equalisation + cosine did not take depth out "
          "of the leading\naxes; a value near 0.1 means it did and the UMAP "
          "reading was wrong.")
    print("\nThe equalised-depth column is the control. It should be near "
          "zero, because\nevery cell was subsampled to one total; if it is "
          "not, the equalisation did not\ndo what it claims and nothing else "
          "in this table can be read.")

    high = pairs.nlargest(min(12, len(pairs)),
                          "max_abs_rho_predownsample_depth")
    print("\nWorst pairs by depth alignment:\n")
    print(high[["dataset", "pair_id", "cells",
                "max_abs_rho_predownsample_depth",
                "argmax_pc_predownsample_depth",
                "max_abs_rho_detected_genes"]].round(3).to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

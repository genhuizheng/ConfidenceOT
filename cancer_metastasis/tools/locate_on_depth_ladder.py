"""Place each real dataset on the depth ladder the screen's arms are indexed by.

The depth screen varies one quantity: ``sd(log depth)``, at 0.0, 0.3, 0.6 and
0.9. Every conclusion in ``SEQUENCING_DEPTH_RESOLUTION.md`` is therefore stated
per rung -- "the depth effect at moderate spread" -- and until now nobody had
measured which rung any real dataset sits on. §4c of that document lists this as
an open item. Without it the screen's numbers are a curve with no point marked
on it: a method that looks acceptable at 0.3 and fails at 0.9 has been neither
endorsed nor rejected for a dataset whose spread is unknown.

Reads the ``predownsample_depth.csv.gz`` that ``27_downsample_counts.py`` emits,
which carries each analysed cell's depth **before** equalisation. That is the
right column: after equalisation the stored depth is nearly constant, so its
spread says nothing about the data the gate had to cope with.

Reported per dataset and per sample. Per sample matters because a pair is two
samples and the gate sees them jointly, so a dataset whose samples sit on
different rungs is not on a rung at all -- and that case is called out rather
than averaged away.

``sd(log depth)`` and not the coefficient of variation: the screen draws depth
from ``lognormal(log(median), sigma)``, so sigma *is* the standard deviation of
the log, and that is the parameter to compare against. The CV of the raw depth
is reported beside it because it is what people usually quote, and for a
lognormal they differ (CV = sqrt(exp(sigma^2) - 1), which is 0.30 at sigma=0.3
but 1.09 at sigma=0.9).

Usage:
  python cancer_metastasis/tools/locate_on_depth_ladder.py
      --dataset GSE180661=.../downsampled_GSE180661_20260914
      --dataset GSE225857=.../downsampled_GSE225857_20260921
      --out depth_ladder.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# The rungs the screen's homogeneous arms are built at, as sd(log depth).
LADDER = {0.0: "cv0", 0.3: "low", 0.6: "mid", 0.9: "high"}


def dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "expected LABEL=PATH, where PATH holds predownsample_depth.csv.gz"
        )
    label, path = value.split("=", 1)
    return label, Path(path)


def nearest_rung(sigma: float) -> str:
    """The arm whose sigma is closest, and how far off it is.

    Named rather than interpolated: the screen measured four points, so a
    dataset between two of them has been measured at neither, and saying which
    it is nearest to is a pointer, not a result.
    """
    rung = min(LADDER, key=lambda value: abs(value - sigma))
    return f"{LADDER[rung]} ({rung:.1f}), off by {abs(rung - sigma):+.2f}"


def summarise(depths: np.ndarray) -> dict[str, float]:
    positive = depths[depths > 0]
    if positive.size < 2:
        return {}
    logged = np.log(positive)
    sigma = float(logged.std(ddof=1))
    return {
        "cells": int(positive.size),
        "zero_depth_cells": int(depths.size - positive.size),
        "median_depth": float(np.median(positive)),
        "sd_log_depth": sigma,
        # The standard error of a standard deviation is about sigma/sqrt(2n),
        # which at these cell counts is small enough that the rung is not in
        # doubt -- worth printing so that is visible rather than assumed.
        "sd_log_depth_se": sigma / np.sqrt(2.0 * positive.size),
        "cv_depth": float(positive.std(ddof=1) / positive.mean()),
        "p01_depth": float(np.quantile(positive, 0.01)),
        "p99_depth": float(np.quantile(positive, 0.99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=dataset_argument, action="append",
                        required=True, metavar="LABEL=EQUALISED_ROOT",
                        help="may be repeated")
    parser.add_argument(
        "--depth-column", default="predownsample_total_counts",
        help="Column holding each cell's pre-equalisation depth",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    for label, root in args.dataset:
        path = root if root.is_file() else root / "predownsample_depth.csv.gz"
        if not path.is_file():
            print(f"{label}: no {path}; skipped")
            continue
        frame = pd.read_csv(path)
        if args.depth_column not in frame.columns:
            print(f"{label}: {path} has no {args.depth_column!r}; columns are "
                  f"{list(frame.columns)}")
            continue
        depths = frame[args.depth_column].to_numpy(dtype=np.float64)
        whole = summarise(depths)
        if not whole:
            print(f"{label}: fewer than two cells with depth; skipped")
            continue
        rows.append({"dataset": label, "scope": "all samples", **whole})
        if "sample_id" in frame.columns:
            for sample, block in frame.groupby("sample_id", sort=True):
                values = block[args.depth_column].to_numpy(dtype=np.float64)
                per_sample = summarise(values)
                if per_sample:
                    rows.append({"dataset": label, "scope": str(sample),
                                 **per_sample})
    if not rows:
        raise SystemExit("no dataset yielded a depth distribution")

    table = pd.DataFrame(rows)
    table["nearest_arm"] = [nearest_rung(value) for value in table.sd_log_depth]
    pd.set_option("display.width", 220)
    show = ["dataset", "scope", "cells", "median_depth", "sd_log_depth",
            "sd_log_depth_se", "cv_depth", "nearest_arm"]
    print(table[show].round(4).to_string(index=False))

    print("\nThe screen's rungs, as sd(log depth): "
          + ", ".join(f"{name}={value:.1f}" for value, name in LADDER.items()))
    overall = table[table.scope.eq("all samples")]
    print()
    for _, row in overall.iterrows():
        samples = table[table.dataset.eq(row.dataset)
                        & ~table.scope.eq("all samples")]
        spread = ""
        if len(samples) > 1:
            low, high = samples.sd_log_depth.min(), samples.sd_log_depth.max()
            spread = (f"; its samples run {low:.2f} to {high:.2f}")
            if high - low > 0.3:
                # A third of a rung apart is not one rung. The gate sees two
                # samples jointly, so this dataset is not located by one number.
                spread += " -- more than a rung apart, so this dataset is not"
                spread += " on a single rung and the per-sample rows are the"
                spread += " ones to read"
        print(f"{row.dataset}: sd(log depth) = {row.sd_log_depth:.3f}, nearest "
              f"{row.nearest_arm}{spread}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

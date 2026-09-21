"""Collect the depth screen into one table, and say which configurations pass.

Reads every ``<root>/<label>/depth_null_arm_summary.csv`` the array wrote and
prints the two columns a configuration has to clear together: the depth effect
on the homogeneous arms, and the positive control. Either alone is satisfiable
by a method that does nothing -- a gate that rejects no cell has no depth
dependence and no power.

Usage:
  python scripts/collect_depth_screen.py /scratch/.../depth_screen
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace", line_buffering=True)

HOMOGENEOUS = ("homogeneous_depth_cv0", "homogeneous_depth_cv_low",
               "homogeneous_depth_cv_mid", "homogeneous_depth_cv_high")
CONTROL = "perturbed_depth_cv0"
SHORT = {"homogeneous_depth_cv0": "cv0", "homogeneous_depth_cv_low": "low",
         "homogeneous_depth_cv_mid": "mid", "homogeneous_depth_cv_high": "high"}
# Display order: ours first, then the external packages, each block reading
# untreated, equalised, equalised with the cosine cost. This is deliberately
# not the array's submission order, which has to stay append-only because the
# earlier indices have already run.
ORDER = ["logcpm", "logcpm_ds", "logcpm_cos",
         "rank256", "rank256_ds", "rank256_ds_cos",
         "pearson", "pearson_ds", "pearson_ds_cos",
         "scanpy_pearson", "scanpy_pearson_ds",
         "sct", "sct_ds", "sct_ds_cos"]

# The screen job appends a suffix to a label when a run means something other
# than the default, so two runs cannot overwrite each other. Parsing them back
# out stops a table listing `sct_ds` and `sct_ds_splatter` as two different
# methods, which they are not -- they are one method on two simulators.
# Matched against the known names rather than by regex, because base labels
# contain underscores themselves.
def split_variant(label: str) -> tuple[str, str]:
    """Return (base configuration, human-readable variant)."""
    for candidate in sorted(ORDER, key=len, reverse=True):
        if label == candidate:
            return candidate, ""
        if label.startswith(candidate + "_"):
            rest = label[len(candidate) + 1:]
            pieces = []
            for token in rest.split("_"):
                if token == "splatter":
                    pieces.append("splatter counts")
                elif token.startswith("r") and token[1:].isdigit():
                    pieces.append(f"{token[1:]} replicates")
                elif token.startswith("n") and token[1:].isdigit():
                    pieces.append(f"N={token[1:]}")
                elif token:
                    pieces.append(token)
            return candidate, ", ".join(pieces)
    return label, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="depth effect below this counts as cleared")
    parser.add_argument("--minimum-f1", type=float, default=0.60,
                        help="positive-control F1 above this counts as intact")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the two CSVs; defaults to the "
                             "root, but point it elsewhere when reading a "
                             "directory that is not ours to write into")
    args = parser.parse_args()
    out = args.out or args.root
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    for path in sorted(args.root.glob("*/depth_null_arm_summary.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "configuration", path.parent.name)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no depth_null_arm_summary.csv under {args.root}")

    arms = pd.concat(frames, ignore_index=True)
    arms.to_csv(out / "screen_all_arms.csv", index=False)
    if "auc_total_counts" not in arms:
        raise SystemExit("summaries carry no auc_total_counts column")
    arms["depth_effect"] = (arms["auc_total_counts"] - 0.5).abs()

    # Group by variant, and inside a variant keep the submission order, so
    # the default rows read as before and any comparison run reads beneath
    # them rather than interleaved.
    found = set(arms.configuration)
    order_index = {name: index for index, name in enumerate(ORDER)}
    def sort_key(label):
        base, variant = split_variant(label)
        return (variant, order_index.get(base, len(ORDER)), label)
    present = sorted(found, key=sort_key)
    missing = [c for c in ORDER if c not in found]

    effect = (arms[arms.arm.isin(HOMOGENEOUS)]
              .pivot_table(index="configuration", columns="arm",
                           values="depth_effect", aggfunc="median")
              .reindex(present)
              .rename(columns=SHORT))
    effect = effect[[c for c in ("cv0", "low", "mid", "high")
                     if c in effect.columns]]

    control = arms[arms.arm.eq(CONTROL)].set_index("configuration")
    for column in ("perturbed_f1", "perturbed_recall", "source_rejection_rate"):
        effect[column] = control[column].reindex(effect.index) \
            if column in control else pd.NA

    worst_columns = [c for c in ("cv0", "low", "mid", "high")
                     if c in effect.columns]
    worst = effect[worst_columns].max(axis=1)
    effect["worst_depth_effect"] = worst

    # The range over replicates on that same worst arm. Without it the pass
    # column reads as a property of the method, when at three replicates the
    # tolerance can fall inside the scatter -- two runs of one method landed
    # either side of it in the production screen.
    replicate_frames = []
    for path in sorted(args.root.glob("*/depth_null_replicates.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "configuration", path.parent.name)
        replicate_frames.append(frame)
    if replicate_frames:
        per_replicate = pd.concat(replicate_frames, ignore_index=True)
        per_replicate["depth_effect"] = \
            (per_replicate["auc_total_counts"] - 0.5).abs()
        long_to_short = {long: short for long, short in SHORT.items()}
        per_replicate["short"] = per_replicate.arm.map(long_to_short)
        spread = []
        for configuration in effect.index:
            arm = effect.loc[configuration, worst_columns].idxmax()
            points = per_replicate[per_replicate.configuration.eq(configuration)
                                   & per_replicate.short.eq(arm)].depth_effect
            spread.append(f"{points.min():.3f}-{points.max():.3f}"
                          if len(points) else "")
        effect["worst_arm_over_replicates"] = spread
    variants = [split_variant(name)[1] for name in effect.index]
    if any(variants):
        effect.insert(0, "variant", [v or "default" for v in variants])
    effect["passes"] = [
        "yes" if (w == w and w <= args.tolerance
                  and f == f and f >= args.minimum_f1) else "no"
        for w, f in zip(worst, effect.get("perturbed_f1", worst * float("nan")))
    ]

    pd.set_option("display.width", 220)
    print("depth effect |AUC - 0.5| by depth spread, lower is better; "
          "then the positive control\n")
    print(effect.round(3).to_string())
    print(f"\npasses = depth effect <= {args.tolerance} on every homogeneous "
          f"arm AND control F1 >= {args.minimum_f1}")
    if missing:
        print("\nnot present yet: " + ", ".join(missing))
    effect.to_csv(out / "screen_depth_effect.csv")
    print(f"\nwrote {out / 'screen_all_arms.csv'}")
    print(f"wrote {out / 'screen_depth_effect.csv'}")


if __name__ == "__main__":
    main()

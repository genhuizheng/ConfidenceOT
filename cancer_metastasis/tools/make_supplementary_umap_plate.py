"""Every pair's gate on one plate, for the supplementary material.

Draws from the frames `make_individual_gate_umap.py` exports, not from the
embeddings themselves, so this runs anywhere pandas and matplotlib do and a
change of layout costs seconds instead of the fifty-one minutes the 127 UMAPs
took. That separation is the point: heavy compute writes coordinates once on a
compute node, and the figure is redrawn locally as often as it needs to be.

Two colourings of the same panels, because the supplementary has to carry both
halves of the reading:

* ``gate`` -- retained and rejected primary cells over a metastatic backdrop.
* ``depth`` -- every cell by its pre-equalisation depth. Side by side with the
  gate plate, a reader can see for themselves which pairs split along their own
  depth gradient. On the 2026-09-21 run the leading axes still track detection
  breadth at rho 0.645 (GSE180661), so a gate plate published without its depth
  counterpart would be inviting a biological reading of something partly
  technical.

Panels are ordered by retained fraction rather than by name, so the range a
dataset spans is visible in the layout instead of having to be looked up: on
GSE180661 that range runs from 5% to 88% between pairs of the same patient.

127 panels do not fit one page at a readable size, so the plate pages. The page
count is reported, and `--columns` and `--rows` set the shape.

Usage:
  python cancer_metastasis/tools/make_supplementary_umap_plate.py
      FRAMES_DIR OUT_DIR --colour gate
      --colour depth --columns 8 --rows 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RETAINED = "#1f6f8b"
REJECTED = "#d1603d"
BACKDROP = "#d5d5cf"
# Full names for the accessions, so a supplementary plate does not require the
# reader to hold a lookup table.
TISSUE = {"GSE180661": "Ovarian", "GSE271675": "Prostate",
          "GSE225857": "Colorectal", "GSE181919": "Head and neck"}


def short_pair(pair_id: str, limit: int = 30) -> str:
    """Keep the end of a pair id: the samples, not the shared patient prefix."""
    return pair_id if len(pair_id) <= limit else "..." + pair_id[-(limit - 3):]


def marker_size(n: int, panel_inches: float) -> float:
    """Scale with both the crowd and the panel, since the plate shrinks both."""
    base = np.clip(2.4 * (4_000.0 / max(n, 1)) ** 0.35, 0.3, 10.0)
    return float(base * (panel_inches / 4.4) ** 2)


def load_frames(root: Path, datasets: list[str] | None) -> list[dict]:
    records = []
    for path in sorted(root.glob("*/*.csv.gz")):
        dataset = path.parent.name
        if datasets and dataset not in datasets:
            continue
        frame = pd.read_csv(path)
        needed = {"x", "y", "side", "retained"}
        if not needed <= set(frame.columns):
            print(f"{path}: missing {needed - set(frame.columns)}; skipped")
            continue
        primary = frame[frame.side.eq("primary")]
        records.append({
            "dataset": dataset,
            "pair_id": path.name[: -len(".csv.gz")],
            "frame": frame,
            "retained_fraction": (float(primary.retained.mean())
                                  if len(primary) else float("nan")),
            "cells": len(frame),
        })
    return records


def draw_panel(axis, frame: pd.DataFrame, colour: str, size: float,
               vmin: float, vmax: float):
    primary = frame[frame.side.eq("primary")]
    if colour == "gate":
        metastasis = frame[frame.side.eq("metastasis")]
        axis.scatter(metastasis.x, metastasis.y, s=size, c=BACKDROP,
                     linewidths=0)
        axis.scatter(primary[~primary.retained].x,
                     primary[~primary.retained].y, s=size, c=REJECTED,
                     linewidths=0)
        axis.scatter(primary[primary.retained].x, primary[primary.retained].y,
                     s=size, c=RETAINED, linewidths=0)
        return None
    if "depth" not in frame:
        axis.text(0.5, 0.5, "no depth", ha="center", va="center", fontsize=5,
                  color="#888880", transform=axis.transAxes)
        return None
    depth = frame.depth.to_numpy(dtype=np.float64)
    values = np.full(len(frame), np.nan)
    finite = np.isfinite(depth) & (depth > 0)
    values[finite] = np.log10(depth[finite])
    # One colour scale across the whole plate, so a panel's colour means the
    # same depth in every panel. Per-panel scaling would make a shallow pair
    # and a deep pair look identical, which is the comparison the plate is for.
    return axis.scatter(frame.x, frame.y, s=size, c=values, cmap="viridis",
                        vmin=vmin, vmax=vmax, linewidths=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("frames_dir", type=Path,
                        help="the `frames` directory the UMAP tool wrote")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--colour", action="append",
                        choices=("gate", "depth"), default=None,
                        help="may be repeated; defaults to both")
    parser.add_argument("--dataset", action="append", default=None,
                        help="restrict to these dataset labels, in this order")
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--rows", type=int, default=7)
    parser.add_argument("--panel-inches", type=float, default=1.7)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--separate-datasets", action="store_true",
                        help="Start a new page at each dataset instead of "
                             "packing them together. Clearer, and costs pages.")
    args = parser.parse_args()
    colours = args.colour or ["gate", "depth"]

    records = load_frames(args.frames_dir, args.dataset)
    if not records:
        raise SystemExit(f"no frames under {args.frames_dir}; run "
                         f"make_individual_gate_umap.py without --no-frames")
    order = (args.dataset if args.dataset
             else sorted({r["dataset"] for r in records}))
    records.sort(key=lambda r: (order.index(r["dataset"])
                                if r["dataset"] in order else len(order),
                                r["retained_fraction"]))

    # One depth scale for the whole plate; see draw_panel.
    everything = []
    for record in records:
        if "depth" in record["frame"]:
            values = record["frame"].depth.to_numpy(dtype=np.float64)
            everything.append(values[np.isfinite(values) & (values > 0)])
    if everything:
        pooled = np.log10(np.concatenate(everything))
        vmin, vmax = np.percentile(pooled, [1, 99])
    else:
        vmin, vmax = 0.0, 1.0

    per_page = args.columns * args.rows
    if args.separate_datasets:
        groups = [[r for r in records if r["dataset"] == name]
                  for name in order]
        groups = [g for g in groups if g]
    else:
        groups = [records]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for colour in colours:
        page = 0
        for group in groups:
            for start in range(0, len(group), per_page):
                chunk = group[start:start + per_page]
                page += 1
                # Reserved space in inches, then converted, because fixed
                # fractions only fit one page shape: at --rows 1 the same
                # 0.03 of the height that comfortably held a title on a
                # 13-inch page became 0.09 inches and the title, the caption
                # and the panel labels printed on top of each other.
                # The depth plate needs more foot than the gate plate: a colour
                # bar carries ticks and an axis label where a legend carries one
                # row of text, and at 0.50 inches the label printed off the page.
                top_inches = 0.92
                bottom_inches = 0.50 if colour == "gate" else 0.80
                height = (args.rows * args.panel_inches + top_inches
                          + bottom_inches)
                figure, axes = plt.subplots(
                    args.rows, args.columns,
                    figsize=(args.columns * args.panel_inches, height),
                    squeeze=False,
                )
                mappable = None
                for index, axis in enumerate(axes.ravel()):
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if index >= len(chunk):
                        axis.axis("off")
                        continue
                    record = chunk[index]
                    size = marker_size(record["cells"], args.panel_inches)
                    drawn = draw_panel(axis, record["frame"], colour, size,
                                       vmin, vmax)
                    mappable = mappable or drawn
                    tissue = TISSUE.get(record["dataset"], record["dataset"])
                    axis.set_title(
                        f"{tissue} {short_pair(record['pair_id'], 26)}\n"
                        f"{record['retained_fraction']:.0%} kept, "
                        f"{record['cells']:,} cells",
                        fontsize=4.6, linespacing=1.3)
                    for spine in axis.spines.values():
                        spine.set_color("#e2e2dc")
                if colour == "gate":
                    figure.legend(handles=[
                        Line2D([], [], marker="o", linestyle="",
                               color=RETAINED, label="primary, retained",
                               markersize=5),
                        Line2D([], [], marker="o", linestyle="",
                               color=REJECTED, label="primary, rejected",
                               markersize=5),
                        Line2D([], [], marker="o", linestyle="",
                               color=BACKDROP, label="metastasis",
                               markersize=5),
                    ], loc="lower center", ncol=3, frameon=False, fontsize=8,
                       bbox_to_anchor=(0.5, 0.10 / height))
                elif mappable is not None:
                    # Its own axes at a fixed place, rather than stealing space
                    # from the panel grid, so the grid geometry is the same on
                    # both plates and the two can be read against each other.
                    # Raised within the reserved foot rather than sitting at
                    # its floor: set_label prints below the bar axes, and at
                    # 0.30 inches the label's descenders fell off the page.
                    # Fixed page dimensions are kept instead of reaching for
                    # bbox_inches='tight', so every page of a multi-page plate
                    # is the same size.
                    bar_axes = figure.add_axes(
                        (0.30, 0.46 / height, 0.40, 0.09 / height))
                    bar = figure.colorbar(mappable, cax=bar_axes,
                                          orientation="horizontal")
                    bar.set_label("log10 depth before equalisation",
                                  fontsize=8)
                    bar.ax.tick_params(labelsize=7)
                # Caption under the title, not at the foot. At the foot it
                # competed with the legend and the colour bar for the same
                # strip and was clipped off the page on the depth plate, and
                # `wrap=True` on a figure text does not reflow reliably -- so
                # the lines are broken here and the space is reserved.
                caption = [
                    "Each panel is one primary-metastasis pair, embedded by "
                    "UMAP on that pair's own joint PCA: the geometry its gate "
                    "was computed in.",
                    "Ordered by retained fraction. Configuration "
                    "rank256_ds_cos.",
                ]
                figure.suptitle(f"Supplementary: per-pair {colour} "
                                f"(page {page})",
                                fontsize=12, fontweight="semibold",
                                y=1.0 - 0.20 / height)
                for line, text in enumerate(caption):
                    figure.text(0.5, 1.0 - (0.50 + line * 0.15) / height,
                                text, fontsize=7, color="#55554f",
                                ha="center")
                # One explicit layout for both colourings. tight_layout was
                # applied to only one of them, which is why they disagreed.
                figure.subplots_adjust(top=1.0 - top_inches / height,
                                       bottom=bottom_inches / height,
                                       left=0.01, right=0.99,
                                       hspace=0.45, wspace=0.08)
                out = args.output_dir / f"supplementary_umap_{colour}_p{page}.png"
                figure.savefig(out, dpi=args.dpi)
                plt.close(figure)
                written.append((out, len(chunk)))
                print(f"{out}  ({len(chunk)} panels)")

    print(f"\n{len(records)} pairs over {len(written)} page(s), "
          f"{args.columns}x{args.rows} per page.")
    table = pd.DataFrame([{k: r[k] for k in
                           ("dataset", "pair_id", "cells", "retained_fraction")}
                          for r in records])
    out = args.output_dir / "supplementary_umap_panel_order.csv"
    table.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

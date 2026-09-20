"""Figure 1: a schematic of ConfidenceOT.

A drawn diagram, not a plot. It says what the method is in one picture: two
samples go in, a cost is built in a shared space, the transport plan and a
binary gate on each side are fitted together rather than one after the other,
and the price of leaving a cell unmatched is read off a null instead of tuned.

Every statement in it was taken from the implementation:
  - the cost a pair pays is C_ij when both cells are gated in and a flat c when
    either is gated out (``_block_objective`` in _cpu_uot.py),
  - a cell is kept while its transport cost falls below c, scored over its
    current partners in M4-E and over the full entropic conditional in M4-R
    (the two branches of the gate coefficient),
  - c is the smallest value whose acceptance on two disjoint halves of one
    sample reaches the minimum (``within_side_null_costs`` and
    ``calibrate_confidence_cost``).

Usage:
  python scripts/make_method_schematic.py --out benchmark_results/journal_figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

C_SOURCE = "#0072B2"     # source side, and retained cells
C_TARGET = "#7f7f7f"     # target side
C_DROP = "#D55E00"       # rejected cells
C_PRICE = "#009E73"      # the rejection cost
C_EDGE = "#4d4d4d"
C_FILL = "#f4f4f4"
C_LOOP_FILL = "#eef4f9"


def box(axes, x0, y0, x1, y1, *, fill=C_FILL, edge=C_EDGE, lw=0.8, radius=1.6,
        z=2, dashed=False):
    patch = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=z,
        linestyle=(0, (2.5, 1.8)) if dashed else "solid",
    )
    axes.add_patch(patch)
    return patch


def label(axes, x, y, text, *, size=7.2, weight="normal", colour="black",
          ha="center", va="center", z=5, spacing=1.35):
    axes.text(x, y, text, fontsize=size, fontweight=weight, color=colour,
              ha=ha, va=va, zorder=z, linespacing=spacing)


def arrow(axes, start, end, *, colour=C_EDGE, lw=0.9, style="-|>",
          connection="arc3,rad=0", z=3, mutation=7):
    axes.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, color=colour, linewidth=lw,
        connectionstyle=connection, mutation_scale=mutation, zorder=z,
        shrinkA=0, shrinkB=0,
    ))


def cell_cloud(axes, centre, *, n, spread, colour, seed, size=3.2, z=4):
    rng = np.random.default_rng(seed)
    points = rng.normal(0.0, 1.0, size=(n, 2)) * spread + np.asarray(centre)
    axes.scatter(points[:, 0], points[:, 1], s=size, c=colour, linewidths=0,
                 zorder=z)
    return points


def draw(axes) -> None:
    axes.set_xlim(0, 100)
    axes.set_ylim(0, 100)
    axes.axis("off")

    # ---- inputs -----------------------------------------------------------
    box(axes, 1, 74, 21, 95)
    label(axes, 11, 92, "Primary", size=7.6, weight="bold")
    cell_cloud(axes, (11, 82), n=45, spread=(3.7, 2.3), colour=C_SOURCE,
               seed=1)
    box(axes, 1, 48, 21, 69)
    label(axes, 11, 66, "Metastasis", size=7.6, weight="bold")
    cell_cloud(axes, (11, 56), n=45, spread=(3.7, 2.3), colour=C_TARGET,
               seed=2)

    box(axes, 1, 24, 21, 42)
    label(axes, 11, 36.5, "Shared space", size=7.4, weight="bold")
    label(axes, 11, 30, "joint HVG,\nscale, joint PCA", size=6.6,
          colour="0.3")

    box(axes, 1, 4, 21, 19)
    label(axes, 11, 14.5, "Cost", size=7.4, weight="bold")
    label(axes, 11, 8.6, r"$C_{ij}$ between every pair", size=6.6, colour="0.3")

    arrow(axes, (11, 74), (11, 69.6), connection="arc3,rad=0")
    arrow(axes, (11, 48), (11, 42.4))
    arrow(axes, (11, 24), (11, 19.4))

    # ---- the joint fit ----------------------------------------------------
    box(axes, 30, 24, 68, 95, fill=C_LOOP_FILL, edge=C_SOURCE, lw=1.0,
        radius=2.2, z=1)
    label(axes, 49, 91.5, "Fitted together, not in sequence", size=7.8,
          weight="bold", colour=C_SOURCE)
    label(axes, 49, 87.2,
          "a cell is never scored against a plan that assumes it",
          size=6.4, colour="0.4")

    box(axes, 35, 68, 63, 85, fill="white")
    label(axes, 49, 80.5, "Transport plan", size=7.4, weight="bold")
    label(axes, 49, 74.5, "entropic, unbalanced,\ngiven the current gates",
          size=6.6, colour="0.3")

    box(axes, 35, 36, 63, 58, fill="white")
    label(axes, 49, 54, "Binary gate, each side", size=7.4, weight="bold")
    label(axes, 49, 47.5,
          "keep a cell while it costs\nless to transport than the\nprice of "
          "leaving it out", size=6.6, colour="0.3")
    label(axes, 49, 39.5, r"keep $i$  $\Leftrightarrow$  cost$_i < c$",
          size=7.2, colour=C_PRICE)

    arrow(axes, (37.5, 68), (37.5, 58.4), connection="arc3,rad=0.42",
          colour=C_SOURCE, lw=1.0)
    arrow(axes, (60.5, 58.4), (60.5, 68), connection="arc3,rad=0.42",
          colour=C_SOURCE, lw=1.0)
    # One word each: a two-line caption is wider than the gutter between the
    # inner box and the loop box, and ran onto the gate's border. Two curved
    # arrows between two named boxes need no more than this.
    label(axes, 32.5, 63.2, "score", size=6.4, colour=C_SOURCE)
    label(axes, 65.5, 63.2, "re-fit", size=6.4, colour=C_SOURCE)

    arrow(axes, (21, 11.5), (24, 11.5), style="-")
    arrow(axes, (24, 11.5), (24, 76.5), colour=C_EDGE, style="-")
    arrow(axes, (24, 76.5), (34.6, 76.5), colour=C_EDGE)

    # ---- where the price comes from ---------------------------------------
    box(axes, 27, 1, 71, 21, dashed=True, fill="white", edge=C_PRICE)
    label(axes, 49, 17, "The price $c$ is read off a null, not chosen",
          size=7.2, weight="bold", colour=C_PRICE)
    label(axes, 49, 8.5,
          "split one sample in half — neither half can\n"
          "contain an incompatible cell — and take the\n"
          "smallest $c$ that accepts them",
          size=6.5, colour="0.3")
    arrow(axes, (49, 21), (49, 35.6), colour=C_PRICE, lw=1.0)

    # ---- output -----------------------------------------------------------
    box(axes, 77, 40, 99, 80)
    label(axes, 88, 76, "Every cell labelled", size=7.6, weight="bold")
    cell_cloud(axes, (86, 63), n=48, spread=(3.4, 3.6), colour=C_SOURCE,
               seed=7)
    cell_cloud(axes, (94, 53), n=13, spread=(1.5, 1.7), colour=C_DROP, seed=8)
    label(axes, 85, 46, "retained", size=6.8, colour=C_SOURCE)
    label(axes, 94, 44.5, "rejected", size=6.8, colour=C_DROP)
    arrow(axes, (68, 60), (76.4, 60))

    label(axes, 88, 35.5,
          "a rejection is an estimate\nof having no counterpart,\n"
          "not a fixed quota",
          size=6.5, colour="0.35")


def build(out: Path) -> None:
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(6.5, 3.6))
        axes = figure.add_axes([0, 0, 1, 1])
        draw(axes)
        out.mkdir(parents=True, exist_ok=True)
        stem = "fig_confidenceot_schematic"
        for extension in ("pdf", "png"):
            figure.savefig(out / f"{stem}.{extension}")
        plt.close(figure)
    print(f"wrote {out / (stem + '.pdf')}")
    print(f"wrote {out / (stem + '.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark_results/journal_figure"))
    build(parser.parse_args().out)


if __name__ == "__main__":
    main()

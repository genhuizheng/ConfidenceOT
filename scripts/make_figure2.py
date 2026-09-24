"""Figure 2 and its supplementary, as one composite and as separate panels.

The benchmark panels are imported from ``make_journal_figure`` rather than
rewritten, so the two scripts cannot disagree about what (a) to (d) show. What
is new is the preprocessing ablation, which replaced a thirteen-row list of
transforms with a 2x2x2 factorial, and a schematic that carries the measured
effect of each factor rather than illustrating one.

The ablation's result is what the schematic exists to state: the three
operations are not three attempts at one job. Each removes a different thing,
and the numbers separate by an order of magnitude on the column that belongs to
it and barely move on the others.

  read equalisation  ->  the total-count axis      0.97 -> 0.07
  rank encoding      ->  the detected-gene axis    0.91 -> 0.08
  cosine cost        ->  power                     0.86 -> 0.96

Every panel is also written on its own, because the schematic and the ablation
get used in talks and in the supplementary where the composite does not fit.
Each is drawn twice from one function rather than cropped out of the composite,
so a standalone panel has its own margins and reads at its own size.

Usage:
  python scripts/make_figure2.py
      --benchmark-root benchmark_results/splatter_20260921
      --ablation-root benchmark_results/ablation_20260923
      --out benchmark_results/figure2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_journal_figure import (  # noqa: E402
    STYLE, load_benchmark, panel_f1, panel_runtime, panel_specificity,
)

# The eight cells of the factorial, in the order that reads as an ablation:
# nothing, then one factor at a time, then the pairs, then all three.
ABLATION = ["logcpm", "rank256", "logcpm_ds", "logcpm_cos",
            "rank256_ds", "rank256_cos", "logcpm_ds_cos", "rank256_ds_cos"]
EXTRA = ["rank256_ds_cos_dr10", "rank256_ds_cos_dr25",
         "rank256_rg-genes_ds_cos", "logcpm_rg-genes_cos"]
# Full names rather than a column of filled/open circles beside the axis. The
# circles were drawn outside the axes with clip_on=False, which survives a
# standalone panel and is cropped away inside a gridspec -- leaving eight rows
# labelled only "log CPM" or "rank" with no way to tell which carried what.
PRETTY = {"logcpm": "log CPM",
          "rank256": "rank",
          "logcpm_ds": "log CPM + DS",
          "logcpm_cos": "log CPM + cos",
          "rank256_ds": "rank + DS",
          "rank256_cos": "rank + cos",
          "logcpm_ds_cos": "log CPM + DS + cos",
          "rank256_ds_cos": "rank + DS + cos",
          "rank256_ds_cos_dr10": "rank + DS + cos, gene filter 10%",
          "rank256_ds_cos_dr25": "rank + DS + cos, gene filter 25%",
          "rank256_rg-genes_ds_cos": "rank + DS + cos, regress $n_{genes}$",
          "logcpm_rg-genes_cos": "log CPM + cos, regress $n_{genes}$"}

C_COUNTS = "#b07d2b"
C_GENES = "#1f6f8b"
C_POWER = "#2f7d4f"
C_RULE = "#8c8c86"
# Built from a chr() call rather than written as a escape, because this
# file has been rewritten through shells that ate the backslash-r twice.
ARROW = "$" + chr(92) + "rightarrow$"


def load_ablation(root: Path) -> pd.DataFrame:
    path = root / "screen_ablation.csv"
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run collect_depth_screen.py "
                         f"on the screen root first")
    return pd.read_csv(path).set_index("configuration")


def panel_schematic(axes) -> None:
    """What each of the three operations removes, with the measured numbers.

    Drawn rather than plotted, because the point is the correspondence between
    an operation and an axis, and that correspondence is what the ablation
    established. The numbers on it are from the ablation, so this is a
    statement of the result and not an illustration of an intention.

    Geometry in data units, with the clouds kept narrower than the row pitch.
    The first attempt drew them at a spread wider than the gap between rows,
    so the three rows overlapped into one smear and the cosine circle fell off
    the bottom of the axes.
    """
    axes.set_xlim(0, 11)
    axes.set_ylim(0.15, 3.45)
    axes.axis("off")

    rng = np.random.default_rng(3)
    n = 80
    long_spread, short_spread = 0.16, 0.055      # +-3 sigma = 0.96 and 0.33
    pitch = 1.15                                  # comfortably more than 0.96

    def cloud(x0, y0, spread_x, spread_y, colour_by=None):
        x = rng.normal(0.0, spread_x, size=n)
        y = rng.normal(0.0, spread_y, size=n)
        if colour_by == "counts":
            colour = plt.get_cmap("YlOrBr")(0.35 + 0.5 * _unit(x))
        elif colour_by == "genes":
            colour = plt.get_cmap("PuBu")(0.35 + 0.5 * _unit(y))
        else:
            colour = "#9a9a94"
        axes.scatter(x0 + x, y0 + y, s=2.4, c=colour, linewidths=0, zorder=3)

    def _unit(values):
        span = np.ptp(values)
        return (values - values.min()) / (span if span else 1.0)

    def arrow(y):
        axes.add_patch(FancyArrowPatch(
            (4.25, y), (5.45, y), arrowstyle="-|>", mutation_scale=7,
            color="#5c5c56", linewidth=0.9, zorder=4))

    def caption(y, title, subtitle, colour, metric, before, after):
        axes.text(0.0, y + 0.20, title, fontsize=7.4,
                  fontweight="semibold", color=colour, ha="left", va="center")
        axes.text(0.0, y - 0.04, subtitle, fontsize=6.2, color="#55554f",
                  ha="left", va="center")
        axes.text(7.25, y + 0.16, metric, fontsize=6.2, color="#55554f",
                  ha="left", va="center")
        axes.text(7.25, y - 0.14, f"{before}  " + ARROW + f"  {after}",
                  fontsize=7.6, color=colour, ha="left",
                  va="center", fontweight="semibold")

    top = 2.95
    # Row 1: the total-count axis, horizontal, collapsed by equalisation.
    caption(top, "read equalisation", "removes the total-count axis",
            C_COUNTS, "axis vs total counts", "0.97", "0.07")
    cloud(3.30, top, long_spread, short_spread, "counts")
    arrow(top)
    cloud(6.35, top, short_spread, short_spread)

    # Row 2: the detected-gene axis, vertical, collapsed by rank encoding.
    middle = top - pitch
    caption(middle, "rank encoding", "removes the detected-gene axis",
            C_GENES, "axis vs detected genes", "0.91", "0.08")
    cloud(3.30, middle, short_spread, long_spread, "genes")
    arrow(middle)
    cloud(6.35, middle, short_spread, short_spread)

    # Row 3 is different in kind: cosine removes no axis, it changes what
    # distance means, so it is drawn as a projection rather than a collapse.
    bottom = middle - pitch
    caption(bottom, "cosine cost", "keeps the axes, measures direction",
            C_POWER, "power, planted 20%", "0.86", "0.96")
    # A fan centred on this row, not a quarter arc springing from it: the
    # first version pointed up and to the right, which put it within a hair of
    # the row above and read as part of that row's cloud.
    radius = 0.42
    angles = rng.uniform(-0.62, 0.62, size=24)
    lengths = rng.uniform(0.45, 1.0, size=24) * radius
    axes.scatter(3.30 + lengths * np.cos(angles),
                 bottom + lengths * np.sin(angles),
                 s=2.4, c="#9a9a94", linewidths=0, zorder=3)
    for angle in angles[:9]:
        axes.plot([3.30, 3.30 + radius * np.cos(angle)],
                  [bottom, bottom + radius * np.sin(angle)],
                  color=C_POWER, linewidth=0.4, alpha=0.5, zorder=1)
    arc = np.linspace(-0.66, 0.66, 60)
    axes.plot(3.30 + radius * np.cos(arc), bottom + radius * np.sin(arc),
              color=C_POWER, linewidth=0.9, zorder=2)
    arrow(bottom)
    cloud(6.35, bottom, short_spread, short_spread)

    axes.set_title("(e)  Each operation removes a different thing",
                   loc="left", fontsize=8.4, fontweight="semibold", pad=4)


def panel_ablation(axes, ablation: pd.DataFrame, order: list[str],
                   title: str) -> None:
    """The factorial, with the metric that belongs to each factor beside it.

    Three columns rather than one, because a single score would hide that the
    factors do different jobs: a configuration can clear the depth axis and
    leave the gene axis untouched, and only showing both makes that visible.
    ``f1_deep`` is with them because specificity without power is free -- log
    CPM keeps its specificity ranking while its F1 falls from 0.88 to 0.25 once
    depth is spread.
    """
    present = [name for name in order if name in ablation.index]
    y_positions = np.arange(len(present))[::-1]
    axes.set_ylim(-0.75, len(present) - 0.25)
    axes.set_yticks(y_positions)
    axes.set_yticklabels([PRETTY.get(name, name) for name in present],
                         fontsize=6.8)
    axes.set_xlim(0, 1.0)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.tick_params(axis="x", labelsize=6.8)
    axes.grid(axis="x", color="#e6e6e0", linewidth=0.6)
    axes.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axes.spines[spine].set_visible(False)

    for y, name in zip(y_positions, present):
        row = ablation.loc[name]
        axes.axhline(y, color="#eeeee8", linewidth=0.7, zorder=0)
        axes.plot([row.axis_vs_counts], [y], marker="o", markersize=5,
                  color=C_COUNTS, linestyle="none", zorder=4)
        axes.plot([row.axis_vs_genes], [y], marker="s", markersize=4.6,
                  color=C_GENES, linestyle="none", zorder=4)
        axes.plot([row.f1_deep], [y], marker="D", markersize=4.2,
                  color=C_POWER, linestyle="none", zorder=4)

    axes.set_xlabel("left two: leading-axis correlation with the covariate "
                    "(lower is better)\nright: F1 at sd(log depth) 0.9 "
                    "(higher is better)", fontsize=6.6)
    # Short labels and a small face: the first version ran "power at high
    # depth spread" off the right edge of the panel.
    axes.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=C_COUNTS,
               label="axis vs counts", markersize=4.6),
        Line2D([], [], marker="s", linestyle="", color=C_GENES,
               label="axis vs genes", markersize=4.2),
        Line2D([], [], marker="D", linestyle="", color=C_POWER,
               label="power, deep", markersize=3.8),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False,
        fontsize=6.0, handletextpad=0.3, columnspacing=0.9)
    axes.set_title(title, loc="left", fontsize=8.4, fontweight="semibold",
                   pad=22)


def save_panel(name: str, out: Path, draw, size: tuple[float, float]) -> None:
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=size)
        axes = figure.add_subplot(111)
        draw(axes)
        figure.tight_layout()
        for suffix in ("png", "pdf"):
            figure.savefig(out / f"{name}.{suffix}", dpi=400)
        plt.close(figure)
    print(f"  {out / (name + '.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    f1, rejection, runtime = load_benchmark(args.benchmark_root)
    ablation = load_ablation(args.ablation_root)

    draws = {
        "panel_a_f1_no_batch": (
            lambda ax: panel_f1(ax, f1, "none", "(a)  No batch effect", True),
            (3.4, 2.5)),
        "panel_b_f1_mild_batch": (
            lambda ax: panel_f1(ax, f1, "mild", "(b)  Mild batch effect",
                                False), (3.4, 2.5)),
        "panel_c_specificity": (
            lambda ax: panel_specificity(ax, rejection), (3.4, 2.7)),
        "panel_d_runtime": (lambda ax: panel_runtime(ax, runtime), (3.4, 2.7)),
        "panel_e_schematic": (panel_schematic, (6.6, 2.6)),
        "panel_f_ablation": (
            lambda ax: panel_ablation(ax, ablation, ABLATION,
                                      "(f)  Preprocessing ablation"),
            (5.0, 3.1)),
        "supplementary_ablation_all": (
            lambda ax: panel_ablation(ax, ablation, ABLATION + EXTRA,
                                      "Supplementary: every configuration"),
            (5.6, 4.1)),
    }

    print("individual panels:")
    for name, (draw, size) in draws.items():
        save_panel(name, args.out, draw, size)

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.1, 9.6))
        # hspace at 0.70 left a band of white above (e) and (f) as tall as
        # the panels themselves, because the ratio is of the *row* height and
        # the lower rows are the tall ones.
        grid = figure.add_gridspec(
            4, 2, height_ratios=[2.4, 2.6, 2.5, 3.2],
            hspace=0.44, wspace=0.30,
            left=0.16, right=0.98, top=0.97, bottom=0.05)
        panel_f1(figure.add_subplot(grid[0, 0]), f1, "none",
                 "(a)  No batch effect", legend=True)
        panel_f1(figure.add_subplot(grid[0, 1]), f1, "mild",
                 "(b)  Mild batch effect", legend=False)
        panel_specificity(figure.add_subplot(grid[1, 0]), rejection)
        panel_runtime(figure.add_subplot(grid[1, 1]), runtime)
        panel_schematic(figure.add_subplot(grid[2, :]))
        panel_ablation(figure.add_subplot(grid[3, :]), ablation, ABLATION,
                       "(f)  Preprocessing ablation")
        for suffix in ("png", "pdf"):
            figure.savefig(args.out / f"figure2.{suffix}", dpi=400)
        plt.close(figure)
    print(f"\ncomposite: {args.out / 'figure2.png'}")


if __name__ == "__main__":
    main()

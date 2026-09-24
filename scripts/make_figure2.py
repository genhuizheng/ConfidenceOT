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
C_PLANT = "#c4553b"
# Built from a chr() call rather than written as a escape, because this
# file has been rewritten through shells that ate the backslash-r twice.
ARROW = "$" + chr(92) + "rightarrow$"


def load_ablation(root: Path) -> pd.DataFrame:
    path = root / "screen_ablation.csv"
    if not path.is_file():
        raise SystemExit(f"{path} is missing; run collect_depth_screen.py "
                         f"on the screen root first")
    return pd.read_csv(path).set_index("configuration")


def panel_validation(axes) -> None:
    """How the ground truth is built, which is what makes the rest readable.

    Every other panel reports whether a method got the answer right, and
    nothing in the figure says where the answer comes from. It comes from a
    construction, not from a simulator parameter: both sides of a pair are
    independent draws from one population, so in the homogeneous arms no cell
    has a counterpart it lacks and the correct rejection rate is zero by
    design. A method that rejects there has failed, and no biological reading
    can rescue it.

    The nuisance is drawn once, below both rows, because applying it to one
    side only would make it a batch effect between samples -- which is a
    different problem, and one that existing batch correction addresses. Here
    both sides carry the same spread, and the question is whether a gate asked
    "which source cells have a counterpart" answers with the nuisance instead.
    """
    axes.set_xlim(0, 12)
    axes.set_ylim(-1.05, 4.5)
    axes.axis("off")
    rng = np.random.default_rng(5)

    def cells(x0, y0, n=26, planted=0, spread=0.30):
        angle = rng.uniform(0, 2 * np.pi, size=n)
        radius = np.sqrt(rng.uniform(0, 1, size=n)) * spread
        x, y = x0 + radius * np.cos(angle) * 2.0, y0 + radius * np.sin(angle)
        keep = np.ones(n, dtype=bool)
        if planted:
            keep[rng.choice(n, planted, replace=False)] = False
        axes.scatter(x[keep], y[keep], s=5.0, c="#9a9a94", linewidths=0,
                     zorder=3)
        if planted:
            axes.scatter(x[~keep], y[~keep], s=12.0, c=C_PLANT, linewidths=0,
                         zorder=4)

    def arrow(x0, x1, y):
        axes.add_patch(FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7,
            color="#5c5c56", linewidth=0.9, zorder=4))

    for x, header in ((1.05, "one population"),
                      (4.15, "two independent draws"),
                      (7.75, "correct answer"), (10.5, "measures")):
        axes.text(x, 4.30, header, fontsize=6.6, color="#3c3c38",
                  ha="center", fontweight="semibold")
    # Named once here rather than under every row: the columns do not change
    # between the rows, and repeating them costs the vertical room that the
    # composite has least of.
    axes.text(3.35, 3.98, "source", fontsize=5.8, color="#55554f", ha="center")
    axes.text(4.95, 3.98, "target", fontsize=5.8, color="#55554f", ha="center")

    for row, (label, planted, answer, measures) in enumerate((
            ("homogeneous -- both sides from the same population", 0,
             "reject nothing", "specificity"),
            ("perturbed -- red cells added to the source only", 6,
             "reject exactly those", "power"))):
        y = 3.20 - row * 1.35
        cells(1.05, y, spread=0.34)
        arrow(1.95, 2.75, y)
        axes.text(4.25, y + 0.45, label, fontsize=6.2, color="#55554f",
                  ha="center", style="italic")
        cells(3.35, y, planted=planted)
        cells(4.95, y)
        arrow(5.75, 6.55, y)
        axes.text(7.75, y, answer, fontsize=7.0, color="#3c3c38",
                  ha="center", va="center")
        arrow(9.05, 9.75, y)
        axes.text(10.5, y, measures, fontsize=7.4, color=C_POWER,
                  ha="center", va="center", fontweight="semibold")

    axes.plot([0.35, 11.65], [1.35, 1.35], color="#d8d8d2", linewidth=0.8)
    axes.text(0.35, 1.06, "both sides then carry the same nuisance:",
              fontsize=6.4, color="#3c3c38", ha="left", fontweight="semibold")
    axes.text(0.55, 0.70,
              "depth spread, sd(log depth) 0 to 0.9 -- panels (a) to (d), (f) left",
              fontsize=6.2, color=C_COUNTS, ha="left")
    axes.text(0.55, 0.34,
              "detection breadth at fixed total counts -- panel (f), hollow squares",
              fontsize=6.2, color=C_GENES, ha="left")
    axes.text(7.5, 0.70, "applied to one side only it would be a batch",
              fontsize=5.8, color="#77776f", ha="left")
    axes.text(7.5, 0.34, "effect, which is a different problem",
              fontsize=5.8, color="#77776f", ha="left")

    # What the two rows above correspond to in the study the method exists for.
    # Without this the panel explains a construction and leaves the reader to
    # guess what it stands in for, and the guess that matters is the wrong one:
    # the gate is a geometric filter against one named lesion, not a test of
    # whether a cell can metastasise. Saying so here is cheaper than saying it
    # in a caption nobody reads beside the figure.
    axes.plot([0.35, 11.65], [0.02, 0.02], color="#d8d8d2", linewidth=0.8)
    axes.text(0.35, -0.29, "what this stands in for:", fontsize=6.4,
              color="#3c3c38", ha="left", fontweight="semibold")
    axes.text(0.55, -0.63,
              "source = one patient's primary tumour   target = that patient's "
              "matched metastasis",
              fontsize=6.2, color="#3c3c38", ha="left")
    axes.text(0.55, -0.94,
              "retained = primary cells closer to the metastasis than to the "
              "primary's own spread",
              fontsize=6.2, color=C_POWER, ha="left")
    axes.text(7.9, -0.94,
              "a filter, not a test of metastatic competence",
              fontsize=5.8, color="#77776f", ha="left", style="italic")

    axes.set_title("(e)  How the ground truth is built",
                   loc="left", fontsize=8.4, fontweight="semibold", pad=4)


def panel_ablation(axes, ablation: pd.DataFrame, order: list[str],
                   title: str) -> None:
    """The factorial, with the metric that belongs to each factor beside it.

    Four columns rather than one, because a single score would hide that the
    factors do different jobs: a configuration can clear the depth axis and
    leave the gene axis untouched, and only showing both makes that visible.
    ``f1_deep`` is with them because specificity without power is free -- log
    CPM keeps its specificity ranking while its F1 falls from 0.88 to 0.25 once
    depth is spread.

    The fourth column is the one the round exists for. ``axis_vs_genes`` is
    measured where detection breadth rides on depth, and the full model clears
    it at 0.085. ``axis_vs_genes_split`` is the same correlation where total
    counts are held at 3,119 and breadth varies on its own, and there the full
    model sits at 0.730. Plotting only the first would show three solved
    problems and hide that the configuration does not act on breadth at all
    once the totals are equalised -- which is what the real data's 0.645 is.
    The two are drawn as a filled and a hollow square of the same colour,
    because they are one quantity under two regimes rather than two quantities.
    """
    present = [name for name in order if name in ablation.index]
    y_positions = np.arange(len(present))[::-1]
    axes.set_ylim(-0.85, len(present) - 0.15)
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

    # Three sub-levels within each row. The series collide at real values --
    # rank+DS puts the count and gene axes both near 0.08, and rank puts the
    # fixed-count square within 0.008 of the power diamond -- so without the
    # offset two different measurements render as one marker.
    OFFSET_COUNTS, OFFSET_GENES, OFFSET_POWER = 0.17, 0.0, -0.21
    for y, name in zip(y_positions, present):
        row = ablation.loc[name]
        axes.axhline(y, color="#eeeee8", linewidth=0.7, zorder=0)
        axes.plot([row.axis_vs_counts], [y + OFFSET_COUNTS], marker="o",
                  markersize=5, color=C_COUNTS, linestyle="none", zorder=4)
        axes.plot([row.axis_vs_genes], [y + OFFSET_GENES], marker="s",
                  markersize=4.6, color=C_GENES, linestyle="none", zorder=4)
        if "axis_vs_genes_split" in row.index and row.axis_vs_genes_split == row.axis_vs_genes_split:
            # Hollow, and joined to its filled twin by a hairline, so the eye
            # reads the gap as one measurement moving between two regimes.
            axes.plot([row.axis_vs_genes, row.axis_vs_genes_split],
                      [y + OFFSET_GENES] * 2, color=C_GENES, linewidth=0.7,
                      alpha=0.45, zorder=3)
            axes.plot([row.axis_vs_genes_split], [y + OFFSET_GENES],
                      marker="s", markersize=4.6, markerfacecolor="white",
                      markeredgecolor=C_GENES, markeredgewidth=1.0,
                      linestyle="none", zorder=4)
        axes.plot([row.f1_deep], [y + OFFSET_POWER], marker="D",
                  markersize=4.2, color=C_POWER, linestyle="none", zorder=4)

    axes.set_xlabel("circles and squares: leading-axis correlation with the "
                    "covariate (lower is better)\ndiamonds: F1 at sd(log "
                    "depth) 0.9 (higher is better)", fontsize=6.6)
    # Short labels and a small face: the first version ran "power at high
    # depth spread" off the right edge of the panel.
    axes.legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=C_COUNTS,
               label="axis vs counts", markersize=4.6),
        Line2D([], [], marker="s", linestyle="", color=C_GENES,
               label="axis vs genes", markersize=4.2),
        Line2D([], [], marker="s", linestyle="", markerfacecolor="white",
               markeredgecolor=C_GENES, markeredgewidth=1.0, color=C_GENES,
               label="same, at fixed counts", markersize=4.2),
        Line2D([], [], marker="D", linestyle="", color=C_POWER,
               label="power, deep", markersize=3.8),
    ], loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False,
        fontsize=6.0, handletextpad=0.3, columnspacing=0.8)
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
        "panel_e_validation": (panel_validation, (6.9, 3.1)),
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
        figure = plt.figure(figsize=(7.1, 10.2))
        # hspace at 0.70 left a band of white above (e) and (f) as tall as
        # the panels themselves, because the ratio is of the *row* height and
        # the lower rows are the tall ones.
        grid = figure.add_gridspec(
            4, 2, height_ratios=[2.4, 2.6, 3.7, 3.2],
            hspace=0.44, wspace=0.30,
            left=0.16, right=0.98, top=0.97, bottom=0.05)
        panel_f1(figure.add_subplot(grid[0, 0]), f1, "none",
                 "(a)  No batch effect", legend=True)
        panel_f1(figure.add_subplot(grid[0, 1]), f1, "mild",
                 "(b)  Mild batch effect", legend=False)
        panel_specificity(figure.add_subplot(grid[1, 0]), rejection)
        panel_runtime(figure.add_subplot(grid[1, 1]), runtime)
        panel_validation(figure.add_subplot(grid[2, :]))
        panel_ablation(figure.add_subplot(grid[3, :]), ablation, ABLATION,
                       "(f)  Preprocessing ablation")
        for suffix in ("png", "pdf"):
            figure.savefig(args.out / f"figure2.{suffix}", dpi=400)
        plt.close(figure)
    print(f"\ncomposite: {args.out / 'figure2.png'}")


if __name__ == "__main__":
    main()

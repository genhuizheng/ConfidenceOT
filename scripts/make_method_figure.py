"""Figure 1: what ConfidenceOT does, drawn from a worked example.

Every panel is real solver output on one small problem, not a drawing of what
the solver is believed to do. The problem is two-dimensional so the geometry
can be seen directly: a target sample, and a source sample made of the same
population plus a satellite cluster that has no counterpart on the other side.
Recovering exactly that satellite is the task.

  (a) Why a gate is needed. Balanced OT conserves mass, so every source cell is
      transported whatever it costs, and the satellite is forced onto whichever
      target cells happen to be nearest.
  (b) What the gate changes. The cost a pair pays becomes C_ij when both cells
      are gated in and a flat rejection cost c when either is gated out, so c is
      the price of leaving a cell unmatched rather than a tuning knob.
  (c) The decision. Alternating with the transport solve, each cell is scored by
      what it costs to keep, and kept only while that is below c. M4-E scores a
      cell by its current partners; M4-R scores it by the full entropic
      conditional, so a rejected cell can return and a rejection is not
      self-confirming.
  (d) Where c comes from. Two disjoint halves of one sample contain no
      incompatible cell by construction while carrying that sample's own
      technical spread, so c is the smallest value that accepts them.

Usage:
  python scripts/make_method_figure.py --out benchmark_results/journal_figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from confidenceot import (  # noqa: E402
    ConfidenceOT,
    calibrate_confidence_cost,
    within_side_null_costs,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Matches scripts/make_journal_figure.py so the two plates sit together.
STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.1,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

C_TARGET = "#999999"      # the other side
C_KEEP = "#0072B2"        # retained
C_DROP = "#D55E00"        # rejected
C_ACCENT = "#009E73"      # the calibrated cost


def build_example(seed: int = 11, n: int = 220, satellite: int = 50):
    """A target sample, and a source sample with a satellite that has no match.

    Two dimensions, so the figure can show the geometry the solver actually
    sees rather than a projection of it.
    """
    rng = np.random.default_rng(seed)
    shared = n - satellite
    target = rng.normal(0.0, 1.0, size=(n, 2))
    source = np.vstack([
        rng.normal(0.0, 1.0, size=(shared, 2)),
        rng.normal(0.0, 0.42, size=(satellite, 2)) + np.array([4.4, 2.5]),
    ])
    truth = np.zeros(n, dtype=bool)
    truth[shared:] = True      # True = no counterpart on the target side
    return source, target, truth


def squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = (np.sum(left * left, axis=1)[:, None]
             + np.sum(right * right, axis=1)[None, :]
             - 2.0 * left @ right.T)
    return np.maximum(value, 0.0)


def panel_problem(axes, source, target, truth, coupling):
    """(a) Balanced OT conserves mass, so the satellite is transported anyway."""
    # One arrow per source cell, to the target cell it sends most mass to.
    partner = np.argmax(coupling, axis=1)
    for index in range(source.shape[0]):
        start, end = source[index], target[partner[index]]
        axes.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-", linewidth=0.35,
            color=C_DROP if truth[index] else "0.8",
            alpha=0.85 if truth[index] else 0.55, zorder=2,
        ))
    axes.scatter(target[:, 0], target[:, 1], s=9, c=C_TARGET, linewidths=0,
                 label="target sample", zorder=3)
    axes.scatter(source[~truth, 0], source[~truth, 1], s=9, c=C_KEEP,
                 linewidths=0, label="source, shared population", zorder=4)
    axes.scatter(source[truth, 0], source[truth, 1], s=11, c=C_DROP,
                 linewidths=0, label="source, no counterpart", zorder=5)
    axes.legend(loc="upper left", handlelength=0.9, handletextpad=0.4,
                borderpad=0.2, labelspacing=0.3, fontsize=6.3,
                bbox_to_anchor=(-0.02, 1.02))
    axes.set_title("(a)  Balanced OT must move every cell", loc="left",
                   fontweight="bold")
    axes.set_xticks([])
    axes.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        axes.spines[side].set_visible(False)
    # adjustable="datalim": with the default "box" the axes shrink to satisfy
    # the aspect and leave half the panel empty.
    axes.set_aspect("equal", adjustable="datalim")


def panel_substitution(axes, rejection_cost):
    """(b) The cost a pair pays, and what a gate does to it."""
    axes.axis("off")
    axes.set_xlim(-0.12, 1)
    axes.set_ylim(0, 1)
    # The panel is about 2.4 in wide at 7.4 pt, so roughly 46 characters
    # a line. Wrapped by hand, because a long single line simply runs off.
    axes.text(0.0, 0.95,
              "Each side carries a binary gate, and the\n"
              "cost a pair pays becomes",
              fontsize=7.4, va="top", linespacing=1.5)
    # Laid out as two aligned lines rather than a cases environment:
    # matplotlib's mathtext has no \begin{cases}.
    axes.plot([0.055, 0.055], [0.595, 0.775], color="0.35", lw=0.9,
              clip_on=False)
    for offset, (expression, condition) in enumerate((
            (r"$C_{ij}$", "both cells gated in"),
            (r"$c$", "either cell gated out"))):
        height = 0.735 - 0.125 * offset
        axes.text(0.095, height, expression, fontsize=9, va="center")
        axes.text(0.26, height, condition, fontsize=7.4, va="center",
                  color="0.25")
    axes.text(0.0, 0.675, r"$\widetilde{C}_{ij} =$", fontsize=9, va="center",
              ha="right")
    # One block, not two: at 7.2 pt this panel holds about five lines below the
    # equation, and two blocks of four and three ran into each other.
    axes.text(0.0, 0.46,
              "so $c$ is a price, not a tuning knob: what the\n"
              "fit pays rather than leave a cell unmatched.\n"
              f"Here $c$ = {rejection_cost:.3f}, in units of the median\n"
              "cost. Gates and plan are optimised together,\n"
              "so no cell is scored against a plan that\n"
              "already assumes it.",
              fontsize=7.2, va="top", linespacing=1.55)
    axes.set_title("(b)  The gate enters the cost", loc="left",
                   fontweight="bold")


def panel_decision(axes, score, truth, gate):
    """(c) The per-cell score, and the rule that reads it."""
    order = np.argsort(score)
    position = np.arange(score.size)
    ranked_truth, ranked = truth[order], score[order]
    axes.axhline(0.0, color=C_ACCENT, lw=1.0, ls=(0, (3, 2)), zorder=2)
    axes.scatter(position[~ranked_truth], ranked[~ranked_truth], s=9,
                 c=C_KEEP, linewidths=0, zorder=3,
                 label="shared population")
    axes.scatter(position[ranked_truth], ranked[ranked_truth], s=11,
                 c=C_DROP, linewidths=0, zorder=4, label="no counterpart")
    axes.text(0.02, 0.0, "  keep below, reject above", transform=axes.get_yaxis_transform(),
              fontsize=6.4, color=C_ACCENT, va="bottom")
    axes.set_xlabel("Source cells, ordered by score")
    axes.set_ylabel(r"Gate score,  mass $\times\,(\mathrm{cost} - c)$")
    axes.legend(loc="upper left", handlelength=0.9, handletextpad=0.4,
                borderpad=0.2, labelspacing=0.3, fontsize=6.4)
    recovered = int(np.sum(truth & ~gate))
    wrong = int(np.sum(~truth & ~gate))
    axes.set_title(f"(c)  One score per cell: {recovered} of {int(truth.sum())} "
                   f"found, {wrong} false", loc="left", fontweight="bold")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.grid(axis="y", lw=0.4, color="0.92", zorder=0)
    axes.set_axisbelow(True)


def panel_calibration(axes, calibration, minimum):
    """(d) c is read off the null, not chosen."""
    costs = np.asarray(calibration.curve_costs, dtype=float)
    acceptance = np.asarray(calibration.source_raw_acceptance_curve, dtype=float)
    keep = np.isfinite(costs) & np.isfinite(acceptance)
    costs, acceptance = costs[keep], acceptance[keep]
    order = np.argsort(costs)
    costs, acceptance = costs[order], acceptance[order]

    axes.plot(costs, acceptance, marker="o", ms=3.2, color=C_KEEP, zorder=3,
              clip_on=False)
    axes.axhline(minimum, color="0.45", lw=0.8, ls=(0, (2, 1.6)), zorder=2)
    axes.axvline(calibration.rejection_cost, color=C_ACCENT, lw=1.0,
                 ls=(0, (3, 2)), zorder=2)
    axes.text(calibration.rejection_cost, 0.06,
              f"  $c$ = {calibration.rejection_cost:.3f}", color=C_ACCENT,
              fontsize=6.8, va="bottom", ha="left")
    axes.text(costs.max(), minimum, f"accept {minimum:.0%}  ", fontsize=6.4,
              color="0.35", va="bottom", ha="right")
    axes.set_xlabel("Candidate rejection cost")
    axes.set_ylabel("Null halves accepted")
    axes.set_ylim(-0.03, 1.05)
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_title("(d)  Two halves of one sample set $c$", loc="left",
                   fontweight="bold")
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.grid(axis="y", lw=0.4, color="0.92", zorder=0)
    axes.set_axisbelow(True)


def build(out: Path, seed: int) -> None:
    source, target, truth = build_example(seed=seed)
    cost = squared_euclidean(source, target)
    scale = float(np.median(cost))
    cost = cost / scale

    # c is calibrated against two disjoint halves of the target sample, which
    # contain no incompatible cell by construction.
    minimum = 0.90
    nulls = within_side_null_costs(target, observed_scale=scale, seed=seed,
                                   n_replicates=6)
    calibration = calibrate_confidence_cost(
        nulls[:4], nulls[4:], backbone="uot",
        null_semantics="within_side_split",
        within_side_acceptance_minimum=minimum,
        grid_size=9, workers=1, emit_warnings=False,
    )
    print(f"calibrated c = {calibration.rejection_cost:.4f}  "
          f"({calibration.selection_status}, valid={calibration.calibration_valid})")

    # The balanced fit for (a): every cell transported, nothing gated.
    balanced = ConfidenceOT(
        backbone="balanced", variant="exact",
        rejection_cost=calibration.rejection_cost,
        source_rejection_bounds=(0.0, 0.0), target_rejection_bounds=(0.0, 0.0),
    ).fit(cost)

    # The gated fit for (c). The target side never rejects here: the question
    # asked of this example is which source cells have a counterpart.
    gated = ConfidenceOT(
        backbone="uot", variant="reversible",
        rejection_cost=calibration.rejection_cost,
        source_rejection_bounds=(0.0, 1.0), target_rejection_bounds=(0.0, 0.0),
    ).fit(cost)
    found = int(np.sum(truth & ~gated.source_gate))
    false = int(np.sum(~truth & ~gated.source_gate))
    print(f"rejected {int(np.sum(~gated.source_gate))} cells: "
          f"{found} of {int(truth.sum())} planted, {false} false positives")

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(6.5, 4.55))
        grid = figure.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                                   height_ratios=[0.95, 1.0],
                                   left=0.105, right=0.975, top=0.935,
                                   bottom=0.095)
        panel_problem(figure.add_subplot(grid[0, 0]), source, target, truth,
                      balanced.coupling)
        panel_substitution(figure.add_subplot(grid[0, 1]),
                           calibration.rejection_cost)
        panel_decision(figure.add_subplot(grid[1, 0]), gated.source_score,
                       truth, gated.source_gate)
        panel_calibration(figure.add_subplot(grid[1, 1]), calibration, minimum)

        out.mkdir(parents=True, exist_ok=True)
        stem = "fig_confidenceot_method"
        for extension in ("pdf", "png"):
            figure.savefig(out / f"{stem}.{extension}")
        plt.close(figure)
    print(f"wrote {out / (stem + '.pdf')}")
    print(f"wrote {out / (stem + '.png')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark_results/journal_figure"))
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()
    build(arguments.out, arguments.seed)


if __name__ == "__main__":
    main()

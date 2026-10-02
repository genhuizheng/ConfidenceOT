"""Every preprocessing arm against every way the gate can be wrong, in one grid.

Rows are the 32 arms of the factorial, in its own order: normalisation
outermost, then the binary tags, so the effect of one axis is a comparison
between adjacent rows. Columns are eight readouts, each at the three technical
levels.

**Every column is an error with 0 as its ideal.** Rates of being right are
turned into rates of being wrong (a recall becomes the fraction missed), so the
whole figure has one reading: darker is worse, white is correct. That is what
lets a single sequential scale serve all eight, and it is also the scale's
honesty: a 0.28 means 28% of something went wrong in every column, so the
scale runs 0 to 1 throughout rather than being stretched per column to make
small errors look large.

The readouts are grouped by the benchmark's cases, each under its ground truth,
because the arms fail in different ways and a single score would hide which:

* **Case 1: all populations shared** -- reject nothing. Source rejected, target
  rejected, and the kept mass between different populations.
* **Case 2: one population unmatched** -- reject only that population. The
  unmatched cells kept, on whichever side has them, and the source cells
  rejected when the unmatched population is on the target.
* **Case 3: no population shared** -- reject everything, so any kept mass is
  wrong. On its own that column rewards rejecting everything, which is why it
  is never read without the rejection columns.
* **depends on nCount** -- |rho(decision cost, nCount)| on the target in
  Case 1, the side the levels degrade.

The levels print as L0 / L1 / L2, always in that order; the caption defines
them (matched, depth mismatch, depth + dropout).

White is zero and grey is undefined; they are different claims. One kind of
undefined cell is drawn rather than left grey: every source cell rejected in
every run, so the target's correlation has nothing to be computed against.
That is the worst outcome the gate has, so it is drawn at the maximum error of
1.0 and crossed. Only the drawing changes; the values table keeps it undefined.

    python benchmark/45_plot_arm_heatmap.py benchmark_results/native/benchmark_metrics.csv \\
        benchmark_results/native/heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ARMS = [f"{n}{'_noscale' * s}{'_ds' * q}{'_cos' * c}"
        for n in ("raw", "logcpm", "rank256", "ranknm256")
        for s in (0, 1) for q in (0, 1) for c in (0, 1)]
# The factorial's order as the rest of the project writes it.
ARMS = [a for n in ("raw", "logcpm", "rank256", "ranknm256") for a in (
    n, f"{n}_cos", f"{n}_ds", f"{n}_ds_cos", f"{n}_noscale", f"{n}_noscale_cos",
    f"{n}_noscale_ds", f"{n}_noscale_ds_cos")]
# Shown as L0 / L1 / L2 under every block, always in that order. What each
# means -- matched, depth mismatch, depth + dropout -- goes in the caption once
# rather than under each of twenty-four columns.
LEVELS = [("L0_matched", "L0"), ("L1_depth", "L1"), ("L2_depth_dropout", "L2")]

# Named with the benchmark schematic's own words (figures/schematic/benchmark):
# its three cases with their ground truth, its levels, kept and rejected,
# shared and unmatched, nCount. The one quantity the schematic has no word for
# is the transport the gate leaves standing between cells it keeps, called
# kept mass here and defined in the subtitle.
GROUPS = {
    "Case 1: all populations shared": "reject nothing",
    "Case 2: one population unmatched": "reject only that population",
    "Case 3: no population shared": "reject everything",
    "depends on nCount": "Case 1",
}
# (group, column title, case, column, transform)
READOUTS = [
    ("Case 1: all populations shared", "source\nrejected", "all_shared", "source_false_rejection_rate", None),
    ("Case 1: all populations shared", "target\nrejected", "all_shared", "target_false_rejection_rate", None),
    ("Case 1: all populations shared", "kept mass,\nwrong population", "all_shared", "false_asserted_mass", None),
    ("Case 2: one population unmatched", "unmatched kept\n(on source)", "population_lost", "source_recall", "missed"),
    ("Case 2: one population unmatched", "unmatched kept\n(on target)", "population_emerged", "target_recall", "missed"),
    ("Case 2: one population unmatched", "source rejected\n(target unmatched)", "population_emerged", "source_false_rejection_rate", None),
    ("Case 3: no population shared", "kept\nmass", "populations_disjoint", "false_asserted_mass", None),
    ("depends on nCount", "|\u03c1(cost, nCount)|\ntarget", "all_shared", "target_rho_cost_total_counts", "abs"),
]

# The reference sequential ramp, light to dark, with white prepended so an
# exact zero -- the ideal -- reads as nothing to see.
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
        "#184f95", "#0d366b"]
UNDEFINED = "#bdbdbd"
INK, MUTED, RULE = "#1a1a1a", "#5f5f5a", "#d8d8d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--method", default="M4-E")
    # No arm is marked. On these 24 columns every arm is Pareto-undominated,
    # and "best" changes with the criterion -- the smallest worst-case error and
    # the smallest mean error pick different arms -- so a mark would be a
    # judgement printed as a finding. --mark outlines an arm without naming it
    # best; say what the outline means in the caption that goes with it.
    parser.add_argument("--mark", action="append", default=[],
                        help="Outline this arm's row; may be repeated")
    return parser.parse_args()


def build_matrix(table: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str, str]]]:
    columns = []
    values = np.full((len(ARMS), len(READOUTS) * len(LEVELS)), np.nan)
    for j, (group, title, case, column, transform) in enumerate(READOUTS):
        for k, (level, short) in enumerate(LEVELS):
            col = j * len(LEVELS) + k
            columns.append((group, title, short))
            part = table[(table["biological_case"] == case)
                         & (table["technical_level"] == level)]
            means = part.groupby("preprocessing")[column].mean()
            for i, arm in enumerate(ARMS):
                v = means.get(arm, np.nan)
                if transform == "missed" and np.isfinite(v):
                    v = 1.0 - v
                elif transform == "abs" and np.isfinite(v):
                    v = abs(v)
                values[i, col] = v
    return values, columns


def collapsed_cells(table: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """Which undefined cells are a collapse, checked against the runs.

    A cell is a collapse when its metric is undefined and, in every run behind
    it, the gate rejected every source cell. Then no target cell has a kept
    partner and the target's cost has nothing to be measured against, so the
    correlation cannot be computed. That is the worst outcome the gate has,
    and leaving it grey would make it look like missing data next to arms that
    were merely bad, so it is DRAWN at the maximum error of 1.0.

    Drawn, not stored: the values table keeps the metric undefined, because a
    correlation of 1.0 is a claim the data does not make. An undefined cell
    with any other cause is not a collapse and stays grey.
    """
    collapsed = np.zeros(values.shape, dtype=bool)
    for i, j in zip(*np.where(np.isnan(values))):
        _group, _title, case, _column, _transform = READOUTS[j // len(LEVELS)]
        level = LEVELS[j % len(LEVELS)][0]
        runs = table[(table["preprocessing"] == ARMS[i])
                     & (table["biological_case"] == case)
                     & (table["technical_level"] == level)]
        collapsed[i, j] = len(runs) > 0 and bool(
            (runs["source_rejected_n"] == runs["source_cells_n"]).all())
    return collapsed


def main() -> None:
    args = parse_args()
    table = pd.read_csv(args.metrics_csv)
    table = table[table["method"] == args.method]
    missing = sorted(set(ARMS) - set(table["preprocessing"].unique()))
    if missing:
        raise SystemExit(f"arms absent from {args.metrics_csv}: {missing}")
    values, columns = build_matrix(table)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The table view, so no value has to be read off a colour.
    frame = pd.DataFrame(values, index=ARMS,
                         columns=[f"{g} | {c.replace(chr(10), ' ')} | {s}"
                                  for g, c, s in columns])
    frame.index.name = "arm"
    frame.round(4).to_csv(args.output_dir / "arm_error_heatmap_values.csv")

    # The plotting layer only: a collapse is drawn at the maximum error. The
    # table above was written from the unaltered values and keeps it undefined.
    collapsed = collapsed_cells(table, values)
    plotted = values.copy()
    plotted[collapsed] = 1.0

    n_rows, n_cols = values.shape
    cell_w, cell_h = 0.38, 0.235
    left, right, top, bottom = 2.25, 1.55, 1.85, 0.55
    fig_w = left + n_cols * cell_w + right
    fig_h = top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       n_cols * cell_w / fig_w, n_rows * cell_h / fig_h])

    cmap = LinearSegmentedColormap.from_list("error", RAMP)
    cmap.set_bad(UNDEFINED)
    masked = np.ma.masked_invalid(plotted)
    # A 2px surface gap between cells, drawn as white cell edges.
    mesh = ax.pcolormesh(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                         edgecolors="#ffffff", linewidth=1.4)
    # Marked, so a drawn 1.0 cannot be read as a measured one: on the depth
    # column a bare 1.0 says "perfectly depth-driven", which these are not.
    for i, j in zip(*np.where(collapsed)):
        ax.text(j + 0.5, i + 0.5, "\u00d7", ha="center", va="center",
                fontsize=8.0, color="#ffffff", fontweight="bold")
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(ARMS, fontsize=7.4, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Separate the normalisation families and the readouts with wider gaps.
    for boundary in range(8, n_rows, 8):
        ax.axhline(boundary, color="#ffffff", linewidth=4.5)
    for boundary in range(len(LEVELS), n_cols, len(LEVELS)):
        ax.axvline(boundary, color="#ffffff", linewidth=4.5)

    # Outline any row asked for, so it can be followed across the figure.
    unknown = [arm for arm in args.mark if arm not in ARMS]
    if unknown:
        raise SystemExit(f"--mark names arms that are not in the factorial: {unknown}")
    for arm in args.mark:
        i = ARMS.index(arm)
        style = "-"
        # Above the separators, which would otherwise cut the outline into
        # pieces at every readout boundary.
        ax.add_patch(Rectangle((0, i), n_cols, 1, fill=False, edgecolor=INK,
                               linewidth=1.1, linestyle=style, clip_on=False,
                               zorder=6))

    # Level labels under every block, readout titles above, group names above those.
    for col, (_, _, short) in enumerate(columns):
        ax.text(col + 0.5, n_rows + 0.45, short, ha="center", va="top",
                fontsize=6.6, color=MUTED)
    groups: dict[str, list[int]] = {}
    for j, (group, title, *_rest) in enumerate(READOUTS):
        start = j * len(LEVELS)
        ax.text(start + len(LEVELS) / 2, -0.55, title, ha="center", va="bottom",
                fontsize=6.8, color=INK, linespacing=1.05)
        groups.setdefault(group, []).append(start)
    for group, starts in groups.items():
        a, b = min(starts), max(starts) + len(LEVELS)
        ax.plot([a + 0.15, b - 0.15], [-2.45, -2.45], color=RULE, linewidth=1.0,
                clip_on=False)
        ax.text((a + b) / 2, -2.7, GROUPS[group], ha="center", va="bottom",
                fontsize=6.8, color=MUTED)
        # A title over a three-column block is wider than the block, and two
        # such blocks sit side by side, so theirs are broken onto two lines.
        title = group
        if b - a <= len(LEVELS):
            title = (group.replace(": ", ":\n") if ": " in group
                     else group.replace(" on ", "\non ", 1))
        ax.text((a + b) / 2, -3.55, title, ha="center", va="bottom",
                fontsize=7.2, fontweight="semibold", color=INK, linespacing=1.05)

    fig.text(left / fig_w, 1 - 0.18 / fig_h,
             "Gate error for each preprocessing arm",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    fig.text(left / fig_w, 1 - 0.42 / fig_h,
             f"Native-depth simulation, {args.method}, mean over 3 sizes x 5 replicates. "
             "Every cell is an error against the case's ground truth; 0 is ideal, darker is worse.",
             fontsize=7.4, color=MUTED, va="top")
    fig.text(left / fig_w, 1 - 0.62 / fig_h,
             "kept mass: transport the gate leaves between cells it keeps on both sides. "
             "In Case 3 any of it is wrong; in Case 1 the part between different populations is.",
             fontsize=7.4, color=MUTED, va="top")

    cax = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w, bottom / fig_h,
                        0.13 / fig_w, min(2.6, n_rows * cell_h) / fig_h])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    bar.ax.tick_params(labelsize=6.6, colors=MUTED, length=2)
    bar.outline.set_visible(False)
    bar.set_label("error (0 = ideal)", fontsize=7.0, color=INK)
    # Legend entries for whatever the figure actually contains, set above the
    # colour bar and clear of its top tick.
    bar_top = bottom + min(2.6, n_rows * cell_h)
    entries = []
    if collapsed.any():
        entries.append(("collapse", "every source cell\nrejected: shown as\nmax error 1.0,\nmetric is NA"))
    if np.isnan(plotted).any():
        entries.append(("undefined", "undefined"))
    x_swatch = (left + n_cols * cell_w + 0.32) / fig_w
    x_note = (left + n_cols * cell_w + 0.52) / fig_w
    y = bar_top + 0.62
    for kind, note in entries:
        swatch = fig.add_axes([x_swatch, y / fig_h, 0.13 / fig_w, 0.13 / fig_h])
        swatch.set_xticks([])
        swatch.set_yticks([])
        for spine in swatch.spines.values():
            spine.set_visible(False)
        if kind == "collapse":
            swatch.set_facecolor(RAMP[-1])
            swatch.text(0.5, 0.5, "\u00d7", ha="center", va="center",
                        fontsize=7.0, color="#ffffff", fontweight="bold",
                        transform=swatch.transAxes)
        else:
            swatch.set_facecolor(UNDEFINED)
        fig.text(x_note, (y + 0.065) / fig_h, note, fontsize=6.4, color=MUTED,
                 va="center", linespacing=1.1)
        y += 0.55

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"arm_error_heatmap.{suffix}", dpi=220)
    plt.close(fig)
    print(f"wrote {args.output_dir / 'arm_error_heatmap.png'}, .pdf and the values table")


if __name__ == "__main__":
    main()

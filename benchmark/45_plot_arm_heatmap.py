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

The eight readouts are grouped by the kind of mistake, because the arms fail in
different ways and a single score would hide which:

* **asserts a wrong match** -- false asserted mass, the cross-population mass
  between retained cells. Its ideal is 0 in every case. On its own it rewards
  rejecting everything, which is why it is never read without the next group.
* **rejects a matched cell** -- the false rejection rate where a match exists,
  including the source side of population_emerged.
* **keeps an unmatched cell** -- one minus recall on the side that has the
  unmatched population.
* **follows depth** -- |rho(decision cost, total counts)| on the target, the
  side the levels degrade.

Grey is undefined and white is zero. They are different claims, and the
project's other heatmaps already draw them that way.

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
LEVELS = [("L0_matched", "matched"), ("L1_depth", "depth mismatch"),
          ("L2_depth_dropout", "depth + dropout")]

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


def undefined_reason(table: pd.DataFrame, values: np.ndarray) -> str:
    """Name the cause of the grey cells, but only if every one of them has it.

    The one cause seen so far: in every run behind the cell the gate kept no
    source cell, so no target cell has a kept partner and the target's cost
    has nothing to be measured against -- the correlation is undefined, not
    zero. That is checked per cell rather than assumed, and if any grey cell
    has another cause the legend says only that it is undefined.
    """
    reasons = []
    for i, j in zip(*np.where(np.isnan(values))):
        _group, _title, case, _column, _transform = READOUTS[j // len(LEVELS)]
        level = LEVELS[j % len(LEVELS)][0]
        runs = table[(table["preprocessing"] == ARMS[i])
                     & (table["biological_case"] == case)
                     & (table["technical_level"] == level)]
        every_source_rejected = (len(runs) > 0 and bool(
            (runs["source_rejected_n"] == runs["source_cells_n"]).all()))
        reasons.append(every_source_rejected)
    if reasons and all(reasons):
        return "undefined: every\nsource cell rejected,\nso the target has\nnothing to measure\nagainst"
    return "undefined"


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

    n_rows, n_cols = values.shape
    cell_w, cell_h = 0.38, 0.235
    left, right, top, bottom = 2.25, 1.55, 1.85, 1.05
    fig_w = left + n_cols * cell_w + right
    fig_h = top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       n_cols * cell_w / fig_w, n_rows * cell_h / fig_h])

    cmap = LinearSegmentedColormap.from_list("error", RAMP)
    cmap.set_bad(UNDEFINED)
    masked = np.ma.masked_invalid(values)
    # A 2px surface gap between cells, drawn as white cell edges.
    mesh = ax.pcolormesh(masked, cmap=cmap, vmin=0.0, vmax=1.0,
                         edgecolors="#ffffff", linewidth=1.4)
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
        ax.text(col + 0.5, n_rows + 0.35, short, ha="center", va="top",
                rotation=90, fontsize=6.4, color=MUTED)
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
    # The note can run to five lines, so it is centred on a swatch set well
    # above the colour bar rather than hung below one sitting on top of it.
    bar_top = bottom + min(2.6, n_rows * cell_h)
    swatch_y = bar_top + 0.62
    swatch = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w,
                           swatch_y / fig_h, 0.13 / fig_w, 0.13 / fig_h])
    swatch.set_facecolor(UNDEFINED)
    swatch.set_xticks([])
    swatch.set_yticks([])
    for spine in swatch.spines.values():
        spine.set_visible(False)
    fig.text((left + n_cols * cell_w + 0.52) / fig_w,
             (swatch_y + 0.065) / fig_h,
             undefined_reason(table, values), fontsize=6.4, color=MUTED,
             va="center", linespacing=1.1)

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"arm_error_heatmap.{suffix}", dpi=220)
    plt.close(fig)
    print(f"wrote {args.output_dir / 'arm_error_heatmap.png'}, .pdf and the values table")


if __name__ == "__main__":
    main()

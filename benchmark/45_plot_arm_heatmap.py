"""Every preprocessing arm against what the gate rejected, in one grid.

Rows are the 32 arms of the factorial, in its own order: normalisation
outermost, then the binary tags, so the effect of one axis is a comparison
between adjacent rows. Columns are seven readouts, each at the three technical
levels.

**Every value is the same quantity: the fraction of a named set of cells that
the gate rejected.** The set is what changes between columns -- the source
cells, the target cells, or the cells of the one population with no
counterpart -- and under each column the ground truth says what that fraction
should be: 0 where those cells all have a match and should be kept, 1 where
they have none and should be rejected. One quantity is what lets one colour
scale serve every column: 0.28 is 28% of that set rejected, wherever it is.

Two kinds of readout were dropped to get there, because neither is a rejected
fraction: the kept mass, which is transport-plan mass rather than a share of
cells, and |rho(decision cost, nCount)|, which is a correlation. Both stay in
the metrics table.

The readouts are grouped by the benchmark's cases, each under its ground truth:

* **Case 1: all populations shared** -- reject nothing. Source and target
  rejected, ground truth 0 on both.
* **Case 2: one population unmatched** -- reject only that population. The
  unmatched population rejected, on whichever side has it (ground truth 1),
  and the source rejected when the unmatched population is on the target
  (ground truth 0).
* **Case 3: no population shared** -- reject everything. Source and target
  rejected, ground truth 1 on both. Read the two together: once every source
  cell is rejected no target cell has a kept partner, and the target gate
  cannot move off "keep", so a target at 0 beside a source at 1 means nothing
  was matched, not that the target side failed.

The levels print as L0 / L1 / L2, always in that order; the caption defines
them (matched, depth mismatch, depth + dropout).

Grey is undefined and white is zero; they are different claims.

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

# The factorial's order as the rest of the project writes it.
ARMS = [a for n in ("raw", "logcpm", "rank256", "ranknm256") for a in (
    n, f"{n}_cos", f"{n}_ds", f"{n}_ds_cos", f"{n}_noscale", f"{n}_noscale_cos",
    f"{n}_noscale_ds", f"{n}_noscale_ds_cos")]
# Shown as L0 / L1 / L2 under every block, always in that order. What each
# means -- matched, depth mismatch, depth + dropout -- goes in the caption once
# rather than under each column.
LEVELS = [("L0_matched", "L0"), ("L1_depth", "L1"), ("L2_depth_dropout", "L2")]

# Named with the benchmark schematic's own words (figures/schematic/benchmark):
# its three cases with their ground truth, its levels, kept and rejected,
# shared and unmatched.
GROUPS = {
    "Case 1: all populations shared": "reject nothing",
    "Case 2: one population unmatched": "reject only that population",
    "Case 3: no population shared": "reject everything",
}
# (group, column title, case, column, ground truth)
#
# Each column is read straight off the metrics table. A recall is the rejected
# fraction of the cells that should be rejected, and a false rejection rate is
# the rejected fraction of the cells that should be kept, so both already are
# the one quantity this figure shows and neither needs a transform.
READOUTS = [
    ("Case 1: all populations shared", "source\nrejected", "all_shared", "source_false_rejection_rate", 0),
    ("Case 1: all populations shared", "target\nrejected", "all_shared", "target_false_rejection_rate", 0),
    ("Case 2: one population unmatched", "unmatched rejected\n(on source)", "population_lost", "source_recall", 1),
    ("Case 2: one population unmatched", "unmatched rejected\n(on target)", "population_emerged", "target_recall", 1),
    ("Case 2: one population unmatched", "source rejected\n(target unmatched)", "population_emerged", "source_false_rejection_rate", 0),
    ("Case 3: no population shared", "source\nrejected", "populations_disjoint", "source_recall", 1),
    ("Case 3: no population shared", "target\nrejected", "populations_disjoint", "target_recall", 1),
]

# The reference sequential ramp, light to dark, with white prepended so an
# exact zero reads as nothing rejected.
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
    # No arm is marked: which arm is best depends on the criterion, so a mark
    # would be a judgement printed as a finding. --mark outlines an arm without
    # naming it best; say what the outline means in the caption that goes
    # with it.
    parser.add_argument("--mark", action="append", default=[],
                        help="Outline this arm's row; may be repeated")
    return parser.parse_args()


def build_matrix(table: pd.DataFrame) -> tuple[np.ndarray, list[tuple[str, str, int, str]]]:
    columns = []
    values = np.full((len(ARMS), len(READOUTS) * len(LEVELS)), np.nan)
    for j, (group, title, case, column, truth) in enumerate(READOUTS):
        for k, (level, short) in enumerate(LEVELS):
            col = j * len(LEVELS) + k
            columns.append((group, title, truth, short))
            part = table[(table["biological_case"] == case)
                         & (table["technical_level"] == level)]
            means = part.groupby("preprocessing")[column].mean()
            for i, arm in enumerate(ARMS):
                values[i, col] = means.get(arm, np.nan)
    return values, columns


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
                         columns=[f"{g} | {c.replace(chr(10), ' ')} | ground truth {t} | {s}"
                                  for g, c, t, s in columns])
    frame.index.name = "arm"
    frame.round(4).to_csv(args.output_dir / "arm_error_heatmap_values.csv")

    n_rows, n_cols = values.shape
    cell_w, cell_h = 0.38, 0.235
    left, right, top, bottom = 2.25, 1.55, 1.85, 0.55
    fig_w = left + n_cols * cell_w + right
    fig_h = top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       n_cols * cell_w / fig_w, n_rows * cell_h / fig_h])

    cmap = LinearSegmentedColormap.from_list("rejected", RAMP)
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
        # Above the separators, which would otherwise cut the outline into
        # pieces at every readout boundary.
        ax.add_patch(Rectangle((0, i), n_cols, 1, fill=False, edgecolor=INK,
                               linewidth=1.1, clip_on=False, zorder=6))

    # Level labels under every block; above it the ground truth, the readout
    # title, and the group name with its ground truth in words.
    for col, (_, _, _, short) in enumerate(columns):
        ax.text(col + 0.5, n_rows + 0.45, short, ha="center", va="top",
                fontsize=6.6, color=MUTED)
    groups: dict[str, list[int]] = {}
    for j, (group, title, _case, _column, truth) in enumerate(READOUTS):
        start = j * len(LEVELS)
        ax.text(start + len(LEVELS) / 2, -0.2, f"ground truth {truth}",
                ha="center", va="bottom", fontsize=6.3, color=MUTED)
        ax.text(start + len(LEVELS) / 2, -0.95, title, ha="center", va="bottom",
                fontsize=6.8, color=INK, linespacing=1.05)
        groups.setdefault(group, []).append(start)
    for group, starts in groups.items():
        a, b = min(starts), max(starts) + len(LEVELS)
        ax.plot([a + 0.15, b - 0.15], [-2.45, -2.45], color=RULE, linewidth=1.0,
                clip_on=False)
        ax.text((a + b) / 2, -2.7, GROUPS[group], ha="center", va="bottom",
                fontsize=6.8, color=MUTED)
        ax.text((a + b) / 2, -3.55, group, ha="center", va="bottom",
                fontsize=7.2, fontweight="semibold", color=INK)

    fig.text(left / fig_w, 1 - 0.18 / fig_h,
             "Rejected fraction for each preprocessing arm",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    fig.text(left / fig_w, 1 - 0.42 / fig_h,
             f"Native-depth simulation, {args.method}, mean over 3 sizes x 5 replicates. "
             "Every cell is the fraction of the named cells that the gate rejected.",
             fontsize=7.4, color=MUTED, va="top")
    fig.text(left / fig_w, 1 - 0.62 / fig_h,
             "Ground truth under each column: 0 = those cells all have a match and "
             "should be kept; 1 = they have none and should be rejected.",
             fontsize=7.4, color=MUTED, va="top")

    cax = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w, bottom / fig_h,
                        0.13 / fig_w, min(2.6, n_rows * cell_h) / fig_h])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    bar.ax.tick_params(labelsize=6.6, colors=MUTED, length=2)
    bar.outline.set_visible(False)
    bar.set_label("fraction rejected", fontsize=7.0, color=INK)
    if np.isnan(values).any():
        # Only when the figure has one: a legend entry for nothing on the page
        # is a claim that something is undefined.
        y = bottom + min(2.6, n_rows * cell_h) + 0.62
        swatch = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w, y / fig_h,
                               0.13 / fig_w, 0.13 / fig_h])
        swatch.set_xticks([])
        swatch.set_yticks([])
        for spine in swatch.spines.values():
            spine.set_visible(False)
        swatch.set_facecolor(UNDEFINED)
        fig.text((left + n_cols * cell_w + 0.52) / fig_w, (y + 0.065) / fig_h,
                 "undefined", fontsize=6.4, color=MUTED, va="center")

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"arm_error_heatmap.{suffix}", dpi=220)
    plt.close(fig)
    print(f"wrote {args.output_dir / 'arm_error_heatmap.png'}, .pdf and the values table")


if __name__ == "__main__":
    main()

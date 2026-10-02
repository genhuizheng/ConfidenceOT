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
LEVELS = [("L0_matched", "L0"), ("L1_depth", "L1"), ("L2_depth_dropout", "L2")]

# (group, column title, case, column, transform)
READOUTS = [
    ("asserts a wrong match", "nothing\nshared", "populations_disjoint", "false_asserted_mass", None),
    ("asserts a wrong match", "all\nshared", "all_shared", "false_asserted_mass", None),
    ("rejects a matched cell", "all shared\nsource", "all_shared", "source_false_rejection_rate", None),
    ("rejects a matched cell", "all shared\ntarget", "all_shared", "target_false_rejection_rate", None),
    ("rejects a matched cell", "emerged\nsource", "population_emerged", "source_false_rejection_rate", None),
    ("keeps an unmatched cell", "lost\nsource", "population_lost", "source_recall", "missed"),
    ("keeps an unmatched cell", "emerged\ntarget", "population_emerged", "target_recall", "missed"),
    ("follows depth", "all shared\ntarget", "all_shared", "target_rho_cost_total_counts", "abs"),
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
                         columns=[f"{g} | {t.replace(chr(10), ' ')} | {s}"
                                  for g, t, s in columns])
    frame.index.name = "arm"
    frame.round(4).to_csv(args.output_dir / "arm_error_heatmap_values.csv")

    n_rows, n_cols = values.shape
    cell_w, cell_h = 0.30, 0.235
    left, right, top, bottom = 2.25, 1.15, 1.55, 0.55
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
        ax.text(col + 0.5, n_rows + 0.55, short, ha="center", va="top",
                fontsize=6.6, color=MUTED)
    groups: dict[str, list[int]] = {}
    for j, (group, title, *_rest) in enumerate(READOUTS):
        start = j * len(LEVELS)
        ax.text(start + len(LEVELS) / 2, -0.55, title, ha="center", va="bottom",
                fontsize=7.0, color=INK, linespacing=1.05)
        groups.setdefault(group, []).append(start)
    for group, starts in groups.items():
        a, b = min(starts), max(starts) + len(LEVELS)
        ax.text((a + b) / 2, -2.55, group, ha="center", va="bottom",
                fontsize=7.6, fontweight="semibold", color=INK)
        ax.plot([a + 0.15, b - 0.15], [-2.35, -2.35], color=RULE, linewidth=1.0,
                clip_on=False)

    fig.text(left / fig_w, 1 - 0.18 / fig_h,
             "Where each preprocessing arm goes wrong",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    fig.text(left / fig_w, 1 - 0.42 / fig_h,
             f"Native-depth simulation, {args.method}. Every cell is an error with 0 "
             "as its ideal; darker is worse. Mean over 3 sizes x 5 replicates.",
             fontsize=7.4, color=MUTED, va="top")

    cax = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w, bottom / fig_h,
                        0.13 / fig_w, min(2.6, n_rows * cell_h) / fig_h])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    bar.ax.tick_params(labelsize=6.6, colors=MUTED, length=2)
    bar.outline.set_visible(False)
    bar.set_label("error (0 = ideal)", fontsize=7.0, color=INK)
    swatch = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w,
                           (bottom + min(2.6, n_rows * cell_h) + 0.25) / fig_h,
                           0.13 / fig_w, 0.13 / fig_h])
    swatch.set_facecolor(UNDEFINED)
    swatch.set_xticks([])
    swatch.set_yticks([])
    for spine in swatch.spines.values():
        spine.set_visible(False)
    fig.text((left + n_cols * cell_w + 0.52) / fig_w,
             (bottom + min(2.6, n_rows * cell_h) + 0.315) / fig_h,
             "undefined", fontsize=6.6, color=MUTED, va="center")

    for suffix in ("png", "pdf"):
        fig.savefig(args.output_dir / f"arm_error_heatmap.{suffix}", dpi=220)
    plt.close(fig)
    print(f"wrote {args.output_dir / 'arm_error_heatmap.png'}, .pdf and the values table")


if __name__ == "__main__":
    main()

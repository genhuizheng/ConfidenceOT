"""Build the journal figure: the OT benchmark and the depth screen in one plate.

Two bodies of simulation work answer two different questions, and a reader
needs both to believe the method:

  (a)-(d)  Splatter dual-backbone benchmark. Does the binary gate find a
           planted population change, does it leave the unaffected
           populations alone, and what does it cost? Partial OT and vanilla
           UOT are the comparators.
  (e)-(f)  Depth screen. Sequencing depth is the confounder that drove the
           real-data gate, so every candidate preprocessing is scored on the
           same simulated ground truth: how much depth dependence is left on
           populations that contain no incompatible cell, against whether the
           gate can still recover a known perturbed 20%.

Panels (e) and (f) share their rows, so one horizontal band is one
preprocessing configuration read left (specificity) to right (power). A
configuration is only usable if it clears both; either column alone is
satisfiable by a gate that does nothing.

Inputs:
  --benchmark-root  the splatter report directory, holding
                    directional_f1_all_sizes_summary.csv,
                    population_rejection_all_sizes_summary.csv and
                    deployment_runtime_all_sizes_summary.csv
  --screen-root     the depth screen root, holding <label>/depth_null_arm_summary.csv

Missing screen configurations are reported and left out rather than faked, so
the plate can be drawn while the array is still running.

Usage:
  python scripts/make_journal_figure.py \
      --benchmark-root benchmark_results/splatter_n1000_5000_10000/report \
      --screen-root /scratch/.../depth_screen \
      --out benchmark_results/journal_figure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Print style: 6.5 in text width, serif, sizes that survive a 400 dpi raster.
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
    "lines.markersize": 4,
    "savefig.dpi": 400,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Okabe-Ito, colourblind safe.
C_TRAD, C_VUOT, C_CONF, C_PART = "#999999", "#E69F00", "#0072B2", "#009E73"
C_ME, C_MR = "#0072B2", "#D55E00"

SIZES = [1000, 5000, 10000]
ANOMALY_SCENARIOS = ["S1_extinction", "S2_emergence", "S3_source_outlier"]
PLANTED = {("S1_extinction", "source", "A"),
           ("S2_emergence", "target", "G"),
           ("S3_source_outlier", "source", "O")}

HOMOGENEOUS = ["homogeneous_depth_cv0", "homogeneous_depth_cv_low",
               "homogeneous_depth_cv_mid", "homogeneous_depth_cv_high"]
CONTROL = "perturbed_depth_cv0"

# Submission order, ours then the external packages, which is also how the
# rows should read: each block goes untreated, equalised, then equalised with
# the cosine cost.
SCREEN_ORDER = ["logcpm", "logcpm_ds", "logcpm_cos",
                "rank256", "rank256_ds", "rank256_ds_cos",
                "pearson_ds",
                "scanpy_pearson", "scanpy_pearson_ds", "sct", "sct_ds"]
SCREEN_LABEL = {
    "logcpm": "log CPM",
    "logcpm_ds": "log CPM + equalise",
    "logcpm_cos": "log CPM + cosine",
    "rank256": "rank-value",
    "rank256_ds": "rank-value + equalise",
    "rank256_ds_cos": "rank-value + eq. + cosine",
    "pearson_ds": "Pearson resid. + equalise",
    "scanpy_pearson": "scanpy Pearson resid.",
    "scanpy_pearson_ds": "scanpy Pearson + equalise",
    "sct": "sctransform v2",
    "sct_ds": "sctransform v2 + equalise",
}
# Everything from here down is somebody else's implementation, which is the
# point of including it: the transform comparison should not rest only on ours.
FIRST_EXTERNAL = "scanpy_pearson"


def group_of(method: str) -> str:
    if method == "Traditional OT":
        return "Traditional OT"
    if method.startswith("Vanilla"):
        return "Vanilla UOT"
    if method == "Partial OT":
        return "Partial OT"
    return "ConfidenceOT"


def tidy(axes) -> None:
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.grid(axis="y", lw=0.4, color="0.9", zorder=0)
    axes.set_axisbelow(True)


def plain(value, _=None) -> str:
    """Log tick label without the trailing .0 that stacks ticks together."""
    return f"{value:g}"


def load_benchmark(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = ("directional_f1_all_sizes_summary.csv",
             "population_rejection_all_sizes_summary.csv",
             "deployment_runtime_all_sizes_summary.csv")
    missing = [n for n in names if not (root / n).exists()]
    if missing:
        raise SystemExit(f"{root} is missing: {', '.join(missing)}")
    f1, rejection, runtime = (pd.read_csv(root / n) for n in names)
    for frame in (f1, rejection, runtime):
        frame["group"] = frame.display_method.map(group_of)
    rejection["planted"] = [(s, side, pop) in PLANTED for s, side, pop
                            in zip(rejection.scenario, rejection.side,
                                   rejection.population)]
    return f1, rejection, runtime


def load_screen(root: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    frames = []
    for path in sorted(root.glob("*/depth_null_arm_summary.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "configuration", path.parent.name)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no depth_null_arm_summary.csv under {root}")
    arms = pd.concat(frames, ignore_index=True)
    if "auc_total_counts" not in arms:
        raise SystemExit("screen summaries carry no auc_total_counts column")
    arms["depth_effect"] = (arms["auc_total_counts"] - 0.5).abs()
    have = set(arms.configuration)
    present = [c for c in SCREEN_ORDER if c in have]
    present += sorted(have - set(present))
    return arms, present, [c for c in SCREEN_ORDER if c not in have]


def panel_f1(axes, f1: pd.DataFrame, batch: str, title: str, legend: bool) -> None:
    anomalies = f1[f1.scenario.isin(ANOMALY_SCENARIOS)
                   & f1.batch_condition.eq(batch)]
    for name, colour, marker in (("Traditional OT", C_TRAD, "s"),
                                 ("Vanilla UOT", C_VUOT, "^"),
                                 ("ConfidenceOT", C_CONF, "o"),
                                 ("Partial OT", C_PART, "D")):
        curve = (anomalies[anomalies.group == name]
                 .groupby("n_cells").directional_f1_mean.mean().reindex(SIZES))
        axes.plot(SIZES, curve.values, marker=marker, color=colour, label=name,
                  clip_on=False, zorder=3)
    axes.set_xscale("log")
    axes.set_xticks(SIZES)
    axes.xaxis.set_major_formatter(FuncFormatter(plain))
    axes.xaxis.set_minor_locator(FixedLocator([]))
    axes.set_xlim(900, 11200)
    axes.set_ylim(-0.03, 1.03)
    axes.set_yticks(np.arange(0, 1.01, 0.2))
    axes.set_xlabel("Cells per side, $N$")
    axes.set_ylabel("Directional F1")
    axes.set_title(title, loc="left", fontweight="bold")
    tidy(axes)
    if legend:
        axes.legend(loc="center right", handlelength=1.6, borderpad=0.2,
                    labelspacing=0.32)


def panel_specificity(axes, rejection: pd.DataFrame) -> None:
    groups = ["Traditional OT", "Vanilla UOT", "ConfidenceOT"]
    positions, width = np.arange(len(groups)), 0.36
    largest = rejection[rejection.n_cells.eq(rejection.n_cells.max())
                        & rejection.scenario.isin(ANOMALY_SCENARIOS)
                        & rejection.batch_condition.eq("none")]
    planted = [largest[largest.group.eq(g) & largest.planted]
               .rejection_signal_mean.mean() for g in groups]
    unaffected = [largest[largest.group.eq(g) & ~largest.planted]
                  .rejection_signal_mean.mean() for g in groups]
    worst = [largest[largest.group.eq(g) & ~largest.planted]
             .rejection_signal_mean.max() for g in groups]

    bars_planted = axes.bar(positions - width / 2, planted, width,
                            color=C_CONF, label="Planted anomaly", zorder=3)
    bars_other = axes.bar(positions + width / 2, unaffected, width,
                          color="0.78", label="Unaffected populations",
                          zorder=3)
    for position, value in zip(positions + width / 2, worst):
        axes.plot([position - width / 2, position + width / 2], [value, value],
                  color="0.2", lw=1.0, zorder=5, solid_capstyle="butt")
    worst_handle, = axes.plot([], [], color="0.2", lw=1.0,
                              label="Worst unaffected population")

    for position, value in zip(positions - width / 2, planted):
        axes.text(position, value + 0.03, f"{value:.2f}", ha="center",
                  va="bottom", fontsize=6.5)
    for position, value in zip(positions + width / 2, unaffected):
        if value > 0.12:            # inside the bar, so it clears the mark
            axes.text(position, value - 0.035, f"{value:.2f}", ha="center",
                      va="top", fontsize=6.5, color="0.15")
        else:
            axes.text(position, value + 0.03, f"{value:.2f}", ha="center",
                      va="bottom", fontsize=6.5)

    axes.set_xticks(positions)
    axes.set_xticklabels(groups)
    axes.set_ylim(0, 1.18)
    axes.set_yticks(np.arange(0, 1.01, 0.2))
    axes.set_ylabel("Rejection signal")
    size = int(largest.n_cells.max())
    axes.set_title(f"(c)  Specificity at $N={size:,}$".replace(",", "{,}"),
                   loc="left", fontweight="bold")
    tidy(axes)
    axes.legend(handles=[bars_planted, bars_other, worst_handle],
                loc="upper left", handlelength=1.3, borderpad=0.2,
                labelspacing=0.32)


def panel_runtime(axes, runtime: pd.DataFrame) -> None:
    for name, colour, marker, key in (("Traditional OT", C_TRAD, "s", None),
                                      ("Vanilla UOT", C_VUOT, "^", None),
                                      ("ConfidenceOT (M4-E)", C_ME, "o", "M4-E"),
                                      ("ConfidenceOT (M4-R)", C_MR, "v", "M4-R")):
        if key is None:
            curve = (runtime[runtime.group == name]
                     .groupby("n_cells").single_fit_seconds_mean.mean())
        else:
            curve = (runtime[runtime.display_method.str.contains(key, regex=False)]
                     .groupby("n_cells").single_fit_seconds_mean.mean())
        axes.plot(SIZES, curve.reindex(SIZES).values, marker=marker,
                  color=colour, label=name, clip_on=False, zorder=3)
    partial = (runtime[runtime.group == "Partial OT"]
               .groupby("n_cells").single_fit_seconds_mean.mean())
    if 1000 in partial.index:
        axes.plot([1000], [partial.loc[1000]], marker="D", color=C_PART,
                  ls="none", label="Partial OT ($N=1000$ only)", clip_on=False,
                  zorder=3)

    reference = np.array(SIZES, float)
    axes.plot(reference, 0.22 * (reference / 1000.0) ** 2, ls=":", lw=0.9,
              color="0.45", zorder=2)
    axes.text(3400, 0.22 * (3400 / 1000.0) ** 2 * 0.42, r"$\mathcal{O}(N^{2})$",
              fontsize=6.8, color="0.35", ha="center", va="top", rotation=31,
              rotation_mode="anchor")

    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xticks(SIZES)
    axes.xaxis.set_major_formatter(FuncFormatter(plain))
    axes.xaxis.set_minor_locator(FixedLocator([]))
    axes.set_xlim(900, 11200)
    axes.set_yticks([0.1, 1, 10, 100, 1000])
    axes.yaxis.set_major_formatter(FuncFormatter(plain))
    axes.yaxis.set_minor_locator(FixedLocator([]))
    axes.set_ylim(0.08, 2600)
    axes.set_xlabel("Cells per side, $N$")
    axes.set_ylabel("Seconds per OT fit")
    axes.set_title("(d)  Cost per fit", loc="left", fontweight="bold")
    tidy(axes)
    axes.legend(loc="upper left", handlelength=1.6, borderpad=0.2,
                labelspacing=0.32)


def screen_tables(arms: pd.DataFrame, present: list[str],
                  tolerance: float, minimum_f1: float):
    effect = (arms[arms.arm.isin(HOMOGENEOUS)]
              .pivot_table(index="configuration", columns="arm",
                           values="depth_effect", aggfunc="median")
              .reindex(index=present, columns=HOMOGENEOUS))
    spread = (arms[arms.arm.isin(HOMOGENEOUS)]
              .groupby("arm").observed_depth_cv.median().reindex(HOMOGENEOUS))
    control = arms[arms.arm.eq(CONTROL)].set_index("configuration")
    power = pd.DataFrame(index=present)
    for column in ("perturbed_f1", "perturbed_recall", "perturbed_precision"):
        power[column] = (control[column].reindex(present) if column in control
                         else np.nan)
    worst = effect.max(axis=1)
    passes = (worst <= tolerance) & (power.perturbed_f1 >= minimum_f1)
    return effect, spread, power, worst, passes.fillna(False)


def panel_depth_effect(axes, effect: pd.DataFrame, spread: pd.Series,
                       passes: pd.Series, tolerance: float) -> None:
    values = effect.to_numpy(dtype=float)
    axes.imshow(values, cmap="Blues", vmin=0.0, vmax=0.5,
                aspect="auto", interpolation="nearest")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if not np.isfinite(value):
                continue
            axes.text(column, row, f"{value:.3f}", ha="center", va="center",
                      fontsize=6.2,
                      color="white" if value > 0.28 else "0.15")
    axes.set_xticks(range(values.shape[1]))
    axes.set_xticklabels([f"{cv:.2f}" for cv in spread.values], fontsize=7)
    axes.set_xlabel("Sequencing-depth CV across cells")
    axes.set_yticks(range(values.shape[0]))
    axes.set_yticklabels([SCREEN_LABEL.get(name, name) for name in effect.index],
                         fontsize=6.6)
    axes.tick_params(length=0)
    axes.set_ylim(values.shape[0] - 0.5, -0.5)
    for label, ok in zip(axes.get_yticklabels(), passes.reindex(effect.index)):
        if bool(ok):
            label.set_color(C_PART)
            label.set_fontweight("bold")
    if FIRST_EXTERNAL in list(effect.index):
        boundary = list(effect.index).index(FIRST_EXTERNAL) - 0.5
        # The line alone: the rows below it are named after the packages that
        # implement them, so labelling the two blocks in words would only
        # repeat that, and there is no space inside the cells for it.
        axes.axhline(boundary, color="0.25", lw=0.8, ls=(0, (2, 1.6)))
    axes.set_title("(e)  Residual depth dependence, $|$AUC $-$ 0.5$|$",
                   loc="left", fontweight="bold", fontsize=8)
    # "Passes" means the whole row is at or under the tolerance, so it belongs
    # beside the rows rather than in the caption; the row label itself carries
    # the colour.
    # Left of the panel, over the row-label gutter: the title is left-aligned
    # to the axes and runs most of its width, so the free space is that side.
    axes.text(-0.02, 1.012, "green = passes", transform=axes.transAxes,
              ha="right", va="bottom", fontsize=6.4, color=C_PART)


def panel_power(axes, power: pd.DataFrame, passes: pd.Series,
                minimum_f1: float, true_fraction: float) -> None:
    positions = np.arange(len(power))
    axes.axvline(minimum_f1, color="0.6", lw=0.8, ls=(0, (2, 1.6)), zorder=1)
    for position, (_, row) in zip(positions, power.iterrows()):
        f1, recall = row.perturbed_f1, row.perturbed_recall
        if np.isfinite(f1):
            axes.plot([0, f1], [position, position], color="0.85", lw=0.9,
                      zorder=2)
            axes.plot([f1], [position], marker="o", ms=4.2, color=C_CONF,
                      zorder=4, clip_on=False)
            axes.text(1.10, position, f"{f1:.2f}", ha="left",
                      va="center", fontsize=6.2, color="0.2")
        if np.isfinite(recall):
            axes.plot([recall], [position], marker="o", ms=4.2, mfc="none",
                      mec="0.45", mew=0.8, zorder=3, clip_on=False)
    axes.set_xlim(0, 1.30)
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    # The two marks are named in the axis label rather than in a legend: this
    # panel is only 1.8 in wide, and a legend box wide enough to read collides
    # with the title on one side and with every row's stem on the other.
    axes.set_xlabel(f"Detection of the planted {true_fraction:.0%}\n"
                    "filled F1, open recall")
    axes.tick_params(axis="y", length=0, labelleft=False)
    axes.set_ylim(len(power) - 0.5, -0.5)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.set_title("(f)  Positive control", loc="left", fontweight="bold",
                   fontsize=8)


def build(benchmark_root: Path, screen_root: Path, out: Path,
          tolerance: float, minimum_f1: float) -> None:
    f1, rejection, runtime = load_benchmark(benchmark_root)
    arms, present, missing = load_screen(screen_root)
    effect, spread, power, worst, passes = screen_tables(
        arms, present, tolerance, minimum_f1)
    control_fraction = float(arms[arms.arm.eq(CONTROL)].perturbed_fraction.median())

    # The bottom block carries one band per configuration, so it grows with
    # the screen while the top block stays a fixed size. Positions are set in
    # inches and converted, rather than left to a shared gridspec: the row
    # labels in (e) need a gutter three times wider than any axis above them,
    # and a gridspec would impose that gutter on every panel.
    rows = max(len(present), 4)
    pad_top, top_block, gap, pad_bottom = 0.28, 4.30, 0.50, 0.68
    bands = 0.26 * rows
    height = pad_top + top_block + gap + bands + 0.34 + pad_bottom
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(6.5, height))
        grid = figure.add_gridspec(
            2, 2, hspace=0.52, wspace=0.27, left=0.075, right=0.985,
            top=1 - pad_top / height,
            bottom=1 - (pad_top + top_block) / height)

        panel_f1(figure.add_subplot(grid[0, 0]), f1, "none",
                 "(a)  No batch effect", legend=True)
        panel_f1(figure.add_subplot(grid[0, 1]), f1, "mild",
                 "(b)  Mild batch effect", legend=False)
        panel_specificity(figure.add_subplot(grid[1, 0]), rejection)
        panel_runtime(figure.add_subplot(grid[1, 1]), runtime)

        left, right, between = 0.235, 0.985, 0.012
        width_effect = (right - left - between) / 1.62
        axes_effect = figure.add_axes([left, pad_bottom / height,
                                      width_effect, bands / height])
        panel_depth_effect(axes_effect, effect, spread, passes, tolerance)
        axes_power = figure.add_axes(
            [left + width_effect + between, pad_bottom / height,
             width_effect * 0.62, bands / height], sharey=axes_effect)
        panel_power(axes_power, power, passes, minimum_f1, control_fraction)

        out.mkdir(parents=True, exist_ok=True)
        stem = "fig_simulation_benchmark"
        for extension in ("pdf", "png"):
            figure.savefig(out / f"{stem}.{extension}")
        plt.close(figure)

    table = effect.rename(columns={name: f"depth_cv_{spread[name]:.2f}"
                                   for name in HOMOGENEOUS})
    table = table.join(power)
    table["worst_depth_effect"] = worst
    table["passes"] = passes.map({True: "yes", False: "no"})
    table.to_csv(out / "fig_simulation_benchmark_source.csv")

    print(f"wrote {out / (stem + '.pdf')}")
    print(f"wrote {out / (stem + '.png')}")
    print(f"wrote {out / 'fig_simulation_benchmark_source.csv'}")
    print()
    print(table.round(3).to_string())
    if missing:
        print("\nnot in the screen yet, so absent from (e) and (f): "
              + ", ".join(missing))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path,
                        default=Path("benchmark_results/journal_figure"))
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="depth effect at or below this counts as cleared")
    parser.add_argument("--minimum-f1", type=float, default=0.60,
                        help="positive-control F1 at or above this is intact")
    arguments = parser.parse_args()
    build(arguments.benchmark_root, arguments.screen_root, arguments.out,
          arguments.tolerance, arguments.minimum_f1)


if __name__ == "__main__":
    main()

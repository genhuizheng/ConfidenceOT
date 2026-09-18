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

# Row order: ours then the external packages, each block reading untreated,
# equalised, then equalised with the cosine cost. Not the array's submission
# order -- that has to stay append-only, since the earlier indices have run --
# so the divider below keeps meaning what it says.
SCREEN_ORDER = ["logcpm", "logcpm_ds", "logcpm_cos",
                "rank256", "rank256_ds", "rank256_ds_cos",
                "pearson_ds", "pearson_ds_cos",
                "scanpy_pearson", "scanpy_pearson_ds",
                "sct", "sct_ds", "sct_ds_cos"]
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
    "pearson_ds_cos": "Pearson resid. + eq. + cosine",
    "sct_ds_cos": "sctransform v2 + eq. + cosine",
}
# Everything from here down is somebody else's implementation, which is the
# point of including it: the transform comparison should not rest only on ours.
FIRST_EXTERNAL = "scanpy_pearson"


def split_variant(label: str) -> tuple[str, str]:
    """Split a screen label into its configuration and its variant suffix.

    The screen job appends a suffix when a run means something other than the
    default -- _splatter for splatter's counts, _r<n> for a different replicate
    count, _n<cells> for a different size -- so that a comparison run cannot
    overwrite what it is compared against. One plate shows one variant: mixing
    them would put two simulators in the same panel with nothing to tell them
    apart, and would double the row count.

    Matched against the known names rather than by regex, because the base
    labels contain underscores themselves.
    """
    for candidate in sorted(SCREEN_ORDER, key=len, reverse=True):
        if label == candidate:
            return candidate, ""
        if label.startswith(candidate + "_"):
            return candidate, label[len(candidate) + 1:]
    return label, ""


def variant_title(variant: str) -> str:
    """A panel-title phrase for a variant suffix, built from its tokens.

    Compound suffixes are the normal case -- a splatter run at another size is
    `splatter_n5000` -- so this composes rather than looking up whole strings.
    """
    phrases = []
    for token in variant.split("_"):
        if token == "splatter":
            phrases.append("splatter counts")
        elif token.startswith("r") and token[1:].isdigit():
            phrases.append(f"{token[1:]} replicates")
        elif token.startswith("n") and token[1:].isdigit():
            phrases.append(f"$N={token[1:]}$")
        elif token:
            phrases.append(token)
    return ", ".join(phrases)


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


def _read_screen(root: Path, name: str) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"*/{name}")):
        frame = pd.read_csv(path)
        frame.insert(0, "configuration", path.parent.name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "auc_total_counts" in combined:
        combined["depth_effect"] = (combined["auc_total_counts"] - 0.5).abs()
    return combined


def load_screen(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str],
                                     list[str]]:
    arms = _read_screen(root, "depth_null_arm_summary.csv")
    if arms.empty:
        raise SystemExit(f"no depth_null_arm_summary.csv under {root}")
    if "depth_effect" not in arms:
        raise SystemExit("screen summaries carry no auc_total_counts column")
    # Per-replicate rows, for the scatter in (e). The summary is already a
    # median over them, and a median alone hides how much of the difference
    # between two configurations is replicate noise.
    replicates = _read_screen(root, "depth_null_replicates.csv")
    return arms, replicates


def choose_variant(arms: pd.DataFrame, wanted: str) -> tuple[list[str], list[str], list[str]]:
    """Rows for one variant, in display order, plus what is missing and what else exists."""
    seen = {}
    for label in set(arms.configuration):
        base, variant = split_variant(label)
        seen.setdefault(variant, {})[base] = label
    others = sorted(v for v in seen if v != wanted)
    chosen = seen.get(wanted, {})
    present = [chosen[base] for base in SCREEN_ORDER if base in chosen]
    present += sorted(set(chosen.values()) - set(present))
    missing = [base for base in SCREEN_ORDER if base not in chosen]
    return present, missing, others


def panel_f1(axes, f1: pd.DataFrame, batch: str, title: str, legend: bool) -> None:
    anomalies = f1[f1.scenario.isin(ANOMALY_SCENARIOS)
                   & f1.batch_condition.eq(batch)]
    for name, colour, marker in (("Traditional OT", C_TRAD, "s"),
                                 ("Vanilla UOT", C_VUOT, "^"),
                                 ("ConfidenceOT", C_CONF, "o"),
                                 ("Partial OT", C_PART, "D")):
        curve = (anomalies[anomalies.group == name]
                 .groupby("n_cells").directional_f1_mean.mean().reindex(SIZES))
        # A method present at one size only draws one marker and no line, so
        # the legend says why rather than leaving a reader to wonder whether
        # the run is missing.
        drawn = name
        if curve.notna().sum() == 1:
            only = int(curve.dropna().index[0])
            drawn = f"{name} ($N={only}$ only)"
        axes.plot(SIZES, curve.values, marker=marker, color=colour,
                  label=drawn, clip_on=False, zorder=3)
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


GROUPS = ["Traditional OT", "Vanilla UOT", "ConfidenceOT", "Partial OT"]
# Why Partial OT has only one size, stated under the bars it belongs to
# rather than left to the caption. The comparator is the exact augmented
# Hungarian assignment, so 5,000 and 10,000 are out of reach by construction,
# not missing runs -- the benchmark's own audit records all 20 of its expected
# cases as complete.
GROUP_NOTE = {"Partial OT": "exact, $\\mathcal{O}(N^{3})$"}


def panel_specificity(axes, rejection: pd.DataFrame) -> None:
    anomalies = rejection[rejection.scenario.isin(ANOMALY_SCENARIOS)
                          & rejection.batch_condition.eq("none")]
    # Draw the bars at the largest size where every method is present, so the
    # comparison is complete, and overlay the largest size of all for the
    # methods that reach it. Drawing only the largest size drops Partial OT
    # out of the panel silently, which is how it went missing.
    sizes = sorted(anomalies.n_cells.unique())
    complete = [s for s in sizes
                if set(GROUPS) <= set(anomalies[anomalies.n_cells.eq(s)].group)]
    primary = complete[-1] if complete else sizes[-1]
    overlay = sizes[-1] if sizes[-1] != primary else None

    def summarise(size):
        frame = anomalies[anomalies.n_cells.eq(size)]
        out = {}
        for name in GROUPS:
            rows = frame[frame.group.eq(name)]
            if rows.empty:
                out[name] = None
                continue
            other = rows[~rows.planted].rejection_signal_mean
            out[name] = (rows[rows.planted].rejection_signal_mean.mean(),
                         other.mean(), other.max())
        return out

    at_primary = summarise(primary)
    at_overlay = summarise(overlay) if overlay else {}

    present = [name for name in GROUPS if at_primary.get(name)]
    positions, width = np.arange(len(present)), 0.36
    planted = [at_primary[name][0] for name in present]
    unaffected = [at_primary[name][1] for name in present]
    worst = [at_primary[name][2] for name in present]

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

    # Three things can sit at a bar's top edge: the value, the worst-population
    # mark, and the larger-size marker. So the value goes inside a bar tall
    # enough to hold it, and above the worst mark rather than above the bar.
    for position, value in zip(positions - width / 2, planted):
        if value > 0.25:
            axes.text(position, value - 0.04, f"{value:.2f}", ha="center",
                      va="top", fontsize=6.2, color="white", zorder=7)
        else:
            axes.text(position, value + 0.03, f"{value:.2f}", ha="center",
                      va="bottom", fontsize=6.2)
    for position, value, top in zip(positions + width / 2, unaffected, worst):
        axes.text(position, max(value, top) + 0.035, f"{value:.2f}",
                  ha="center", va="bottom", fontsize=6.2, color="0.15")

    handles = [bars_planted, bars_other, worst_handle]
    if at_overlay:
        for index, name in enumerate(present):
            values = at_overlay.get(name)
            if not values:
                continue
            axes.plot([index - width / 2, index + width / 2],
                      [values[0], values[1]], marker="D", ms=3.0, ls="none",
                      mfc="none", mec="0.15", mew=0.8, zorder=6, clip_on=False)
        overlay_handle, = axes.plot([], [], marker="D", ms=3.0, ls="none",
                                    mfc="none", mec="0.15", mew=0.8,
                                    label=f"at $N={overlay}$")
        handles.append(overlay_handle)

    axes.set_xticks(positions)
    axes.set_xticklabels(
        [name.replace(" ", "\n") + ("\n" + GROUP_NOTE[name]
                                    if name in GROUP_NOTE else "")
         for name in present], fontsize=6.4)
    axes.set_ylim(0, 1.34)
    axes.set_yticks(np.arange(0, 1.01, 0.2))
    axes.set_ylabel("Rejection signal")
    axes.set_title(f"(c)  Specificity at $N={primary}$",
                   loc="left", fontweight="bold")
    tidy(axes)
    axes.legend(handles=handles, loc="upper left", handlelength=1.2,
                borderpad=0.2, labelspacing=0.3, fontsize=6.4)


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


# A sequential single-hue ramp, light to dark with depth spread, because the
# thing it encodes is a magnitude. Four steps for the four homogeneous arms.
DEPTH_SHADES = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]


def panel_depth_effect(axes, effect: pd.DataFrame, replicates: pd.DataFrame,
                       spread: pd.Series, passes: pd.Series,
                       tolerance: float, title: str) -> None:
    rows = list(effect.index)
    # Every replicate, not just the median. Two runs of the same method landed
    # either side of the tolerance in the production screen, so the scatter is
    # the finding: at three replicates the threshold cuts inside the noise.
    scatter = replicates if not replicates.empty else effect.reset_index()
    for row_index, configuration in enumerate(rows):
        axes.plot([0, 0.52], [row_index, row_index], color="0.93", lw=0.7,
                  zorder=1)
        for arm_index, arm in enumerate(HOMOGENEOUS):
            offset = (arm_index - 1.5) * 0.17
            if replicates.empty:
                points = effect.loc[[configuration], arm].to_numpy(dtype=float)
            else:
                points = (scatter[scatter.configuration.eq(configuration)
                                  & scatter.arm.eq(arm)]
                          .depth_effect.to_numpy(dtype=float))
            points = points[np.isfinite(points)]
            if not len(points):
                continue
            axes.plot(points, np.full(len(points), row_index + offset),
                      marker="o", ms=3.0, ls="none",
                      mfc=DEPTH_SHADES[arm_index], mec="white", mew=0.4,
                      zorder=3, clip_on=True)

    axes.axvline(tolerance, color="0.45", lw=0.8, ls=(0, (2, 1.6)), zorder=2)
    axes.set_xlim(-0.012, 0.52)
    axes.set_xticks([0, 0.05, 0.1, 0.2, 0.3, 0.4])
    axes.set_xticklabels(["0", f"{tolerance:g}", "0.1", "0.2", "0.3", "0.4"])
    axes.set_xlabel("Residual depth dependence, $|$AUC $-$ 0.5$|$")
    axes.set_yticks(range(len(rows)))
    # Look the label up by base name: one plate is one variant, and the
    # variant is named in the title, so repeating its suffix on all thirteen
    # rows would only push the readable part out of the gutter.
    axes.set_yticklabels(
        [SCREEN_LABEL.get(split_variant(name)[0], name) for name in rows],
        fontsize=6.6)
    axes.tick_params(axis="y", length=0)
    axes.set_ylim(len(rows) - 0.5, -0.5)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    for label, ok in zip(axes.get_yticklabels(), passes.reindex(rows)):
        if bool(ok):
            label.set_color(C_PART)
            label.set_fontweight("bold")
    if FIRST_EXTERNAL in rows:
        # The line alone: the rows below it are named after the packages that
        # implement them, so labelling the two blocks in words would repeat
        # that and cost space the panel does not have.
        axes.axhline(rows.index(FIRST_EXTERNAL) - 0.5, color="0.25", lw=0.8,
                     ls=(0, (2, 1.6)), zorder=2)

    handles = [plt.Line2D([], [], marker="o", ms=3.0, ls="none",
                          mfc=shade, mec="white", mew=0.4, label=f"{cv:.2f}")
               for shade, cv in zip(DEPTH_SHADES, spread.values)]
    axes.set_title(title, loc="left", fontweight="bold", fontsize=8)
    # Left of the panel, over the row-label gutter: the title is left-aligned
    # to the axes and runs most of its width, so the free space is that side.
    axes.text(-0.02, 1.012, "green = passes", transform=axes.transAxes,
              ha="right", va="bottom", fontsize=6.4, color=C_PART)
    return handles


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
          tolerance: float, minimum_f1: float, variant: str = "") -> None:
    f1, rejection, runtime = load_benchmark(benchmark_root)
    arms, replicates = load_screen(screen_root)
    present, missing, others = choose_variant(arms, variant)
    if not present:
        available = ", ".join(repr(name) for name in others) or "none"
        raise SystemExit(
            f"no configurations with variant {variant!r} under {screen_root}; "
            f"available variants: {available}"
        )
    arms = arms[arms.configuration.isin(present)]
    if not replicates.empty:
        replicates = replicates[replicates.configuration.isin(present)]
    effect, spread, power, worst, passes = screen_tables(
        arms, present, tolerance, minimum_f1)
    control_fraction = float(arms[arms.arm.eq(CONTROL)].perturbed_fraction.median())
    # Every output carries the variant, plate and source data alike. Naming
    # only the plate would let a comparison run overwrite the source data of
    # the run it is being compared against.
    stem = "fig_simulation_benchmark" + (f"_{variant}" if variant else "")

    # The bottom block carries one band per configuration, so it grows with
    # the screen while the top block stays a fixed size. Positions are set in
    # inches and converted, rather than left to a shared gridspec: the row
    # labels in (e) need a gutter three times wider than any axis above them,
    # and a gridspec would impose that gutter on every panel.
    rows = max(len(present), 4)
    pad_top, top_block, gap, pad_bottom, header = 0.28, 4.30, 0.50, 0.68, 0.64
    bands = 0.26 * rows
    height = pad_top + top_block + gap + bands + header + pad_bottom
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

        # 0.27, not 0.235: a passing row is drawn bold, and bold is wider,
        # which clipped the longest label once sct_ds_cos started passing.
        left, right, between = 0.27, 0.985, 0.022
        width_effect = (right - left - between) / 1.62
        axes_effect = figure.add_axes([left, pad_bottom / height,
                                      width_effect, bands / height])
        named = variant_title(variant) if variant else ""
        title = "(e)  Specificity: no cell here is incompatible"
        if named:
            title = f"(e)  Specificity, {named}"
        handles = panel_depth_effect(axes_effect, effect, replicates, spread,
                                     passes, tolerance, title)
        axes_power = figure.add_axes(
            [left + width_effect + between, pad_bottom / height,
             width_effect * 0.62, bands / height], sharey=axes_effect)
        panel_power(axes_power, power, passes, minimum_f1, control_fraction)

        # Above (e), right-aligned to it: inside the axes the legend lands on
        # whichever rows happen to have low values, which is a property of the
        # results, not of the layout.
        legend = figure.legend(
            handles=handles, ncol=4, loc="lower right",
            bbox_to_anchor=(left + width_effect,
                            (pad_bottom + bands + 0.22) / height),
            handlelength=0.8, handletextpad=0.35, columnspacing=0.9,
            borderpad=0.2, fontsize=5.8, title="depth CV across cells")
        legend.get_title().set_fontsize(5.8)

        out.mkdir(parents=True, exist_ok=True)
        for extension in ("pdf", "png"):
            figure.savefig(out / f"{stem}.{extension}")
        plt.close(figure)

    table = effect.rename(columns={name: f"depth_cv_{spread[name]:.2f}"
                                   for name in HOMOGENEOUS})
    table = table.join(power)
    table["worst_depth_effect"] = worst
    table["passes"] = passes.map({True: "yes", False: "no"})
    table.to_csv(out / f"{stem}_source.csv")
    if not replicates.empty:
        columns = [c for c in ("configuration", "arm", "replicate",
                               "observed_depth_cv", "auc_total_counts",
                               "depth_effect", "source_rejection_rate",
                               "perturbed_f1", "perturbed_recall",
                               "perturbed_precision") if c in replicates]
        (replicates[replicates.configuration.isin(present)][columns]
         .to_csv(out / f"{stem}_replicates.csv", index=False))

    print(f"wrote {out / (stem + '.pdf')}")
    print(f"wrote {out / (stem + '.png')}")
    print(f"wrote {out / (stem + '_source.csv')}")
    print()
    print(table.round(3).to_string())
    if missing:
        named = f"variant {variant!r}" if variant else "the default variant"
        print(f"\nnot in the screen for {named}, so absent from (e) and (f): "
              + ", ".join(missing))
    if others:
        print("other variants in this root, each needing its own --variant: "
              + ", ".join(repr(name) for name in others))


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
    parser.add_argument("--variant", default="",
                        help="which screen variant to plate: empty for the "
                             "default runs, 'splatter' for the ones built on "
                             "splatter's counts, 'r10' for a 10-replicate "
                             "round. One plate shows one variant, so two "
                             "simulators never share a panel.")
    arguments = parser.parse_args()
    build(arguments.benchmark_root, arguments.screen_root, arguments.out,
          arguments.tolerance, arguments.minimum_f1, arguments.variant)


if __name__ == "__main__":
    main()

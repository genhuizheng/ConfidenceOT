"""Four solvers at three preprocessing arms, on the benchmark the arm heatmap used.

The input is what ``benchmark/tacc/solvers.slurm`` writes: one directory per
arm under the solvers root, one directory per pair under it, each holding the
``solver_comparison.csv`` of ``50_solver_comparison.py``. The pairs are the arm
heatmap's own -- the native-depth simulation, four cases, three technical
levels, three sizes, five replicates -- and every solver is scored by the
fixed rule: ConfidenceOT by its M4-E gate, the three POT solvers by the 0.5
mass-deficit cutoff of ``comparator_rejection.py``.

Two figures.

**The heatmap** has a row per solver and arm, and its columns are
``45_plot_arm_heatmap.py``'s, imported rather than copied, so each column here
is the same readout at the same level and with the same ground truth as
there. Every cell is the fraction of the named cells rejected, the mean over
the pairs of that case and level.

**The bar plot** is directional F1, defined as the earlier Splatter benchmark
defined it (``scripts/run_splatter_dual_backbone_calibrated_case.py``): the F1
of rejection on the side that holds the cells to reject -- target for
population_emerged, source otherwise. That puts population_lost on the
source, population_emerged on the target, and populations_disjoint on the
source, which is also the only fair side there: once a solver has rejected
every source cell nothing is matched, and its target cells cannot be told
apart from kept ones. all_shared has nothing to reject, so no F1; its false
rejections are on the heatmap. F1 is 2TP / (2TP + FP + FN), so a solver that
rejects nothing scores 0 rather than dropping out; each bar is the mean over
pairs and its whisker the standard deviation.

``--check-against`` takes the arm heatmap's ``benchmark_metrics.csv`` and
requires ConfidenceOT's rejected fractions here to equal its M4-E rows there,
pair by pair: the two come from one gate, so any difference means the solver
comparison read the wrong run or misaligned its cells.

    python benchmark/55_plot_method_comparison.py SOLVERS_ROOT OUT_DIR \\
        --check-against benchmark_results/native/benchmark_metrics.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("arm_heatmap", HERE / "45_plot_arm_heatmap.py")
arm_heatmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(arm_heatmap)
READOUTS, LEVELS, GROUPS = arm_heatmap.READOUTS, arm_heatmap.LEVELS, arm_heatmap.GROUPS
RAMP, UNDEFINED = arm_heatmap.RAMP, arm_heatmap.UNDEFINED
INK, MUTED, RULE = arm_heatmap.INK, arm_heatmap.MUTED, arm_heatmap.RULE

ARMS = ["ranknm256_noscale_ds_cos", "ranknm256_noscale_cos", "raw"]
# (name in solver_comparison.csv, name on the figure), in the figure's order.
SOLVERS = [("ConfidenceOT M4-E", "ConfidenceOT"), ("Vanilla UOT", "Vanilla UOT"),
           ("Partial OT m=0.85", "Partial OT (m = 0.85)"),
           ("Traditional OT", "Balanced OT")]
CASES = ("all_shared", "population_lost", "population_emerged", "populations_disjoint")
PAIR = re.compile(r"^N(\d+)_rep(\d+)_(L\d_[a-z_]+?)_(" + "|".join(CASES) + r")$")
# The side directional F1 is read on, as the earlier benchmark chose it.
F1_SIDE = {"population_lost": "source", "population_emerged": "target",
           "populations_disjoint": "source"}
F1_CASE = {"population_lost": "Case 2, unmatched on source",
           "population_emerged": "Case 2, unmatched on target",
           "populations_disjoint": "Case 3, nothing shared"}
# Categorical slots 1-3 of the reference palette, in order.
ARM_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("solvers_root", type=Path,
                        help="The directory holding one subdirectory per arm")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--check-against", type=Path, default=None,
                        metavar="BENCHMARK_METRICS_CSV")
    return parser.parse_args()


def collect(root: Path) -> pd.DataFrame:
    """One row per arm, pair and solver, with both sides' counts and rates."""
    rows = []
    for arm in ARMS:
        for path in sorted((root / arm).glob("*/solver_comparison.csv")):
            pair_id = path.parent.name
            match = PAIR.match(pair_id)
            if match is None:
                raise SystemExit(f"cannot read the condition of {pair_id!r}")
            table = pd.read_csv(path)
            for solver, shown in SOLVERS:
                block = table[table["method"] == solver]
                if block.empty:
                    continue
                record = {"arm": arm, "pair_id": pair_id, "solver": shown,
                          "n_cells": int(match.group(1)),
                          "replicate": int(match.group(2)),
                          "technical_level": match.group(3),
                          "biological_case": match.group(4)}
                for side in ("source", "target"):
                    one = block[block["side"] == side]
                    if len(one) != 1:
                        raise SystemExit(f"{path}: {len(one)} {side} rows for {solver}")
                    one = one.iloc[0]
                    record[f"{side}_false_rejection_rate"] = one["fixed_false_rejection_rate"]
                    record[f"{side}_recall"] = one["fixed_recall"]
                    for count in ("tp", "fp", "fn"):
                        record[f"{side}_{count}"] = int(one[f"fixed_{count}"])
                rows.append(record)
    if not rows:
        raise SystemExit(f"no solver_comparison.csv under {root}/<arm>/<pair>/")
    return pd.DataFrame(rows)


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    return (table.groupby(["arm", "solver"]).size().rename("pairs")
            .reindex(pd.MultiIndex.from_product([ARMS, [s for _, s in SOLVERS]],
                                                names=["arm", "solver"]), fill_value=0)
            .reset_index())


def check(table: pd.DataFrame, metrics_csv: Path) -> None:
    """ConfidenceOT here against the arm heatmap's own M4-E rows."""
    reference = pd.read_csv(metrics_csv)
    reference = reference[reference["method"] == "M4-E"]
    ours = table[table["solver"] == "ConfidenceOT"]
    columns = ["source_false_rejection_rate", "source_recall",
               "target_false_rejection_rate", "target_recall"]
    joined = ours.merge(reference, left_on=["arm", "pair_id"],
                        right_on=["preprocessing_label", "pair_id"],
                        suffixes=("", "_reference"), validate="one_to_one")
    if len(joined) != len(ours):
        raise SystemExit(f"{len(ours) - len(joined)} ConfidenceOT pairs are not in "
                         f"{metrics_csv}")
    worst = 0.0
    for column in columns:
        here, there = joined[column].to_numpy(float), joined[f"{column}_reference"].to_numpy(float)
        both = ~(np.isnan(here) | np.isnan(there))
        if (np.isnan(here) != np.isnan(there)).any():
            raise SystemExit(f"{column}: defined here and undefined there for some pairs")
        if both.any():
            worst = max(worst, float(np.max(np.abs(here[both] - there[both]))))
    if worst > 1e-9:
        raise SystemExit(f"ConfidenceOT differs from {metrics_csv} by up to {worst:.3g}: "
                         f"the solver comparison did not read the gate the arm heatmap "
                         f"scored")
    print(f"check: ConfidenceOT equals the arm heatmap's M4-E rows on all "
          f"{len(joined)} pairs")


def heatmap(table: pd.DataFrame, out: Path) -> None:
    rows = [(shown, arm) for _, shown in SOLVERS for arm in ARMS]
    n_cols = len(READOUTS) * len(LEVELS)
    values = np.full((len(rows), n_cols), np.nan)
    counts = np.zeros((len(rows), n_cols), dtype=int)
    columns = []
    for j, (group, title, case, column, truth) in enumerate(READOUTS):
        for k, (level, short) in enumerate(LEVELS):
            col = j * len(LEVELS) + k
            columns.append((group, title, truth, short))
            part = table[(table["biological_case"] == case)
                         & (table["technical_level"] == level)]
            means = part.groupby(["solver", "arm"])[column].mean()
            sizes = part.groupby(["solver", "arm"])[column].count()
            for i, key in enumerate(rows):
                values[i, col] = means.get(key, np.nan)
                counts[i, col] = int(sizes.get(key, 0))
    frame = pd.DataFrame(values, index=pd.MultiIndex.from_tuples(rows, names=["solver", "arm"]),
                         columns=[f"{g} | {c.replace(chr(10), ' ')} | ground truth {t} | {s}"
                                  for g, c, t, s in columns])
    frame.round(4).to_csv(out / "method_comparison_heatmap_values.csv")
    complete = bool((counts == 15).all())

    n_rows = len(rows)
    cell_w, cell_h = 0.38, 0.26
    # The solver names get a column of their own, left of the arm names.
    left, right, top, bottom = 3.95, 1.55, 1.85, 0.55
    fig_w = left + n_cols * cell_w + right
    fig_h = top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([left / fig_w, bottom / fig_h,
                       n_cols * cell_w / fig_w, n_rows * cell_h / fig_h])
    cmap = LinearSegmentedColormap.from_list("rejected", RAMP)
    cmap.set_bad(UNDEFINED)
    mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=1.0,
                         edgecolors="#ffffff", linewidth=1.4)
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows, 0)
    ax.set_xticks([])
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels([arm for _, arm in rows], fontsize=7.4, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in range(len(ARMS), n_rows, len(ARMS)):
        ax.axhline(boundary, color="#ffffff", linewidth=4.5)
    for boundary in range(len(LEVELS), n_cols, len(LEVELS)):
        ax.axvline(boundary, color="#ffffff", linewidth=4.5)
    # The solver, once per block of arms, left of the arm names.
    for g, (_, shown) in enumerate(SOLVERS):
        middle = bottom + (n_rows - g * len(ARMS) - len(ARMS) / 2) * cell_h
        fig.text(0.3 / fig_w, middle / fig_h, shown, ha="left", va="center",
                 fontsize=7.8, fontweight="semibold", color=INK)
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

    fig.text(0.3 / fig_w, 1 - 0.18 / fig_h,
             "Fixed operating point: rejected fraction for each solver and arm",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    over = ("mean over 3 sizes x 5 replicates" if complete else
            "mean over the pairs that finished (15 per cell when complete)")
    fig.text(0.3 / fig_w, 1 - 0.42 / fig_h,
             f"Native-depth simulation, {over}. ConfidenceOT by its M4-E gate; UOT and "
             "Partial OT reject a cell with more than half its mass untransported; Balanced "
             "OT transports all of it.",
             fontsize=7.4, color=MUTED, va="top")
    fig.text(0.3 / fig_w, 1 - 0.62 / fig_h,
             "Ground truth under each column: 0 = those cells all have a match and "
             "should be kept; 1 = they have none and should be rejected.",
             fontsize=7.4, color=MUTED, va="top")
    cax = fig.add_axes([(left + n_cols * cell_w + 0.32) / fig_w, bottom / fig_h,
                        0.13 / fig_w, min(2.6, n_rows * cell_h) / fig_h])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    bar.ax.tick_params(labelsize=6.6, colors=MUTED, length=2)
    bar.outline.set_visible(False)
    cax.set_title("fraction\nrejected", fontsize=7.0, color=INK, loc="left", pad=6)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"method_comparison_heatmap.{suffix}", dpi=220)
    plt.close(fig)


def directional_f1(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in table.itertuples(index=False):
        side = F1_SIDE.get(record.biological_case)
        if side is None:
            continue
        tp, fp, fn = (getattr(record, f"{side}_{c}") for c in ("tp", "fp", "fn"))
        if tp + fn == 0:
            raise SystemExit(f"{record.pair_id}: nothing to reject on the {side} side")
        rows.append({"arm": record.arm, "solver": record.solver,
                     "pair_id": record.pair_id, "case": record.biological_case,
                     "technical_level": record.technical_level, "side": side,
                     "f1": 2 * tp / (2 * tp + fp + fn)})
    return pd.DataFrame(rows)


def bar_plot(f1: pd.DataFrame, out: Path) -> None:
    summary = (f1.groupby(["solver", "arm"])["f1"].agg(["mean", "std", "count"])
               .reindex(pd.MultiIndex.from_product([[s for _, s in SOLVERS], ARMS],
                                                   names=["solver", "arm"])).reset_index())
    by_case = (f1.groupby(["solver", "arm", "case"])["f1"].agg(["mean", "std", "count"])
               .reset_index())
    summary.round(4).to_csv(out / "method_comparison_f1.csv", index=False)
    by_case.round(4).to_csv(out / "method_comparison_f1_by_case.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.11)
    width, gap = 0.26, 0.02
    handles = []
    for k, (arm, colour) in enumerate(zip(ARMS, ARM_COLOURS)):
        part = summary[summary["arm"] == arm]
        x = np.arange(len(SOLVERS)) + (k - 1) * (width + gap)
        handles.append(ax.bar(x, part["mean"].fillna(0), width=width, color=colour,
                              label=arm, edgecolor="#ffffff", linewidth=1.0, zorder=2))
        ax.errorbar(x, part["mean"], yerr=part["std"], fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2, zorder=3)
        # Above the whisker, so the number never sits on the line.
        for xi, value, spread in zip(x, part["mean"], part["std"].fillna(0)):
            if np.isfinite(value):
                ax.text(xi, min(value + spread, 1.0) + 0.025, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=6.6, color=INK)
    ax.set_xticks(np.arange(len(SOLVERS)))
    ax.set_xticklabels([shown for _, shown in SOLVERS], fontsize=8, color=INK)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=7, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("directional F1", fontsize=8, color=INK)
    ax.grid(axis="y", color="#e6e6e1", linewidth=0.6, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.legend(handles=handles, frameon=False, fontsize=7.4, ncol=3,
               loc="upper left", bbox_to_anchor=(0.08, 0.835))
    pairs = int(summary["count"].max()) if summary["count"].notna().any() else 0
    fig.text(0.09, 0.97, "Fixed operating point: directional F1 for each solver and arm",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    fig.text(0.09, 0.915,
             "F1 of rejection on the side holding the cells to reject: the source in "
             "Case 2 (unmatched on source) and Case 3,",
             fontsize=7.0, color=MUTED, va="top")
    fig.text(0.09, 0.88,
             f"the target in Case 2 (unmatched on target). Mean over up to {pairs} pairs "
             "(3 cases x 3 levels x 3 sizes x 5 replicates); whisker 1 SD.",
             fontsize=7.0, color=MUTED, va="top")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"method_comparison_f1.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table = collect(args.solvers_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    present = coverage(table)
    present.to_csv(args.output_dir / "method_comparison_coverage.csv", index=False)
    short = present[present["pairs"] < 180]
    if len(short):
        print("pairs present, of 180 per arm and solver:")
        print(short.to_string(index=False))
    if args.check_against is not None:
        check(table, args.check_against)
    table.to_csv(args.output_dir / "method_comparison_pairs.csv", index=False)
    heatmap(table, args.output_dir)
    bar_plot(directional_f1(table), args.output_dir)
    print(f"wrote {args.output_dir / 'method_comparison_heatmap.png'} and "
          f"{args.output_dir / 'method_comparison_f1.png'}")


if __name__ == "__main__":
    main()

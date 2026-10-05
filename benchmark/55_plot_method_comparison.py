"""Five methods at three preprocessing arms, on the benchmark the arm heatmap used.

The input is ``main_decisions_per_pair.csv`` from ``56_comparator_metrics.py``:
one row per pair, side and strategy, holding the hard decisions' counts. The
pairs are the arm heatmap's own -- the native-depth simulation, four cases,
three technical levels, three sizes, five replicates. Thirteen strategies:

    ConfidenceOT M4-E (acceptance >= 0.90, 0.95, 0.99): native gate
    ConfidenceOT M4-R (acceptance >= 0.90, 0.95, 0.99): native gate
                                        (the same runs' reversible gate)
    Vanilla UOT: fixed cutoff (u > 0.5)
    Vanilla UOT: global-optimal threshold
    Partial OT (truth-derived m): fixed cutoff (u > 0.5)
    Partial OT (truth-derived m): global-optimal threshold
    IC-POT (c_s = c_t = 0.5): fixed cutoff (u > 0.5)
    IC-POT (c_s = c_t = 0.5): global-optimal threshold
    Balanced OT: no rejection

u is the share of a cell's own mass left untransported. The global-optimal
threshold is one per method and arm, chosen with the truth to maximise the
mean pair F1 on the sides holding both classes, then frozen and applied to
every pair, case and side: an upper bound, not a usable rule. Partial OT is
given its m from the truth of each pair, so its rejected mass was supplied,
not found. IC-POT runs at the literature's constant unmatched cost (Tripathi
et al., arXiv:2605.20030), and nothing in it comes from the truth. ConfidenceOT's
acceptance is the share of a within-side null its calibrated rejection cost
must accept; M4-R takes the cost M4-E was calibrated to.

Two figures.

**The heatmap** has a row per strategy and arm, and its columns are
``45_plot_arm_heatmap.py``'s, imported rather than copied, so each column here
is the same readout at the same level and with the same ground truth as
there. Every cell is the fraction of the named cells rejected, the mean over
the pairs of that case and level.

**The bar plot** is directional F1, as the earlier Splatter benchmark defined
it: the F1 of rejection on the side that holds the cells to reject -- the
source in population_lost and populations_disjoint, the target in
population_emerged. all_shared has nothing to reject and no F1; its false
rejections are on the heatmap. A strategy that rejects nothing scores 0
rather than dropping out; each bar is the mean over pairs, whisker 1 SD.

``--check-against`` takes the arm heatmap's ``benchmark_metrics.csv`` and
requires ConfidenceOT's rates here to equal its M4-E rows there, pair by
pair: the two come from one gate.

    python benchmark/55_plot_method_comparison.py \\
        comparator_metrics/main_decisions_per_pair.csv OUT_DIR \\
        --check-against benchmark_results/native/benchmark_metrics.csv
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


arm_heatmap = _load("arm_heatmap", "45_plot_arm_heatmap.py")
metrics = _load("comparator_metrics", "56_comparator_metrics.py")
READOUTS, LEVELS, GROUPS = arm_heatmap.READOUTS, arm_heatmap.LEVELS, arm_heatmap.GROUPS
RAMP, UNDEFINED = arm_heatmap.RAMP, arm_heatmap.UNDEFINED
INK, MUTED, RULE = arm_heatmap.INK, arm_heatmap.MUTED, arm_heatmap.RULE
STRATEGIES, DIRECTIONAL = metrics.STRATEGIES, metrics.DIRECTIONAL
ARMS = list(metrics.ARMS)
KEYS = ["strategy", "arm", "pair_id", "n_cells_nominal", "replicate", "technical_level",
        "biological_case"]
# Categorical slots 1-3 of the reference palette, in order.
ARM_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decisions_csv", type=Path,
                        help="main_decisions_per_pair.csv from 56_comparator_metrics.py")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--check-against", type=Path, default=None,
                        metavar="BENCHMARK_METRICS_CSV")
    return parser.parse_args()


def collect(path: Path) -> pd.DataFrame:
    """One row per strategy, arm and pair, with both sides' counts and rates."""
    long = pd.read_csv(path)
    unknown = sorted(set(long["strategy"]) - set(STRATEGIES))
    if unknown:
        raise SystemExit(f"{path} has strategies this figure does not know: {unknown}")
    sides = []
    for side in ("source", "target"):
        part = long[long["side"] == side].set_index(KEYS)
        sides.append(part[["false_rejection_rate", "recall", "tp", "fp", "fn"]]
                     .add_prefix(f"{side}_"))
    return sides[0].join(sides[1], how="outer").reset_index()


def coverage(table: pd.DataFrame) -> pd.DataFrame:
    return (table.groupby(["arm", "strategy"]).size().rename("pairs")
            .reindex(pd.MultiIndex.from_product([ARMS, STRATEGIES], names=["arm", "strategy"]),
                     fill_value=0)
            .reset_index())


def check(table: pd.DataFrame, metrics_csv: Path) -> None:
    """ConfidenceOT here against the arm heatmap's own M4-E rows."""
    reference = pd.read_csv(metrics_csv)
    reference = reference[reference["method"] == "M4-E"]
    ours = table[table["strategy"] == STRATEGIES[0]]
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
        if (np.isnan(here) != np.isnan(there)).any():
            raise SystemExit(f"{column}: defined here and undefined there for some pairs")
        both = ~np.isnan(here)
        if both.any():
            worst = max(worst, float(np.max(np.abs(here[both] - there[both]))))
    if worst > 1e-9:
        raise SystemExit(f"ConfidenceOT differs from {metrics_csv} by up to {worst:.3g}: "
                         f"the comparison did not read the gate the arm heatmap scored")
    print(f"check: ConfidenceOT equals the arm heatmap's M4-E rows on all {len(joined)} pairs")


def heatmap(table: pd.DataFrame, out: Path) -> None:
    rows = [(strategy, arm) for strategy in STRATEGIES for arm in ARMS]
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
            means = part.groupby(["strategy", "arm"])[column].mean()
            sizes = part.groupby(["strategy", "arm"])[column].count()
            for i, key in enumerate(rows):
                values[i, col] = means.get(key, np.nan)
                counts[i, col] = int(sizes.get(key, 0))
    frame = pd.DataFrame(values, index=pd.MultiIndex.from_tuples(rows, names=["strategy", "arm"]),
                         columns=[f"{g} | {c.replace(chr(10), ' ')} | ground truth {t} | {s}"
                                  for g, c, t, s in columns])
    frame.round(4).to_csv(out / "method_comparison_heatmap_values.csv")
    complete = bool((counts == 15).all())

    n_rows = len(rows)
    cell_w, cell_h = 0.38, 0.26
    # The strategy gets a column of its own, left of the arm names.
    left, right, top, bottom = 4.6, 1.55, 2.65, 0.55
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
    # The method and its strategy, once per block of arms, left of the arm names.
    for g, strategy in enumerate(STRATEGIES):
        method, rule = strategy.split(": ", 1)
        middle = bottom + (n_rows - g * len(ARMS) - len(ARMS) / 2) * cell_h
        fig.text(0.3 / fig_w, (middle + 0.07) / fig_h, method, ha="left", va="bottom",
                 fontsize=7.8, fontweight="semibold", color=INK)
        fig.text(0.3 / fig_w, (middle + 0.02) / fig_h, rule, ha="left", va="top",
                 fontsize=7.2, color=MUTED)
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
             "Rejected fraction for each method, strategy and preprocessing arm",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    over = ("mean over 3 sizes x 5 replicates" if complete else
            "mean over the pairs present (15 per cell when complete)")
    for line, text in enumerate((
            f"Native-depth simulation, {over}. u = the share of a cell's own mass left "
            "untransported; Balanced OT transports all of it and rejects nothing.",
            "Global-optimal threshold: one per method and arm, chosen with the truth (largest "
            "mean pair F1 on the sides holding both classes), then applied unchanged "
            "everywhere -- an upper bound.",
            "Partial OT is given m from each pair's truth. Ground truth under each column: "
            "0 = those cells all have a match and should be kept; 1 = they have none.",
            "IC-POT runs at the literature's constant unmatched cost, c_s = c_t = 0.5 "
            "(Tripathi et al., arXiv:2605.20030); nothing in it comes from the truth.",
            "ConfidenceOT acceptance: the share of a within-side null its calibrated rejection "
            "cost must accept; M4-R uses the cost M4-E was calibrated to.",
            "ConfidenceOT M4-R is the same run's reversible gate; at acceptance 0.90 its outer "
            "loop cycled and stopped at the iteration cap in most pairs other than "
            "populations_disjoint.")):
        fig.text(0.3 / fig_w, 1 - (0.42 + 0.2 * line) / fig_h, text,
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
        side = DIRECTIONAL.get(record.biological_case)
        if side is None:
            continue
        tp, fp, fn = (getattr(record, f"{side}_{c}") for c in ("tp", "fp", "fn"))
        if tp + fn == 0:
            raise SystemExit(f"{record.pair_id}: nothing to reject on the {side} side")
        rows.append({"arm": record.arm, "strategy": record.strategy,
                     "pair_id": record.pair_id, "case": record.biological_case,
                     "technical_level": record.technical_level, "side": side,
                     "f1": 2 * tp / (2 * tp + fp + fn)})
    return pd.DataFrame(rows)


def bar_plot(f1: pd.DataFrame, out: Path) -> None:
    summary = (f1.groupby(["strategy", "arm"])["f1"].agg(["mean", "std", "count"])
               .reindex(pd.MultiIndex.from_product([STRATEGIES, ARMS],
                                                   names=["strategy", "arm"])).reset_index())
    by_case = (f1.groupby(["strategy", "arm", "case"])["f1"].agg(["mean", "std", "count"])
               .reset_index())
    summary.round(4).to_csv(out / "method_comparison_f1.csv", index=False)
    by_case.round(4).to_csv(out / "method_comparison_f1_by_case.csv", index=False)

    # Wide enough that every strategy's label has its own column.
    fig, ax = plt.subplots(figsize=(1.7 * len(STRATEGIES) + 0.9, 5.1))
    fig.subplots_adjust(left=0.065, right=0.99, top=0.74, bottom=0.17)
    width, gap = 0.26, 0.02
    handles = []
    for k, (arm, colour) in enumerate(zip(ARMS, ARM_COLOURS)):
        part = summary[summary["arm"] == arm]
        x = np.arange(len(STRATEGIES)) + (k - 1) * (width + gap)
        handles.append(ax.bar(x, part["mean"].fillna(0), width=width, color=colour,
                              label=arm, edgecolor="#ffffff", linewidth=1.0, zorder=2))
        ax.errorbar(x, part["mean"], yerr=part["std"], fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2, zorder=3)
        # Above the whisker, so the number never sits on the line.
        for xi, value, spread in zip(x, part["mean"], part["std"].fillna(0)):
            if np.isfinite(value):
                ax.text(xi, min(value + spread, 1.0) + 0.025, f"{value:.2f}",
                        ha="center", va="bottom", fontsize=6.4, color=INK)
    ax.set_xticks(np.arange(len(STRATEGIES)))
    labels = []
    for strategy in STRATEGIES:
        method, rule = strategy.split(": ", 1)
        labels.append(method.replace(" (", "\n(") + "\n" + rule)
    ax.set_xticklabels(labels, fontsize=7.4, color=INK)
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
               loc="upper left", bbox_to_anchor=(0.058, 0.83))
    pairs = int(summary["count"].max()) if summary["count"].notna().any() else 0
    fig.text(0.065, 0.97, "Directional F1 for each method, strategy and preprocessing arm",
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    fig.text(0.065, 0.915,
             "F1 of rejection on the side holding the cells to reject: the source in "
             "population_lost and populations_disjoint, the target in population_emerged. "
             f"Mean over up to {pairs} pairs; whisker 1 SD.",
             fontsize=7.0, color=MUTED, va="top")
    fig.text(0.065, 0.88,
             "The global-optimal threshold and Partial OT's m are taken from the truth: "
             "upper bounds, not usable rules. ConfidenceOT and the fixed cutoff use no truth. "
             "At acceptance 0.90, M4-R's outer loop cycled in most pairs other than "
             "populations_disjoint.",
             fontsize=7.0, color=MUTED, va="top")
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"method_comparison_f1.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table = collect(args.decisions_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    present = coverage(table)
    present.to_csv(args.output_dir / "method_comparison_coverage.csv", index=False)
    short = present[present["pairs"] < 180]
    if len(short):
        print("pairs present, of 180 per arm and strategy:")
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

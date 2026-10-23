"""Patient-by-patient figures for the five CNV / RNA gate strategies, from the 51_ export package
and the 53_ patient UMAPs. No OT is rerun: the pair gates are combined per cell.

    python cancer_metastasis/54_patient_figures.py EXPORT_DIR PATIENT_UMAP_DIR FIGURES_DIR
        [--acceptances 0.99 0.95 0.90] [--datasets ...] [--sides primary] [--skip-umap]

**Per cell (rule c, no threshold).** A cell of a patient is a reference cell of one or more of
that patient's pairs (a primary sample is paired with each metastasis of the patient, and the
reverse). For each strategy and acceptance:
  gated pairs     the pairs whose gate scored the cell (single gates: retain or reject; a two-step
                  gate: its final state, retain / step 2 reject / step 1 reject). A two-step pair
                  whose step 2 did not run has no final state and is left out for that strategy.
  retained pairs  the gated pairs that (finally) retained the cell
  retained_fraction = retained pairs / gated pairs, NaN when no pair gated the cell.
The same cell can be retained against one metastasis and rejected against another; the fraction
says how often. A patient with one pair has fractions 0 or 1 only.

Writes to FIGURES_DIR:
  cell_retained_fraction_acc<a>.csv.gz  one row per cell x strategy: gated / retained pairs, fraction
  patient_summary.tsv   patient x strategy x side x acceptance: cells, pairs, the median number of
                        pairs that gated a cell (a primary sample meets only its own metastases), mean fraction, and the
                        shares of cells retained by every gated pair (f = 1), by some (0 < f < 1)
                        and by none (f = 0)
  umap/acc<a>/<dataset>/<patient>.png   the patient's joint UMAP: row 0 the samples, then one row
                        per strategy coloured by retained_fraction (vermillion 0, grey 0.5, blue 1)
  consistency_acc<a>_<dataset>.png   one row per patient, one column per strategy: stacked shares
                        of cells with f = 1 / 0 < f < 1 / f = 0, n pairs per patient
  overview_acc<a>_<dataset>_<side>.png   every patient of the dataset in one figure: one row per
                        patient on its own UMAP, columns samples + each strategy's retained fraction
  dataset_acc<a>_<dataset>_<side>.png  with --dataset-umap: every patient on one joint UMAP (53_
                        --group dataset), panels: patients, the authors' proliferating subtype, each
                        strategy's retained fraction with mean f in proliferating vs other cells
Titles say which coordinates were drawn (53_ --space raw or gate), read from reference.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

STRATEGIES = [("rna", "RNA only"), ("cnv", "CNV only"), ("rna_cnv", "CNV + RNA"),
              ("cnv_then_rna", "CNV -> RNA, final"), ("rna_then_cnv", "RNA -> CNV, final")]
SCORED = {"retain", "reject", "exclude"}  # exclude only occurs in the two-step final columns
SIDE_COLOURS = {"primary": "#762A83", "metastasis": "#1B7837"}
# diverging, Okabe-Ito poles as in 52_: never retained vermillion, always retained blue, grey midpoint
FRACTION = LinearSegmentedColormap.from_list("retained_fraction", ["#D55E00", "#BFBFBF", "#0072B2"])
CONSISTENCY = [("always", "retained by every gated pair (f = 1)", "#0072B2"),
               ("sometimes", "retained by some (0 < f < 1)", "#BFBFBF"),
               ("never", "retained by none (f = 0)", "#D55E00")]
CONTEXT = "#F0F0F0"
# samples: never the retained-fraction poles (#D55E00, #0072B2) or its grey midpoint
SAMPLE_COLOURS = ["#762A83", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442", "#8C510A", "#01665E",
                  "#000000", "#C51B7D"]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def patient_fractions(files: list[Path], acc: str) -> pd.DataFrame:
    """One row per cell x strategy: gated_pairs, retained_pairs, retained_fraction."""
    rows = []
    for path in files:
        columns = ["cell_uid", "pair_id", "side"] + [f"{s}__acc{acc}" for s, _ in STRATEGIES]
        cells = pd.read_csv(path, usecols=lambda c: c in set(columns))
        for strategy, _ in STRATEGIES:
            states = cells[f"{strategy}__acc{acc}"].astype(str)
            if states.str.startswith("step2_").any():  # step 2 did not run: no final state in this pair
                continue
            scored = states.isin(SCORED)
            rows.append(pd.DataFrame({"cell_uid": cells["cell_uid"], "side": cells["side"], "strategy": strategy,
                                      "gated": scored.astype(int), "retained": (states == "retain").astype(int)}))
    if not rows:
        return pd.DataFrame(columns=["cell_uid", "side", "strategy", "gated_pairs", "retained_pairs", "retained_fraction"])
    long = pd.concat(rows, ignore_index=True)
    table = long.groupby(["cell_uid", "side", "strategy"], as_index=False).agg(
        gated_pairs=("gated", "sum"), retained_pairs=("retained", "sum"))
    table["retained_fraction"] = table["retained_pairs"] / table["gated_pairs"].where(table["gated_pairs"] > 0)
    return table


def consistency(fraction: pd.Series) -> dict:
    f = fraction.dropna()
    if f.empty:
        return {"always": np.nan, "sometimes": np.nan, "never": np.nan}
    return {"always": float((f == 1).mean()), "sometimes": float(((f > 0) & (f < 1)).mean()),
            "never": float((f == 0).mean())}


def space_label(folder: Path) -> str:
    """What the coordinates were computed from, as 53_ recorded it."""
    record = folder / "reference.json"
    space = json.loads(record.read_text()).get("space", "raw") if record.exists() else "raw"
    return ("UMAP in the gate space (ranknm256_noscale_ds_cos, equalised counts)" if space == "gate"
            else "UMAP on raw counts (log-normalised, 48_ recipe)")


def umap_figure(coords: pd.DataFrame, fractions: pd.DataFrame, sides: list[str], acc: str, n_pairs: int,
                title: str, path: Path, space: str = "") -> None:
    x, y = coords["umap_1"].to_numpy(float), coords["umap_2"].to_numpy(float)
    pad_x, pad_y = (np.nanmax(x) - np.nanmin(x)) * 0.04, (np.nanmax(y) - np.nanmin(y)) * 0.04
    limits = ((np.nanmin(x) - pad_x, np.nanmax(x) + pad_x), (np.nanmin(y) - pad_y, np.nanmax(y) + pad_y))
    position = pd.Series(np.arange(len(coords)), index=coords["cell_uid"].to_numpy())
    order = np.random.default_rng(0).permutation(len(coords))
    side_of = coords["side"].to_numpy()
    fig, axes = plt.subplots(len(STRATEGIES) + 1, len(sides),
                             figsize=(4.4 * len(sides) + (0.9 if len(sides) == 1 else 0), 3.1 * (len(STRATEGIES) + 1)),
                             squeeze=False)
    norm = Normalize(0, 1)

    def frame(ax):
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bdbdbd")

    for j, side in enumerate(sides):
        mine = side_of == side
        size = float(np.clip(9000 / max(mine.sum(), 1), 1.0, 14))
        ax = axes[0, j]
        ax.scatter(x[~mine], y[~mine], s=size * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
        samples = sorted(coords.loc[mine, "sample"].astype(str).unique())
        for k, sample in enumerate(samples):
            hit = mine & (coords["sample"].astype(str).to_numpy() == sample)
            ax.scatter(x[hit], y[hit], s=size, c=SAMPLE_COLOURS[k % len(SAMPLE_COLOURS)], linewidths=0,
                       rasterized=True, label=f"{sample} ({int(hit.sum())})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.5, frameon=False, markerscale=2.5,
                  title="sample (cells)", title_fontsize=7)
        ax.set_title(f"{side.upper()}: {int(mine.sum())} cells, {len(samples)} sample(s)", fontsize=9,
                     color=SIDE_COLOURS[side], fontweight="bold")
        frame(ax)
        for i, (strategy, label) in enumerate(STRATEGIES, start=1):
            ax = axes[i, j]
            ax.scatter(x[~mine], y[~mine], s=size * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
            part = fractions[(fractions["strategy"] == strategy) & (fractions["side"] == side)]
            value = np.full(len(coords), np.nan)
            value[position.loc[part["cell_uid"]].to_numpy()] = part["retained_fraction"].to_numpy(float)
            draw = order[mine[order]]
            scored = draw[np.isfinite(value[draw])]
            unscored = draw[~np.isfinite(value[draw])]
            ax.scatter(x[unscored], y[unscored], s=size * 0.4, c="black", linewidths=0, rasterized=True)
            ax.scatter(x[scored], y[scored], s=size, c=value[scored], cmap=FRACTION, norm=norm, linewidths=0,
                       rasterized=True)
            share = consistency(part["retained_fraction"])
            text = (f"f=1 {share['always']:.0%}   0<f<1 {share['sometimes']:.0%}   f=0 {share['never']:.0%}"
                    if np.isfinite(share["always"]) else "not run")
            if len(unscored):
                text += f"\nno gated pair {len(unscored)} (black)"
            ax.set_title(text, fontsize=7.5)
            frame(ax)
            if j == 0:
                ax.set_ylabel(label, fontsize=9)
    axes[0, 0].set_ylabel("samples", fontsize=9)
    cax = fig.add_axes([0.25, 0.012, 0.5, 0.008])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FRACTION), cax=cax, orientation="horizontal")
    bar.set_label("retained fraction f = pairs that retained the cell / pairs that gated it", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    per_cell = fractions.loc[fractions["side"].isin(sides) & (fractions["gated_pairs"] > 0), "gated_pairs"]
    fig.suptitle(f"{title}  -  {n_pairs} pair(s); a cell is gated by {per_cell.median():g} (median, "
                 f"range {per_cell.min()}-{per_cell.max()}); acceptance {acc}\n{space}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def overview_figure(items: list, dataset: str, side: str, acc: str, space: str, path: Path) -> None:
    """Every patient of a dataset in one figure: one row per patient (its own UMAP), columns =
    samples, then each strategy's retained fraction; the same colours as the per-patient figures."""
    n = len(items)
    columns = ["samples"] + [label for _, label in STRATEGIES]
    fig, axes = plt.subplots(n, len(columns), figsize=(1.75 * len(columns) + 1.6, 1.75 * n + 1.2), squeeze=False)
    norm = Normalize(0, 1)
    for r, (patient, coords, fractions, n_pairs) in enumerate(items):
        x, y = coords["umap_1"].to_numpy(float), coords["umap_2"].to_numpy(float)
        mine = coords["side"].to_numpy() == side
        size = float(np.clip(2500 / max(mine.sum(), 1), 0.3, 6))
        position = pd.Series(np.arange(len(coords)), index=coords["cell_uid"].to_numpy())
        order = np.random.default_rng(0).permutation(len(coords))
        draw = order[mine[order]]
        for c in range(len(columns)):
            ax = axes[r, c]
            ax.scatter(x[~mine], y[~mine], s=size * 0.4, c=CONTEXT, linewidths=0, rasterized=True)
            if c == 0:
                samples = sorted(coords.loc[mine, "sample"].astype(str).unique())
                colour = {s: SAMPLE_COLOURS[k % len(SAMPLE_COLOURS)] for k, s in enumerate(samples)}
                ax.scatter(x[draw], y[draw], s=size, c=[colour[s] for s in coords["sample"].astype(str).to_numpy()[draw]],
                           linewidths=0, rasterized=True)
                ax.set_ylabel(f"{patient}\n{n_pairs} pair(s), {len(samples)} sample(s)", fontsize=6.5)
            else:
                strategy = STRATEGIES[c - 1][0]
                part = fractions[(fractions["strategy"] == strategy) & (fractions["side"] == side)]
                value = np.full(len(coords), np.nan)
                value[position.loc[part["cell_uid"]].to_numpy()] = part["retained_fraction"].to_numpy(float)
                scored = draw[np.isfinite(value[draw])]
                ax.scatter(x[scored], y[scored], s=size, c=value[scored], cmap=FRACTION, norm=norm, linewidths=0,
                           rasterized=True)
                share = consistency(part["retained_fraction"])
                if np.isfinite(share["always"]):
                    ax.text(0.02, 0.02, f"{share['always']:.0%} / {share['sometimes']:.0%} / {share['never']:.0%}",
                            transform=ax.transAxes, fontsize=5.5, color="#333333")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d0d0d0")
            if r == 0:
                ax.set_title(columns[c], fontsize=8)
    height = fig.get_size_inches()[1]  # margins in inches, so a tall dataset keeps room for its title
    cax = fig.add_axes([0.3, 0.25 / height, 0.4, 0.07 / height])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FRACTION), cax=cax, orientation="horizontal")
    bar.set_label("retained fraction f (corner: share of cells f=1 / 0<f<1 / f=0)", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    fig.suptitle(f"{dataset}  -  {side.upper()}, acceptance {acc}: every patient on its own joint UMAP\n{space}",
                 fontsize=9, color=SIDE_COLOURS[side])
    fig.tight_layout(rect=(0, 0.75 / height, 1, 1 - 0.75 / height))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# the authors' proliferating subtypes, as tools/make_gate_umap_figures.py matches them
PROLIFERATION = r"MKI67|MK167|PCNA|TOP2A|prolifer"


def dataset_figure(coords: pd.DataFrame, fractions: pd.DataFrame, annotation: pd.Series, dataset: str,
                   side: str, acc: str, space: str, path: Path) -> None:
    """Every patient of a dataset on one joint UMAP: the other side as backdrop, this side coloured
    by retained fraction per strategy; plus the patients and the authors' proliferating subtype."""
    x, y = coords["umap_1"].to_numpy(float), coords["umap_2"].to_numpy(float)
    mine = coords["side"].to_numpy() == side
    size = float(np.clip(30000 / max(mine.sum(), 1), 0.15, 5))
    order = np.random.default_rng(0).permutation(len(coords))
    draw = order[mine[order]]
    position = pd.Series(np.arange(len(coords)), index=coords["cell_uid"].to_numpy())
    labels = annotation.reindex(coords["cell_uid"]).astype(str).to_numpy()
    proliferating = pd.Series(labels).str.contains(PROLIFERATION, case=False, regex=True).to_numpy() & mine
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), squeeze=False)
    flat = axes.ravel()

    def frame(ax, title):
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d0d0d0")
        ax.set_title(title, fontsize=9)

    ax = flat[0]
    ax.scatter(x[~mine], y[~mine], s=size * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
    patients = sorted(coords.loc[mine, "patient_id"].astype(str).unique())
    colour = {p: SAMPLE_COLOURS[k % len(SAMPLE_COLOURS)] for k, p in enumerate(patients)}
    ax.scatter(x[draw], y[draw], s=size, c=[colour[p] for p in coords["patient_id"].astype(str).to_numpy()[draw]],
               linewidths=0, rasterized=True)
    if len(patients) <= 10:
        for p in patients:
            ax.scatter([], [], s=20, c=colour[p], label=p)
        ax.legend(loc="lower left", fontsize=6, frameon=False, markerscale=1)
    frame(ax, f"{side} cells by patient ({len(patients)} patients, {int(mine.sum()):,} cells)")
    ax = flat[1]
    ax.scatter(x[~mine], y[~mine], s=size * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
    ax.scatter(x[mine & ~proliferating], y[mine & ~proliferating], s=size, c="#BDBDBD", linewidths=0, rasterized=True)
    ax.scatter(x[proliferating], y[proliferating], s=size * 1.3, c="black", linewidths=0, rasterized=True)
    frame(ax, f"authors' proliferating subtype (black, {int(proliferating.sum()):,} cells)"
              if proliferating.any() else "authors' labels name no proliferating subtype")
    norm = Normalize(0, 1)
    for ax, (strategy, label) in zip(flat[2:7], STRATEGIES):
        ax.scatter(x[~mine], y[~mine], s=size * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
        part = fractions[(fractions["strategy"] == strategy) & (fractions["side"] == side)]
        value = np.full(len(coords), np.nan)
        known = part["cell_uid"].isin(position.index)
        value[position.loc[part.loc[known, "cell_uid"]].to_numpy()] = part.loc[known, "retained_fraction"].to_numpy(float)
        scored = draw[np.isfinite(value[draw])]
        ax.scatter(x[scored], y[scored], s=size, c=value[scored], cmap=FRACTION, norm=norm, linewidths=0,
                   rasterized=True)
        share = consistency(part["retained_fraction"])
        text = f"f=1 {share['always']:.0%}   0<f<1 {share['sometimes']:.0%}   f=0 {share['never']:.0%}"
        if proliferating.any():
            inside = np.nanmean(value[proliferating])
            outside = np.nanmean(value[mine & ~proliferating])
            text += f"\nmean f: proliferating {inside:.2f} vs other {outside:.2f}"
        frame(ax, f"{label}\n{text}")
    flat[7].axis("off")
    flat[7].text(0.0, 0.95, f"{side.upper()} cells coloured; the other side is the grey backdrop.\n"
                 "f = pairs that retained the cell / pairs that gated it\n(a primary cell meets each metastasis "
                 "its sample was paired with).\nPatients usually separate on a joint UMAP;\nthat is between-patient "
                 "difference, not the gate.", fontsize=8, va="top", transform=flat[7].transAxes)
    cax = fig.add_axes([0.78, 0.22, 0.18, 0.02])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=FRACTION), cax=cax, orientation="horizontal")
    bar.set_label("retained fraction f", fontsize=8)
    fig.suptitle(f"{dataset}  -  all patients, acceptance {acc}\n{space}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def consistency_figure(summary: pd.DataFrame, dataset: str, side: str, acc: str, path: Path) -> None:
    part = summary[(summary["dataset_id"] == dataset) & (summary["side"] == side) & (summary["acceptance"] == acc)]
    patients = sorted(part["patient_id"].unique())
    fig, axes = plt.subplots(1, len(STRATEGIES), figsize=(2.9 * len(STRATEGIES) + 1.6, 0.32 * len(patients) + 1.8),
                             sharey=True, squeeze=False)
    pairs = part.groupby("patient_id")["pairs"].max()
    per_cell = part.groupby("patient_id")["median_gated_pairs_per_cell"].median()
    for ax, (strategy, label) in zip(axes[0], STRATEGIES):
        here = part[part["strategy"] == strategy].set_index("patient_id")
        for i, patient in enumerate(patients):
            if patient not in here.index or not np.isfinite(here.loc[patient, "always"]):
                ax.text(0.01, i, "not run", va="center", fontsize=6.5, color="#636363")
                continue
            left = 0.0
            for key, _, colour in CONSISTENCY:
                value = float(here.loc[patient, key])
                ax.barh(i, value, left=left, height=0.75, color=colour, edgecolor="white", linewidth=0.6)
                if value >= 0.12:
                    ax.text(left + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=6,
                            color="black" if colour == "#BFBFBF" else "white")
                left += value
        ax.set_xlim(0, 1); ax.set_ylim(len(patients) - 0.5, -0.5)
        ax.set_title(label, fontsize=8.5)
        ax.tick_params(axis="x", labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0][0].set_yticks(range(len(patients)))
    axes[0][0].set_yticklabels([f"{p} ({int(pairs[p])} pair{'s' if pairs[p] > 1 else ''}; "
                                f"{per_cell[p]:g} per cell)" for p in patients], fontsize=7)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=t) for _, t, c in CONSISTENCY]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"{dataset}  -  {side.upper()}, acceptance {acc}: share of each patient's cells by retained fraction",
                 fontsize=9.5, color=SIDE_COLOURS[side])
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path)
    parser.add_argument("patient_umap", type=Path, help="53_patient_umap.py output (holds patients/)")
    parser.add_argument("figures", type=Path)
    parser.add_argument("--acceptances", nargs="+", default=["0.99", "0.95", "0.90"])
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--sides", nargs="+", choices=["primary", "metastasis"], default=["primary", "metastasis"])
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--dataset-umap", type=Path,
                        help="53_ --group dataset output (holds datasets/): adds one all-patient figure per dataset")
    args = parser.parse_args()
    sides = [s for s in ("primary", "metastasis") if s in args.sides]
    args.figures.mkdir(parents=True, exist_ok=True)
    files = sorted((args.export / "cells").glob("*/*.csv.gz"))
    index = pd.concat([pd.read_csv(f, usecols=["dataset_id", "patient_id", "pair_id"], nrows=1).assign(path=str(f))
                       for f in files], ignore_index=True)
    if args.datasets:
        index = index[index["dataset_id"].isin(args.datasets)]
    summary_rows, missing = [], []
    overview_space = ""
    for acc in args.acceptances:
        all_cells = []
        overview: dict = {}
        for (dataset, patient), part in index.groupby(["dataset_id", "patient_id"], sort=True):
            fractions = patient_fractions([Path(p) for p in part["path"]], acc)
            fractions.insert(0, "patient_id", patient)
            fractions.insert(0, "dataset_id", dataset)
            all_cells.append(fractions)
            for (strategy, side), group in fractions.groupby(["strategy", "side"]):
                summary_rows.append({"dataset_id": dataset, "patient_id": patient, "acceptance": acc,
                                     "strategy": strategy, "side": side, "pairs": len(part),
                                     "cells": int(group["retained_fraction"].notna().sum()),
                                     "median_gated_pairs_per_cell": float(group.loc[group["gated_pairs"] > 0, "gated_pairs"].median()),
                                     "mean_retained_fraction": float(group["retained_fraction"].mean()),
                                     **consistency(group["retained_fraction"])})
            if args.skip_umap:
                continue
            umap_file = args.patient_umap / "patients" / dataset / str(patient) / "patient_umap.csv.gz"
            if not umap_file.exists():
                missing.append(f"{dataset} {patient}")
                continue
            coords = pd.read_csv(umap_file)
            overview_space = space_label(umap_file.parent)
            placed = fractions["cell_uid"].isin(set(coords["cell_uid"]))
            unplaced = fractions[~placed & fractions["retained_fraction"].notna()]
            if len(unplaced):  # a gated cell must have coordinates; say so loudly if one does not
                print(f"WARNING {dataset} {patient} acc {acc}: {unplaced['cell_uid'].nunique()} gated cells "
                      f"have no coordinates in {umap_file}")
            fractions = fractions[placed]
            umap_figure(coords, fractions, sides, acc, len(part), f"{dataset}  {patient}",
                        args.figures / "umap" / f"acc{acc}" / dataset / f"{patient}.png", overview_space)
            overview.setdefault(dataset, []).append((patient, coords, fractions, len(part)))
        for dataset, items in overview.items():
            for side in sides:
                overview_figure(items, dataset, side, acc, overview_space,
                                args.figures / f"overview_acc{acc}_{dataset}_{side}.png")
        every = pd.concat(all_cells, ignore_index=True)
        every.to_csv(args.figures / f"cell_retained_fraction_acc{acc}.csv.gz", index=False, compression="gzip")
        if args.dataset_umap is not None:
            for dataset in sorted(index["dataset_id"].unique()):
                umap_file = args.dataset_umap / "datasets" / dataset / "patient_umap.csv.gz"
                if not umap_file.exists():
                    print(f"no dataset UMAP for {dataset}")
                    continue
                coords = pd.read_csv(umap_file)
                annotation = pd.concat([pd.read_csv(f, usecols=["cell_uid", "annotation"]) for f in
                                        index.loc[index["dataset_id"] == dataset, "path"]]).drop_duplicates("cell_uid")
                for side in sides:
                    dataset_figure(coords, every[every["dataset_id"] == dataset],
                                   annotation.set_index("cell_uid")["annotation"], dataset, side, acc,
                                   space_label(umap_file.parent), args.figures / f"dataset_acc{acc}_{dataset}_{side}.png")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.figures / "patient_summary.tsv", sep="\t", index=False)
    for acc in args.acceptances:
        for dataset in sorted(summary["dataset_id"].unique()):
            for side in sides:
                consistency_figure(summary, dataset, side, acc,
                                   args.figures / f"consistency_acc{acc}_{dataset}_{side}.png")
    if missing:
        print(f"no patient UMAP for {len(sorted(set(missing)))} patient(s): {sorted(set(missing))[:10]}")
    print(f"done: {args.figures}")


if __name__ == "__main__":
    main()

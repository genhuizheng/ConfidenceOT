"""Figures for the five CNV / RNA gate strategies, from the 51_strategy_export.py package.

    python cancer_metastasis/52_strategy_figures.py EXPORT_DIR FIGURES_DIR [--acceptances 0.99]
        [--datasets GSE180661 ...] [--pairs PAIR ...] [--min-patients 3] [--skip-umap]

Runs anywhere the package is (a laptop after download, or TACC): it reads only the package.

1. umap/acc<a>/<dataset>/<pair>.png: one figure per pair and acceptance. Columns: primary,
   metastasis, headed with the side's sample and cell count. Row 0 shows where each side sits
   (primary #762A83, metastasis #1B7837); then rna, cnv, rna_cnv, cnv_then_rna step 1,
   cnv_then_rna final, rna_then_cnv step 1, rna_then_cnv final, each panel titled with counts
   and % of the side's gated cells. One frozen joint UMAP per pair, so every panel of a pair has
   the same coordinates and axis limits. Okabe-Ito, the bar plot's colours and names: single
   gates retain #0072B2 / reject #D55E00; two-step final rows step 1+2 retain #56B4E9 / step 2
   reject #E69F00 / step 1 reject #CC79A7; no state #b3b3b3; drawn in random order, the other
   side's cells #ececec behind.
2. rejected_fraction_acc<a>.png (+ .tsv): pairs x (strategy, side), one quantity: the share of
   the side's gated cells not retained (1 - retain), so a two-step column counts step 1
   exclusions and step 2 rejections together. Gated = retain + reject + exclude: not_gated
   reference cells are not counted, and a two-step pair whose step 2 did not run is blank, as is
   not run.
5. state_rates_acc<a>.png (+ .tsv): primary panels above, metastasis below, one column per
   dataset, one bar per strategy: single gates retain / reject, two-step gates step 1 retain +
   step 2 retain / step 1 retain + step 2 reject / step 1 reject; each the mean over pairs of the
   pair's gated-cell share, n pairs per bar; the .tsv adds the pooled cell shares.
3. gsea/<accession>_acc<a>.png: rows = strategy x side, columns = the 50 hallmark sets in
   47_gsea_dotplot.py's categories; NES blue-white-orange at +/-2, size = -log10 FDR in 47_'s bins,
   black outline = FDR < 0.05. A row with fewer than --min-patients paired patients is drawn
   without outlines and labelled "no significance (n < 3)"; a row with no result is labelled
   with the reason (FAILED, not run).
4. deg/<accession>_<side>_acc<a>.png: genes x strategies, log2 fold change (RdBu_r, symmetric),
   black dot = DESeq2 FDR < 0.05; genes are the union of each strategy's 10 largest |Wald
   statistic|, the same rule whether or not any gene is significant (a comparison with none is
   drawn and says so). A strategy with no DEG table keeps its column, grey, labelled with the
   reason; a comparison with no table at all is listed in figures_index.tsv with the reason.
figures_index.tsv lists every figure with its inputs and flags.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("dotplot47", HERE / "47_gsea_dotplot.py")
dotplot = importlib.util.module_from_spec(_spec)
sys.modules["dotplot47"] = dotplot
_spec.loader.exec_module(dotplot)

LABEL = "ranknm256_noscale_ds_cos"
ROWS = [("rna", "RNA only"), ("cnv", "CNV only"), ("rna_cnv", "CNV + RNA"),
        ("cnv_then_rna_step1", "CNV -> RNA, step 1 (CNV)"), ("cnv_then_rna", "CNV -> RNA, final"),
        ("rna_then_cnv_step1", "RNA -> CNV, step 1 (RNA)"), ("rna_then_cnv", "RNA -> CNV, final")]
STRATEGIES = [("rna", "RNA only"), ("cnv", "CNV only"), ("rna_cnv", "CNV + RNA"),
              ("cnv_then_rna", "CNV -> RNA, step 2"), ("rna_then_cnv", "RNA -> CNV, step 2")]
SHOW = ["primary", "metastasis"]  # the sides drawn (--sides); the tables always keep both
OPPOSITE = {"primary": "metastasis", "metastasis": "primary"}
SIDES = [("primary", "primary_rejected_vs_primary_retained"),
         ("metastasis", "metastasis_rejected_vs_metastasis_retained")]
# Okabe-Ito: blue / vermillion / black stay apart under every common colour-vision deficiency
COLOURS = {"retain": "#0072B2", "reject": "#D55E00", "exclude": "#000000"}
OTHER = "#b3b3b3"          # this side's cells with no state (not gated, not run)
CONTEXT = "#ececec"        # the other side's cells, behind
SIDE_COLOURS = {"primary": "#762A83", "metastasis": "#1B7837"}
# bar plot, Okabe-Ito: blues = retained at the end, oranges = rejected by the last gate, purple =
# rejected at step 1; single-gate and two-step states keep their own names and colours
BAR_COLOURS = {"retain": "#0072B2", "reject": "#D55E00", "both_retain": "#56B4E9",
               "step2_reject": "#E69F00", "step1_reject": "#CC79A7"}
LIGHT_FILLS = {"#56B4E9", "#E69F00"}
TWO_STEP_NAMES = {"cnv_then_rna", "rna_then_cnv"}
BLOCKS = {"GSE271675": "prostate"}
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def umap_figure(cells: pd.DataFrame, acc: str, path: Path) -> None:
    """Row 0: where each side sits on the pair's joint UMAP. Rows 1-7: one gate view each; the
    column's side in state colours (drawn in random order, so no state hides another), the
    other side pale grey behind. Same coordinates and limits in every panel."""
    x, y = cells["umap_1"].to_numpy(float), cells["umap_2"].to_numpy(float)
    pad_x, pad_y = (np.nanmax(x) - np.nanmin(x)) * 0.04, (np.nanmax(y) - np.nanmin(y)) * 0.04
    limits = ((np.nanmin(x) - pad_x, np.nanmax(x) + pad_x), (np.nanmin(y) - pad_y, np.nanmax(y) + pad_y))
    sides = ("primary", "metastasis")
    mask = {side: cells["side"].eq(side).to_numpy() for side in sides}
    sample = {side: ", ".join(sorted(cells.loc[mask[side], "sample"].astype(str).unique())) for side in sides}
    size = {side: float(np.clip(9000 / max(mask[side].sum(), 1), 1.5, 14)) for side in sides}
    order = np.random.default_rng(0).permutation(len(cells))
    fig, axes = plt.subplots(len(ROWS) + 1, len(SHOW), figsize=(4.2 * len(SHOW) + (0.8 if len(SHOW) == 1 else 0),
                                                              3.0 * (len(ROWS) + 1)), squeeze=False)

    def frame(ax):
        ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#bdbdbd")

    for j, side in enumerate(SHOW):
        ax = axes[0, j]
        other = mask[OPPOSITE[side]]
        ax.scatter(x[other], y[other], s=size[side] * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
        ax.scatter(x[mask[side]], y[mask[side]], s=size[side], c=SIDE_COLOURS[side], linewidths=0, rasterized=True)
        ax.set_title(f"{side.upper()}: {sample[side]}\n{int(mask[side].sum())} cells "
                     f"(grey: the {OPPOSITE[side]} cells)", fontsize=9, color=SIDE_COLOURS[side], fontweight="bold")
        frame(ax)
    axes[0, 0].set_ylabel("sample", fontsize=9)
    # a two-step final row names each state by what each step did, in the bar plot's colours
    single_view = {"retain": ("retain", BAR_COLOURS["retain"]), "reject": ("reject", BAR_COLOURS["reject"])}
    final_view = {"retain": ("step 1+2 retain", BAR_COLOURS["both_retain"]),
                  "reject": ("step 2 reject", BAR_COLOURS["step2_reject"]),
                  "exclude": ("step 1 reject", BAR_COLOURS["step1_reject"])}
    for i, (column, title) in enumerate(ROWS, start=1):
        view = final_view if column in TWO_STEP_NAMES else single_view
        states = cells[f"{column}__acc{acc}"].astype(str).to_numpy()
        for j, side in enumerate(SHOW):
            ax = axes[i, j]
            other = mask[OPPOSITE[side]]
            ax.scatter(x[other], y[other], s=size[side] * 0.5, c=CONTEXT, linewidths=0, rasterized=True)
            mine = order[mask[side][order]]
            colours = [view[s][1] if s in view else OTHER for s in states[mine]]
            ax.scatter(x[mine], y[mine], s=size[side], c=colours, linewidths=0, rasterized=True)
            tally = pd.Series(states[mask[side]]).value_counts()
            scored = sum(int(tally.get(k, 0)) for k in view)
            parts = [f"{view[k][0]} {int(tally[k])} ({tally[k] / scored:.0%})" for k in view if k in tally and scored]
            rest = int(mask[side].sum()) - scored
            if rest:
                parts.append(f"no state {rest}")
            text = "  ".join(parts)
            if len(text) > 48:  # two lines, so neighbouring panels' titles never touch
                text = "  ".join(parts[:2]) + "\n" + "  ".join(parts[2:])
            ax.set_title(text, fontsize=7.5)
            frame(ax)
        axes[i, 0].set_ylabel(title, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=7, color=c, label=k)
               for k, c in list(single_view.values()) + list(final_view.values())]
    handles += [plt.Line2D([], [], marker="o", ls="", ms=7, color=OTHER, label="this side, no state"),
                plt.Line2D([], [], marker="o", ls="", ms=7, color=CONTEXT, label="the other side")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(f"{cells['dataset_id'].iloc[0]}  {cells['pair_id'].iloc[0]}  -  acceptance {acc}; "
                 f"% of the side's gated cells", fontsize=9.5)
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


KEY = ["pair_id", "dataset_id", "strategy", "side"]


def gated_fractions(counts: pd.DataFrame, acc: str) -> pd.DataFrame:
    """Per pair x strategy x side: retain / reject / exclude as shares of the cells the gates
    scored. Denominator = retain + reject + exclude: not_gated cells (reference cells no gate
    scored, the same in every strategy) are not counted as rejected. A two-step pair whose step 2
    did not run has no final state, so it is left out of that strategy (not counted as excluded)."""
    part = counts[counts["acceptance"].astype(str) == acc]
    unfinished = part.loc[part["state"].astype(str).str.startswith("step2_"), ["pair_id", "strategy"]].drop_duplicates()
    part = part.merge(unfinished.assign(unfinished=True), on=["pair_id", "strategy"], how="left")
    part = part[part["unfinished"].isna() & part["state"].isin(list(COLOURS))]
    wide = part.pivot_table(index=KEY, columns="state", values="cells", aggfunc="sum", fill_value=0)
    wide = wide.reindex(columns=list(COLOURS), fill_value=0)
    gated = wide.sum(axis=1)
    shares = wide.div(gated, axis=0)[gated > 0]
    shares["gated_cells"] = gated[gated > 0]
    return shares.reset_index()


def state_rates(counts: pd.DataFrame, acc: str, out: Path) -> Path:
    """Stacked bars per dataset: mean over pairs of each pair's retain / reject / exclude share
    (each pair weighs the same; datasets are never pooled). The .tsv also has the pooled cell
    shares and the number of pairs behind each bar."""
    shares = gated_fractions(counts, acc)
    mean = shares.groupby(["dataset_id", "strategy", "side"])[list(COLOURS)].mean()
    pooled = shares.assign(**{k: shares[k] * shares["gated_cells"] for k in COLOURS}).groupby(
        ["dataset_id", "strategy", "side"])[list(COLOURS) + ["gated_cells"]].sum()
    pooled[list(COLOURS)] = pooled[list(COLOURS)].div(pooled["gated_cells"], axis=0)
    pairs = shares.groupby(["dataset_id", "strategy", "side"]).size().rename("pairs")
    table = mean.add_prefix("mean_").join(pooled.add_prefix("pooled_")).join(pairs).reset_index()
    table.to_csv(out / f"state_rates_acc{acc}.tsv", sep="\t", index=False)
    datasets = sorted(table["dataset_id"].unique())
    # one bar per strategy; a two-step bar is split by what each step did to the cell
    bars = [("rna", "RNA only"), ("cnv", "CNV only"), ("rna_cnv", "CNV + RNA"),
            ("cnv_then_rna", "CNV -> RNA (step 1 CNV, step 2 RNA)"),
            ("rna_then_cnv", "RNA -> CNV (step 1 RNA, step 2 CNV)")]
    single = [("retain", "retain", BAR_COLOURS["retain"]), ("reject", "reject", BAR_COLOURS["reject"])]
    two_step = [("retain", "step 1 retain + step 2 retain", BAR_COLOURS["both_retain"]),
                ("reject", "step 1 retain + step 2 reject", BAR_COLOURS["step2_reject"]),
                ("exclude", "step 1 reject", BAR_COLOURS["step1_reject"])]
    fig, axes = plt.subplots(len(SHOW), len(datasets), figsize=(4.8 * len(datasets),
                                                               len(SHOW) * (0.5 * len(bars) + 0.9) + 1.2),
                             sharey=True, squeeze=False)
    for r, side in enumerate(SHOW):
        for ax, dataset in zip(axes[r], datasets):
            here = table[table["dataset_id"] == dataset].set_index(["strategy", "side"])
            for i, (strategy, _) in enumerate(bars):
                key = (strategy, side)
                if key not in here.index:
                    ax.text(0.01, i, "not run", va="center", fontsize=7, color="#636363")
                    continue
                left = 0.0
                for state, _, colour in (two_step if strategy in TWO_STEP_NAMES else single):
                    value = float(here.loc[key, f"mean_{state}"])
                    ax.barh(i, value, left=left, height=0.7, color=colour, edgecolor="white", linewidth=1)
                    if value >= 0.06:
                        ink = "black" if colour in LIGHT_FILLS else "white"
                        ax.text(left + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=7, color=ink)
                    left += value
                ax.text(1.01, i, f"n={int(here.loc[key, 'pairs'])}", va="center", fontsize=7, color="#636363")
            ax.axhline(2.5, color="#bdbdbd", lw=0.8, ls=":")  # single gates above, two-step below
            ax.set_xlim(0, 1); ax.set_ylim(len(bars) - 0.5, -0.5)
            ax.set_title(f"{dataset}  -  {side.upper()}", fontsize=9.5, color=SIDE_COLOURS[side], fontweight="bold")
            ax.tick_params(axis="x", labelsize=7)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if r == len(SHOW) - 1:
                ax.set_xlabel("mean share of the pair's gated cells", fontsize=8)
        axes[r][0].set_yticks(range(len(bars)))
        axes[r][0].set_yticklabels([label for _, label in bars], fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=label) for _, label, c in single + two_step]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"Gate states at acceptance {acc}: mean over pairs of each pair's share (n = pairs behind each bar)",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=2.0)
    path = out / f"state_rates_acc{acc}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def rejected_fraction(counts: pd.DataFrame, acc: str, out: Path) -> Path:
    shares = gated_fractions(counts, acc)
    table = shares.assign(not_retained=1 - shares["retain"])[KEY + ["not_retained"]]
    columns = [(s, side) for s, _ in ROWS for side in SHOW]
    wide = table.pivot_table(index=["dataset_id", "pair_id"], columns=["strategy", "side"], values="not_retained")
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(columns)).sort_index()
    wide.to_csv(out / f"rejected_fraction_acc{acc}.tsv", sep="\t")
    fig, ax = plt.subplots(figsize=(2.5 + 0.42 * len(columns), max(4.0, 2.6 + 0.13 * len(wide))))
    mesh = ax.pcolormesh(np.ma.masked_invalid(wide.to_numpy(float)), cmap="Blues", vmin=0, vmax=1,
                         edgecolors="white", linewidth=0.3)
    ax.set_ylim(len(wide), 0)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels([f"{dict(ROWS)[s]}\n{side}" for s, side in columns], rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(wide)) + 0.5)
    ax.set_yticklabels([f"{d}  {p}" for d, p in wide.index], fontsize=5)
    fig.colorbar(mesh, ax=ax, fraction=0.03, label="share of the side's cells\nnot retained")
    fig.tight_layout()
    path = out / f"rejected_fraction_acc{acc}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def unit_dir(export: Path, acc: str, side: str, strategy: str, accession: str) -> Path:
    block = BLOCKS.get(accession, "uniform")
    return export / "deg_gsea" / f"acc{acc}" / side / strategy / "arm_A" / block / accession / LABEL


def paired_patients(unit: Path, contrast: str) -> int | None:
    for path in unit.rglob("pseudobulk_sample_metadata.csv"):
        meta = pd.read_csv(path)
        if "contrast" in meta:
            meta = meta[meta["contrast"].astype(str) == contrast]
        if {"patient_id", "comparison_status"} <= set(meta.columns):
            both = meta.groupby("patient_id")["comparison_status"].nunique()
            return int((both >= 2).sum())
    return None


def unit_status(unit: Path) -> str:
    if (unit / "DONE").exists():
        return "done"
    if (unit / "FAILED").exists():
        return "FAILED: " + (unit / "FAILED").read_text(encoding="utf-8").strip().splitlines()[-1][:70]
    return "not run"


def gsea_figure(export: Path, accession: str, acc: str, min_patients: int, path: Path) -> dict:
    pathways = dotplot.PATHWAYS
    rows, labels, flags = [], [], []
    for side, contrast in [s for s in SIDES if s[0] in SHOW]:
        for strategy, title in STRATEGIES:
            unit = unit_dir(export, acc, side, strategy, accession)
            status = unit_status(unit)
            table_path = unit / "gsea" / "contrasts" / contrast / "all_gene_discovery_gsea_results.csv"
            n = paired_patients(unit, contrast)
            label = f"{title}, {side}"
            frame = None
            if status == "done" and table_path.exists():
                frame = pd.read_csv(table_path)
                if "collection" in frame:
                    frame = frame[frame["collection"].astype(str).eq("Hallmark")]
                frame["pathway"] = frame["pathway"].astype(str).str.replace("^HALLMARK_", "", regex=True)
                frame = frame.set_index("pathway").reindex(pathways)
                label += f"  (n={n})" if n is not None else ""
                if n is not None and n < min_patients:
                    label += f"  no significance (n < {min_patients})"
            else:
                label += f"  [{status if status != 'done' else 'no GSEA table'}]"
            rows.append((frame, n is not None and n >= min_patients))
            labels.append(label)
            flags.append({"side": side, "strategy": strategy, "status": status, "patients": n})
    limit = 2.0
    cmap = LinearSegmentedColormap.from_list("nes", [dotplot.NEGATIVE, dotplot.MIDPOINT, dotplot.POSITIVE])
    norm = Normalize(vmin=-limit, vmax=limit)
    fig = plt.figure(figsize=(16, 1.9 + 0.34 * len(rows)))
    ax = fig.add_axes([0.23, 0.30, 0.70, 0.62])
    for i, (frame, reliable) in enumerate(rows):
        if frame is None:
            ax.axhspan(i - 0.5, i + 0.5, color="#f4f4f4", zorder=0)
            continue
        nes = frame["NES"].to_numpy(float)
        fdr = frame["fdr"].to_numpy(float) if "fdr" in frame else frame["FDR q-val"].to_numpy(float)
        strength = -np.log10(np.clip(fdr, 1.0 / dotplot.PERMUTATIONS, None))
        sizes = np.asarray(dotplot.SIZE_AREAS)[np.digitize(np.nan_to_num(strength), dotplot.SIZE_BREAKS)] * 0.55
        called = (fdr < dotplot.FDR_CALL) & reliable
        xs = np.arange(len(pathways))
        ok = np.isfinite(nes)
        ax.scatter(xs[ok & ~called], np.full((ok & ~called).sum(), i), s=sizes[ok & ~called],
                   c=np.clip(nes[ok & ~called], -limit, limit), cmap=cmap, norm=norm, linewidths=0)
        ax.scatter(xs[ok & called], np.full((ok & called).sum(), i), s=sizes[ok & called],
                   c=np.clip(nes[ok & called], -limit, limit), cmap=cmap, norm=norm, edgecolors="black", linewidths=0.8)
    if len(SHOW) == 2:
        ax.axhline(len(STRATEGIES) - 0.5, color="black", lw=0.8)
    ax.set_xlim(-0.6, len(pathways) - 0.4); ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(range(len(pathways)))
    ax.set_xticklabels([dotplot.heatmap43.display_name("HALLMARK_" + p) for p in pathways], rotation=60,
                       ha="right", rotation_mode="anchor", fontsize=7)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=7.5)
    cax = fig.add_axes([0.94, 0.55, 0.008, 0.3])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label="NES")
    fig.text(0.23, 0.96, f"{accession}, acceptance {acc}: GSEA of rejected vs retained (hallmark); "
             f"size -log10 FDR, outline FDR < {dotplot.FDR_CALL:g}", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"rows": flags}


def deg_figure(export: Path, accession: str, side: str, contrast: str, acc: str, path: Path) -> str:
    """Draws the figure; returns "" when drawn, otherwise why not."""
    tables, missing = {}, {}
    for strategy, title in STRATEGIES:
        unit = unit_dir(export, acc, side, strategy, accession)
        table_path = unit / "deg" / "contrasts" / contrast / "pydeseq2_all_gene_discovery.csv"
        status = unit_status(unit)
        if status == "done" and table_path.exists():
            tables[title] = pd.read_csv(table_path).set_index("gene")
        else:
            missing[title] = status if status != "done" else "no DEG table"
    if not tables:
        return "no DEG table: " + "; ".join(f"{t} {s}" for t, s in missing.items())
    # the same rule in every figure, significant or not: each strategy's 10 largest |Wald|;
    # significance is the dot, so a comparison with no FDR < 0.05 gene is still drawn
    genes = []
    for table in tables.values():
        strongest = table["wald_statistic"].abs().dropna().sort_values(ascending=False)
        genes += list(strongest.index[:10])
    genes = list(dict.fromkeys(genes))
    if not genes:
        return "no gene has a Wald statistic"
    significant = sum(int((tab["fdr"] < 0.05).sum()) for tab in tables.values())
    # every strategy keeps its column; one without a table is left empty and says why
    columns = [t if t in tables else f"{t}\n[{missing[t][:40]}]" for _, t in STRATEGIES]
    lfc = pd.DataFrame({c: tables[t]["log2_fold_change"].reindex(genes) if t in tables else np.nan
                        for c, (_, t) in zip(columns, STRATEGIES)}, index=genes)
    padj = pd.DataFrame({c: tables[t]["fdr"].reindex(genes) if t in tables else np.nan
                         for c, (_, t) in zip(columns, STRATEGIES)}, index=genes)
    bound = float(np.nanmax(np.abs(lfc.to_numpy()))) or 1.0
    fig, ax = plt.subplots(figsize=(2.2 + 1.0 * lfc.shape[1], 1.4 + 0.18 * len(genes)))
    ax.set_facecolor("#d9d9d9")  # no table: grey, never the white of a zero fold change
    mesh = ax.pcolormesh(np.ma.masked_invalid(lfc.to_numpy(float)), cmap="RdBu_r", vmin=-bound, vmax=bound)
    rr, cc = np.where(padj.to_numpy(float) < 0.05)
    ax.scatter(cc + 0.5, rr + 0.5, s=6, c="black")
    ax.set_ylim(len(genes), 0)
    ax.set_xticks(np.arange(lfc.shape[1]) + 0.5); ax.set_xticklabels(lfc.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(genes)) + 0.5); ax.set_yticklabels(genes, fontsize=6)
    fig.colorbar(mesh, ax=ax, fraction=0.05, label="log2 fold change\nrejected vs retained")
    note = "" if significant else "\nno gene at FDR < 0.05 in any strategy"
    ax.set_title(f"{accession}, {side}, acceptance {acc}; each strategy's 10 largest |Wald|, "
                 f"dot = FDR < 0.05{note}", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", type=Path)
    parser.add_argument("figures", type=Path)
    parser.add_argument("--acceptances", nargs="+", default=["0.99", "0.95", "0.90"])
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--pairs", nargs="+", default=None)
    parser.add_argument("--min-patients", type=int, default=3)
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--sides", nargs="+", choices=["primary", "metastasis"], default=["primary", "metastasis"],
                        help="sides to draw; every .tsv still carries both")
    args = parser.parse_args()
    SHOW[:] = [s for s in ("primary", "metastasis") if s in args.sides]
    args.figures.mkdir(parents=True, exist_ok=True)
    counts = pd.read_csv(args.export / "state_counts.tsv", sep="\t", dtype={"acceptance": str})
    datasets = args.datasets or sorted(counts["dataset_id"].astype(str).unique())
    index = []
    for acc in args.acceptances:
        index.append({"figure": str(rejected_fraction(counts[counts["dataset_id"].isin(datasets)], acc, args.figures)),
                      "kind": "rejected_fraction", "acceptance": acc})
        index.append({"figure": str(state_rates(counts[counts["dataset_id"].isin(datasets)], acc, args.figures)),
                      "kind": "state_rates", "acceptance": acc})
        if not args.skip_umap:
            for path in sorted((args.export / "cells").glob("*/*.csv.gz")):
                dataset, pair = path.parent.name, path.name[:-len(".csv.gz")]
                if dataset not in datasets or (args.pairs and pair not in args.pairs):
                    continue
                cells = pd.read_csv(path)
                if cells["umap_1"].isna().all() or f"rna__acc{acc}" not in cells:
                    index.append({"figure": "", "kind": "umap", "acceptance": acc, "pair": pair, "note": "no coordinates"})
                    continue
                target = args.figures / "umap" / f"acc{acc}" / dataset / f"{pair}.png"
                umap_figure(cells, acc, target)
                index.append({"figure": str(target), "kind": "umap", "acceptance": acc, "pair": pair})
        for accession in datasets:
            target = args.figures / "gsea" / f"{accession}_acc{acc}.png"
            info = gsea_figure(args.export, accession, acc, args.min_patients, target)
            index.append({"figure": str(target), "kind": "gsea", "acceptance": acc, "dataset": accession,
                          "note": json.dumps(info["rows"])})
            for side, contrast in [s for s in SIDES if s[0] in SHOW]:
                target = args.figures / "deg" / f"{accession}_{side}_acc{acc}.png"
                why = deg_figure(args.export, accession, side, contrast, acc, target)
                index.append({"figure": "" if why else str(target), "kind": "deg", "acceptance": acc,
                              "dataset": accession, "side": side, "note": why})
    previous = args.figures / "figures_index.tsv"
    if args.skip_umap and previous.exists():  # keep the UMAP rows of the run that drew them
        old = pd.read_csv(previous, sep="\t", dtype=str)
        index = old[old["kind"] == "umap"].to_dict("records") + index
    pd.DataFrame(index).to_csv(previous, sep="\t", index=False)
    print(f"{sum(1 for r in index if r['figure'])} figures written to {args.figures}")


if __name__ == "__main__":
    main()

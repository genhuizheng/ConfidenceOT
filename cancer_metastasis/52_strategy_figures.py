"""Figures for the five CNV / RNA gate strategies, from the 51_strategy_export.py package.

    python cancer_metastasis/52_strategy_figures.py EXPORT_DIR FIGURES_DIR [--acceptances 0.99]
        [--datasets GSE180661 ...] [--pairs PAIR ...] [--min-patients 3] [--skip-umap]

Runs anywhere the package is (a laptop after download, or TACC): it reads only the package.

1. umap/acc<a>/<dataset>/<pair>.png: one figure per pair and acceptance. Rows: rna, cnv,
   rna_cnv, cnv_then_rna step 1, cnv_then_rna final, rna_then_cnv step 1, rna_then_cnv final;
   columns: primary, metastasis. One frozen joint UMAP per pair, so every panel of a pair has the
   same coordinates and axis limits. Colours: retain #2b8cbe, reject #d7301f, exclude (step 1
   rejected) #8c8c8c, not run / not gated #e3e3e3; the pair's other cells in light grey behind.
2. rejected_fraction_acc<a>.png (+ .tsv): pairs x (strategy, side), one quantity: the share of
   the side's gated cells not retained (1 - retain), so a two-step column counts step 1
   exclusions and step 2 rejections together. Gated = retain + reject + exclude: not_gated
   reference cells are not counted, and a two-step pair whose step 2 did not run is blank, as is
   not run.
5. state_rates_acc<a>.png (+ .tsv): stacked bars per dataset, retain / reject / exclude as the
   mean over pairs of each pair's gated-cell shares, n pairs per bar; the .tsv adds the pooled
   cell shares.
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
SIDES = [("primary", "primary_rejected_vs_primary_retained"),
         ("metastasis", "metastasis_rejected_vs_metastasis_retained")]
COLOURS = {"retain": "#2b8cbe", "reject": "#d7301f", "exclude": "#8c8c8c"}
OTHER = "#e3e3e3"
BLOCKS = {"GSE271675": "prostate"}
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def umap_figure(cells: pd.DataFrame, acc: str, path: Path) -> None:
    x, y = cells["umap_1"].to_numpy(float), cells["umap_2"].to_numpy(float)
    pad_x, pad_y = (np.nanmax(x) - np.nanmin(x)) * 0.04, (np.nanmax(y) - np.nanmin(y)) * 0.04
    limits = ((np.nanmin(x) - pad_x, np.nanmax(x) + pad_x), (np.nanmin(y) - pad_y, np.nanmax(y) + pad_y))
    size = max(0.5, min(6.0, 6000 / max(len(cells), 1)))
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(7.2, 2.55 * len(ROWS)), squeeze=False)
    for i, (column, title) in enumerate(ROWS):
        states = cells[f"{column}__acc{acc}"].astype(str)
        for j, side in enumerate(("primary", "metastasis")):
            ax = axes[i, j]
            ax.scatter(x, y, s=size * 0.6, c="#f2f2f2", linewidths=0, rasterized=True)
            mine = cells["side"].eq(side).to_numpy()
            state = states[mine].to_numpy()
            other = ~np.isin(state, list(COLOURS))
            ax.scatter(x[mine][other], y[mine][other], s=size, c=OTHER, linewidths=0, rasterized=True)
            for name, colour in COLOURS.items():
                hit = state == name
                ax.scatter(x[mine][hit], y[mine][hit], s=size, c=colour, linewidths=0, rasterized=True)
            tally = pd.Series(state).value_counts()
            text = "  ".join(f"{k} {int(tally.get(k, 0))}" for k in ("retain", "reject", "exclude") if k in tally)
            if other.any():
                text += f"  other {int(other.sum())}"
            ax.set_title(f"{title} - {side}\n{text}", fontsize=8)
            ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#bdbdbd")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in COLOURS.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="", color=OTHER, label="not run / not gated"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
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
    rows = [(s, side) for s, _ in ROWS for side in ("primary", "metastasis")]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 0.36 * len(rows) + 1.6),
                             sharey=True, squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        here = table[table["dataset_id"] == dataset].set_index(["strategy", "side"])
        for i, key in enumerate(rows):
            if key not in here.index:
                ax.text(0.01, i, "not run", va="center", fontsize=7, color="#636363")
                continue
            left = 0.0
            for state, colour in COLOURS.items():
                value = float(here.loc[key, f"mean_{state}"])
                ax.barh(i, value, left=left, height=0.72, color=colour, edgecolor="white", linewidth=1)
                if value >= 0.06:
                    ax.text(left + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=6.5, color="white")
                left += value
            ax.text(1.01, i, f"n={int(here.loc[key, 'pairs'])}", va="center", fontsize=6.5, color="#636363")
        ax.set_xlim(0, 1); ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("mean share of the pair's gated cells", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0][0].set_yticks(range(len(rows)))
    axes[0][0].set_yticklabels([f"{dict(ROWS)[s]}, {side}" for s, side in rows], fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Gate states at acceptance {acc}: mean over pairs (n = pairs behind each bar)", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out / f"state_rates_acc{acc}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def rejected_fraction(counts: pd.DataFrame, acc: str, out: Path) -> Path:
    shares = gated_fractions(counts, acc)
    table = shares.assign(not_retained=1 - shares["retain"])[KEY + ["not_retained"]]
    columns = [(s, side) for s, _ in ROWS for side in ("primary", "metastasis")]
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
    for side, contrast in SIDES:
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
    args = parser.parse_args()
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
            for side, contrast in SIDES:
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

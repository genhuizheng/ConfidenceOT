"""RETrace2 case-report figures (one PDF + PNG per panel, nbt style).

    python -I lineage_validation/retrace2_20261112/03_figures.py --out results/retrace2_20261112 \
        --data results/retrace2_20261112/extracted/2025_RETrace2_MSH2_Female_Combined

Panels (OUT/figures/):
  r1_cells_coverage      cells per tissue (all / in tree) and captured loci per cell by tissue
  r2_coverage_confound   lineage distance vs shared loci per pair, coloured by organ pair
  r3_tree_tracks         the authors' tree (leaf order) with tissue, organ, authors' clade and brain cell-type tracks
  r4_clade_composition   tissue composition of each authors' clade (10-20 cells), with V vs null for organ
  r5_distance_heatmaps   lineage distance raw and coverage-residual, cells in tree order (one quantity per heatmap)
  r6_ot_gates            retained fraction per pair under homopolymer-OT and methylation-OT
  r7_ot_overlap          top-5 source overlap of homopolymer-OT vs methylation-OT against its chance level
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402

TIS_COL = {"Brain_LeftAnteriorCortex": "#2C6E9B", "Brain_LeftPosteriorCortex": "#7FB3D9", "Kidney_Left-Cortex-1": "#4E8A6B",
           "Kidney_Right-Cortex-1": "#9CC5A1", "Liver_LeftLobe": "#B4702A", "Liver_MedianLobe": "#D9A066", "Liver_RightLobe": "#EBC9A0"}
ORG_COL = {"Brain": "#2C6E9B", "Kidney": "#4E8A6B", "Liver": "#B4702A"}
CT_COL = ["#7D3C98", "#C0392B", "#A23B5B", "#5B4A8A", "#8C7A5B", "#3F7F7F", "#D1495B", "#E3E6E8"]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)


def short(t):
    return t.replace("Brain_Left", "Brain ").replace("Cortex-1", "").replace("Kidney_", "Kidney ").replace("Liver_", "Liver ").replace("Cortex", " cortex").replace("-", " ").strip()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    fo = a.out / "figures"; fo.mkdir(exist_ok=True); st.apply()
    C = pd.read_csv(a.out / "prep" / "cells.tsv", sep="\t"); Cr = pd.read_csv(a.out / "analysis" / "cells_residual_clades.tsv", sep="\t")
    att = pd.read_csv(a.out / "analysis" / "attrition.tsv", sep="\t", index_col=0)
    ms = pd.read_excel(a.data / "ms_stats" / "Supplementary_Table_MSH2_Per_Cell_Statistics.xlsx")
    zl = np.load(a.out / "prep" / "D_lineage.npz"); DL, NT = zl["D"], zl["shared_targets"]; n = len(DL)
    tissues = list(TIS_COL)
    # r1
    fig = plt.figure(figsize=(6.4, 2.4))
    ax = fig.add_axes([0.08, 0.32, 0.38, 0.55]); st.tidy(ax)
    x = np.arange(len(tissues))
    ax.bar(x, att.reindex(tissues).cells_all, color=st.NEUTRAL, width=0.7, label="all cells")
    ax.bar(x, att.reindex(tissues).cells_in_tree, color=[TIS_COL[t] for t in tissues], width=0.7, label="in authors' tree (QC pass)")
    ax.set_xticks(x); ax.set_xticklabels([short(t) for t in tissues], rotation=45, ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
    ax.set_ylabel("cells", fontsize=st.FS_AXLAB); ax.legend(frameon=False, fontsize=st.FS_MIN, loc="upper right", bbox_to_anchor=(1, 1.25))
    ax2 = fig.add_axes([0.58, 0.32, 0.38, 0.55]); st.tidy(ax2)
    for k, t in enumerate(tissues):
        v = ms[ms.Tissue == t].MS_Targets_10X.dropna().to_numpy() if "Tissue" in ms else np.array([])
        if len(v):
            ax2.scatter(k + (np.random.default_rng(k).random(len(v)) - 0.5) * 0.4, v / 1e3, s=4, color=TIS_COL[t], lw=0)
            ax2.plot([k - 0.3, k + 0.3], [np.median(v) / 1e3] * 2, color=st.INK, lw=0.8)
    ax2.axhline(10, **st.REF_KW)
    ax2.set_xticks(x); ax2.set_xticklabels([short(t) for t in tissues], rotation=45, ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
    ax2.set_ylabel("homopolymer loci captured (x1000, 10X)", fontsize=st.FS_AXLAB)
    fig.text(0.01, 0.97, "MSH2 female mouse: 216 cells, 152 pass the authors' QC (dashed: 10k-locus threshold)", fontsize=st.FS_TITLE, va="top")
    save(fig, fo, "r1_cells_coverage")
    # r2
    org = C.organ.to_numpy(); iu = np.triu_indices(n, 1)
    pair = np.array(["-".join(sorted([org[i], org[j]])) for i, j in zip(*iu)])
    fig = plt.figure(figsize=(3.4, 2.8)); ax = fig.add_axes([0.17, 0.16, 0.78, 0.66]); st.tidy(ax)
    pc = {"Brain-Brain": "#2C6E9B", "Kidney-Kidney": "#4E8A6B", "Liver-Liver": "#B4702A", "Brain-Kidney": "#7FB3D9", "Brain-Liver": "#B5A6C9",
          "Kidney-Liver": "#C9C08A"}
    for k_ in pc:
        m = pair == k_
        ax.scatter(NT[iu][m] / 1e3, DL[iu][m], s=1.2, color=pc[k_], lw=0, alpha=0.5, label=k_, rasterized=True)
    xs = np.linspace(NT[iu].min(), NT[iu].max(), 50); b = np.polyfit(np.log(NT[iu]), DL[iu], 1)
    ax.plot(xs / 1e3, np.polyval(b, np.log(xs)), color=st.INK, lw=0.8)
    from scipy.stats import spearmanr
    rho = spearmanr(NT[iu], DL[iu]).correlation
    ax.set_xlabel("loci shared by the two cells (x1000)", fontsize=st.FS_AXLAB); ax.set_ylabel("authors' lineage distance", fontsize=st.FS_AXLAB)
    ax.legend(frameon=False, fontsize=st.FS_MIN, markerscale=5, ncol=2, loc="upper right")
    fig.text(0.02, 0.97, f"Lineage distance falls with shared coverage (Spearman {rho:.2f})", fontsize=st.FS_TITLE, va="top")
    save(fig, fo, "r2_coverage_confound")
    # tree leaf order
    from Bio import Phylo
    tree = Phylo.read(a.data / "results_10000targets_3000cpgs" / "phylo" / "2025_MSH2_female_combined.buildPhylo.newick-bootstrap.txt", "newick")
    leaves = [t.name for t in tree.get_terminals()]
    Ci = C.set_index("cell").reindex(leaves)
    # r3 tracks + tree
    fig = plt.figure(figsize=(7.2, 3.2))
    tx = fig.add_axes([0.03, 0.12, 0.45, 0.76]); tx.axis("off")
    Phylo.draw(tree, axes=tx, label_func=lambda c: None, do_show=False, show_confidence=False)
    for ln in tx.lines:
        ln.set_linewidth(0.4); ln.set_color(st.INK)
    for coll in tx.collections:
        coll.set_linewidth(0.4); coll.set_color(st.INK)
    tx.set_ylim(len(leaves) + 0.5, 0.5); tx.set_xlabel(""); tx.set_ylabel("")
    clades = sorted([c for c in Ci.clade.unique() if c != "unclustered"], key=lambda s: int(s[5:]))
    ccol = {c: ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#C0392B", "#5E8FA8", "#8C7A5B", "#A3A86B", "#D1495B"][i % 9] for i, c in enumerate(clades)}
    ccol["unclustered"] = "#E3E6E8"
    cts = [c for c in Ci.cell_type_assignment.value_counts().index if c != "Unassigned"]
    ctcol = {c: CT_COL[i % len(CT_COL)] for i, c in enumerate(cts)}; ctcol["Unassigned"] = "#FFFFFF"
    tracks = [("tissue", [TIS_COL[t] for t in Ci.tissue]), ("organ", [ORG_COL[o] for o in Ci.organ]), ("authors' clade", [ccol[c] for c in Ci.clade]),
              ("brain cell type", [ctcol.get(c, "#FFFFFF") if o == "Brain" else "#F4F5F6" for c, o in zip(Ci.cell_type_assignment, Ci.organ)])]
    for k, (nm, cols) in enumerate(tracks):
        ax = fig.add_axes([0.49 + k * 0.035, 0.12, 0.03, 0.76]); ax.set_ylim(len(leaves) + 0.5, 0.5); ax.set_xlim(0, 1); ax.axis("off")
        ax.barh(np.arange(1, len(leaves) + 1), 1, height=1, color=cols, lw=0)
        ax.text(0.5, 0, nm, rotation=60, fontsize=st.FS_MIN, ha="left", va="bottom")
    lg = fig.add_axes([0.66, 0.12, 0.33, 0.76]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    y = 0.98
    for title, items in (("tissue", [(short(t), c) for t, c in TIS_COL.items()]), ("authors' clade (10-20 cells)", [(c, ccol[c]) for c in clades] + [("unclustered", ccol["unclustered"])]),
                         ("brain cell type (authors)", [(c.replace("Brain_", ""), ctcol[c]) for c in cts] + [("unassigned", "#FFFFFF")])):
        lg.text(0, y, title, fontsize=st.FS_MIN, va="top"); y -= 0.045
        cols_ = 2 if len(items) > 6 else 1; per = (len(items) + cols_ - 1) // cols_
        for k, (nm, c) in enumerate(items):
            cx, cy = (k // per) * 0.5, y - (k % per) * 0.038
            lg.add_patch(Rectangle((cx, cy - 0.025), 0.05, 0.025, facecolor=c, edgecolor=st.INK, lw=0.2)); lg.text(cx + 0.07, cy - 0.012, nm, fontsize=st.FS_MIN, va="center")
        y -= per * 0.038 + 0.03
    fig.text(0.01, 0.98, "Authors' homopolymer tree (152 cells; bootstrap median 0.29) with tissue, clade and brain cell-type tracks", fontsize=st.FS_TITLE, va="top")
    save(fig, fo, "r3_tree_tracks")
    # r4 clade composition
    T = pd.read_csv(a.out / "analysis" / "table_clade_x_tissue.tsv", sep="\t", index_col=0)
    T = T.reindex(sorted(T.index, key=lambda s: int(s[5:])))
    A = pd.read_csv(a.out / "analysis" / "association.tsv", sep="\t").set_index("label")
    fig = plt.figure(figsize=(4.4, 2.6)); ax = fig.add_axes([0.12, 0.2, 0.55, 0.62]); st.tidy(ax)
    bottom = np.zeros(len(T))
    for t in tissues:
        if t in T.columns:
            v = T[t].to_numpy() / T.sum(1).to_numpy()
            ax.bar(np.arange(len(T)), v, bottom=bottom, color=TIS_COL[t], width=0.75, label=short(t), edgecolor="white", lw=0.3); bottom += v
    ax.set_xticks(range(len(T))); ax.set_xticklabels([f"{c[5:]}\n({int(s)})" for c, s in zip(T.index, T.sum(1))], fontsize=st.FS_MIN)
    ax.set_xlabel("authors' clade (cells)", fontsize=st.FS_AXLAB); ax.set_ylabel("share of the clade's cells", fontsize=st.FS_AXLAB)
    ax.legend(frameon=False, fontsize=st.FS_MIN, loc="center left", bbox_to_anchor=(1.0, 0.5))
    o = A.loc["organ"]; ti = A.loc["tissue"]
    fig.text(0.02, 0.97, f"Clade x organ: V {o.cramer_v:.2f} vs null {o.null_mean:.2f}; clade x tissue within organ: {ti.cramer_v:.2f} vs {ti.null_mean:.2f}",
             fontsize=st.FS_MIN, va="top")
    save(fig, fo, "r4_clade_composition")
    # r5 distance heatmaps
    order = pd.Index(C.cell).get_indexer(leaves)
    iu2 = np.triu_indices(n, 1); b = np.polyfit(np.log(NT[iu2]), DL[iu2], 1)
    R = np.zeros((n, n)); R[iu2] = DL[iu2] - np.polyval(b, np.log(NT[iu2])); R = R + R.T
    for nm, M, lab in (("raw", DL, "authors' lineage distance"), ("coverage_residual", R, "residual after log(shared loci)")):
        fig = plt.figure(figsize=(3.4, 3.2)); ax = fig.add_axes([0.1, 0.1, 0.68, 0.72])
        Mo = M[np.ix_(order, order)].copy(); np.fill_diagonal(Mo, np.nan)
        lo_, hi_ = np.nanpercentile(Mo, [2, 98])
        im = ax.imshow(Mo, cmap="viridis_r", vmin=lo_, vmax=hi_, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        tr = fig.add_axes([0.07, 0.1, 0.025, 0.72]); tr.imshow(np.array([[i] for i in range(n)]), aspect="auto",
                                                               cmap=ListedColormap([TIS_COL[t] for t in Ci.tissue]), interpolation="nearest"); tr.axis("off")
        cb = fig.colorbar(im, cax=fig.add_axes([0.81, 0.3, 0.03, 0.4])); cb.ax.tick_params(labelsize=st.FS_MIN); cb.set_label(lab, fontsize=st.FS_MIN)
        fig.text(0.02, 0.97, f"Cells in tree order: {lab}", fontsize=st.FS_TITLE, va="top")
        save(fig, fo, f"r5_distance_heatmap_{nm}")
    # r6 / r7 OT
    F = pd.read_csv(a.out / "ot" / "fits.tsv", sep="\t"); K = pd.read_csv(a.out / "ot" / "compare.tsv", sep="\t")
    pairs = list(dict.fromkeys(F.pair))
    lab = [p_.replace("Brain_Left", "Br ").replace("Liver_", "Li ").replace("Kidney_", "Ki ").replace("-Cortex-1", "").replace("Cortex", "").replace("Lobe", "") for p_ in pairs]
    fig = plt.figure(figsize=(6.6, 2.6))
    for k, side in enumerate(("retained_source", "retained_target")):
        ax = fig.add_axes([0.07 + k * 0.48, 0.42, 0.42, 0.45]); st.tidy(ax)
        for j, (cn, colr, off) in enumerate((("lineage", st.PAL["tcr"], -0.2), ("methylation", st.PAL["peptide"], 0.2))):
            v = F[F.cost == cn].set_index("pair").reindex(pairs)[side]
            valid = F[F.cost == cn].set_index("pair").reindex(pairs).m4e_inference_valid.astype(bool)
            ax.bar(np.arange(len(pairs)) + off, v, 0.38, color=colr, label=f"{cn}-OT" if k == 0 else None, alpha=1.0)
            for i_, ok in enumerate(valid):
                if not ok:
                    ax.text(i_ + off, v.iloc[i_] + 0.02, "*", ha="center", fontsize=st.FS_MIN)
        ax.set_ylim(0, 1.12); ax.set_xticks(range(len(pairs))); ax.set_xticklabels(lab, rotation=60, ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
        ax.set_title(side.replace("retained_", "retained ") + " cells", fontsize=st.FS_TITLE, pad=16)
        if k == 0:
            ax.legend(frameon=False, fontsize=st.FS_MIN, loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2)
    fig.text(0.01, 0.02, "* calibration flagged m4e_inference_valid = False. Same pairs, same solver; only the cost differs.", fontsize=st.FS_MIN, color=st.SUBTLE)
    save(fig, fo, "r6_ot_gates")
    k_ = K[(K.a == "lineage") & (K.b == "methylation")].set_index("pair").reindex(pairs)
    fig = plt.figure(figsize=(3.4, 2.7)); ax = fig.add_axes([0.17, 0.17, 0.78, 0.66]); st.tidy(ax)
    ax.scatter(k_.chance, k_.top5_overlap, s=14, color=st.PAL["tcr"], lw=0)
    for i_, (cx, cy) in enumerate(zip(k_.chance, k_.top5_overlap)):
        ax.text(cx + 0.01, cy, lab[i_], fontsize=st.FS_MIN, va="center")
    lim = [0, 1]; ax.plot(lim, lim, **st.REF_KW); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("chance overlap (5 / common sources)", fontsize=st.FS_AXLAB); ax.set_ylabel("top-5 source overlap", fontsize=st.FS_AXLAB)
    fig.text(0.02, 0.97, "Homopolymer-OT vs methylation-OT: per-target top sources", fontsize=st.FS_TITLE, va="top")
    save(fig, fo, "r7_ot_overlap")
    print("figures done")


if __name__ == "__main__":
    main()

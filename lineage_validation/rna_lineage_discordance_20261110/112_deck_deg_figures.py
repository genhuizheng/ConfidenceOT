"""Slide versions of the DEG v3 results (RNA-OT- vs CRISPR-OT-preferred cells within a clade): large type, plain labels.
Reads deg_v3/summary (104_) and, for D9, the 105_ F3 block's counts. Writes ROOT/figures/deck_20261112/D7-D9.

  D7_preference_depth_and_rna    left: depth ratio per contrast; right: Spearman of its 50 NES with the RNA-similarity baseline
  D8_hallmark_all_sets           all 50 hallmark sets x contrast type, median over mice of the mouse-mean NES, raw | thinned;
                                 dot = same sign in >= 75% of mice in both runs
  D9_example_genes_3724_c45      group mean z of the top 10 genes up / down (F3 block)

    python lineage_validation/rna_lineage_discordance_20261110/112_deck_deg_figures.py --root results/rna_lineage_discordance_20261110
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

C_RNA, C_CR, C_BOTH, GREY, INK = "#2C6E9B", "#D98C2B", "#A6A6A6", "#9AA0A6", "#262626"
OT_TYPES = [("A1_rna_vs_crispr", "RNA-OT- vs\nCRISPR-OT-\npreferred"), ("A1_rna_vs_both", "RNA-OT-\npreferred\nvs both"),
            ("A1_crispr_vs_both", "CRISPR-OT-\npreferred\nvs both"), ("A2_cross_vs_own", "targets:\nRNA-cross\nvs RNA-own")]
HM_TYPES = [("A1_rna_vs_crispr", "RNA-OT- vs CRISPR-OT-\npreferred"), ("A1_rna_vs_both", "RNA-OT-preferred\nvs both"),
            ("A1_crispr_vs_both", "CRISPR-OT-preferred\nvs both"), ("A1_baseline", "RNA-closest vs\nRNA-farthest (control)"),
            ("A2_cross_vs_own", "targets: RNA-cross\nvs RNA-own"), ("A2_baseline", "targets: RNA control")]


def slide_style():
    matplotlib.rcParams.update({"font.family": "Arial", "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "xtick.labelsize": 12,
                                "ytick.labelsize": 12, "legend.fontsize": 12, "axes.spines.top": False, "axes.spines.right": False,
                                "axes.linewidth": 1.0, "pdf.fonttype": 42})


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=250 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    S = a.root / "deg_v3" / "summary"; out = a.root / "figures" / "deck_20261112"; out.mkdir(parents=True, exist_ok=True)
    rd = lambda f: pd.read_csv(S / f, sep="\t", dtype={"mouse": str})  # noqa: E731
    QC, BP, MN, CO = rd("contrast_qc.tsv"), rd("baseline_pairs.tsv"), rd("mouse_nes.tsv"), rd("consistency.tsv")
    slide_style(); rng = np.random.default_rng(20261113)

    # D7 depth ratio and RNA-similarity baseline
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for i, (t, lab) in enumerate(OT_TYPES):
        v = QC[QC.type == t].depth_ratio.dropna().to_numpy()
        axs[0].scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=30, color=GREY, zorder=2)
        axs[0].plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=2.2, zorder=3)
        axs[0].text(i, 4.6, f"x{np.median(v):.2f}", ha="center", fontsize=13, fontweight="bold")
        w = BP[BP.type == t].nes_spearman_raw.dropna().to_numpy()
        axs[1].scatter(i + rng.uniform(-0.17, 0.17, len(w)), w, s=30, color=GREY, zorder=2)
        axs[1].plot([i - 0.28, i + 0.28], [np.median(w)] * 2, color=INK, lw=2.2, zorder=3)
        axs[1].text(i, 1.12, f"{np.median(w):.2f}", ha="center", fontsize=13, fontweight="bold")
    axs[0].axhline(1, ls="--", color="#777777", lw=1); axs[0].set_yscale("log"); axs[0].minorticks_off()
    axs[0].set_yticks([0.25, 0.5, 1, 2, 4]); axs[0].set_yticklabels(["0.25", "0.5", "1", "2", "4"]); axs[0].set_ylim(0.18, 5.5)
    axs[0].set_ylabel("sequencing depth,\nfirst group / second (median)")
    axs[0].set_title("Which cells are sequenced deeper?\n(above 1 = the first group)")
    axs[1].axhline(0, ls="--", color="#777777", lw=1); axs[1].set_ylim(-1.05, 1.25)
    axs[1].set_ylabel("agreement with the RNA-similarity control\n(Spearman of 50 hallmark NES)")
    axs[1].set_title("Is the difference just 'closer in RNA'?\n(1 = same as the control)")
    for ax in axs:
        ax.set_xticks(range(len(OT_TYPES))); ax.set_xticklabels([l for _, l in OT_TYPES], fontsize=12)
    fig.text(0.5, -0.03, "One dot per contrast (one clade, one site). Groups chosen by OT difference within the clade; "
             "control = cells closest vs farthest in RNA from the same targets.", ha="center", fontsize=11, color="#555555")
    fig.tight_layout(); save(fig, out, "D7_preference_depth_and_rna")

    # D8 all 50 hallmark sets, median over mice of the mouse-mean NES, raw | thinned
    med = {k: MN.groupby(["type", "term"])[f"nes_{k}"].median() for k in ("raw", "ds")}
    ok = CO[~CO.without_3724].set_index(["type", "term"]).sign_ok
    terms = sorted(MN.term.unique(), key=lambda t: -np.nan_to_num(med["raw"].get(("A1_rna_vs_crispr", t), 0)))
    rows = [(k, t) for k in ("raw", "ds") for t, _ in HM_TYPES]
    M = np.array([[med[k].get((t, term), np.nan) for term in terms] for (k, t) in rows])
    fig = plt.figure(figsize=(16, 7.2)); ax = fig.add_axes([0.21, 0.30, 0.72, 0.58])
    cm = plt.get_cmap("RdBu_r").copy(); cm.set_bad("#EEEEEE")
    im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cm, vmin=-1.5, vmax=1.5, interpolation="nearest")
    for i, (k, t) in enumerate(rows):
        for j, term in enumerate(terms):
            if bool(ok.get((t, term), False)):
                ax.scatter(j, i, s=16, color=INK, zorder=3)
    ax.axhline(len(HM_TYPES) - 0.5, color="white", lw=5)
    ax.set_xticks(range(len(terms))); ax.set_xticklabels([t.replace("HALLMARK_", "").replace("_", " ").lower() for t in terms], rotation=60,
                                                         ha="right", fontsize=11)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([l.replace("\n", " ") for _ in range(2) for _, l in HM_TYPES], fontsize=11)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for i0, name in ((0, "raw counts"), (len(HM_TYPES), "depth-thinned")):
        ax.text(-0.235, i0 + len(HM_TYPES) / 2 - 0.5, name, transform=ax.get_yaxis_transform(), rotation=90, ha="right", va="center",
                fontsize=13, fontweight="bold")
    cb = fig.add_axes([0.945, 0.45, 0.008, 0.3]); fig.colorbar(im, cax=cb, ticks=[-1.5, 0, 1.5])
    cb.set_title("NES", fontsize=12, loc="left"); cb.tick_params(labelsize=11)
    fig.text(0.05, 0.93, "All 50 hallmark sets (columns, ordered by the first row). Colour = median over mice of the mouse-mean NES; red = higher "
             "in the first-named group.\nDot = same sign in >= 75% of mice in both runs. No set is also outside the random-split range in half of "
             "its contrasts, so none is called consistent.", fontsize=12, color="#555555")
    save(fig, out, "D8_hallmark_all_sets")

    # D9 example genes of the F3 block, group mean z
    import anndata as ad
    import scanpy as sc
    G = pd.read_csv(a.root / "deg_v3" / "groups.tsv", sep="\t")
    C = pd.read_csv(a.root / "deg_v3" / "contrasts.tsv", sep="\t", dtype={"mouse": str})
    m = C[(C.type == "A1_rna_vs_crispr") & C.mouse.isin(["3457", "3460", "3724", "3451"])]
    blk = m.assign(n=m.case_n + m.reference_n).sort_values("n", ascending=False).iloc[0]; cid = blk.contrast_id
    g = G[G.contrast_id == cid].copy(); g["grp"] = g.group.map({"case": "RNA-OT-preferred", "reference": "CRISPR-OT-preferred"})
    gb = G[(G.contrast_id == cid.replace("A1_rna_vs_crispr", "A1_rna_vs_both")) & (G.group == "reference")].copy(); gb["grp"] = "both supported"
    g = pd.concat([g, gb])
    deg = pd.read_csv(a.root / "deg_v3" / "deg_gsea_raw" / "deg" / f"{cid}.tsv.gz", sep="\t")
    up = list(deg.sort_values("scores", ascending=False).names.head(10)); dn = list(deg.sort_values("scores").names.head(10))
    A = ad.read_h5ad(a.root / "kp_counts_20261112" / f"KPTracer_{blk.mouse}.h5ad"); A.obs_names = A.obs_names.astype(str)
    x = A[g.cell.astype(str).to_numpy()].copy(); depth = np.asarray(x.X.sum(1)).ravel()
    sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
    X = x[:, up + dn].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X); Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    order = ["RNA-OT-preferred", "both supported", "CRISPR-OT-preferred"]; grp = g.grp.to_numpy()
    Mz = np.array([Z[grp == o].mean(0) for o in order]).T
    fig, ax = plt.subplots(figsize=(10, 6.6))
    im = ax.imshow(Mz, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_yticks(range(20)); ax.set_yticklabels(up + dn, fontsize=13); ax.axhline(9.5, color="white", lw=4)
    ax.set_xticks(range(3))
    brk = {o: o.replace("-preferred", "-\npreferred").replace("both supported", "both\nsupported") for o in order}
    ax.set_xticklabels([f"{brk[o]}\n{(grp == o).sum()} cells\ndepth {np.median(depth[grp == o]) / 1000:.1f}k" for o in order], fontsize=12)
    ax.xaxis.tick_top()
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(3.0, 4.5, "higher in\nRNA-OT-\npreferred", va="center", fontsize=12, color=C_RNA)
    ax.text(3.0, 14.5, "higher in\nCRISPR-OT-\npreferred", va="center", fontsize=12, color=C_CR)
    cb = fig.add_axes([0.925, 0.43, 0.015, 0.14]); fig.colorbar(im, cax=cb, ticks=[-0.8, 0, 0.8])
    cb.set_title("mean z", fontsize=11, loc="left"); cb.tick_params(labelsize=10)
    fig.text(0.02, 0.01, f"KP 3724 primary, clade {blk.clade.split('|')[-1]} (Mesenchymal-2), one example: the contrast with the most cells.\n"
             "Top 10 genes each way, raw counts; groups chosen by OT difference within the clade.", fontsize=10, color="#555555")
    save(fig, out, "D9_example_genes_3724_c45")
    print("deck DEG figures ->", out)


if __name__ == "__main__":
    main()

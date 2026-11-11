"""Exploratory figure panels of the three-view comparison (PLAN_v2.md section 6), one PDF + PNG per panel.

    python lineage_validation/rna_lineage_discordance_20261110/50_figures.py --root results/rna_lineage_discordance_20261110 [--tag l5best]

Reads ROOT/summary (40_summarize.py) and ROOT/compare. Panels:
  f1_gate_kappa          A vs B gate agreement (Cohen's kappa) per unit, source and target, KP / Quinn
  f1_coupling_tv         distribution of the per-target total-variation distance between the A and B couplings
  f1_top_sources         edit agreement s (left) and RNA similarity r (right) of each coupling's top-15 sources, per unit
  f2_resolution          lineage resolution: median informative events per cell vs distinct event sets / cells
  f3_discordance_null    divergence and convergence per lineage: observed vs within-site permutation null
  f4_quadrant_mass       coupling mass on lineage-close / RNA-far and RNA-close / lineage-far pairs, A and B vs independent
  f5_state_kp            KP: share of divergent / convergent cells per authors' cell state (one quantity per heatmap)
  f6_gate_auc            AUROC of divergence / convergence for rejected vs retained cells, A and B
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402

COL = {"KP": "#2C6E9B", "Quinn": "#B4702A"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tag", default="l5best")
    a = p.parse_args()
    S = a.root / "summary"; out = a.root / "figures" / a.tag; out.mkdir(parents=True, exist_ok=True)
    st.apply()
    ab = pd.read_csv(S / "a_vs_b_units.tsv", sep="\t", dtype={"mouse": str}); ab = ab[ab.tag == a.tag]
    C = pd.read_csv(S / "cells_all.tsv.gz", sep="\t", dtype={"mouse": str}, low_memory=False); C = C[C.tag == a.tag]
    lin = pd.read_csv(S / "discordance_lineages.tsv", sep="\t", dtype={"mouse": str}); lin = lin[lin.tag == a.tag]
    gate = pd.read_csv(S / "gate_vs_discordance.tsv", sep="\t", dtype={"mouse": str}); gate = gate[gate.tag == a.tag]
    lab = ab.dataset_short + " " + ab.stratum

    # f1 gate kappa
    sh = st.Sheet(3.2, 2.2); ax = sh.ax(0.5, 0.45, 2.55, 1.55); st.tidy(ax)
    groups = sorted(lab.unique())
    for k, g in enumerate(groups):
        d = ab[lab == g]
        for j, sd in enumerate(("source", "target")):
            y = d[f"kappa_{sd}"].to_numpy(float)
            x = k * 2.6 + j + (np.random.default_rng(1).random(len(y)) - 0.5) * 0.35
            ax.scatter(x, y, s=8, color=COL[g.split()[0]], alpha=0.8 if sd == "target" else 0.45, lw=0)
    ax.axhline(0, **st.REF_KW)
    ax.set_xticks([k * 2.6 + j for k in range(len(groups)) for j in (0, 1)])
    ax.set_xticklabels([f"{g}\n{sd}" for g in groups for sd in ("src", "tgt")], fontsize=st.FS_MIN)
    ax.set_ylabel("Cohen's kappa, RNA-OT vs CRISPR-OT gate", fontsize=st.FS_AXLAB)
    sh.text(0.05, 2.15, "Gate agreement per unit (M4-E)", fontsize=st.FS_TITLE, va="top"); sh.save(out, "f1_gate_kappa")

    # f1 coupling TV
    t = C[(C.side == "target") & C.tv.notna()]
    sh = st.Sheet(3.0, 2.2); ax = sh.ax(0.5, 0.45, 2.35, 1.55); st.tidy(ax)
    for ds, g in t.groupby("dataset"):
        ax.hist(g.tv, bins=np.linspace(0, 1, 41), histtype="step", density=True, color=COL[ds], lw=1.0, label=f"{ds} ({len(g):,} cells)")
    ax.set_xlabel("total variation, RNA-OT vs CRISPR-OT source distribution", fontsize=st.FS_AXLAB); ax.set_ylabel("density", fontsize=st.FS_AXLAB)
    ax.legend(frameon=False, fontsize=st.FS_LEGEND)
    sh.text(0.05, 2.15, "How differently the two OTs match a target cell", fontsize=st.FS_TITLE, va="top"); sh.save(out, "f1_coupling_tv")

    # f1 top sources: s and r of each coupling's top-15 sources (unit means)
    sh = st.Sheet(5.0, 2.2)
    for k, (col_a, col_b, name) in enumerate((("s_top_rnaot", "s_top_crisprot", "edit agreement s of top-15 sources"),
                                              ("r_top_rnaot", "r_top_crisprot", "RNA similarity r of top-15 sources"))):
        ax = sh.ax(0.5 + k * 2.45, 0.45, 1.95, 1.55); st.tidy(ax)
        for ds, g in ab.groupby("dataset_short"):
            ax.scatter(g[col_a], g[col_b], s=10, color=COL[ds], lw=0, label=ds)
        lim = [np.nanmin(ab[[col_a, col_b]].to_numpy()), np.nanmax(ab[[col_a, col_b]].to_numpy())]
        ax.plot(lim, lim, **st.REF_KW)
        ax.set_xlabel("RNA-OT", fontsize=st.FS_AXLAB); ax.set_ylabel("CRISPR-OT", fontsize=st.FS_AXLAB)
        ax.set_title(name, fontsize=st.FS_TITLE)
        if k == 0:
            ax.legend(frameon=False, fontsize=st.FS_LEGEND)
    sh.save(out, "f1_top_sources")

    # f2 resolution
    R = pd.read_csv(a.root / "lineages" / "resolution.tsv", sep="\t", dtype={"mouse": str})
    sh = st.Sheet(3.0, 2.4); ax = sh.ax(0.5, 0.45, 2.35, 1.75); st.tidy(ax)
    for ds, g in R.dropna(subset=["median_events_per_cell"]).groupby("dataset"):
        d = "KP" if ds == "KPTracer" else "Quinn"
        ax.scatter(g.median_events_per_cell, g.distinct_sets_share, s=np.clip(g.cells / 40, 4, 60), color=COL[d],
                   edgecolors=np.where(g.resolved, "#333333", "none"), lw=0.4, alpha=0.8, label=d)
        for r in g[g.dataset == "KPTracer"].itertuples():
            ax.text(r.median_events_per_cell, r.distinct_sets_share, f" {r.mouse}", fontsize=st.FS_MIN, va="center")
    ax.axvline(3, **st.REF_KW); ax.axhline(0.1, **st.REF_KW)
    ax.set_xlabel("median informative events per cell", fontsize=st.FS_AXLAB); ax.set_ylabel("distinct event sets / cells", fontsize=st.FS_AXLAB)
    ax.legend(frameon=False, fontsize=st.FS_LEGEND)
    sh.text(0.05, 2.35, "Lineage resolution (outlined: passes the fixed rule; size: cells)", fontsize=st.FS_TITLE, va="top"); sh.save(out, "f2_resolution")

    # f3 discordance vs null
    sh = st.Sheet(5.0, 2.3)
    for k, (o, nl, name) in enumerate((("div", "div_null", "lineage-close / RNA-far (div)"), ("conv", "conv_null", "RNA-close / lineage-far (conv)"))):
        ax = sh.ax(0.5 + k * 2.45, 0.45, 1.95, 1.6); st.tidy(ax)
        for ds, g in lin.groupby("dataset"):
            ax.scatter(g[nl], g[o], s=np.clip(g.cells / 30, 3, 60), color=COL[ds], lw=0, alpha=0.7, label=ds)
        ax.plot([0, 1], [0, 1], **st.REF_KW)
        ax.set_xlabel("permutation null", fontsize=st.FS_AXLAB); ax.set_ylabel("observed", fontsize=st.FS_AXLAB); ax.set_title(name, fontsize=st.FS_TITLE)
        if k == 0:
            ax.legend(frameon=False, fontsize=st.FS_LEGEND)
    sh.text(0.05, 2.28, "Per unit x lineage (size: cells); below the diagonal = RNA and lineage agree more than chance",
            fontsize=st.FS_MIN, va="top", color=st.SUBTLE)
    sh.save(out, "f3_discordance_null")

    # f4 quadrant mass
    sh = st.Sheet(5.0, 2.2)
    for k, (cr, ri, ind, name) in enumerate((("lcrf_cr", "lcrf_rna", "lcrf_ind", "lineage-close / RNA-far pairs"),
                                             ("rclf_cr", "rclf_rna", "rclf_ind", "RNA-close / lineage-far pairs"))):
        ax = sh.ax(0.5 + k * 2.45, 0.45, 1.95, 1.55); st.tidy(ax)
        for ds, g in ab.groupby("dataset_short"):
            ax.scatter(g[ind], g[ri], s=10, color=COL[ds], lw=0, label=f"{ds} RNA-OT")
            ax.scatter(g[ind], g[cr], s=10, facecolors="none", edgecolors=COL[ds], lw=0.6, label=f"{ds} CRISPR-OT")
        m = np.nanmax(ab[[cr, ri, ind]].to_numpy()) if len(ab) else 1
        ax.plot([0, m], [0, m], **st.REF_KW)
        ax.set_xlabel("independent coupling", fontsize=st.FS_AXLAB); ax.set_ylabel("OT coupling", fontsize=st.FS_AXLAB)
        ax.set_title(f"mass on {name}", fontsize=st.FS_TITLE)
        if k == 0:
            ax.legend(frameon=False, fontsize=st.FS_LEGEND)
    sh.save(out, "f4_quadrant_mass")

    # f5 KP states: share divergent / convergent per state (cells >= 20)
    if "cell_type" in C:
        kp = C[(C.dataset == "KP") & C.cell_type.notna()]
        for q, name in (("divergent", "divergent (lineage-close / RNA-far)"), ("convergent", "convergent (RNA-close / lineage-far)")):
            tab = kp.groupby(["cell_type", "mouse"])[q].agg(["mean", "size"]).reset_index()
            tab = tab[tab["size"] >= 20]
            W = tab.pivot(index="cell_type", columns="mouse", values="mean")
            if W.empty:
                continue
            sh = st.Sheet(0.9 + 0.32 * W.shape[1] + 0.8, 0.6 + 0.16 * W.shape[0] + 0.5)
            ax = sh.ax(1.6, 0.5, 0.32 * W.shape[1], 0.16 * W.shape[0])
            im = ax.imshow(W.to_numpy(), cmap="Blues", vmin=0, vmax=0.3, aspect="auto")
            ax.set_xticks(range(W.shape[1])); ax.set_xticklabels(W.columns, fontsize=st.FS_MIN)
            ax.set_yticks(range(W.shape[0])); ax.set_yticklabels(W.index, fontsize=st.FS_MIN)
            for (i, j), v in np.ndenumerate(W.to_numpy()):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=st.FS_MIN, color="white" if v > 0.2 else st.INK)
            cax = sh.ax(1.6 + 0.32 * W.shape[1] + 0.1, 0.5, 0.08, min(1.0, 0.16 * W.shape[0]))
            cb = sh.fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=st.FS_MIN)
            sh.text(0.05, 0.6 + 0.16 * W.shape[0] + 0.45, f"KP: share of {name} cells per state (expected 0.10)", fontsize=st.FS_TITLE, va="top")
            sh.save(out, f"f5_state_kp_{q}")

    # f6 gate AUROC
    sh = st.Sheet(4.6, 2.2); ax = sh.ax(0.5, 0.55, 4.0, 1.45); st.tidy(ax)
    cols = [c for c in gate.columns if c.startswith("auc_")]
    for k, c in enumerate(cols):
        for ds, g in gate.groupby("dataset_short"):
            y = g[c].dropna()
            ax.scatter(np.full(len(y), k) + (0.15 if ds == "Quinn" else -0.15), y, s=7, color=COL[ds], lw=0, alpha=0.7)
    ax.axhline(0.5, **st.REF_KW)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("auc_", "").replace("_rejected_", "\n") for c in cols], fontsize=st.FS_MIN, rotation=0)
    ax.set_ylabel("AUROC, rejected vs retained", fontsize=st.FS_AXLAB)
    sh.text(0.05, 2.15, "Do the gates reject discordant cells? (> 0.5: rejected cells score higher)", fontsize=st.FS_TITLE, va="top")
    sh.save(out, "f6_gate_auc")


if __name__ == "__main__":
    main()

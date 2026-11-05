"""Hallmark GSEA heatmaps from pdx_fate_deg_gsea.py: retained cells of one state, grouped by dominant fate (source) or
dominant origin (target).

    python realdata_benchmark/v3_grid/figures/pdx_fate_gsea_figures.py \
        --gsea results/v3_grid_20261104/pdx_fate_deg_gsea_20261105 --out results/v3_grid_20261104/figures/pdx_fate_gsea

One panel per configuration x side x expression (rawqc, ds), every analysed contrast and all 50 hallmark sets:
  rows     contrasts in time order, then the published state order; left columns: the cells' time point and the one
           they are coupled to, the state, the two groups compared, their cell counts, and the AUC of
           pre-equalisation depth for the fate group vs the reference (> 0.5: the fate group is deeper);
  dots     colour = NES of the fate group vs the reference (orange: higher in the fate group), one scale (+-2.5) for
           every panel; area = -log10 FDR in four bins, outlined FDR < 0.05 (pseudo-replicated cells: a ranking aid,
           not a test).
Depth check (one point per row: mean NES over the 50 sets vs the depth AUC of its two groups):
pdx_depth_check_fate_<expression> and, with --rejected-gsea, pdx_depth_check_rejected_rawqc (pdx_deg_gsea.py units).
Also writes fate_composition_<cfg>.tsv: every pair x side x state, retained cells by dominant fate / origin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbt_style as st  # noqa: E402
from pdx_gsea_figures import (CATEGORIES, NAME, RULE_LIGHT, SAMPLES, WEEK, PATHWAYS, dot_matrix, size_legend,
                               term_key)  # noqa: E402
from pdx_prepare import STATES  # noqa: E402

EXPR = {"rawqc": "original counts", "ds": "depth-equalised counts (2,817 UMIs per cell)"}
VERB = {"source": "->", "target": "from"}
TITLE = {"source": "Retained cells of one state, split by where ConfidenceOT sends them at the NEXT time point",
         "target": "Retained cells of one state, split by where ConfidenceOT brings them from at the PREVIOUS time point"}
EXPLAIN = {
    "source": ("Each row: retained cells of one state at one time point. Each cell's fate = the next-time-point state "
               "that receives most of its coupling mass. Row = cells with fate X vs cells with the reference fate "
               "(their own state when\nthat group has >= 20 cells, else the largest group). Each dot: hallmark "
               "GSEA NES on genes ranked by Wilcoxon, X vs reference. One tumour per time point: dot size (FDR) ranks "
               "the sets, it is not a test."),
    "target": ("Each row: retained cells of one state at one time point. Each cell's origin = the previous-time-point "
               "state that sends most of its coupling mass. Row = cells from X vs cells from the reference origin "
               "(their own state when\nthat group has >= 20 cells, else the largest group). Each dot: hallmark "
               "GSEA NES on genes ranked by Wilcoxon, X vs reference. One tumour per time point: dot size (FDR) ranks "
               "the sets, it is not a test."),
}


def panel(gs: pd.DataFrame, con: pd.DataFrame, cfg: str, side: str, kind: str, out: Path) -> int:
    u = con[(con.config == cfg) & (con.side == side) & (con.status == "analysed")].copy()
    if u.empty:
        return 0
    u["pi"] = u.pair.map(lambda p: SAMPLES.index(p.split("__")[0])); u["si"] = u.state.map(STATES.index)
    u["fi"] = u.fate.map(STATES.index)
    u = u.sort_values(["pi", "si", "fi"]).reset_index(drop=True)
    g = gs[(gs.config == cfg) & (gs.side == side) & (gs.expression == kind)].copy(); g["key"] = g.term.map(term_key)
    nes = np.full((len(u), 50), np.nan); fdr = np.full_like(nes, np.nan)
    for i, r in u.iterrows():
        h = g[(g.pair == r.pair) & (g.state == r.state) & (g.fate == r.fate)].set_index("key")
        for j, pw in enumerate(PATHWAYS):
            if pw in h.index:
                nes[i, j] = float(h.at[pw, "nes"]); fdr[i, j] = float(h.at[pw, "fdr"])

    row_h, col_w = 0.15, 0.105
    left, top_pad, bottom = 5.05, 0.92, max(1.40, 2.25 - len(u) * row_h)
    W = left + 50 * col_w + 1.35; H = top_pad + len(u) * row_h + bottom
    sh = st.Sheet(W, H)
    ax = sh.ax(left, bottom, 50 * col_w, len(u) * row_h)
    im = dot_matrix(sh, ax, nes, fdr)

    x_week, x_state, x_fate, x_ref, x_nf, x_nr, x_auc = 0.08, 1.80, 1.92, 3.05, left - 0.80, left - 0.46, left - 0.08
    y_of = lambda k: bottom + (len(u) - k - 0.5) * row_h  # noqa: E731
    y_head = bottom + len(u) * row_h + 0.07
    for x, t, ha in ((x_week, "cells at", "left"), (x_state, "state", "right"),
                     (x_fate, "fate X" if side == "source" else "origin X", "left"),
                     (x_ref, "reference" if side == "source" else "reference origin", "left"),
                     (x_nf, "n X", "right"), (x_nr, "n ref", "right"), (x_auc, "depth AUC", "right")):
        sh.text(x, y_head, t, fontsize=st.FS_MIN, color=st.SUBTLE, ha=ha, va="bottom")
    for k, r in enumerate(u.itertuples()):
        a_, b_ = r.pair.split("__")
        here, other = (a_, b_) if side == "source" else (b_, a_)
        new_block = k == 0 or u.pi.iat[k] != u.pi.iat[k - 1]
        if new_block:
            sh.text(x_week, y_of(k), f"{WEEK[here]}  ({VERB[side]} {WEEK[other]})", fontsize=st.FS_MIN, va="center")
            if k:
                ax.axhline(k - 0.5, color=RULE_LIGHT, lw=0.5)
                sh.fig.add_artist(plt.Line2D([x_week / W, left / W], [(y_of(k) + row_h / 2) / H] * 2, color=st.GUIDE,
                                             lw=0.5, transform=sh.fig.transFigure))
        if new_block or u.si.iat[k] != u.si.iat[k - 1]:
            sh.text(x_state, y_of(k), r.state, fontsize=st.FS_MIN, ha="right", va="center")
        sh.text(x_fate, y_of(k), f"{VERB[side]} {r.fate}", fontsize=st.FS_MIN, va="center")
        sh.text(x_ref, y_of(k), f"{VERB[side]} {r.reference}", fontsize=st.FS_MIN, color=st.SUBTLE, va="center")
        sh.text(x_nf, y_of(k), f"{r.fate_n:,}", fontsize=st.FS_MIN, ha="right", va="center")
        sh.text(x_nr, y_of(k), f"{r.reference_n:,}", fontsize=st.FS_MIN, ha="right", va="center")
        sh.text(x_auc, y_of(k), f"{r.auc_depth_fate:.2f}", fontsize=st.FS_MIN, ha="right", va="center")

    x0 = 0
    for cat, m in CATEGORIES:
        x1 = x0 + len(m) - 1
        ax.annotate("", xy=(x0 - 0.35, -0.85), xytext=(x1 + 0.35, -0.85), xycoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK), annotation_clip=False)
        ax.text((x0 + x1) / 2, -1.1, cat, ha="center", va="bottom", fontsize=st.FS_MIN, clip_on=False)
        if x0:
            ax.axvline(x0 - 0.5, color=RULE_LIGHT, lw=0.5)
        x0 = x1 + 1

    cx, ch = left + 50 * col_w + 0.22, min(1.0, max(0.6, len(u) * row_h))
    cy = bottom + len(u) * row_h - ch
    cax = sh.ax(cx, cy, 0.08, ch)
    cb = sh.fig.colorbar(im, cax=cax, extend="both"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=st.FS_MIN, length=2); cb.set_ticks([-2, 0, 2])
    sh.text(cx, cy + ch + 0.06, "NES", fontsize=st.FS_MIN, va="bottom")
    sh.text(cx + 0.32, cy + ch - 0.02, "higher in\ngroup X", fontsize=st.FS_MIN, color="#c06a08", va="top")
    sh.text(cx + 0.32, cy + 0.02, "higher in\nreference", fontsize=st.FS_MIN, color="#2b6cb4", va="bottom")
    size_legend(sh, cx, cy - 0.12)

    sh.text(0.08, H - 0.08, f"{TITLE[side]}  ·  ConfidenceOT {NAME[cfg]}  ·  {EXPR[kind]}", fontsize=st.FS_TITLE,
            va="top")
    sh.text(0.08, H - 0.26, EXPLAIN[side], fontsize=st.FS_MIN, color=st.SUBTLE, va="top", linespacing=1.4)
    sh.save(out, f"pdx_fate_gsea_{cfg}_{side}_{kind}")
    return len(u)


def depth_check(nes: pd.Series, auc: pd.Series, title: str, xlabel: str, ylabel: str, name: str, out: Path) -> None:
    """One unit / contrast per point: mean NES over the 50 hallmark sets vs the depth AUC of the two groups."""
    from scipy.stats import spearmanr
    sh = st.Sheet(2.35, 2.05)
    ax = sh.ax(0.48, 0.40, 1.72, 1.30); st.tidy(ax)
    ax.axhline(0, **st.REF_KW); ax.axvline(0.5, **st.REF_KW)
    ax.scatter(auc, nes, s=9, color=st.PAL["model"], lw=0, alpha=0.8)
    rho = spearmanr(auc, nes)[0]
    ax.set_xlim(0.1, 1.0); ax.set_ylim(-1.6, 1.6); ax.set_xticks([0.2, 0.5, 0.8]); ax.set_yticks([-1, 0, 1])
    ax.set_xlabel(xlabel, fontsize=st.FS_AXLAB); ax.set_ylabel(ylabel, fontsize=st.FS_AXLAB)
    ax.text(0.12, 1.45, f"Spearman {rho:.2f}, n = {len(nes)}", fontsize=st.FS_ANNOT, va="top")
    sh.text(0.06, 2.0, title, fontsize=st.FS_TITLE, va="top")
    sh.save(out, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gsea", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rejected-gsea", type=Path, help="pdx_deg_gsea.py output (rejected vs retained), for the depth check")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.apply()
    con = pd.read_csv(a.gsea / "contrasts.tsv", sep="\t")
    gs = pd.read_csv(a.gsea / "gsea_hallmark.tsv", sep="\t")
    for cfg in con.config.unique():
        for side in ("source", "target"):
            for kind in EXPR:
                print(f"  {cfg} {side} {kind}: {panel(gs, con, cfg, side, kind, a.out)} contrasts")
        f = pd.read_csv(a.gsea / f"fates_{cfg}.tsv.gz", sep="\t")
        f.groupby(["pair", "side", "state", "fate"]).size().rename("cells").reset_index().to_csv(
            a.out / f"fate_composition_{cfg}.tsv", sep="\t", index=False)
    for kind in EXPR:  # depth check: mean NES of a row vs the depth AUC of its two groups
        m = gs[gs.expression == kind].groupby(["config", "pair", "side", "state", "fate"]).agg(
            nes=("nes", "mean"), auc=("auc_depth_fate", "first"))
        depth_check(m.nes, m.auc, f"Fate contrasts, {EXPR[kind].split(' (')[0]}", "depth AUC, X vs reference",
                    "mean NES (50 sets)", f"pdx_depth_check_fate_{kind}", a.out)
    if a.rejected_gsea:
        r = pd.read_csv(a.rejected_gsea / "gsea_hallmark.tsv", sep="\t").groupby(["config", "pair", "side", "state"]).agg(
            nes=("nes", "mean"), auc=("auc_depth_rejected", "first"))
        depth_check(r.nes, r.auc, "Rejected vs retained, original counts", "depth AUC, rejected vs retained",
                    "mean NES (50 sets)", "pdx_depth_check_rejected_rawqc", a.out)


if __name__ == "__main__":
    main()

"""Figures and tables from pdx_deg_gsea.py: hallmark GSEA dot panels and the top DEG of every analysed unit.

    python realdata_benchmark/v3_grid/figures/pdx_gsea_figures.py --gsea results/v3_grid_20261104/pdx_deg_gsea_20261105 \
        --out results/v3_grid_20261104/figures/pdx_gsea

Per configuration and side (every analysed unit, both sides, all 50 hallmark sets; nothing filtered):
  pdx_gsea_<cfg>_<side>    rows = pair x state units (time order, then the published state order), columns = the 50
                           hallmark sets by the process categories of Liberzon et al. 2015; colour = NES of rejected vs
                           retained (orange: higher in rejected), one colour scale (+-2.5) for every panel; the NES only.
                           The cells are pseudo-replicates (one tumour per time point), so the FDR is not drawn (it is
                           in gsea_hallmark.tsv). Left columns: the time point of the gated cells and the one it is
                           gated against, the state, the rejected and retained cell counts, and the AUC of pre-equalisation
                           depth for rejected vs retained (> 0.5: rejected cells are deeper).
  pdx_deg_top.tsv          every unit: the 10 genes highest and lowest by Wilcoxon score, with log fold change and the
                           detected fraction in rejected cells
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nbt_style as st  # noqa: E402
from pdx_prepare import STATES  # noqa: E402

# Liberzon et al. 2015, Fig. 2 (the order of cancer_metastasis/47_gsea_dotplot.py)
CATEGORIES = [
    ("Cellular\ncomponent", ["APICAL_JUNCTION", "APICAL_SURFACE", "PEROXISOME"]),
    ("Development", ["ADIPOGENESIS", "ANGIOGENESIS", "EPITHELIAL_MESENCHYMAL_TRANSITION", "MYOGENESIS", "SPERMATOGENESIS",
                     "PANCREAS_BETA_CELLS"]),
    ("DNA\ndamage", ["DNA_REPAIR", "UV_RESPONSE_DN", "UV_RESPONSE_UP"]),
    ("Immune", ["ALLOGRAFT_REJECTION", "COAGULATION", "COMPLEMENT", "INTERFERON_ALPHA_RESPONSE", "INTERFERON_GAMMA_RESPONSE",
                "IL6_JAK_STAT3_SIGNALING", "INFLAMMATORY_RESPONSE"]),
    ("Metabolic", ["BILE_ACID_METABOLISM", "CHOLESTEROL_HOMEOSTASIS", "FATTY_ACID_METABOLISM", "GLYCOLYSIS",
                   "HEME_METABOLISM", "OXIDATIVE_PHOSPHORYLATION", "XENOBIOTIC_METABOLISM"]),
    ("Pathway", ["APOPTOSIS", "HYPOXIA", "PROTEIN_SECRETION", "UNFOLDED_PROTEIN_RESPONSE", "REACTIVE_OXYGEN_SPECIES_PATHWAY"]),
    ("Proliferation", ["E2F_TARGETS", "G2M_CHECKPOINT", "MYC_TARGETS_V1", "MYC_TARGETS_V2", "P53_PATHWAY", "MITOTIC_SPINDLE"]),
    ("Signaling", ["ANDROGEN_RESPONSE", "ESTROGEN_RESPONSE_EARLY", "ESTROGEN_RESPONSE_LATE", "IL2_STAT5_SIGNALING",
                   "KRAS_SIGNALING_UP", "KRAS_SIGNALING_DN", "MTORC1_SIGNALING", "NOTCH_SIGNALING", "PI3K_AKT_MTOR_SIGNALING",
                   "HEDGEHOG_SIGNALING", "TGF_BETA_SIGNALING", "TNFA_SIGNALING_VIA_NFKB", "WNT_BETA_CATENIN_SIGNALING"]),
]
PATHWAYS = [p for _, m in CATEGORIES for p in m]
assert len(PATHWAYS) == 50 and len(set(PATHWAYS)) == 50
CM = LinearSegmentedColormap.from_list("nes", ["#2b6cb4", "#ffffff", "#f28c1c"])
SAMPLES = ["w00_preCX_1", "w02_2wkCX_2", "w04_4wkCX_1", "w08_8wkCX_1", "w12_12wkCX_2", "w16_16wkCX_1", "w20_20wkCX_1_R",
           "w22_22wkCX_2_R"]
WEEK = dict(zip(SAMPLES, ["wk 0", "wk 2", "wk 4", "wk 8", "wk 12", "wk 16", "wk 20", "wk 22"]))
NAME = {"baseline": "production baseline (ranknm256 production noscale cos PCA30)",
        "splatter_l5_best": "Splatter L5 best (ranknm256 midrank scale cos PCA50)",
        "logcpm": "logCPM (logcpm noscale cos PCA30)"}
NES_LIM = 2.5  # one colour scale for every panel; |NES| max 3.2, 99th percentile 2.2
WEEK_HEAD = {"source": "cells at", "target": "cells at"}
TITLE = {"source": "What differs between the cells ConfidenceOT rejects and keeps, gated against the NEXT time point",
         "target": "What differs between the cells ConfidenceOT rejects and keeps, gated against the PREVIOUS time point"}
EXPLAIN = {
    "source": ("Each row: the cells of one state at one time point. ConfidenceOT (time point -> next time point) splits them "
               "into rejected (no counterpart at the next time point) and retained (a counterpart there).\n"
               "Each square: hallmark GSEA NES on genes ranked by Wilcoxon, rejected vs retained cells of that row. "
               "One tumour per time point, so the cells are pseudo-replicates: read the NES, not a p-value (FDR in "
               "gsea_hallmark.tsv). Depth AUC: rejected vs retained cells by sequencing depth (> 0.5: rejected deeper)."),
    "target": ("Each row: the cells of one state at one time point. ConfidenceOT (previous time point -> time point) splits "
               "them into rejected (no counterpart at the previous time point) and retained (a counterpart there).\n"
               "Each square: hallmark GSEA NES on genes ranked by Wilcoxon, rejected vs retained cells of that row. "
               "One tumour per time point, so the cells are pseudo-replicates: read the NES, not a p-value (FDR in "
               "gsea_hallmark.tsv). Depth AUC: rejected vs retained cells by sequencing depth (> 0.5: rejected deeper)."),
}


UPPER = {"DNA", "UV", "DN", "IL6", "JAK", "STAT3", "IL2", "STAT5", "KRAS", "MTORC1", "PI3K", "AKT", "MTOR", "TGF", "TNFA",
         "NFKB", "WNT", "E2F", "G2M", "MYC", "V1", "V2", "P53"}


def pretty(p: str) -> str:
    w = [x if x in UPPER else x.lower() for x in p.split("_")]
    w[0] = w[0] if w[0] in UPPER else w[0].capitalize()
    return " ".join(w).replace("TNFA", "TNFα").replace("TGF beta", "TGF-β").replace("NFKB", "NF-κB")


def term_key(t: str) -> str:
    return t.replace("HALLMARK_", "").upper()


def panel(gs: pd.DataFrame, units: pd.DataFrame, cfg: str, side: str, out: Path) -> int:
    u = units[(units.config == cfg) & (units.side == side) & (units.status == "analysed")].copy()
    if u.empty:
        return 0
    u["pi"] = u.pair.map(lambda p: SAMPLES.index(p.split("__")[0])); u["si"] = u.state.map(STATES.index)
    u = u.sort_values(["pi", "si"]).reset_index(drop=True)
    g = gs[(gs.config == cfg) & (gs.side == side)].copy(); g["key"] = g.term.map(term_key)
    nes = np.full((len(u), 50), np.nan)
    for i, r in u.iterrows():
        h = g[(g.pair == r.pair) & (g.state == r.state)].set_index("key")
        for j, pw in enumerate(PATHWAYS):
            if pw in h.index:
                nes[i, j] = float(h.at[pw, "nes"])

    row_h, col_w = 0.13, 0.105
    left, top_pad, bottom = 3.15, 0.88, 1.40
    W = left + 50 * col_w + 1.35; H = top_pad + len(u) * row_h + bottom
    sh = st.Sheet(W, H)
    ax = sh.ax(left, bottom, 50 * col_w, len(u) * row_h)
    cm = CM.copy(); cm.set_bad("#EEEFF0")
    im = ax.pcolormesh(np.arange(51) - 0.5, np.arange(len(u) + 1) - 0.5, np.ma.masked_invalid(nes), cmap=cm,
                       norm=TwoSlopeNorm(0, -NES_LIM, NES_LIM), edgecolors="white", lw=0.3)
    ax.set_xlim(-0.5, 49.5); ax.set_ylim(len(u) - 0.5, -0.5)
    ax.set_yticks([])
    ax.set_xticks(range(50)); ax.set_xticklabels([pretty(p) for p in PATHWAYS], rotation=60,
                                                 ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)

    # left columns: time point (one label per block), state, rejected and retained cell counts
    x_week, x_state, x_rej, x_ret, x_auc = 0.08, left - 1.02, left - 0.74, left - 0.44, left - 0.08
    y_of = lambda k: bottom + (len(u) - k - 0.5) * row_h  # noqa: E731
    y_head = bottom + len(u) * row_h + 0.07
    for x, t, ha in ((x_week, f"{WEEK_HEAD[side]}", "left"), (x_state, "state", "right"),
                     (x_rej, "rejected", "right"), (x_ret, "retained", "right"),
                     (x_auc, "depth AUC", "right")):
        sh.text(x, y_head, t, fontsize=st.FS_MIN, color=st.SUBTLE, ha=ha, va="bottom")
    for k, r in enumerate(u.itertuples()):
        a_, b_ = r.pair.split("__")
        here, other = (a_, b_) if side == "source" else (b_, a_)
        if k == 0 or u.pi.iat[k] != u.pi.iat[k - 1]:
            sh.text(x_week, y_of(k), f"{WEEK[here]}  (vs {WEEK[other]})", fontsize=st.FS_MIN, va="center")
            if k:
                ax.axhline(k - 0.5, color=st.INK, lw=0.5)
                sh.fig.add_artist(plt.Line2D([x_week / W, left / W], [(y_of(k) + row_h / 2) / H] * 2, color=st.GUIDE,
                                             lw=0.5, transform=sh.fig.transFigure))
        sh.text(x_state, y_of(k), r.state, fontsize=st.FS_MIN, ha="right", va="center")
        sh.text(x_rej, y_of(k), f"{r.rejected_n:,}", fontsize=st.FS_MIN, ha="right", va="center")
        sh.text(x_ret, y_of(k), f"{r.retained_n:,}", fontsize=st.FS_MIN, ha="right", va="center")
        sh.text(x_auc, y_of(k), f"{r.auc_depth_rejected:.2f}", fontsize=st.FS_MIN, ha="right", va="center")

    x0 = 0
    for cat, m in CATEGORIES:
        x1 = x0 + len(m) - 1
        ax.annotate("", xy=(x0 - 0.35, -0.85), xytext=(x1 + 0.35, -0.85), xycoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK), annotation_clip=False)
        ax.text((x0 + x1) / 2, -1.1, cat, ha="center", va="bottom", fontsize=st.FS_MIN, clip_on=False)
        if x0:
            ax.axvline(x0 - 0.5, color="white", lw=1.2)
        x0 = x1 + 1

    cx, ch = left + 50 * col_w + 0.22, min(1.1, len(u) * row_h)
    cy = bottom + len(u) * row_h - ch
    cax = sh.ax(cx, cy, 0.08, ch)
    cb = sh.fig.colorbar(im, cax=cax, extend="both"); cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=st.FS_MIN, length=2); cb.set_ticks([-2, -1, 0, 1, 2])
    sh.text(cx, cy + ch + 0.06, "NES", fontsize=st.FS_MIN, va="bottom")
    sh.text(cx + 0.32, cy + ch - 0.02, "higher in\nrejected cells", fontsize=st.FS_MIN, color="#c06a08", va="top")
    sh.text(cx + 0.32, cy + 0.02, "higher in\nretained cells", fontsize=st.FS_MIN, color="#2b6cb4", va="bottom")
    sh.text(cx, cy - 0.12, "grey: set not\nscored (size\noutside 15-500)", fontsize=st.FS_MIN, color=st.SUBTLE, va="top")

    sh.text(0.08, H - 0.08, f"{TITLE[side]}  ·  {NAME[cfg]}", fontsize=st.FS_TITLE, va="top")
    sh.text(0.08, H - 0.26, EXPLAIN[side], fontsize=st.FS_MIN, color=st.SUBTLE, va="top", linespacing=1.4)
    sh.save(out, f"pdx_gsea_{cfg}_{side}")
    return len(u)


def deg_top(root: Path, units: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    rows = []
    for r in units[units.status == "analysed"].itertuples():
        f = root / "deg" / r.config / f"{r.pair}__{r.side}__{r.state.replace(' ', '_')}.tsv.gz"
        d = pd.read_csv(f, sep="\t").sort_values("scores", ascending=False)
        for direction, part in (("higher in rejected", d.head(n)), ("higher in retained", d.tail(n)[::-1])):
            for k, x in enumerate(part.itertuples(), start=1):
                rows.append({"config": r.config, "pair": r.pair, "side": r.side, "state": r.state, "direction": direction,
                             "rank": k, "gene": x.names, "score": x.scores, "logfoldchange": x.logfoldchanges,
                             "pct_rejected": x.pct_nz_group})  # scanpy writes no reference fraction for a groups subset
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gsea", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    st.apply()
    units = pd.read_csv(a.gsea / "units.tsv", sep="\t")
    gs = pd.read_csv(a.gsea / "gsea_hallmark.tsv", sep="\t")
    for cfg in units.config.unique():
        for side in ("source", "target"):
            k = panel(gs, units, cfg, side, a.out)
            print(f"  {cfg} {side}: {k} units")
    deg_top(a.gsea, units).to_csv(a.out / "pdx_deg_top.tsv", sep="\t", index=False)
    print(units.groupby(["config", "status"]).size().to_string())


if __name__ == "__main__":
    main()

"""Figures and tables from pdx_deg_gsea.py: hallmark GSEA dot panels and the top DEG of every analysed unit.

    python realdata_benchmark/v3_grid/figures/pdx_gsea_figures.py --gsea results/v3_grid_20261104/pdx_deg_gsea_20261105 \
        --out results/v3_grid_20261104/figures/pdx_gsea

Per configuration and side (every analysed unit, both sides, all 50 hallmark sets; nothing filtered):
  pdx_gsea_<cfg>_<side>    rows = pair x state units (time order, then the published state order), columns = the 50
                           hallmark sets by the process categories of Liberzon et al. 2015; colour = NES of rejected vs
                           retained (orange: higher in rejected); size = -log10 FDR in four bins. The cells are
                           pseudo-replicates (one tumour per time point), so the size is a ranking aid, not a test.
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
SIZE_BREAKS, SIZE_AREAS, SIZE_LABELS = (1.0, 2.0, 3.0), (3.0, 12.0, 26.0, 46.0), ("< 1", "1-2", "2-3", ">= 3")
CM = LinearSegmentedColormap.from_list("nes", ["#2b6cb4", "#ffffff", "#f28c1c"])
SAMPLES = ["w00_preCX_1", "w02_2wkCX_2", "w04_4wkCX_1", "w08_8wkCX_1", "w12_12wkCX_2", "w16_16wkCX_1", "w20_20wkCX_1_R",
           "w22_22wkCX_2_R"]
WEEK = dict(zip(SAMPLES, ["wk 0", "wk 2", "wk 4", "wk 8", "wk 12", "wk 16", "wk 20", "wk 22"]))
NAME = {"baseline": "production baseline (ranknm256 production noscale cos PCA30)",
        "splatter_l5_best": "Splatter L5 best (ranknm256 midrank scale cos PCA50)",
        "logcpm": "logCPM (logcpm noscale cos PCA30)"}
SIDE = {"source": "as source (does the cell have a counterpart at the next time point?)",
        "target": "as target (does the cell have a counterpart at the previous time point?)"}


UPPER = {"DNA", "UV", "DN", "IL6", "JAK", "STAT3", "IL2", "STAT5", "KRAS", "MTORC1", "PI3K", "AKT", "MTOR", "TGF", "TNFA",
         "NFKB", "WNT", "E2F", "G2M", "MYC", "V1", "V2", "P53"}


def pretty(p: str) -> str:
    w = [x if x in UPPER else x.lower() for x in p.split("_")]
    w[0] = w[0] if w[0] in UPPER else w[0].capitalize()
    return " ".join(w).replace("TNFA", "TNFα").replace("TGF beta", "TGF-β").replace("NFKB", "NF-κB")


def term_key(t: str) -> str:
    return t.replace("HALLMARK_", "").upper()


def label(r) -> str:
    a, b = r.pair.split("__")
    return f"{WEEK[a]} -> {WEEK[b]} · {r.state}  ({r.rejected_n} / {r.retained_n})"


def panel(gs: pd.DataFrame, units: pd.DataFrame, cfg: str, side: str, out: Path) -> int:
    u = units[(units.config == cfg) & (units.side == side) & (units.status == "analysed")].copy()
    if u.empty:
        return 0
    u["pi"] = u.pair.map(lambda p: SAMPLES.index(p.split("__")[0])); u["si"] = u.state.map(STATES.index)
    u = u.sort_values(["pi", "si"]).reset_index(drop=True)
    g = gs[(gs.config == cfg) & (gs.side == side)].copy(); g["key"] = g.term.map(term_key)
    nes = np.full((len(u), 50), np.nan); fdr = np.full_like(nes, np.nan)
    for i, r in u.iterrows():
        h = g[(g.pair == r.pair) & (g.state == r.state)].set_index("key")
        for j, pw in enumerate(PATHWAYS):
            if pw in h.index:
                nes[i, j] = float(h.at[pw, "nes"]); fdr[i, j] = float(h.at[pw, "fdr"])
    strength = -np.log10(np.clip(fdr, 1e-3, 1))
    area = np.asarray(SIZE_AREAS)[np.digitize(np.nan_to_num(strength), SIZE_BREAKS)]

    row_h, col_w = 0.115, 0.098
    left, top_pad, bottom = 2.55, 0.80, 1.35
    W = left + 50 * col_w + 1.05; H = top_pad + len(u) * row_h + bottom
    sh = st.Sheet(W, H)
    ax = sh.ax(left, bottom, 50 * col_w, len(u) * row_h)
    yy, xx = np.meshgrid(np.arange(len(u)), np.arange(50), indexing="ij")
    ok = np.isfinite(nes)
    lim = max(2.0, float(np.nanmax(np.abs(nes))) if ok.any() else 2.0)
    sc_ = ax.scatter(xx[ok], yy[ok], s=area[ok], c=nes[ok], cmap=CM, norm=TwoSlopeNorm(0, -lim, lim), lw=0.25,
                     edgecolors=np.where(fdr[ok] < 0.05, "#333333", "none"))
    ax.set_xlim(-0.6, 49.6); ax.set_ylim(len(u) - 0.5, -0.5)
    ax.set_yticks(range(len(u))); ax.set_yticklabels([label(r) for r in u.itertuples()], fontsize=st.FS_MIN)
    ax.set_xticks(range(50)); ax.set_xticklabels([pretty(p) for p in PATHWAYS], rotation=60,
                                                 ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4); s_.set_color("#B0B5B9")
    for k in range(1, len(u)):  # a rule between time-point pairs
        if u.pi.iat[k] != u.pi.iat[k - 1]:
            ax.axhline(k - 0.5, color="#D5D8DB", lw=0.4)
    x0 = 0
    for cat, m in CATEGORIES:
        x1 = x0 + len(m) - 1
        ax.annotate("", xy=(x0 - 0.3, -0.9), xytext=(x1 + 0.3, -0.9), xycoords="data",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=st.INK), annotation_clip=False)
        ax.text((x0 + x1) / 2, -1.2, cat, ha="center", va="bottom", fontsize=st.FS_MIN, clip_on=False)
        if x0:
            ax.axvline(x0 - 0.5, color="#D5D8DB", lw=0.4)
        x0 = x1 + 1
    cax = sh.ax(left + 50 * col_w + 0.25, bottom + len(u) * row_h - 0.9, 0.08, 0.85)
    cb = sh.fig.colorbar(sc_, cax=cax); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=st.FS_MIN, length=2)
    cb.set_label("NES (orange: higher in rejected)", fontsize=st.FS_MIN)
    lx, ly = left + 50 * col_w + 0.25, bottom + len(u) * row_h - 1.25
    sh.text(lx, ly, "-log10 FDR", fontsize=st.FS_MIN)
    for k, (a_, lab) in enumerate(zip(SIZE_AREAS, SIZE_LABELS)):
        sh.fig.add_artist(plt.Line2D([(lx + 0.06) / W], [(ly - 0.16 * (k + 1)) / H], marker="o", ms=np.sqrt(a_),
                                     color="#808080", lw=0, transform=sh.fig.transFigure))
        sh.text(lx + 0.16, ly - 0.16 * (k + 1), lab, fontsize=st.FS_MIN, va="center")
    sh.text(left, H - 0.08, f"Hallmark GSEA, rejected vs retained within each state, {side}, {NAME[cfg]}",
            fontsize=st.FS_TITLE, va="top")
    sh.text(left, H - 0.26, SIDE[side] + "; row: pair · state (rejected / retained cells); outlined: FDR < 0.05",
            fontsize=st.FS_MIN, color=st.SUBTLE, va="top")
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

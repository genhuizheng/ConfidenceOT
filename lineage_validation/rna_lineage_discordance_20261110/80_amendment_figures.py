"""PLAN_v2.md amendment A6: figure panels, one PDF + PNG each (nbt style).

    python lineage_validation/rna_lineage_discordance_20261110/80_amendment_figures.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs [--tag l5best] [--representative 3457_Apc_T1]

Panels (ROOT/figures/<tag>/amendment/):
  g1_residuals_<mouse>   clade x state standardised residuals (sensitivity scope, every mouse with >= 2 clades and states)
  g2_state_hill_z        per state and mouse: z of the clade Hill diversity against the within-site null
  g2_state_maxshare_z    the same for the largest clade share
  g3_primary_vs_met      per mouse: Cramer's V excess over the null, primary vs metastasis, all cells and per tertile
  g3_viewc_side_state    KP view C: divergence excess per state, source (primary) vs target (metastasis)
  g4_common_tv           per unit: median TV on each coupling's own retained block vs on the common block
  g4_common_overlap      per unit: top-k overlap on the common block vs its chance level
  g5_rep_<family>_{state,clade,site}  UMAP of the family's RNA coordinates (its L2/L3 unit, the tag's configuration)
  g5_rep_<family>_edits  informative-edit matrix (cells by clade x events in >= 2% of cells)
The representative family defaults to 3457_Apc_T1: the resolved family with the most states of >= 50 cells.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
sys.path.insert(0, str(REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"))
import matrix_core as mc  # noqa: E402
import nbt_style as st  # noqa: E402

CAT = ["#2C6E9B", "#B4702A", "#4E8A6B", "#C0392B", "#7D3C98", "#8C7A5B", "#5E8FA8", "#A3A86B", "#6B7280", "#D19A9A"]
CONFIG_FILE = {"l5best": "ranknm256-midrank__none__scale__cos__pca50", "logcpm_pca": "logcpm__none__noscale__sqeuc__pca30"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def heat(df, value, out, name, title, cbar_label, annot=None):
    rows, cols = list(df.index), list(df.columns)
    cw, rh = 0.32, 0.16
    left = 0.25 + 0.042 * max(len(str(r)) for r in rows)
    bottom = 0.25 + 0.036 * max(len(str(c)) for c in cols)
    pw, ph = cw * len(cols), rh * len(rows)
    W, H = left + pw + 0.75, bottom + ph + 0.3
    sh = st.Sheet(max(W, 2.4), H); ax = sh.ax(left, bottom, pw, ph)
    v = df.to_numpy(float); lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1
    im = ax.imshow(v, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=st.FS_MIN)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annot is not None:
        for i in range(len(rows)):
            for j in range(len(cols)):
                if pd.notna(annot.iat[i, j]):
                    dark = np.isfinite(v[i, j]) and abs(v[i, j]) > 0.6 * lim
                    ax.text(j, i, f"{int(annot.iat[i, j])}", ha="center", va="center", fontsize=st.FS_MIN, color="white" if dark else st.INK)
    cax = sh.ax(left + pw + 0.12, bottom, 0.07, min(0.9, ph))
    cb = plt.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=st.FS_MIN); cb.set_label(cbar_label, fontsize=st.FS_MIN)
    cb.outline.set_visible(False)
    sh.text(0.05, H - 0.05, title, fontsize=st.FS_TITLE, va="top"); sh.save(out, name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--tag", default="l5best")
    p.add_argument("--representative", default="3457_Apc_T1")
    a = p.parse_args()
    out = a.root / "figures" / a.tag / "amendment"; out.mkdir(parents=True, exist_ok=True)
    st.apply()
    rd = lambda f: pd.read_csv(f, sep="\t", dtype={"mouse": str}, low_memory=False)  # noqa: E731
    SL, AS = a.root / "state_lineage", a.root / "amendment_summary"
    res = rd(a.root / "lineages" / "resolution.tsv")
    resolved_mice = set(res[(res.dataset == "KPTracer") & res.resolved].mouse)

    # g1 residual heatmaps
    R = rd(SL / "residuals.tsv"); R = R[R.scope == "sensitivity_all"]
    for mouse, g in R.groupby("mouse"):
        M = g.pivot(index="clade", columns="state", values="residual"); N = g.pivot(index="clade", columns="state", values="count")
        M.index = [c.split("|")[-1] + " (" + c.split("|")[0] + ")" for c in M.index]; N.index = M.index
        heat(M, "residual", out, f"g1_residuals_{mouse}", f"KP {mouse}: clade x state, standardised residual (counts in cells)",
             "(obs - exp) / sqrt(exp)", annot=N)

    # g2 state diversity z
    S = rd(SL / "states.tsv"); S = S[S.scope == "sensitivity_all"]
    for col, name, lab in (("hill_z", "g2_state_hill_z", "z, clade Hill diversity vs within-site null"),
                           ("max_share_z", "g2_state_maxshare_z", "z, largest clade share vs within-site null")):
        M = S.pivot(index="state", columns="mouse", values=col).clip(-10, 10)
        heat(M, col, out, name, lab + " (clipped at +/-10)", "z", annot=None)

    # g3 primary vs metastasis
    pm = rd(SL / "primary_vs_met.tsv")
    if len(pm):
        pm["lab"] = np.where(pm.stratum == "all", "all", pm.stratum + " " + pm.tertile.astype(str))
        mice = sorted(pm.mouse.unique()); labs = ["all"] + [f"{a_} {b_}" for a_ in ("depth", "events") for b_ in ("low", "mid", "high")]
        sh = st.Sheet(1.0 + 1.25 * len(mice), 2.3)
        for k, mouse in enumerate(mice):
            ax = sh.ax(0.55 + 1.25 * k, 0.75, 1.0, 1.25); st.tidy(ax)
            d = pm[pm.mouse == mouse].set_index("lab").reindex(labs)
            x = np.arange(len(labs))
            ax.vlines(x, d.primary_excess, d.metastasis_excess, color=st.GUIDE, lw=st.LW_GUIDE * 2)
            ax.scatter(x, d.primary_excess, s=10, color=st.PAL["tcr"], lw=0, label="primary", zorder=3)
            ax.scatter(x, d.metastasis_excess, s=10, color=st.PAL["peptide"], lw=0, label="metastasis", zorder=3)
            ax.axhline(0, **st.REF_KW)
            ax.set_xticks(x); ax.set_xticklabels(labs, rotation=70, ha="right", fontsize=st.FS_MIN)
            ax.set_title(f"KP {mouse}" + ("" if mouse in resolved_mice else " (unresolved)"), fontsize=st.FS_TITLE)
            if "metastasis_permutable_share" in d:
                a0 = d.loc["all"] if "all" in d.index else None
                if a0 is not None and pd.notna(a0.metastasis_permutable_share):
                    ax.text(0.02, 0.98, f"permutable: pri {a0.primary_permutable_share:.2f}, met {a0.metastasis_permutable_share:.2f}",
                            transform=ax.transAxes, fontsize=st.FS_MIN, color=st.SUBTLE, va="top")
            if k == 0:
                ax.set_ylabel("Cramer's V - null V", fontsize=st.FS_AXLAB); ax.legend(frameon=False, fontsize=st.FS_LEGEND)
        sh.text(0.05, 2.25, "Clade x state coupling beyond the within-site null, equal n per side (200 draws)", fontsize=st.FS_TITLE, va="top")
        sh.save(out, "g3_primary_vs_met")
    v = rd(AS / "a3_viewc_side_state.tsv")
    M = v.assign(col=v.mouse + " " + v.side.map({"source": "pri", "target": "met"})).pivot(index="cell_type", columns="col", values="div_excess")
    heat(M, "div_excess", out, "g3_viewc_side_state", "KP view C: divergence excess (observed - null) by state and side", "div excess")

    # g4 common support
    cs = rd(a.root / "common_support" / "units.tsv"); cs = cs[cs.config == a.tag]
    ab = rd(a.root / "summary" / "a_vs_b_units.tsv"); ab = ab[ab.tag == a.tag][["unit_id", "tv_median"]].rename(columns={"tv_median": "tv_own"})
    cs = cs.merge(ab, on="unit_id")
    colr = {"KPTracer": st.PAL["tcr"], "Quinn2021": st.PAL["peptide"]}
    for name, xcol, ycol, xl, yl, title in (("g4_common_tv", "tv_own", "tv_median", "TV, each coupling's own retained block", "TV, common block",
                                             "Coupling disagreement before / after restricting to the common support"),
                                            ("g4_common_overlap", "chance_overlap", "overlap_mean", "chance overlap k / common sources",
                                             "top-k overlap, common block", "Top-source overlap on the common block vs chance")):
        sh = st.Sheet(2.6, 2.4); ax = sh.ax(0.55, 0.45, 1.85, 1.65); st.tidy(ax)
        for ds, g in cs.groupby("dataset"):
            ax.scatter(g[xcol], g[ycol], s=np.where(g.mouse == "3724", 16, 9), color=colr[ds], lw=0, label=ds)
        lim = [0, max(cs[[xcol, ycol]].max().max(), 0.05) * 1.05]
        ax.plot(lim, lim, **st.REF_KW); ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(xl, fontsize=st.FS_AXLAB); ax.set_ylabel(yl, fontsize=st.FS_AXLAB); ax.legend(frameon=False, fontsize=st.FS_LEGEND)
        sh.text(0.05, 2.35, title, fontsize=st.FS_TITLE, va="top"); sh.save(out, name)

    # g5 representative family
    fam = a.representative; mouse = fam.split("_")[0]
    cells = rd(SL / "cells.tsv.gz"); fc = cells[cells.family == fam].set_index("cell")
    u = f"L23__KPTracer_{mouse}__{fam}__family_metastases"
    z = np.load(a.root / "inputs_A" / "rna_lineage_discordance_20261110" / "rna_coords" / u / f"{CONFIG_FILE[a.tag]}.npz", allow_pickle=True)
    ids = np.r_[z["source_ids"].astype(str), z["target_ids"].astype(str)]; Z = np.vstack([z["X"], z["Y"]])
    keep = pd.Index(ids).isin(fc.index); ids, Z = ids[keep], Z[keep]
    import umap
    E2 = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine" if bool(z["cos"]) else "euclidean", random_state=20261111).fit_transform(Z)
    f = fc.loc[ids]
    for col in ("state", "clade", "site"):
        labs = f[col].astype(str).str.split("|").str[-1]
        vc = labs.value_counts(); small = vc[vc < 20].index.union(vc.index[len(CAT):])
        labs = labs.where(~labs.isin(small) | (labs == "minor"), "other (< 20 cells)")
        order = [x for x in labs.value_counts().index if x not in ("minor", "other (< 20 cells)")] +                 [x for x in ("minor", "other (< 20 cells)") if (labs == x).any()]
        sh = st.Sheet(3.4, 2.4); ax = sh.ax(0.15, 0.15, 2.0, 2.0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        for k, lv in enumerate(order):
            m = (labs == lv).to_numpy()
            ax.scatter(E2[m, 0], E2[m, 1], s=2.5, color=CAT[k] if lv not in ("minor", "other (< 20 cells)") else st.NEUTRAL, lw=0, label=f"{lv} ({m.sum()})")
        ax.legend(frameon=False, fontsize=st.FS_MIN, loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2.5)
        sh.text(0.05, 2.38, f"{fam}: RNA UMAP ({a.tag} coordinates), colour = {col}", fontsize=st.FS_TITLE, va="top")
        sh.save(out, f"g5_rep_{fam}_{col}")
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); t, prior = bb.load_tables(a.inputs)["KPTracer"]
    sub = t[(t.lineage == fam) & t.cell.isin(set(ids))]
    L = mc.Lineage(sub, ids, np.repeat("source", len(ids)), prior)
    I = L.incidence.toarray() > 0
    freq = I.mean(0); ev = np.flatnonzero(freq >= 0.02); ev = ev[np.argsort(-freq[ev])][:60]
    clade = f.clade.astype(str).to_numpy(); co = pd.Series(clade).value_counts().index
    from scipy.cluster.hierarchy import leaves_list, linkage
    order = []
    for c in co:
        m = np.flatnonzero(clade == c)
        order += list(m[leaves_list(linkage(I[np.ix_(m, ev)], "average", metric="jaccard"))] if len(m) > 2 else m)
    order = np.array(order)
    M = I[np.ix_(order, ev)]
    sh = st.Sheet(4.2, 3.2); ax = sh.ax(0.7, 0.35, 3.3, 2.5)
    ax.imshow(M, aspect="auto", cmap=plt.matplotlib.colors.ListedColormap(["#FFFFFF", st.INK]), interpolation="nearest")
    bounds = np.cumsum([np.sum(clade == c) for c in co])
    for b_ in bounds[:-1]:
        ax.axhline(b_ - 0.5, color=st.PAL["model"], lw=st.LW_REF)
    mids = np.r_[0, bounds[:-1]] + np.diff(np.r_[0, bounds]) / 2
    ax.set_yticks(mids); ax.set_yticklabels([c.split("|")[-1] for c in co], fontsize=st.FS_MIN)
    ax.set_xticks([]); ax.set_xlabel(f"{len(ev)} informative events present in >= 2% of cells (by frequency)", fontsize=st.FS_AXLAB)
    for s in ax.spines.values():
        s.set_visible(False)
    sh.text(0.05, 3.15, f"{fam}: informative-edit matrix ({len(ids)} cells, rows by clade)", fontsize=st.FS_TITLE, va="top")
    sh.save(out, f"g5_rep_{fam}_edits")


if __name__ == "__main__":
    main()

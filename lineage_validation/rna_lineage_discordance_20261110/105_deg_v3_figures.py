"""PLAN_v3_deg.md figures from deg_v3/summary (104_). One panel per file, all 50 hallmark sets, every figure says how
its groups were chosen. Writes ROOT/figures/deg_v3/<name>.{png,pdf}.

  F1_nes_heatmap_{raw,ds}         mouse-mean NES, set x (contrast type x mouse)
  F2_across_mice_<type>           NES per mouse, raw (filled) and thinned (open); 3724 in red
  F3_within_clade_heatmap         the PLAN block (largest A1 contrast among the focus mice), top 20 up / 20 down genes
  F4_baseline_spearman            per A contrast: Spearman of its 50 NES with its RNA-proximity baseline
  F5_random_null_<type>           share of contrasts outside the random-split NES range, per set (A1 main, A2)
  F6_bc_null_<type>               B1 / C1: share of contrasts with |NES| outside the random |NES| range, raw and thinned
  F7_depth_ratio                  case / reference median depth per contrast, by type

    python lineage_validation/rna_lineage_discordance_20261110/105_deg_v3_figures.py --root results/rna_lineage_discordance_20261110
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402

TYPES = [("A1_rna_vs_crispr", "RNA-OT- vs CRISPR-OT-preferred", "OT difference"),
         ("A1_rna_vs_both", "RNA-OT-preferred vs both", "OT difference"),
         ("A1_crispr_vs_both", "CRISPR-OT-preferred vs both", "OT difference"),
         ("A1_baseline", "RNA-closest vs RNA-farthest (baseline)", "RNA similarity"),
         ("A2_cross_vs_own", "RNA-cross vs RNA-own (targets)", "OT (RNA-OT column mass)"),
         ("A2_baseline", "closer to other vs own clade (baseline)", "RNA similarity")]
LAB = {t: l for t, l, _ in TYPES}; BY = {t: b for t, _, b in TYPES}
C_RNA, C_CR, C_BOTH = "#2C6E9B", "#D98C2B", "#A6A6A6"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def nice(t):
    return t.replace("HALLMARK_", "").replace("_", " ").lower()


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=400 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    S = a.root / "deg_v3" / "summary"; out = a.root / "figures" / "deg_v3"; out.mkdir(parents=True, exist_ok=True)
    st.apply()
    rd = lambda f: pd.read_csv(S / f, sep="\t", dtype={"mouse": str})  # noqa: E731
    MN, CO, L, QC, BP, BC = rd("mouse_nes.tsv"), rd("consistency.tsv"), rd("nes_long.tsv"), rd("contrast_qc.tsv"), rd("baseline_pairs.tsv"), rd("bc_null.tsv")
    terms = sorted(L.term.unique()); tix = {t: i for i, t in enumerate(terms)}
    cmap = plt.get_cmap("RdBu_r").copy(); cmap.set_bad("#EEEEEE")

    # F1 NES heatmap, mouse mean, raw and thinned
    for k in ("raw", "ds"):
        cols = [(t, m) for t, _, _ in TYPES for m in sorted(MN[MN.type == t].mouse.unique())]
        M = np.full((len(terms), len(cols)), np.nan)
        piv = MN.set_index(["type", "mouse", "term"])[f"nes_{k}"]
        for j, (t, m) in enumerate(cols):
            for term in terms:
                M[tix[term], j] = piv.get((t, m, term), np.nan)
        fig = plt.figure(figsize=(7.2, 7.6)); ax = fig.add_axes([0.30, 0.06, 0.62, 0.78])
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=-3, vmax=3, interpolation="nearest")
        ax.set_yticks(range(len(terms))); ax.set_yticklabels([nice(t) for t in terms], fontsize=st.FS_MIN)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([m for _, m in cols], rotation=90, fontsize=st.FS_MIN)
        x0 = 0
        for t, lab, by in TYPES:
            n = sum(1 for c in cols if c[0] == t)
            if n:
                ax.axvline(x0 - 0.5, color="white", lw=1.5)
                ax.text(x0 + n / 2 - 0.5, -1.2, f"{lab}\n(groups by {by})", rotation=60, ha="left", va="bottom", fontsize=st.FS_MIN)
            x0 += n
        for s_ in ax.spines.values():
            s_.set_visible(False)
        cb = fig.add_axes([0.935, 0.30, 0.012, 0.25]); fig.colorbar(im, cax=cb, ticks=[-3, 0, 3])
        cb.set_title("NES\n(mouse mean)", fontsize=st.FS_MIN, loc="left"); cb.tick_params(labelsize=st.FS_MIN)
        fig.text(0.01, 0.995, f"Hallmark NES per mouse ({'raw counts' if k == 'raw' else 'depth-thinned counts'}); "
                 "positive = higher in the first-named group; grey = not evaluable", fontsize=st.FS_TITLE, va="top")
        save(fig, out, f"F1_nes_heatmap_{k}")

    # F2 across-mouse dots per type
    for t, lab, by in TYPES:
        s = MN[MN.type == t]
        if s.empty:
            continue
        fig, ax = plt.subplots(figsize=(4.6, 7.2))
        for term in terms:
            y = len(terms) - 1 - tix[term]
            for k, face in (("raw", True), ("ds", False)):
                v = s[s.term == term]
                for r in v.itertuples():
                    x = getattr(r, f"nes_{k}")
                    if pd.isna(x):
                        continue
                    colr = st.PAL["model"] if r.mouse == "3724" else "#6B7280"
                    ax.scatter(x, y + (0.15 if k == "raw" else -0.15), s=9, facecolor=colr if face else "white", edgecolor=colr, lw=0.6, zorder=3)
        ax.axvline(0, **st.REF_KW)
        ax.set_yticks(range(len(terms))); ax.set_yticklabels([nice(x) for x in terms[::-1]], fontsize=st.FS_MIN)
        ax.set_ylim(-0.7, len(terms) - 0.3); ax.set_xlabel("NES (mouse mean)", fontsize=st.FS_AXLAB); st.tidy(ax)
        c = CO[(CO.type == t) & (~CO.without_3724)].set_index("term")
        for term in terms:
            if term in c.index and c.at[term, "sign_ok"]:
                ax.text(1.01, len(terms) - 1 - tix[term], "same sign, both runs", transform=ax.get_yaxis_transform(), fontsize=st.FS_MIN, va="center",
                        color=st.SUBTLE)
        ax.set_title(f"{lab}\ngroups chosen by {by}; filled = raw, open = thinned; red = 3724", fontsize=st.FS_TITLE, loc="left")
        save(fig, out, f"F2_across_mice_{t}")

    # F3 within-clade heatmap of the PLAN block
    G = pd.read_csv(a.root / "deg_v3" / "groups.tsv", sep="\t")
    C = pd.read_csv(a.root / "deg_v3" / "contrasts.tsv", sep="\t", dtype={"mouse": str})
    m = C[(C.type == "A1_rna_vs_crispr") & C.mouse.isin(["3457", "3460", "3724", "3451"])]
    blk = m.assign(n=m.case_n + m.reference_n).sort_values("n", ascending=False).iloc[0]
    cid = blk.contrast_id; both_id = cid.replace("A1_rna_vs_crispr", "A1_rna_vs_both")
    g = G[G.contrast_id == cid].copy(); g["grp"] = g.group.map({"case": "RNA-OT-preferred", "reference": "CRISPR-OT-preferred"})
    if (G.contrast_id == both_id).any():
        gb = G[(G.contrast_id == both_id) & (G.group == "reference")].copy(); gb["grp"] = "both supported"; g = pd.concat([g, gb])
    deg = pd.read_csv(a.root / "deg_v3" / "deg_gsea_raw" / "deg" / f"{cid}.tsv.gz", sep="\t")
    genes = list(deg.sort_values("scores", ascending=False).names.head(20)) + list(deg.sort_values("scores").names.head(20))
    import anndata as ad
    import scanpy as sc
    A = ad.read_h5ad(a.root / "kp_counts_20261112" / f"KPTracer_{blk.mouse}.h5ad")
    A.obs_names = A.obs_names.astype(str)
    x = A[g.cell.astype(str).to_numpy()].copy(); sc.pp.normalize_total(x, target_sum=1e4); sc.pp.log1p(x)
    X = x[:, genes].X; X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    AC = pd.read_csv(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str}).set_index("cell")
    CA = pd.read_csv(a.root / "deg_v3" / "cells_A.tsv.gz", sep="\t")
    dmap = CA[(CA.kind == "A1") & (CA.unit_id == blk.unit) & (CA.clade == blk.clade)].drop_duplicates("cell").set_index("cell").d
    g["d"] = g.cell.map(dmap).to_numpy()
    order_g = ["RNA-OT-preferred", "both supported", "CRISPR-OT-preferred"]
    g["o"] = g.grp.map({k: i for i, k in enumerate(order_g)}); idx = np.lexsort((-g.d.fillna(0).to_numpy(), g.o.to_numpy()))
    g = g.iloc[idx]; Z = Z[idx]
    F92 = load("f92", HERE / "92_flow_figure.py")
    meta = AC.reindex(g.cell.astype(str))
    fig = plt.figure(figsize=(7.2, 5.2))
    hx = fig.add_axes([0.16, 0.08, 0.66, 0.69])
    hx.imshow(np.clip(Z.T, -2.5, 2.5), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    hx.set_yticks(range(len(genes))); hx.set_yticklabels(genes, fontsize=st.FS_MIN); hx.set_xticks([])
    hx.axhline(19.5, color="white", lw=1.2)
    hx.set_xlabel(f"{len(g)} cells, ordered by group, then by d = n x (q_RNA - q_CRISPR)", fontsize=st.FS_AXLAB)
    for s_ in hx.spines.values():
        s_.set_visible(False)
    gc = {"RNA-OT-preferred": C_RNA, "both supported": C_BOTH, "CRISPR-OT-preferred": C_CR}
    sc_ = meta.state.fillna("NA").to_numpy(); dep = np.log10(meta.depth.to_numpy(float)); ev = meta.events.to_numpy(float)
    tracks = [("group", [gc[v] for v in g.grp]), ("RNA state", [F92.STATE_COL.get(v, "#E3E6E8") for v in sc_])]
    for k_, (nm, cols) in enumerate(tracks):
        tx = fig.add_axes([0.16, 0.86 - k_ * 0.035, 0.66, 0.028]); tx.set_xlim(0, len(g)); tx.axis("off")
        tx.bar(np.arange(len(g)) + 0.5, 1, width=1.0, color=cols, lw=0)
        tx.text(-0.01, 0.5, nm, transform=tx.transAxes, ha="right", va="center", fontsize=st.FS_MIN)
    # numeric tracks (grey scale per track, labelled with their range)
    for k_, (nm, v) in enumerate((("log10 depth", dep), ("edits", ev))):
        tx = fig.add_axes([0.16, 0.80 - k_ * 0.025, 0.66, 0.018])
        tx.imshow(v[None, :], aspect="auto", cmap="Greys", interpolation="nearest"); tx.axis("off")
        tx.text(-0.01, 0.5, nm, transform=tx.transAxes, ha="right", va="center", fontsize=st.FS_MIN)
        tx.text(1.01, 0.5, f"{np.nanmin(v):.1f}-{np.nanmax(v):.1f}", transform=tx.transAxes, ha="left", va="center", fontsize=st.FS_MIN, color=st.SUBTLE)
    lg = fig.add_axes([0.88, 0.38, 0.12, 0.40]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1); y = 1.0
    for nm, colr in gc.items():
        lg.add_patch(plt.Rectangle((0, y - 0.05), 0.1, 0.045, color=colr)); lg.text(0.14, y - 0.027, nm, fontsize=st.FS_MIN, va="center"); y -= 0.07
    y -= 0.03
    for s_ in [s_ for s_ in F92.STATE_COL if s_ in set(sc_)]:
        lg.add_patch(plt.Rectangle((0, y - 0.05), 0.1, 0.045, color=F92.STATE_COL[s_])); lg.text(0.14, y - 0.027, s_, fontsize=st.FS_MIN, va="center"); y -= 0.06
    cb = fig.add_axes([0.89, 0.10, 0.012, 0.18]); fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-2.5, 2.5)), cax=cb, ticks=[-2, 0, 2])
    cb.set_title("z (log1p)", fontsize=st.FS_MIN, loc="left"); cb.tick_params(labelsize=st.FS_MIN)
    fig.text(0.01, 0.995, f"{blk.unit.split('__')[2]} clade {blk.clade.split('|')[-1]}, site {blk.site}: groups chosen by OT difference within one clade\n"
             f"top 20 genes up and 20 down, RNA-OT- vs CRISPR-OT-preferred (raw counts); depth ratio {blk.depth_ratio:.2f}",
             fontsize=st.FS_TITLE, va="top")
    save(fig, out, "F3_within_clade_heatmap")

    # F4 baseline Spearman per contrast
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    ts = [t for t in ("A1_rna_vs_crispr", "A1_rna_vs_both", "A1_crispr_vs_both", "A2_cross_vs_own") if (BP.type == t).any()]
    rng = np.random.default_rng(20261113)
    for i, t in enumerate(ts):
        v = BP[BP.type == t].nes_spearman_raw.dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=8, color=st.NEUTRAL, zorder=2)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=st.INK, lw=1.2, zorder=3)
        ax.text(i, 1.08, f"median {np.median(v):.2f}", ha="center", fontsize=st.FS_MIN)
    ax.axhline(0, **st.REF_KW); ax.set_ylim(-1.05, 1.15); ax.set_xticks(range(len(ts)))
    ax.set_xticklabels([LAB[t].replace(" vs ", "\nvs ") for t in ts], fontsize=st.FS_MIN)
    ax.set_ylabel("Spearman, 50 NES\ncontrast vs its RNA-similarity baseline", fontsize=st.FS_AXLAB); st.tidy(ax)
    ax.set_title("How much of each contrast is RNA proximity (raw counts; one dot per contrast)", fontsize=st.FS_TITLE, loc="left", pad=12)
    save(fig, out, "F4_baseline_spearman")

    # F5 random-split null per set
    for t in ("A1_rna_vs_crispr", "A2_cross_vs_own"):
        c = CO[(CO.type == t) & (~CO.without_3724)].set_index("term").reindex(terms)
        fig, ax = plt.subplots(figsize=(4.2, 7.2))
        y = np.arange(len(terms))[::-1]
        ax.barh(y, c.outside_null_share.to_numpy(), color=[C_RNA if s_ else st.NEUTRAL for s_ in c.sign_ok.fillna(False)], height=0.7)
        ax.axvline(0.5, **st.REF_KW); ax.set_xlim(0, 1); ax.set_yticks(y); ax.set_yticklabels([nice(x) for x in terms], fontsize=st.FS_MIN)
        ax.set_ylim(-0.7, len(terms) - 0.3)
        ax.set_xlabel("share of contrasts outside the random-split NES range\n(on the majority side; dashed = reading rule 0.5)", fontsize=st.FS_AXLAB)
        st.tidy(ax)
        ax.set_title(f"{LAB[t]}: groups chosen by {BY[t]}\nblue = same sign in >= 75% of mice in both runs", fontsize=st.FS_TITLE, loc="left")
        save(fig, out, f"F5_random_null_{t}")

    # F6 B1 / C1 |NES| outside the random |NES| range
    for t, lab, by in (("B1_state", "B1: state X vs state Y within one clade and site", "RNA state"),
                       ("C1_clade", "C1: clade X vs clade Y within one state and site", "CRISPR edits")):
        c = BC[(BC.type == t) & (~BC.without_3724)].set_index("term").reindex(terms)
        fig, ax = plt.subplots(figsize=(4.2, 7.2))
        y = np.arange(len(terms))[::-1]
        ax.barh(y + 0.18, c.outside_abs_null_share_raw.to_numpy(), height=0.34, color=st.INK, label="raw")
        ax.barh(y - 0.18, c.outside_abs_null_share_ds.to_numpy(), height=0.34, color=st.NEUTRAL, label="thinned")
        ax.axvline(0.5, **st.REF_KW); ax.set_xlim(0, 1); ax.set_yticks(y); ax.set_yticklabels([nice(x) for x in terms], fontsize=st.FS_MIN)
        ax.set_ylim(-0.7, len(terms) - 0.3); ax.legend(frameon=False, fontsize=st.FS_MIN, loc="lower right")
        ax.set_xlabel("share of contrasts with |NES| above the random-split |NES| range", fontsize=st.FS_AXLAB); st.tidy(ax)
        ax.set_title(f"{lab}\ngroups chosen by {by}; {int(c.contrasts.max())} contrasts", fontsize=st.FS_TITLE, loc="left")
        save(fig, out, f"F6_bc_null_{t}")

    # F7 depth ratio per contrast
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    tt = [t for t, _, _ in TYPES] + ["B1_state", "C1_clade"]
    for i, t in enumerate(tt):
        v = QC[QC.type == t].depth_ratio.dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=7, color=st.NEUTRAL, zorder=2)
        if len(v):
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=st.INK, lw=1.2, zorder=3)
    ax.axhline(1, **st.REF_KW); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_yticks([0.25, 0.5, 1, 2, 4]); ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"]); ax.set_xticks(range(len(tt)))
    ax.set_xticklabels([LAB.get(t, t.replace("_", " ")).replace(" vs ", "\nvs ") for t in tt], fontsize=st.FS_MIN, rotation=45, ha="right")
    ax.set_ylabel("median depth, first group / second", fontsize=st.FS_AXLAB); st.tidy(ax)
    ax.set_title("Sequencing-depth imbalance of each contrast (raw counts; line = median)", fontsize=st.FS_TITLE, loc="left")
    save(fig, out, "F7_depth_ratio")
    print("figures ->", out, "; F3 block", cid)


if __name__ == "__main__":
    main()

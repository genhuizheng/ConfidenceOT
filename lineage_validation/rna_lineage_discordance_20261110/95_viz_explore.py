"""Exploratory visualisation round (user brief 2026-11-12): candidate designs on real data, no refits, no new thresholds.

    python lineage_validation/rna_lineage_discordance_20261110/95_viz_explore.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs

OT difference (OT-unit clades of state_lineage/, the flow tables of 92_flow_figure.py):
  B_heatmap_<unit>     source clade x target clade: F_RNA - F_CRISPR (diverging) beside Shared (sequential), with the
                       clade tree, state composition of each clade and the site
  C_bars_<unit>        tree + per-source-clade horizontal bars: Shared / RNA-only / CRISPR-only outgoing mass, split
                       into same-clade and cross-clade, with state composition and the side
  V_levels_<unit>      per target clade: clade-level Shared fraction vs exact cell-level top-15 source overlap
  V_baseline           per unit: clade-level Shared(RNA, CRISPR) against Shared(RNA, independent) and
                       Shared(CRISPR, independent), independent = product of the common-support clade shares
Biology (all eligible KP cells, original annotation: state_lineage_ext/all_cells__original/):
  BA_lineage_states_<family>  clade tree | edit-frequency heatmap (informative events in >= 5% of the family) | state
                       composition | primary / metastasis composition, one row per clade
  BB_state_families_<mouse>   per state: its cells split by family (independent tumours) and clade, with the
                       within-site null's expected largest clade share as a tick
  BC_sites_<mouse>     per site (primary first): clade x state mosaic (width = clade cells, fill = states); sites
                       with < 2 clades or < 2 states (>= 5 cells) are labelled "not identifiable", never "no association"
  BD_discordance_<unit>  RNA UMAP coloured by clade, with view-C divergent / convergent cells outlined
Case audit: explore/case_audit.tsv (events, clades, states, sites, depth and clone size per clade, comparability).
"""

from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
sys.path.insert(0, str(REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"))
import matrix_core as mc  # noqa: E402
import nbt_style as st  # noqa: E402
sys.path.insert(0, str(HERE))
import lineage_tree as LT  # noqa: E402
G = {}   # AC (all-cell clades) and the tree cache, set in main

SEED = 20261111
SPECIAL = ("minor", "Other", "unassigned")
FOCUS = ["Pre-EMT", "Early EMT-1", "Early EMT-2", "Mesenchymal-1", "Mesenchymal-1 (Met)", "Mesenchymal-2", "Mesenchymal-2 (Met)"]
FAM_COL = ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#8C7A5B", "#C0392B", "#5E8FA8"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


F92 = load("f92", HERE / "92_flow_figure.py")
STATE_COL, C_SHARED, C_RNA, C_CR = F92.STATE_COL, F92.C_SHARED, F92.C_RNA, F92.C_CR


def nice(c):
    return c.split("|")[-1] if "|" in c else c


def clade_tree(fam, clades, cells_of, t, prior):
    """The family's shared tree (lineage_tree.py), pruned to `clades`. Returns (ordered clades, handle or None)."""
    tree = LT.family_tree(fam, G["AC"][G["AC"].family == fam][["cell", "clade"]], t, prior, mc, G["cache"])
    tord, pr = LT.pruned_order(tree, [c for c in clades if c not in SPECIAL])
    if pr is None or pr[0] == "leaf":
        return tord, None
    return tord, (pr, tree["names"], tord)


def draw_tree(ax, Z, ycent, x_left, x_right):
    """Pruned family tree, leaf i of the clade order at ycent[i] (rows run downward), scale bar above the top leaf."""
    pr, names, tord = Z
    LT.draw_pruned(ax, pr, names, {c: ycent[i] for i, c in enumerate(tord)}, x_left, x_right, y_bar=min(ycent) - 0.75,
                   color=st.INK, fontsize=st.FS_MIN)
    for ln in ax.lines:
        ln.set_clip_on(False)


def stacked_h(ax, y, h, comp, x0=0.0, w=1.0, cmap=None):
    x = x0
    for k_, fr in comp.items():
        ax.add_patch(Rectangle((x, y - h / 2), fr * w, h, facecolor=(cmap or STATE_COL).get(k_, "#E3E6E8"), edgecolor="none")); x += fr * w


def save(fig, out, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)


def state_legend(fig, states, y, x0=0.02, h=0.06):
    lx = fig.add_axes([0, y, 1, h]); lx.axis("off"); lx.set_xlim(0, 10); lx.set_ylim(0, 2)
    x, row = x0 * 10, 1.5
    for s_ in [s for s in STATE_COL if s in set(states)]:
        w_ = 0.3 + 0.055 * len(s_)
        if x + w_ > 9.95:
            x, row = x0 * 10, row - 1
        lx.add_patch(Rectangle((x, row - 0.3), 0.13, 0.6, facecolor=STATE_COL[s_], edgecolor="none")); lx.text(x + 0.17, row, s_, fontsize=st.FS_MIN, va="center")
        x += w_


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "figures" / "explore"; out.mkdir(parents=True, exist_ok=True)
    tab = a.root / "explore"; tab.mkdir(exist_ok=True)
    st.apply()
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); t, prior = bb.load_tables(a.inputs)["KPTracer"]
    rd = lambda f: pd.read_csv(f, sep="\t", dtype={"mouse": str, "cell": str}, low_memory=False)  # noqa: E731
    G["AC"] = rd(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz"); G["cache"] = a.root / "explore" / "trees"
    ot_cl = G["AC"].set_index("cell")         # every figure uses the all-eligible-cell clades
    flow_units = rd(a.root / "flow" / "units.tsv").set_index("unit_id")
    OT_CASES = list(flow_units.index)

    # ---------- OT difference: B heatmap, C bars, V levels ----------
    for u in OT_CASES:
        fl = rd(a.root / "flow" / "flows" / f"{u}.tsv"); cl = rd(a.root / "flow" / "clades" / f"{u}.tsv")
        order = list(cl.clade); fam = next((c.split("|")[0] for c in order if "|" in c), None)
        named = [c for c in order if c not in SPECIAL]
        cells_of = {c: ot_cl.index[ot_cl.clade == c].to_numpy() for c in named}
        tord, Z = clade_tree(fam, named, cells_of, t, prior)
        order = tord + [c for c in order if c in SPECIAL]
        k = len(order)
        M = {c: fl.pivot(index="source_clade", columns="target_clade", values=c).reindex(index=order, columns=order).fillna(0)
             for c in ("F_rna", "F_crispr", "shared", "rna_enriched", "crispr_enriched")}
        b = pickle.load(open(a.inputs.parent / "bundles" / f"{u}.pkl", "rb")) if (a.inputs.parent / "bundles" / f"{u}.pkl").exists() else None
        comp = cpu = None
        cc = rd(a.root / "compare" / f"{u}__l5best" / "cells.tsv.gz")
        cc["clade"] = cc.cell.map(ot_cl.clade).fillna("unassigned"); cc["state"] = cc.cell.map(ot_cl.state).fillna("NA")
        cc["clade"] = cc.clade.where(cc.clade.isin(order), "Other")
        common = cc[cc.rna_retained & cc.crispr_retained]
        srcc, tgtc = common[common.side == "source"], common[common.side == "target"]
        # B: two heatmaps (difference, shared) + annotations
        cw = 0.32 if k <= 7 else 2.2 / k
        FH = 1.35 + k * cw
        fig = plt.figure(figsize=(7.2, FH))
        for pi, (key, title, cmap, norm) in enumerate((("diff", "F_RNA - F_CRISPR (blue = RNA-OT more, orange = CRISPR-OT more)", None, None),
                                                       ("shared", "Shared = min(F_RNA, F_CRISPR)", "Greys", None))):
            x0 = 1.75 + pi * (k * cw + 1.0)
            ax = fig.add_axes([x0 / 7.2, 0.55 / FH, k * cw / 7.2, (k * cw + 0.15) / FH])
            V = (M["F_rna"] - M["F_crispr"]).to_numpy() if key == "diff" else M["shared"].to_numpy()
            if key == "diff":
                lim = max(np.abs(V).max(), 1e-6)
                from matplotlib.colors import LinearSegmentedColormap
                cm_ = LinearSegmentedColormap.from_list("rc", [C_CR, "#FFFFFF", C_RNA])
                im = ax.imshow(V, cmap=cm_, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
            else:
                im = ax.imshow(V, cmap="Greys", vmin=0, vmax=max(V.max(), 1e-6), aspect="auto")
            for i in range(k):
                for j in range(k):
                    if abs(V[i, j]) >= 0.01:
                        ax.text(j, i, f"{V[i, j]:+.2f}" if key == "diff" else f"{V[i, j]:.2f}", ha="center", va="center", fontsize=st.FS_MIN,
                                color="white" if (key == "shared" and V[i, j] > 0.6 * V.max()) else st.INK)
            ax.set_xticks(range(k)); ax.set_xticklabels([nice(c) for c in order], fontsize=st.FS_MIN, rotation=60, ha="right", rotation_mode="anchor")
            ax.set_yticks([])
            if pi == 0:
                for i, c in enumerate(order):
                    ax.text(-0.6, i, nice(c), ha="right", va="center", fontsize=st.FS_MIN)
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_title(title, fontsize=st.FS_MIN, pad=4)
            ax.set_xlabel("target clade", fontsize=st.FS_MIN)
            # target state composition strip above columns
            for j, c in enumerate(order):
                comp = tgtc[tgtc.clade == c].state.value_counts(normalize=True)
                y = 0
                for s_, fr in comp.items():
                    ax.add_patch(Rectangle((j - 0.45, -0.62 - y - fr * 0.3 + 0.3 - 0.3), 0.9, fr * 0.3, facecolor=STATE_COL.get(s_, "#E3E6E8"), clip_on=False))
                    y += fr * 0.3
            ax.set_ylim(k - 0.5, -0.95)
            if pi == 0:   # source state composition strip left of rows, tree further left
                for i, c in enumerate(order):
                    comp = srcc[srcc.clade == c].state.value_counts(normalize=True)
                    x = 0
                    for s_, fr in comp.items():
                        ax.add_patch(Rectangle((-2.6 + x, i - 0.4), fr * 0.8, 0.8, facecolor=STATE_COL.get(s_, "#E3E6E8"), clip_on=False)); x += fr * 0.8
                ax.set_xlim(-0.5, k - 0.5)
                if Z is not None:
                    tx = fig.add_axes([0.01, ax.get_position().y0, 0.07, ax.get_position().height]); tx.axis("off")
                    tx.set_ylim(k - 0.5, -0.95); tx.set_xlim(0, 1)
                    draw_tree(tx, Z, list(range(len(tord))), 0.0, 1.0)
            cb = fig.colorbar(im, cax=fig.add_axes([(x0 + k * cw + 0.08) / 7.2, 0.55 / FH, 0.08 / 7.2, min(0.9, k * cw) / FH]))
            cb.ax.tick_params(labelsize=st.FS_MIN); cb.outline.set_visible(False)
        tsites = sorted(pd.Series(b["target_site"]).unique()) if b else []
        fig.text(0.01, 0.985, f"{u.split('__')[2]} (primary) -> {', '.join(s.split('_')[-1] for s in tsites)} (metastasis): clade-level coupling, "
                 f"common support {flow_units.loc[u, 'common_sources']} x {flow_units.loc[u, 'common_targets']} cells", fontsize=st.FS_TITLE, va="top")
        fig.text(0.01, 0.025, "Strips: RNA-state composition of each clade (source rows left, target columns top). Each coupling normalised to 1 on "
                 "the common support. Clade-level mass, not cell pairs.", fontsize=st.FS_MIN, color=st.SUBTLE)
        save(fig, out, f"B_heatmap_{u}")
        # C: tree + overlap bars per source clade
        fig = plt.figure(figsize=(7.2, 1.15 + 0.38 * k))
        H = 1.15 + 0.38 * k
        ax = fig.add_axes([2.3 / 7.2, 0.55 / H, 3.4 / 7.2, (0.38 * k) / H])
        st.tidy(ax)
        rows = []
        for i, c in enumerate(order):
            same = c if c not in SPECIAL else None
            rec = {}
            for kind in ("shared", "rna_enriched", "crispr_enriched"):
                rowv = M[kind].loc[c]
                rec[kind + "_same"] = float(rowv.get(same, 0)) if same else 0.0
                rec[kind + "_cross"] = float(rowv.sum() - rec[kind + "_same"])
            rows.append(rec)
        segs = [("shared_same", C_SHARED, None), ("shared_cross", C_SHARED, "//"), ("crispr_enriched_same", C_CR, None),
                ("crispr_enriched_cross", C_CR, "//"), ("rna_enriched_same", C_RNA, None), ("rna_enriched_cross", C_RNA, "//")]
        for i, rec in enumerate(rows):
            x = 0
            for key, colr, hatch in segs:
                v = rec[key]
                ax.barh(i, v, left=x, color=colr, height=0.62, edgecolor="white" if hatch is None else "white", hatch=hatch, lw=0.4, alpha=0.9); x += v
        ax.set_yticks(range(k)); ax.set_yticklabels([nice(c) for c in order], fontsize=st.FS_TICK); ax.set_ylim(k - 0.5, -0.5)
        ax.set_xlabel("outgoing coupling mass (union of both OTs, share of total)", fontsize=st.FS_AXLAB)
        # state composition column and side
        sx = fig.add_axes([1.45 / 7.2, 0.55 / H, 0.5 / 7.2, (0.38 * k) / H]); sx.axis("off"); sx.set_xlim(0, 1); sx.set_ylim(k - 0.5, -0.5)
        for i, c in enumerate(order):
            stacked_h(sx, i, 0.62, srcc[srcc.clade == c].state.value_counts(normalize=True))
        sx.set_title("source\nstates", fontsize=st.FS_MIN)
        tx2 = fig.add_axes([5.85 / 7.2, 0.55 / H, 0.5 / 7.2, (0.38 * k) / H]); tx2.axis("off"); tx2.set_xlim(0, 1); tx2.set_ylim(k - 0.5, -0.5)
        for i, c in enumerate(order):
            stacked_h(tx2, i, 0.62, tgtc[tgtc.clade == c].state.value_counts(normalize=True))
        tx2.set_title("target\nstates", fontsize=st.FS_MIN)
        if Z is not None:
            tr = fig.add_axes([0.2 / 7.2, 0.55 / H, 1.1 / 7.2, (0.38 * k) / H]); tr.axis("off"); tr.set_ylim(k - 0.5, -0.5); tr.set_xlim(0, 1)
            draw_tree(tr, Z, list(range(len(tord))), 0.0, 1.0)
        lg = [Rectangle((0, 0), 1, 1, facecolor=C_SHARED), Rectangle((0, 0), 1, 1, facecolor=C_CR), Rectangle((0, 0), 1, 1, facecolor=C_RNA),
              Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=st.INK, hatch="//")]
        ax.legend(lg, ["Shared", "CRISPR-enriched", "RNA-enriched", "hatched = to another clade"], frameon=False, fontsize=st.FS_MIN,
                  loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4)
        fig.text(0.01, 1 - 0.04 / H, f"{u.split('__')[2]} -> metastasis: where each source clade's coupling goes (common support)", fontsize=st.FS_TITLE, va="top")
        save(fig, out, f"C_bars_{u}")
        # V: clade-level shared vs exact cell-level top-15 overlap per target clade
        tg = cc[(cc.side == "target") & cc.top15_overlap.notna()]
        vrows = []
        for c in order:
            col = M["shared"][c].sum() / max(M["F_rna"][c].sum(), 1e-12) if c in M["shared"].columns else np.nan
            g = tg[tg.clade == c]
            vrows.append({"unit_id": u, "target_clade": c, "cells": len(g), "clade_level_shared_of_rna": col,
                          "cell_top15_overlap_mean": g.top15_overlap.mean() if len(g) else np.nan, "cell_tv_mean": g.tv.mean() if len(g) else np.nan})
        V = pd.DataFrame(vrows); V.to_csv(tab / f"levels_{u}.tsv", sep="\t", index=False)
        fig = plt.figure(figsize=(3.0, 2.3)); ax = fig.add_axes([0.2, 0.2, 0.75, 0.62]); st.tidy(ax)
        x = np.arange(len(V)); ax.bar(x - 0.2, V.clade_level_shared_of_rna, 0.38, color=C_SHARED, label="clade-level Shared / F_RNA")
        ax.bar(x + 0.2, V.cell_top15_overlap_mean, 0.38, color=st.INK, label="cell-level top-15 overlap")
        ax.set_xticks(x); ax.set_xticklabels([f"{nice(c)}\n({n})" for c, n in zip(V.target_clade, V.cells)], fontsize=st.FS_MIN)
        ax.set_ylim(0, 1); ax.set_ylabel("share", fontsize=st.FS_AXLAB); ax.legend(frameon=False, fontsize=st.FS_MIN, loc="upper right", bbox_to_anchor=(1, 1.3))
        fig.text(0.02, 0.98, "Same targets, two levels of agreement", fontsize=st.FS_TITLE, va="top")
        save(fig, out, f"V_levels_{u}")

    # ---------- V baseline: clade-level shared vs the independent coupling ----------
    brow = []
    for u in flow_units.index:
        fl = rd(a.root / "flow" / "flows" / f"{u}.tsv")
        Fr = fl.pivot(index="source_clade", columns="target_clade", values="F_rna").fillna(0)
        Fc = fl.pivot(index="source_clade", columns="target_clade", values="F_crispr").reindex_like(Fr).fillna(0)
        cl = rd(a.root / "flow" / "clades" / f"{u}.tsv").set_index("clade")
        ps = (cl.source_cells / cl.source_cells.sum()).reindex(Fr.index).fillna(0).to_numpy()
        pt = (cl.target_cells / cl.target_cells.sum()).reindex(Fr.columns).fillna(0).to_numpy()
        Fi = np.outer(ps, pt)
        brow.append({"unit_id": u, "shared_rna_crispr": np.minimum(Fr, Fc).to_numpy().sum(), "shared_rna_independent": np.minimum(Fr.to_numpy(), Fi).sum(),
                     "shared_crispr_independent": np.minimum(Fc.to_numpy(), Fi).sum(), "named_clades": int((~Fr.index.isin(SPECIAL)).sum())})
    Bv = pd.DataFrame(brow); Bv.to_csv(tab / "baseline_independent.tsv", sep="\t", index=False)
    fig = plt.figure(figsize=(3.2, 2.6)); ax = fig.add_axes([0.18, 0.17, 0.75, 0.68]); st.tidy(ax)
    ax.scatter(Bv.shared_rna_independent, Bv.shared_rna_crispr, s=9, color=C_RNA, lw=0, label="x = Shared(RNA-OT, independent)")
    ax.scatter(Bv.shared_crispr_independent, Bv.shared_rna_crispr, s=9, color=C_CR, lw=0, label="x = Shared(CRISPR-OT, independent)")
    ax.plot([0, 1], [0, 1], **st.REF_KW); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.set_xlabel("clade-level Shared with the independent coupling", fontsize=st.FS_AXLAB); ax.set_ylabel("Shared(RNA-OT, CRISPR-OT)", fontsize=st.FS_AXLAB)
    ax.legend(frameon=False, fontsize=st.FS_MIN, loc="lower right")
    fig.text(0.02, 0.98, "Is clade-level agreement more than clade sizes alone give? (45 KP units)", fontsize=st.FS_TITLE, va="top")
    save(fig, out, "V_baseline")

    # ---------- Biology on all eligible cells ----------
    AC = rd(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz")
    S_all = rd(a.root / "state_lineage_ext" / "all_cells__original" / "states.tsv"); S_all = S_all[S_all.scope == "sensitivity_all"]
    ident = rd(a.root / "state_lineage_ext" / "all_cells__original" / "site_identifiability.tsv"); ident = ident[ident.scope == "sensitivity_all"]
    res = rd(a.root / "state_lineage_ext" / "all_cells__original" / "resolution.tsv").set_index("lineage")
    audit = []
    fam_named = AC[~AC.clade.isin(SPECIAL)].groupby("family").clade.nunique()
    for fam in sorted(fam_named[fam_named >= 2].index):
        g = AC[AC.family == fam]
        named = sorted(set(g.clade) - set(SPECIAL))
        cells_of = {c: g.cell[g.clade == c].to_numpy() for c in named}
        tord, Z = clade_tree(fam, named, cells_of, t, prior)
        order = tord + (["minor"] if (g.clade == "minor").any() else [])
        k = len(order)
        L = mc.Lineage(t[(t.lineage == fam) & t.cell.isin(set(g.cell))], g.cell.to_numpy(), np.repeat("source", len(g)), prior)
        I = L.incidence.toarray() > 0; freq = I.mean(0); ev = np.flatnonzero(freq >= 0.05); ev = ev[np.argsort(-freq[ev])][:40]
        lab = g.clade.to_numpy()
        E = np.array([I[lab == c][:, ev].mean(0) for c in order])
        # order events by the clade where they peak
        ev_order = np.lexsort((-E.max(0), E.argmax(0))); E = E[:, ev_order]
        H = 1.0 + 0.3 * k; fig = plt.figure(figsize=(7.2, H))
        y0, hh = 0.42 / H, 0.3 * k / H
        if Z is not None:
            tr = fig.add_axes([0.01, y0, 0.08, hh]); tr.axis("off"); tr.set_ylim(k - 0.5, -0.5); tr.set_xlim(0, 1)
            draw_tree(tr, Z, list(range(len(tord))), 0.0, 1.0)
        lx = fig.add_axes([0.095, y0, 0.06, hh]); lx.axis("off"); lx.set_ylim(k - 0.5, -0.5); lx.set_xlim(0, 1)
        for i, c in enumerate(order):
            lx.text(0, i, f"{nice(c)} ({int((lab == c).sum())})", fontsize=st.FS_MIN, va="center")
        ex = fig.add_axes([0.165, y0, 0.36, hh])
        ex.imshow(E, cmap="Greys", vmin=0, vmax=1, aspect="auto", interpolation="nearest"); ex.set_xticks([]); ex.set_yticks([])
        for s_ in ex.spines.values():
            s_.set_visible(False)
        ex.set_title(f"CRISPR edits: frequency of {len(ev)} informative events in each clade", fontsize=st.FS_MIN)
        sx = fig.add_axes([0.55, y0, 0.25, hh]); sx.axis("off"); sx.set_ylim(k - 0.5, -0.5); sx.set_xlim(0, 1)
        for i, c in enumerate(order):
            stacked_h(sx, i, 0.7, g[g.clade == c].state.value_counts(normalize=True))
        sx.set_title("RNA states (share of the clade's cells)", fontsize=st.FS_MIN)
        px = fig.add_axes([0.82, y0, 0.12, hh]); px.axis("off"); px.set_ylim(k - 0.5, -0.5); px.set_xlim(0, 1)
        for i, c in enumerate(order):
            pr = (g[g.clade == c].primary).mean()
            stacked_h(px, i, 0.7, pd.Series({"primary": pr, "metastasis": 1 - pr}), cmap={"primary": "#5F666B", "metastasis": "#D3D7DA"})
        px.set_title("primary | metastasis", fontsize=st.FS_MIN)
        rr = res.loc[fam]
        fig.text(0.01, 1 - 0.06 / H, f"{fam}: one family, its clades, their edits and their RNA states  (median events/cell {rr.median_events_per_cell:.0f}, "
                 f"distinct edit sets {rr.distinct_sets_share:.2f}, {'resolved' if rr.resolved else 'NOT resolved by the frozen rule'})",
                 fontsize=st.FS_TITLE, va="top")
        state_legend(fig, g.state, 0.0, h=0.32 / H)
        save(fig, out, f"BA_lineage_states_{fam}")
        for c in order:
            h = g[g.clade == c]
            audit.append({"family": fam, "clade": c, "cells": len(h), "median_events": h.events.median(), "median_missing": h.missing_share.median(),
                          "median_depth": h.depth.median(), "primary_share": h.primary.mean(), "sites": h.site.nunique(),
                          "top_state": h.state.value_counts().index[0], "top_state_share": h.state.value_counts(normalize=True).iloc[0],
                          "family_resolved": bool(rr.resolved)})
    pd.DataFrame(audit).to_csv(tab / "case_audit.tsv", sep="\t", index=False)
    for mouse in sorted(AC.mouse.unique()):
        g = AC[(AC.mouse == mouse)]
        g = g[g.clade != "minor"]
        stv = g.state.value_counts(); states = list(stv[stv >= 50].index)
        if not states:
            continue
        fams = sorted(g.family.unique()); fcol = {f: FAM_COL[i % len(FAM_COL)] for i, f in enumerate(fams)}
        # BB: per state, cells by family and clade
        fig = plt.figure(figsize=(7.2, 1.0 + 0.24 * len(states))); H = 1.0 + 0.24 * len(states)
        ax = fig.add_axes([1.45 / 7.2, 0.55 / H, 4.3 / 7.2, 0.24 * len(states) / H]); ax.set_xlim(0, 1); ax.set_ylim(len(states) - 0.5, -0.5)
        for s_ in ("top", "right", "left"):
            ax.spines[s_].set_visible(False)
        ss = S_all[S_all.mouse == mouse].set_index("state")
        for i, s_ in enumerate(states):
            h = g[g.state == s_]; comp = h.groupby(["family", "clade"]).size().sort_values(ascending=False)
            comp = comp.reindex(sorted(comp.index, key=lambda x: (fams.index(x[0]), -comp[x])))
            x = 0
            for (f_, c_), n_ in comp.items():
                w_ = n_ / len(h)
                ax.add_patch(Rectangle((x, i - 0.32), w_, 0.64, facecolor=fcol[f_], edgecolor="white", lw=0.6, alpha=0.9)); x += w_
            if s_ in ss.index:
                ax.plot([ss.loc[s_, "max_share_null"]] * 2, [i - 0.42, i + 0.42], color=st.INK, lw=0.9)
                ax.text(1.02, i, f"{int(ss.loc[s_, 'families_ge5'])} fam / {int(ss.loc[s_, 'clades_ge5'])} clades  (n={len(h)})", fontsize=st.FS_MIN, va="center")
        ax.set_yticks(range(len(states))); ax.set_yticklabels(states, fontsize=st.FS_TICK); ax.set_xlabel("share of the state's cells", fontsize=st.FS_AXLAB)
        ax.tick_params(axis="y", length=0)
        hl = [Rectangle((0, 0), 1, 1, facecolor=fcol[f]) for f in fams]
        ax.legend(hl, [f"family {f}" for f in fams], frameon=False, fontsize=st.FS_MIN, ncol=min(5, len(fams)), loc="lower left", bbox_to_anchor=(0, 1.0))
        fig.text(0.01, 1 - 0.05 / H, f"KP {mouse}: which lineages carry each RNA state (segments = clades, colour = independent family; "
                 f"tick = largest clade share expected under the within-site null)", fontsize=st.FS_MIN, va="top")
        save(fig, out, f"BB_state_families_{mouse}")
        # BC: per-site clade x state mosaic
        sites = sorted(g.site.unique(), key=lambda s: (not g[g.site == s].primary.iloc[0], s))
        sites = [s for s in sites if (g.site == s).sum() >= 20]
        idt = ident[ident.mouse == mouse].set_index("site")
        fig = plt.figure(figsize=(7.2, 2.6)); ax = fig.add_axes([0.06, 0.24, 0.92, 0.58]); ax.axis("off")
        n_s = len(sites); gapw = 0.04; wsite = (1 - gapw * (n_s - 1)) / n_s
        allcl = sorted(set(g.clade))
        for si, s_ in enumerate(sites):
            h = g[g.site == s_]; x0 = si * (wsite + gapw)
            cl_ = h.clade.value_counts(); cl_ = cl_[[c for c in sorted(cl_.index, key=lambda c: -cl_[c])]]
            x = x0
            for c_, n_ in cl_.items():
                w_ = n_ / len(h) * wsite
                comp = h[h.clade == c_].state.value_counts(normalize=True); y = 0
                for st_, fr in comp.items():
                    ax.add_patch(Rectangle((x, y), w_, fr, facecolor=STATE_COL.get(st_, "#E3E6E8"), edgecolor="none")); y += fr
                ax.add_patch(Rectangle((x, 0), w_, 1, facecolor="none", edgecolor="white", lw=0.8))
                if w_ > 0.035:
                    ax.text(x + w_ / 2, -0.04, nice(c_), ha="center", va="top", fontsize=st.FS_MIN, rotation=90)
                x += w_
            idf = bool(idt.identifiable.get(s_, False)) if s_ in idt.index else False
            kind = "primary" if h.primary.iloc[0] else "metastasis"
            ax.text(x0 + wsite / 2, 1.04, f"{s_.split('_')[-1]} ({kind[:3]}.)\nn={len(h)}", ha="center", va="bottom", fontsize=st.FS_MIN)
            if not idf:
                ax.text(x0 + wsite / 2, 0.5, "not\nidentifiable", ha="center", va="center", fontsize=st.FS_MIN, color=st.INK,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.25, 1.25)
        fig.text(0.01, 0.985, f"KP {mouse}: within each site, clades (column width = cells) and their RNA states (fill)", fontsize=st.FS_TITLE, va="top")
        fig.text(0.01, 0.02, "not identifiable = fewer than 2 clades or 2 states (>= 5 cells) in the site: the null cannot test it; "
                 "it is not evidence of no association", fontsize=st.FS_MIN, color=st.SUBTLE)
        state_legend(fig, g.state, 0.065)
        save(fig, out, f"BC_sites_{mouse}")
    # BD: discordant cells on the RNA UMAP of one family
    import umap
    bd = []
    for mouse, gm in flow_units.reset_index().groupby("mouse"):
        case = gm[gm.unit_id.str.startswith("L23")]
        pick = case if len(case) else gm
        pick = pick.assign(_n=np.minimum(pick.common_sources, pick.common_targets)).sort_values("_n", ascending=False)
        for u_ in pick.unit_id.head(2 if len(case) > 1 else 1):
            bd.append((u_, u_.split("__")[2]))
    C_all = rd(a.root / "summary" / "cells_all.tsv.gz")
    for u, fam in bd:
        z = np.load(a.root / "inputs_A" / "rna_lineage_discordance_20261110" / "rna_coords" / u / "ranknm256-midrank__none__scale__cos__pca50.npz", allow_pickle=True)
        ids = np.r_[z["source_ids"].astype(str), z["target_ids"].astype(str)]; X = np.vstack([z["X"], z["Y"]])
        C = C_all[(C_all.unit_id == u) & (C_all.tag == "l5best")].drop_duplicates("cell").set_index("cell").reindex(ids)
        rng = np.random.default_rng(SEED); sub = np.arange(len(ids)) if len(ids) <= 6000 else np.sort(rng.choice(len(ids), 6000, replace=False))
        E2 = umap.UMAP(n_neighbors=15, min_dist=0.3, metric="cosine", random_state=SEED).fit_transform(X[sub])
        cl = pd.Series(ids[sub]).map(ot_cl.clade).fillna("unassigned").to_numpy()
        dv = C.divergent.to_numpy()[sub].astype(bool); cv = C.convergent.to_numpy()[sub].astype(bool)
        fig = plt.figure(figsize=(7.2, 2.9))
        for pi, (mask, title) in enumerate(((dv, "divergent cells (lineage-close neighbours are RNA-far)"), (cv, "convergent cells (RNA-close neighbours are lineage-far)"))):
            ax = fig.add_axes([0.02 + pi * 0.5, 0.06, 0.34, 0.78]); ax.axis("off")
            order_c = pd.Series(cl).value_counts().index
            for ki, c in enumerate(order_c):
                m = cl == c
                ax.scatter(E2[m, 0], E2[m, 1], s=1.5, color=(FAM_COL[ki % len(FAM_COL)] if c not in SPECIAL else st.NEUTRAL), lw=0, alpha=0.5,
                           label=nice(c) if pi == 0 else None, rasterized=True)
            ax.scatter(E2[mask, 0], E2[mask, 1], s=5, facecolor="none", edgecolor=st.INK, lw=0.4, rasterized=True)
            ax.set_title(f"{title}: {int(mask.sum())} of {len(mask)}", fontsize=st.FS_MIN)
            if pi == 0:
                ax.legend(frameon=False, fontsize=st.FS_MIN, markerscale=4, loc="center left", bbox_to_anchor=(1.0, 0.5), title="clade", title_fontsize=st.FS_MIN)
        fig.text(0.01, 0.985, f"{fam}: RNA UMAP (L5-best coordinates) coloured by CRISPR clade; outlined = view-C flags (top 10% within lineage)",
                 fontsize=st.FS_TITLE, va="top")
        save(fig, out, f"BD_discordance_{u}")
    # E: across mice, state concentration beyond the within-site null (largest clade share, observed - null)
    S2 = S_all.copy(); S2["excess"] = S2.max_share - S2.max_share_null
    for focus_only in (True, False):
        d = S2[S2.state.isin(FOCUS)] if focus_only else S2
        mice = sorted(d.mouse.unique()); sts = [x for x in (FOCUS if focus_only else sorted(d.state.unique())) if x in set(d.state)]
        fig = plt.figure(figsize=(7.2, 0.9 + 0.22 * len(sts))); H = 0.9 + 0.22 * len(sts)
        ax = fig.add_axes([1.5 / 7.2, 0.55 / H, 5.2 / 7.2, 0.22 * len(sts) / H])
        piv = d.pivot_table(index="state", columns="mouse", values="excess").reindex(index=sts, columns=mice)
        nn = d.pivot_table(index="state", columns="mouse", values="cells").reindex(index=sts, columns=mice)
        lim = float(np.nanmax(np.abs(piv.to_numpy()))) or 0.1
        from matplotlib.colors import LinearSegmentedColormap
        cmx = LinearSegmentedColormap.from_list("conc", ["#5E8FA8", "#FFFFFF", "#A23B5B"])
        for i, s_ in enumerate(sts):
            for j, m in enumerate(mice):
                v = piv.iat[i, j]
                if np.isfinite(v):
                    ax.scatter(j, i, s=8 + 40 * np.sqrt(nn.iat[i, j] / np.nanmax(nn.to_numpy())), c=[v], cmap=cmx, vmin=-lim, vmax=lim, edgecolor=st.INK, lw=0.3)
        ax.set_xticks(range(len(mice))); ax.set_xticklabels(mice, rotation=90, fontsize=st.FS_MIN)
        ax.set_yticks(range(len(sts))); ax.set_yticklabels(sts, fontsize=st.FS_TICK); ax.set_ylim(len(sts) - 0.5, -0.5); ax.set_xlim(-0.6, len(mice) - 0.4)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.tick_params(length=0)
        sm = plt.cm.ScalarMappable(cmap=cmx, norm=plt.Normalize(-lim, lim))
        cb = fig.colorbar(sm, cax=fig.add_axes([6.85 / 7.2, 0.55 / H, 0.08 / 7.2, min(0.9, 0.22 * len(sts)) / H])); cb.ax.tick_params(labelsize=st.FS_MIN)
        cb.set_label("largest clade share - null", fontsize=st.FS_MIN); cb.outline.set_visible(False)
        fig.text(0.01, 1 - 0.04 / H, "Per mouse and state (>= 50 cells, all eligible KP cells): how much more one clade dominates the state than the "
                 "within-site null expects (red = lineage-concentrated, blue = spread over clades; dot area = cells)", fontsize=st.FS_MIN, va="top")
        save(fig, out, "E_state_concentration_" + ("focus" if focus_only else "all"))
    print("done")


if __name__ == "__main__":
    main()

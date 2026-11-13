"""Tree-annotated OT difference flow: RNA-OT vs CRISPR-OT on one Source -> Target unit, at clade level (no refits).

    python lineage_validation/rna_lineage_discordance_20261110/92_flow_figure.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles \
        [--tag l5best] [--figures UNIT ...]

Every KP unit with both couplings (M4-E):
  common support   sources retained by both gates x targets retained by both gates
  F_RNA, F_CRISPR  each coupling restricted to the common support and normalised to total mass 1, then summed into
                   source-clade x target-clade blocks (clades: state_lineage/cells.tsv.gz, the frozen A2 clades of the
                   original annotation; "minor" = pooled clusters < 20 cells; cells without a clade = "unassigned")
  Other            a clade holding < 2% of the common-support cells on both sides is merged into "Other" (rule fixed
                   here, reported per unit in clades.tsv)
  per block        Shared = min(F_RNA, F_CRISPR), RNA-enriched = max(F_RNA - F_CRISPR, 0),
                   CRISPR-enriched = max(F_CRISPR - F_RNA, 0)
  bars             RNA-OT: Shared / RNA-enriched over sum F_RNA (= 1); CRISPR-OT: Shared / CRISPR-enriched over
                   sum F_CRISPR (= 1); overall: Shared / RNA-enriched / CRISPR-enriched over sum max(F_RNA, F_CRISPR).
                   Because both F sum to 1, RNA-enriched total = CRISPR-enriched total, so bars 1 and 2 carry the
                   same split by construction
  gates            per side, cell fractions: both retained, RNA-only, CRISPR-only, both rejected
  clades           --clades (default: the all-eligible-cell A2 clades, state_lineage_ext/all_cells__original), so clade
                   ids match every other figure
  tree             lineage_tree.py: one cached tree per family (average linkage on the mean 1 - s between clades), drawn
                   pruned to the clades of the figure; a summary of the edits, not the authors' Cassiopeia tree
Writes ROOT/flow/{units.tsv, gates.tsv, flows/<unit>.tsv, clades/<unit>.tsv} and, for --figures units (default:
every 3457 and 3724 unit), ROOT/figures/flow/<unit>.{png,pdf,svg}.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath
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

CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
OTHER_SHARE, TREE_CELLS, SEED, MIN_RIBBON = 0.02, 300, 20261111, 0.002
C_SHARED, C_RNA, C_CR = "#A6A6A6", "#2C6E9B", "#D98C2B"
DECK = False  # set by --deck
GATE_COL = {"both retained": "#5F666B", "RNA-only retained": "#9DBAD3", "CRISPR-only retained": "#EBC28F", "both rejected": "#FFFFFF"}
STATE_COL = {
    "Pre-EMT": "#8FB996", "Early EMT-1": "#C9A227", "Early EMT-2": "#E3D08A", "Mesenchymal-1": "#7D3C98",
    "Mesenchymal-1 (Met)": "#BE98CF", "Mesenchymal-2": "#A23B5B", "Mesenchymal-2 (Met)": "#DB9BAE", "Apc Early": "#4E8A6B",
    "Apc Mesenchymal-1 (Met)": "#5B4A8A", "Apc Mesenchymal-2": "#8A4E6B", "Endoderm-like": "#8C7A5B", "Gastric-like": "#9C6644",
    "Early gastric": "#CBB295", "Late Gastric": "#5E4030", "AT1-like": "#6F9FB0", "AT2-like": "#A9C8D6",
    "Lung progenitor-like": "#A3A86B", "High plasticity": "#D1495B", "Lkb1 subtype": "#6B7280", "NA": "#E3E6E8"}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def spread(y, gap, lo, hi):
    """Label positions: same top-to-bottom order as y (descending), at least `gap` apart, kept within [lo, hi]."""
    y = np.asarray(y, float); out = y.copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(out)):
            if out[i - 1] - out[i] < gap:
                d = (gap - (out[i - 1] - out[i])) / 2; out[i - 1] += d; out[i] -= d; moved = True
        out = np.clip(out, lo, hi)
        if not moved:
            break
    return out


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.75):
    """Band from [y0a, y0b] at x0 to [y1a, y1b] at x1 (cubic Bezier edges)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0a), (xm, y0a), (xm, y1a), (x1, y1a), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0a)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, lw=0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--tag", default="l5best")
    p.add_argument("--figures", nargs="*")
    p.add_argument("--clades", type=Path, help="cells.tsv.gz with cell, family, clade, state (default all_cells__original)")
    p.add_argument("--deck", action="store_true", help="slide version of the --figures units: plain labels, larger type, no gate "
                   "strips or overlap bars; writes figures/deck_20261112/flow_<unit>.png only (no tables)")
    a = p.parse_args()
    global DECK
    DECK = a.deck
    out = a.root / "flow"; (out / "flows").mkdir(parents=True, exist_ok=True); (out / "clades").mkdir(exist_ok=True)
    fig_out = a.root / "figures" / ("deck_20261112" if DECK else "flow"); fig_out.mkdir(parents=True, exist_ok=True)
    st.apply()
    if DECK:
        for k_ in [k_ for k_ in dir(st) if k_.startswith("FS_")]:
            setattr(st, k_, getattr(st, k_) * 1.45)
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); t, prior = bb.load_tables(a.inputs)["KPTracer"]
    cl = pd.read_csv(a.clades or a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t",
                     dtype={"cell": str, "mouse": str}).set_index("cell")
    cl_fam = cl.reset_index()[["cell", "family", "clade"]]
    A = a.root / "inputs_A"; cid = CONFIGS[a.tag]
    urows, grows = [], []
    for f in sorted(glob.glob(str(a.bundles / "*.pkl"))):
        b = pickle.load(open(f, "rb")); u = b["unit_id"]
        if b["dataset"] != "KPTracer" or (DECK and u not in (a.figures or [])):
            continue
        crd = a.root / "crispr_ot" / u
        row = A / "v3_grid_20261104" / "kp" / "rows" / u / (cid.split("|")[0] + "__none") / cid.replace("|", "__")
        if not (crd / "audit.json").exists() or not (row / "coupling_m4e.npz").exists():
            continue
        sid, tid = np.asarray(b["source_ids"]).astype(str), np.asarray(b["target_ids"]).astype(str)
        z = np.load(row / "coupling_m4e.npz")
        P = z["coupling"][np.ix_(pd.Index(z["source_ids"].astype(str)).get_indexer(sid), pd.Index(z["target_ids"].astype(str)).get_indexer(tid))].astype(np.float64)
        Pc = np.load(crd / "coupling_m4e.npz")["coupling"].astype(np.float64)
        cc = pd.read_csv(row / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
        cc = cc[cc.method == "M4-E"]; cc["observation_id"] = cc.observation_id.astype(str)
        gt = pd.read_csv(crd / "gates.tsv.gz", sep="\t", dtype={"observation_id": str}); gt = gt[gt.method == "M4-E"]
        gate = {}
        for side, ids in (("source", sid), ("target", tid)):
            r = ~cc[cc.side == side].set_index("observation_id").rejected.reindex(ids).to_numpy(bool)
            c = gt[gt.side == side].set_index("observation_id").retained.reindex(ids).to_numpy(bool)
            gate[side] = (r, c)
            grows.append({"unit_id": u, "side": side, "cells": len(ids), "both retained": float(np.mean(r & c)),
                          "RNA-only retained": float(np.mean(r & ~c)), "CRISPR-only retained": float(np.mean(~r & c)),
                          "both rejected": float(np.mean(~r & ~c))})
        S = np.flatnonzero(gate["source"][0] & gate["source"][1]); T = np.flatnonzero(gate["target"][0] & gate["target"][1])
        if len(S) == 0 or len(T) == 0:
            continue
        lab_s = cl.clade.reindex(sid[S]).fillna("unassigned").to_numpy(); lab_t = cl.clade.reindex(tid[T]).fillna("unassigned").to_numpy()
        st_s = cl.state.reindex(sid[S]).fillna("NA").to_numpy(); st_t = cl.state.reindex(tid[T]).fillna("NA").to_numpy()
        # Other rule
        fs = pd.Series(lab_s).value_counts(normalize=True); ft = pd.Series(lab_t).value_counts(normalize=True)
        allc = sorted(set(fs.index) | set(ft.index))
        small = [c for c in allc if fs.get(c, 0) < OTHER_SHARE and ft.get(c, 0) < OTHER_SHARE]
        mp = {c: ("Other" if c in small else c) for c in allc}
        lab_s = np.array([mp[c] for c in lab_s]); lab_t = np.array([mp[c] for c in lab_t])
        # the family's shared tree, pruned to the named clades of this unit
        named = sorted({c for c in set(lab_s) | set(lab_t) if c not in ("Other", "minor", "unassigned")})
        fam = named[0].split("|")[0] if named else None
        tree = LT.family_tree(fam, cl_fam[cl_fam.family == fam], t, prior, mc, a.root / "explore" / "trees") if fam else None
        order, pr = LT.pruned_order(tree, named)
        Z = (pr, tree["names"]) if tree is not None else None
        order = order + [c for c in ("minor", "Other", "unassigned") if c in set(lab_s) | set(lab_t)]
        k = len(order); ix = {c: i for i, c in enumerate(order)}
        bs, bt = np.array([ix[c] for c in lab_s]), np.array([ix[c] for c in lab_t])
        F = {}
        for name, Q in (("rna", P), ("crispr", Pc)):
            B = Q[np.ix_(S, T)]; B = B / B.sum()
            Os, Ot = np.eye(k)[bs], np.eye(k)[bt]; F[name] = Os.T @ B @ Ot
        sh_, re_, ce_ = np.minimum(F["rna"], F["crispr"]), np.maximum(F["rna"] - F["crispr"], 0), np.maximum(F["crispr"] - F["rna"], 0)
        un = np.maximum(F["rna"], F["crispr"])
        fl = pd.DataFrame([{"source_clade": order[i], "target_clade": order[j], "F_rna": F["rna"][i, j], "F_crispr": F["crispr"][i, j],
                            "shared": sh_[i, j], "rna_enriched": re_[i, j], "crispr_enriched": ce_[i, j]} for i in range(k) for j in range(k)])
        if not DECK:
            fl.to_csv(out / "flows" / f"{u}.tsv", sep="\t", index=False)
        crow = []
        for c in order:
            crow.append({"clade": c, "source_cells": int((lab_s == c).sum()), "target_cells": int((lab_t == c).sum()),
                         "merged_into_other": ", ".join(x.split("|")[-1] for x in small) if c == "Other" else ""})
        if not DECK:
            pd.DataFrame(crow).to_csv(out / "clades" / f"{u}.tsv", sep="\t", index=False)
        rec = {"unit_id": u, "mouse": b["mouse"], "common_sources": len(S), "common_targets": len(T), "clades": k,
               "shared": sh_.sum(), "rna_enriched": re_.sum(), "crispr_enriched": ce_.sum(), "union": un.sum(),
               "bar1_shared": sh_.sum() / F["rna"].sum(), "bar1_rna_enriched": re_.sum() / F["rna"].sum(),
               "bar2_shared": sh_.sum() / F["crispr"].sum(), "bar2_crispr_enriched": ce_.sum() / F["crispr"].sum(),
               "bar3_shared": sh_.sum() / un.sum(), "bar3_rna_enriched": re_.sum() / un.sum(), "bar3_crispr_enriched": ce_.sum() / un.sum(),
               "same_clade_mass_rna": float(sum(F["rna"][i, i] for i in range(k) if order[i] not in ("Other", "unassigned"))),
               "same_clade_mass_crispr": float(sum(F["crispr"][i, i] for i in range(k) if order[i] not in ("Other", "unassigned"))),
               "other_merged": len(small)}
        urows.append(rec)
        want = a.figures if a.figures is not None else None
        if (want is None and b["mouse"] in ("3457", "3724")) or (want and u in want):
            draw(fig_out, u, b, order, Z, F, sh_, re_, ce_, un, lab_s, lab_t, st_s, st_t, gate, rec, sid, tid)
        print(f"{u}: {len(S)} x {len(T)}, {k} clades, shared {rec['bar3_shared']:.2f} / RNA {rec['bar3_rna_enriched']:.2f} / CRISPR {rec['bar3_crispr_enriched']:.2f}", flush=True)
    if DECK:
        return
    pd.DataFrame(urows).to_csv(out / "units.tsv", sep="\t", index=False)
    pd.DataFrame(grows).to_csv(out / "gates.tsv", sep="\t", index=False)



def draw(fig_out, u, b, order, Z, F, sh_, re_, ce_, un, lab_s, lab_t, st_s, st_t, gate, rec, sid, tid):
    k = len(order)
    fig = plt.figure(figsize=(7.2, 4.3) if DECK else (7.2, 5.1))
    ax = fig.add_axes([0.0, 0.13, 1.0, 0.87] if DECK else [0.0, 0.29, 1.0, 0.71]); ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    KEY = (("both OTs", "extra in RNA-OT", "extra in CRISPR-OT") if DECK else ("Shared", "RNA-enriched", "CRISPR-enriched"))
    XS0, XS1, XT0, XT1 = 2.25, 2.5, 7.5, 7.75
    gap = 0.012; top = 0.79; Htot = 0.71 - gap * (k - 1)
    tot = un.sum(); scale = Htot / tot
    hs = un.sum(1) * scale; ht = un.sum(0) * scale
    ys = top - np.r_[0, np.cumsum(hs[:-1] + gap)]; yt = top - np.r_[0, np.cumsum(ht[:-1] + gap)]   # segment tops

    def col_bar(x0, x1, ytops, hh, labs, states):
        for i, c in enumerate(order):
            if hh[i] <= 0:
                continue
            m = labs == c; comp = pd.Series(states[m]).value_counts(normalize=True) if m.any() else pd.Series(dtype=float)
            y = ytops[i]
            for s_, fr in comp.items():
                ax.add_patch(Rectangle((x0, y - fr * hh[i]), x1 - x0, fr * hh[i], facecolor=STATE_COL.get(s_, "#E3E6E8"), edgecolor="none"))
                y -= fr * hh[i]
            ax.add_patch(Rectangle((x0, ytops[i] - hh[i]), x1 - x0, hh[i], facecolor="none", edgecolor=st.INK, lw=0.5))
    col_bar(XS0, XS1, ys, hs, lab_s, st_s); col_bar(XT0, XT1, yt, ht, lab_t, st_t)
    # ports: one fixed order (block i -> j; shared, RNA, CRISPR) on both sides, so the stacks match
    src_port, tgt_port = {}, {}; y = ys.copy()
    for i in range(k):
        for j in range(k):
            for kind, M in (("s", sh_), ("r", re_), ("c", ce_)):
                h = M[i, j] * scale; src_port[(i, j, kind)] = (y[i] - h, y[i]); y[i] -= h
    y = yt.copy()
    for j in range(k):
        for i in range(k):
            for kind, M in (("s", sh_), ("r", re_), ("c", ce_)):
                h = M[i, j] * scale; tgt_port[(i, j, kind)] = (y[j] - h, y[j]); y[j] -= h
    undrawn = 0.0
    for kind, colr, M in (("s", C_SHARED, sh_), ("r", C_RNA, re_), ("c", C_CR, ce_)):
        for i in range(k):
            for j in range(k):
                v = M[i, j]
                if v <= 0:
                    continue
                if v / tot < MIN_RIBBON:
                    undrawn += v; continue
                a0, a1 = src_port[(i, j, kind)]; b0, b1 = tgt_port[(i, j, kind)]
                ribbon(ax, XS1, XT0, a1, a0, b1, b0, colr, alpha=0.6 if kind == "s" else 0.85)

    def nice(c):
        return c.split("|")[-1] if "|" in c else c
    TIP = 1.25                     # tree leaf tips; source labels start here, right before the bar
    sp_ = lambda c: c in ("minor", "Other", "unassigned")  # noqa: E731
    LG = 0.03                      # minimum label spacing (axes units)
    src = [i for i in range(k) if hs[i] > 0]; tgt = [i for i in range(k) if ht[i] > 0]
    ly_s = dict(zip(src, spread([ys[i] - hs[i] / 2 for i in src], LG, 0.06, top)))
    ly_t = dict(zip(tgt, spread([yt[i] - ht[i] / 2 for i in tgt], LG, 0.06, top)))
    for i in src:
        c = order[i]; yc = ys[i] - hs[i] / 2; yl = ly_s[i]
        ax.text(TIP + 0.05, yl, f"{nice(c)} ({int((lab_s == c).sum())})" + ("\nnot in tree" if sp_(c) and Z is not None else ""), ha="left", va="center",
                fontsize=st.FS_TICK if not sp_(c) else st.FS_MIN, color=st.INK if not sp_(c) else st.SUBTLE)
        ax.plot([XS0 - 0.32, XS0 - 0.06, XS0], [yl, yl, yc], color=st.GUIDE if abs(yl - yc) < 1e-6 else "#9AA0A6", lw=0.4)
    for i in tgt:
        c = order[i]; yc = yt[i] - ht[i] / 2; yl = ly_t[i]
        ax.text(XT1 + 0.12, yl, f"{nice(c)} ({int((lab_t == c).sum())})", ha="left", va="center",
                fontsize=st.FS_TICK if not sp_(c) else st.FS_MIN, color=st.INK if not sp_(c) else st.SUBTLE)
        if abs(yl - yc) > 1e-6:
            ax.plot([XT1, XT1 + 0.04, XT1 + 0.1], [yc, yl, yl], color="#9AA0A6", lw=0.4)
    if Z is not None and Z[0] is not None and Z[0][0] != "leaf":
        y_of = {order[i]: ly_s[i] for i in src if not sp_(order[i])}
        for c in [c for c in order if not sp_(c) and c not in y_of]:      # a treed clade with no source mass: its target label row
            y_of[c] = ly_t[order.index(c)]
        LT.draw_pruned(ax, Z[0], Z[1], y_of, 0.1, TIP, y_bar=0.035, color=st.INK, fontsize=st.FS_MIN)
        ax.text(0.1, top + 0.012, "CRISPR clade tree (family-wide, pruned)", fontsize=st.FS_MIN, color=st.SUBTLE, va="bottom")
    # headers: site, side, gate fractions (cells, not coupling mass)
    tsites = sorted(pd.Series(b["target_site"]).unique())
    tsite = tsites[0] if len(tsites) == 1 else f"{len(tsites)} sites ({', '.join(s_.split('_')[-1] for s_ in tsites)})"
    for xc, title, side in ((1.9, f"Source: {u.split('__')[2]} (primary)", "source"),
                            (8.1, f"Target: {tsite} (metastasis)", "target")):
        ax.text(xc, 0.985, title, ha="center", va="top", fontsize=st.FS_TITLE)
        if DECK:
            continue
        r, c = gate[side]
        vals = [("both retained", np.mean(r & c)), ("RNA-only retained", np.mean(r & ~c)), ("CRISPR-only retained", np.mean(~r & c)),
                ("both rejected", np.mean(~r & ~c))]
        x = xc - 0.75
        for kname, v in vals:
            ax.add_patch(Rectangle((x, 0.915), v * 1.5, 0.022, facecolor=GATE_COL[kname], edgecolor=st.INK, lw=0.3)); x += v * 1.5
        ax.text(xc, 0.905, f"gates (cells): both {vals[0][1]:.0%} | RNA-only {vals[1][1]:.0%} | CRISPR-only {vals[2][1]:.0%} | neither {vals[3][1]:.0%}",
                ha="center", va="top", fontsize=st.FS_MIN, color=st.SUBTLE)
    # flow key, centre top
    x = 3.0 if DECK else 3.55
    for nm, colr in zip(KEY, (C_SHARED, C_RNA, C_CR)):
        ax.add_patch(Rectangle((x, 0.852 if not DECK else 0.90), 0.2, 0.024, facecolor=colr, edgecolor="none", alpha=0.85))
        ax.text(x + 0.26, 0.864 if not DECK else 0.912, nm, fontsize=st.FS_TICK, va="center")
        x += 0.42 + (0.105 if DECK else 0.075) * len(nm)
    ax.text(5.0, 0.83 if not DECK else 0.865, "ribbon width = matching between a source clade and a target clade (cells kept by both OTs)"
            if DECK else "flow width = clade-to-clade coupling mass on cells retained by both gates", ha="center", va="center",
            fontsize=st.FS_MIN, color=st.SUBTLE)
    if not DECK:
        note = (f"Common support: {rec['common_sources']} sources x {rec['common_targets']} targets; each coupling normalised to 1 there. "
                "Clade-level mass, not cell-to-cell matches.")
        if undrawn > 0:
            note += f" Ribbons < {MIN_RIBBON:.1%} not drawn ({undrawn / tot:.1%}; kept in the bars)."
        ax.text(5.0, 0.03, note, ha="center", fontsize=st.FS_MIN, color=st.SUBTLE)
    # state legend (bar fill = authors' RNA state of the clade's cells)
    present = [s_ for s_ in STATE_COL if s_ in set(st_s) | set(st_t)]
    lx = fig.add_axes([0.0, 0.0, 1.0, 0.11] if DECK else [0.0, 0.215, 1.0, 0.065]); lx.axis("off"); lx.set_xlim(0, 10); lx.set_ylim(0, 2)
    lx.text(0.15, 1.5, "Bar colour = RNA state:" if DECK else "Bar fill = RNA state:", fontsize=st.FS_MIN, va="center")
    x, row = (1.85 if DECK else 1.35), 1.5
    for s_ in present:
        w_ = (0.42 + 0.085 * len(s_)) if DECK else (0.32 + 0.058 * len(s_))
        if x + w_ > 9.9:
            x, row = 1.35, row - 1
        lx.add_patch(Rectangle((x, row - 0.3), 0.14, 0.6, facecolor=STATE_COL[s_], edgecolor="none"))
        lx.text(x + 0.18, row, s_, fontsize=st.FS_MIN, va="center")
        x += w_
    if DECK:
        for ext in ("png", "pdf"):
            fig.savefig(fig_out / f"flow_{u}.{ext}", dpi=300 if ext == "png" else None)
        plt.close(fig)
        return
    # three bars
    bx = fig.add_axes([0.30, 0.025, 0.50, 0.18]); bx.set_xlim(0, 1); bx.set_ylim(-0.5, 2.9); bx.axis("off")
    bx.text(0, 2.75, "Clade-level overlap of the two couplings (blocks = source clade x target clade)", fontsize=st.FS_TITLE, va="center")
    bars = [("RNA-OT\n100% = all RNA-OT mass", [(rec["bar1_shared"], C_SHARED, "Shared"), (rec["bar1_rna_enriched"], C_RNA, "RNA-enriched")]),
            ("CRISPR-OT\n100% = all CRISPR-OT mass", [(rec["bar2_shared"], C_SHARED, "Shared"), (rec["bar2_crispr_enriched"], C_CR, "CRISPR-enriched")]),
            ("Overall\n100% = sum of block max", [(rec["bar3_shared"], C_SHARED, "Shared"), (rec["bar3_rna_enriched"], C_RNA, "RNA-enriched"),
                                                   (rec["bar3_crispr_enriched"], C_CR, "CRISPR-enriched")])]
    for r_, (name, segs) in enumerate(bars):
        y = 2 - r_; x = 0; tail = []
        for v, colr, nm in segs:
            bx.barh(y, v, left=x, color=colr, height=0.6, edgecolor="white", lw=0.6)
            lab = f"{nm} {v:.0%}"
            if v >= 0.06 + 0.012 * len(lab):
                bx.text(x + v / 2, y, lab, ha="center", va="center", fontsize=st.FS_TICK, color=st.INK if colr == C_SHARED else "white")
            elif v > 0:
                tail.append(lab)
            x += v
        if tail:
            bx.text(1.01, y, "  ".join(tail), ha="left", va="center", fontsize=st.FS_MIN)
        bx.text(-0.015, y, name, ha="right", va="center", fontsize=st.FS_MIN)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fig_out / f"{u}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    main()

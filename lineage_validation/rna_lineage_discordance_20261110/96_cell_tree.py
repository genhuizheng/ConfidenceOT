"""Cell-level CRISPR tree per KP family with aligned per-cell tracks (exploratory visualisation, no new analysis).

    python lineage_validation/rna_lineage_discordance_20261110/96_cell_tree.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs [--families FAM ...] [--max-cells 1200]

Per family (all eligible cells, state_lineage_ext/all_cells__original):
  cells    up to --max-cells, drawn per clade in proportion to its size with at least min(20, clade size) per clade
           (seed 20261111); the figure states how many of how many
  tree     average linkage on 1 - s of those cells (non-evaluable pairs: the family median), the rule that defines the
           A2 clades; the dashed line marks the clade cut (0.5). Leaves are coloured by each cell's A2 clade from the
           full-family clustering, so a subsample tree that splits differently from the full one shows it
  tracks   per cell, aligned with the leaves: A2 clade, RNA state, primary / metastatic site, and the informative-edit
           matrix (events in >= 5% of the drawn cells, ordered by where their carriers sit on the tree; black =
           present, light grey = intBC missing)
Writes ROOT/figures/cell_tree/<family>.{png,pdf,svg}.
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
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
sys.path.insert(0, str(REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"))
import matrix_core as mc  # noqa: E402
import nbt_style as st  # noqa: E402

SEED, CUT = 20261111, 0.5
CLADE_COL = ["#2C6E9B", "#B4702A", "#4E8A6B", "#7D3C98", "#C0392B", "#5E8FA8", "#8C7A5B", "#A3A86B", "#D1495B", "#6F4E37",
             "#3F7F7F", "#9C6644"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def draw_cell_tree(label, gfull, lineage, t, prior, flag, out, max_cells=1200, state_col=None, site_col=None,
                   clade_title="A2 clade", note=""):
    """One figure. gfull: cells of the lineage (cell, clade, state, site, primary). lineage: allele-table lineage name.
    state_col: state -> colour. site_col: site -> colour (default primary dark / metastasis light)."""
    rng = np.random.default_rng(SEED)
    gfull = gfull.reset_index(drop=True)
    n_all = len(gfull); share = min(1.0, max_cells / n_all)
    pick = []
    for c, h in gfull.groupby("clade"):
        m = min(len(h), max(min(20, len(h)), int(round(len(h) * share))))
        pick.append(rng.choice(h.index.to_numpy(), m, replace=False))
    g = gfull.loc[np.sort(np.concatenate(pick))].reset_index(drop=True)
    L = mc.Lineage(t[(t.lineage == lineage) & t.cell.isin(set(g.cell))], g.cell.to_numpy(), np.repeat("source", len(g)), prior)
    s, ok = L.agreement(np.arange(len(g)), np.arange(len(g)))
    d = squareform(1 - np.asarray(s, dtype=np.float64), checks=False); okv = squareform(np.asarray(ok, dtype=np.uint8), checks=False).astype(bool)
    d[~okv] = np.median(d[okv]) if okv.any() else 1.0
    Z = linkage(d, "average")
    dn = dendrogram(Z, no_plot=True, orientation="left"); leaves = np.array(dn["leaves"])
    n = len(g)
    I = L.incidence.toarray() > 0; O = np.asarray(L.observed.toarray()) > 0
    freq = I.mean(0); ev = np.flatnonzero(freq >= 0.05); ev = ev[np.argsort(-freq[ev])][:50]
    rank = np.empty(len(g)); rank[leaves] = np.arange(len(g))       # column order: mean tree position of the carriers
    ev = ev[np.argsort([rank[I[:, e]].mean() for e in ev])]
    ev_intbc = L.event_intbc[ev]
    M = np.where(I[:, ev], 2, np.where(O[:, ev_intbc] if len(ev) and O.shape[1] > ev_intbc.max() else True, 0, 1))[leaves]
    named = [c for c in g.clade.value_counts().index if c != "minor"]
    ccol = {c: CLADE_COL[i % len(CLADE_COL)] for i, c in enumerate(named)}; ccol["minor"] = "#D3D7DA"
    if site_col is None:
        site_col = {}
        site_of = ["#5F666B" if pr else "#D3D7DA" for pr in g.primary.to_numpy()[leaves]]
        site_leg = [("primary", "#5F666B"), ("metastasis", "#D3D7DA")]
    else:
        site_of = [site_col.get(x, "#D3D7DA") for x in g.site.to_numpy()[leaves]]
        site_leg = [(k, v) for k, v in site_col.items() if k in set(g.site)]
    fig = plt.figure(figsize=(7.2, 4.6))
    tx = fig.add_axes([0.03, 0.08, 0.30, 0.80])
    for xs_, ds_ in zip(dn["icoord"], dn["dcoord"]):
        tx.plot(ds_, xs_, color=st.INK, lw=0.3)
    tx.axvline(CUT, color=st.PAL["model"], ls=(0, (3, 2)), lw=0.7)
    tx.text(CUT, n * 10 + 4, "clade cut 0.5", color=st.PAL["model"], fontsize=st.FS_MIN, ha="center", va="bottom")
    tx.set_ylim(0, n * 10); tx.set_xlim(max(1.0, Z[:, 2].max()) * 1.02, 0)
    tx.set_yticks([]); tx.set_xlabel("edit distance 1 - s (average linkage)", fontsize=st.FS_AXLAB)
    for s_ in ("top", "right", "left"):
        tx.spines[s_].set_visible(False)
    yl = np.arange(n) * 10 + 5
    tracks = [(clade_title, [ccol[c] for c in g.clade.to_numpy()[leaves]]),
              ("RNA state", [state_col.get(x, "#E3E6E8") for x in g.state.to_numpy()[leaves]]),
              ("site", site_of)]
    for k_, (name, cols) in enumerate(tracks):
        ax = fig.add_axes([0.335 + k_ * 0.03, 0.08, 0.026, 0.80]); ax.set_ylim(0, n * 10); ax.set_xlim(0, 1); ax.axis("off")
        ax.barh(yl, 1, height=10, color=cols, lw=0)
        ax.text(0.5, n * 10 + 4, name, rotation=60, fontsize=st.FS_MIN, ha="left", va="bottom")
    ex = fig.add_axes([0.435, 0.08, 0.37, 0.80])
    ex.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#FFFFFF", "#E6E8EA", st.INK]), vmin=0, vmax=2,
              extent=(0, max(len(ev), 1), 0, n * 10), origin="lower")
    ex.set_yticks([]); ex.set_xticks([]); ex.set_xlabel(f"{len(ev)} informative edits (>= 5% of drawn cells)", fontsize=st.FS_AXLAB)
    for s_ in ex.spines.values():
        s_.set_visible(False)
    lg = fig.add_axes([0.82, 0.08, 0.17, 0.80]); lg.axis("off"); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    y = 1.0
    lg.text(0, y, f"{clade_title} (cells drawn / all)", fontsize=st.FS_MIN, va="top"); y -= 0.035
    full = gfull.clade.value_counts()
    for c in named + (["minor"] if (g.clade == "minor").any() else []):
        lg.add_patch(plt.Rectangle((0, y - 0.022), 0.08, 0.022, color=ccol[c]))
        lg.text(0.11, y - 0.011, f"{c.split('|')[-1]} ({(g.clade == c).sum()}/{full.get(c, 0)})", fontsize=st.FS_MIN, va="center"); y -= 0.03
        if y < 0.5:
            break
    y -= 0.02; lg.text(0, y, "RNA state", fontsize=st.FS_MIN, va="top"); y -= 0.035
    for x in [x for x in state_col if x in set(g.state)]:
        lg.add_patch(plt.Rectangle((0, y - 0.022), 0.08, 0.022, color=state_col[x])); lg.text(0.11, y - 0.011, x, fontsize=st.FS_MIN, va="center"); y -= 0.028
    y -= 0.015
    for lab, colr in site_leg + [("edit present", st.INK), ("intBC missing", "#E6E8EA")]:
        lg.add_patch(plt.Rectangle((0, y - 0.022), 0.08, 0.022, color=colr, ec=st.INK, lw=0.2)); lg.text(0.11, y - 0.011, lab, fontsize=st.FS_MIN, va="center"); y -= 0.028
    fig.text(0.01, 0.985, f"{label}: cell-level CRISPR tree with each cell's clade, RNA state and site\n{n} of {n_all} cells drawn; {flag}{note}",
             fontsize=st.FS_TITLE, va="top")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"{label}.{ext}", dpi=400 if ext == "png" else None)
    plt.close(fig)
    print(f"{label}: {n} of {n_all} cells, {len(named)} clades, {len(ev)} edits", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--families", nargs="*")
    p.add_argument("--max-cells", type=int, default=1200)
    a = p.parse_args()
    out = a.root / "figures" / "cell_tree"; out.mkdir(parents=True, exist_ok=True)
    st.apply()
    STATE_COL = load("f92", HERE / "92_flow_figure.py").STATE_COL
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); t, prior = bb.load_tables(a.inputs)["KPTracer"]
    AC = pd.read_csv(a.root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str})
    res = pd.read_csv(a.root / "state_lineage_ext" / "all_cells__original" / "resolution.tsv", sep="\t").set_index("lineage")
    named_n = AC[AC.clade != "minor"].groupby("family").clade.nunique()
    for fam in (a.families or sorted(named_n[named_n >= 2].index)):
        rr = res.loc[fam] if fam in res.index else None
        flag = "" if rr is None else (f"median edits/cell {rr.median_events_per_cell:.0f}, distinct edit sets {rr.distinct_sets_share:.2f}, "
                                      + ("resolved" if rr.resolved else "NOT resolved by the frozen rule"))
        draw_cell_tree(fam, AC[AC.family == fam], fam, t, prior, flag, out, a.max_cells, STATE_COL)


if __name__ == "__main__":
    main()

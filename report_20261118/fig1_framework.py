"""Figure 1 - ConfidenceOT: why and how.

a-c are schematics (tagged). d is a real result: one N = 5,000 Splatter unit of the v3 grid (L2 depth + dropout,
replicate 2, Case 2 population lost, main configuration, M4-E), extracted on TACC by tacc_fig1_example.py into
out/Figure1/data/example/{cells.tsv, group_mass.tsv, meta.json}.
The workflow in c follows realdata_benchmark/v3_grid/v3_common.py (representation -> correction -> fit) and
cancer_metastasis/02_run_pair.py / src/confidenceot (UOT backbone, M4-E gate, within-side split null).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402
import glyphs as g  # noqa: E402

OUT = fs.OUT / "Figure1"
EX = OUT / "data" / "example"
POPC = {"A": fs.POP[0], "B": fs.POP[1], "C": fs.POP[2], "D": fs.POP[3]}


def cloud(ax, cx, cy, n, sx, sy, color, rng, s=5, marker="o"):
    x = rng.normal(cx, sx, n); y = rng.normal(cy, sy, n)
    ax.scatter(x, y, s=s, color=color, lw=0, marker=marker, zorder=3)
    return x, y


# ------------------------------------------------------------------ schematic axes in inches
def sax(sh, x, y, w, h):
    ax = sh.ax(x, y, w, h); ax.set_xlim(0, w); ax.set_ylim(0, h); fs.blank(ax)
    return ax


# ------------------------------------------------------------------ a
def panel_a(sh):
    sh.panel("a", 0.0, 3.50, 3.30, 5.40, "Why do we need rejection? (schematic)")
    ax = sax(sh, 0.05, 3.55, 3.22, 1.58)
    for x0, lab, col in ((0.05, "Source sample", fs.SRC), (2.02, "Target sample", fs.TGT)):
        g.frame(ax, x0, 0.22, 1.15, 1.18, color=col, lw=0.7)
        g.label(ax, x0 + 0.575, 1.48, lab, color=col, size=fs.FS_ANNOT)
    rows = [(1.12, "A", fs.POP[0]), (0.78, "B", fs.POP[1]), (0.42, None, None)]
    for y, name, col in rows[:2]:
        g.cell_cluster(ax, 0.42, y, col, n=4, r=0.055, spread=0.11, seed=int(y * 10))
        g.cell_cluster(ax, 2.39, y, col, n=4, r=0.055, spread=0.11, seed=int(y * 10) + 1)
        g.label(ax, 0.72, y, f"population {name}", color=col, ha="left")
        g.label(ax, 2.69, y, f"population {name}", color=col, ha="left")
        g.arrow(ax, 1.22, y, 1.98, y, color=fs.RETAIN, lw=0.8)
    g.label(ax, 1.60, 1.25, "shared: match", color=fs.RETAIN)
    g.cell_cluster(ax, 0.42, 0.42, fs.POP[2], n=4, r=0.055, spread=0.11, seed=7)
    g.label(ax, 0.72, 0.42, "source-only", color=fs.POP[2], ha="left")
    g.cell_cluster(ax, 2.39, 0.42, fs.POP[3], n=4, r=0.055, spread=0.11, seed=8)
    g.label(ax, 2.69, 0.42, "target-only", color=fs.POP[3], ha="left")
    for x in (1.30, 1.90):
        ax.scatter(x, 0.42, marker="x", s=16, color=fs.REJECT, lw=1.0, zorder=5)
    g.label(ax, 1.60, 0.56, "no counterpart:", color=fs.REJECT)
    g.label(ax, 1.60, 0.42, "reject", color=fs.REJECT)
    g.label(ax, 1.61, 0.06, "a population can be lost, gained or changed between two samples", color=fs.SUBTLE)
    fs.schematic_tag(ax, 1.0, 1.06)


# ------------------------------------------------------------------ b
def panel_b(sh):
    sh.panel("b", 3.35, 3.50, 7.20, 5.40, "Standard OT vs ConfidenceOT (schematic)")
    ax = sax(sh, 3.38, 3.55, 3.80, 1.58)
    rows = [(1.24, "Balanced OT", fs.METHOD["Balanced OT"], "balanced", "every cell sends and receives all its mass",
             "cells without a counterpart are forced into matches"),
            (0.76, "Unbalanced OT", fs.METHOD["UOT"], "uot", "less mass can be moved",
             "but no cell is declared unmatched"),
            (0.28, "ConfidenceOT", fs.METHOD["ConfidenceOT"], "cot", "UOT backbone + keep / reject per cell",
             "unmatched cells are rejected (x)")]
    for i, (y, name, col, kind, l1, l2) in enumerate(rows):
        g.pill(ax, 0.0, y - 0.10, 0.92, 0.20, name, col, alpha=0.22 if kind == "balanced" else 0.16,
               tc=fs.INK if kind == "balanced" else col)
        g.arrow(ax, 0.96, y, 1.08, y)
        g.coupling_icon(ax, 1.36, y, 0.21, kind=kind)
        g.arrow(ax, 1.68, y, 1.80, y)
        g.label(ax, 1.86, y + 0.07, l1, color=col if kind != "balanced" else fs.INK, ha="left")
        g.label(ax, 1.86, y - 0.08, l2, color=fs.SUBTLE, ha="left")
        g.ref(ax, 3.80, y, "Fig. 2d")
        if i < 2:
            g.rule(ax, 0.0, 3.80, y - 0.24)
    ax.annotate("", xy=(0.46, 0.41), xytext=(0.46, 0.63), arrowprops=dict(arrowstyle="-|>", color=fs.REJECT, lw=0.6, mutation_scale=5))
    g.label(ax, 0.52, 0.52, "+ gate", color=fs.REJECT, ha="left")
    # legend
    y = 0.01
    ax.plot([0.0, 0.14], [y, y], color=fs.RETAIN, lw=1.0); g.label(ax, 0.17, y, "correct match", ha="left")
    ax.plot([0.85, 0.99], [y, y], color=fs.REJECT, lw=1.0); g.label(ax, 1.02, y, "forced match", ha="left")
    ax.scatter([1.72], [y], marker="x", s=10, color=fs.REJECT, lw=0.8); g.label(ax, 1.78, y, "rejected", ha="left")
    g.label(ax, 2.35, y, "line width = transported mass", ha="left", color=fs.SUBTLE)
    fs.schematic_tag(ax, 1.0, 1.06)


# ------------------------------------------------------------------ c
def panel_c(sh):
    sh.panel("c", 0.0, 2.18, 7.20, 3.45, "ConfidenceOT workflow (schematic of the code path used for every result)")
    ax = sax(sh, 0.05, 2.20, 7.12, 1.00)
    xs = np.linspace(0.42, 6.66, 7)
    yi, yl, ys = 0.64, 0.36, 0.30
    s = 0.18
    # 1 counts
    g.matrix_icon(ax, xs[0] - 0.09, yi + 0.04, 0.12, fs.SRC, rows=4, cols=5, seed=1)
    g.matrix_icon(ax, xs[0] + 0.09, yi - 0.04, 0.12, fs.TGT, rows=4, cols=5, seed=2)
    # 2 rank
    g.rank_icon(ax, xs[1], yi, s, fs.SRC)
    # 3 PCA
    g.scatter_icon(ax, xs[2], yi, s, [fs.POP[0], fs.POP[1], fs.POP[2]])
    # 4 correction: target cloud shifted onto source
    rng = np.random.default_rng(5)
    ax.scatter(xs[3] - 0.06 + rng.normal(0, 0.03, 10), yi + rng.normal(0, 0.03, 10), s=2, color=fs.SRC, lw=0, zorder=4)
    ax.scatter(xs[3] + 0.10 + rng.normal(0, 0.03, 10), yi - 0.07 + rng.normal(0, 0.03, 10), s=2, color=fs.TGT, lw=0, zorder=4)
    g.arrow(ax, xs[3] + 0.08, yi - 0.07, xs[3] - 0.02, yi - 0.01, color=fs.TGT, lw=0.6, ms=4)
    ax.plot([xs[3] - s, xs[3] - s, xs[3] + s], [yi + s, yi - s, yi - s], color=fs.INK, lw=0.45)
    # 5 cost
    g.heat_icon(ax, xs[4], yi, 0.16, fs.SUBTLE, n=4, diag=True)
    # 6 UOT + gate
    g.coupling_icon(ax, xs[5], yi, 0.19, kind="cot")
    # 7 output
    for k, (dy, c) in enumerate(((0.10, fs.POP[0]), (0.0, fs.POP[1]))):
        g.cell(ax, xs[6] - 0.10, yi + dy, 0.035, c, nucleus=False); g.cell(ax, xs[6] + 0.10, yi + dy, 0.035, c, nucleus=False)
        ax.plot([xs[6] - 0.065, xs[6] + 0.065], [yi + dy] * 2, color=fs.RETAIN, lw=0.9)
    g.cell(ax, xs[6] - 0.10, yi - 0.11, 0.035, fs.POP[2], nucleus=False)
    ax.scatter(xs[6] + 0.0, yi - 0.11, marker="x", s=9, color=fs.REJECT, lw=0.8)
    steps = [("Count matrices", "source and target,\ninteger counts"),
             ("Rank encoding", "each cell: top 256 genes,\nvalue 1 - rank / 256"),
             ("Representation", "2,000 top-variance genes,\nscaled, PCA 50"),
             ("Local correction", "target moved only where\ncredible anchors exist"),
             ("Cost matrix", "cosine distance,\n/ median pair cost"),
             ("UOT + keep / reject", "fitted together\n(M4-E gate)"),
             ("Output", "retained matches +\nlist of rejected cells")]
    for i, (h, b) in enumerate(steps):
        if i == 5:
            g.pill(ax, xs[i] - 0.48, yl - 0.08, 0.96, 0.16, h, fs.REJECT, fs_=fs.FS_ANNOT)
        else:
            g.label(ax, xs[i], yl, h, color=fs.INK, size=fs.FS_ANNOT)
        g.label(ax, xs[i], ys - 0.05, b, color=fs.INK, va="top")
        if i < 6:
            g.arrow(ax, xs[i] + 0.27, yi, xs[i + 1] - 0.27, yi)
    g.pill(ax, xs[5] - 0.75, 0.90, 1.50, 0.12, "rejection cost: keeps 99% of a within-sample split null", fs.RETAIN,
           fs_=fs.FS_SMALL)
    g.arrow(ax, xs[5], 0.895, xs[5], yi + 0.215, color=fs.RETAIN)
    g.label(ax, 0.02, 0.96, "same fixed configuration for every data set (Fig. 2c)", color=fs.SUBTLE, ha="left")


# ------------------------------------------------------------------ d (real data)
def panel_d(sh):
    sh.panel("d", 0.0, 0.0, 7.20, 2.14, "Representative result (real computation, Splatter N = 5,000)")
    cells = pd.read_csv(EX / "cells.tsv", sep="\t")
    gm = pd.read_csv(EX / "group_mass.tsv", sep="\t")
    meta = json.loads((EX / "meta.json").read_text())
    emb = EX / "umap_local.tsv"
    if not emb.exists():  # display embedding computed here from the 30 PCs written on TACC
        import umap
        Z = cells[[f"pc{k + 1}" for k in range(30)]].to_numpy()
        E = umap.UMAP(n_neighbors=30, min_dist=0.3, random_state=0).fit_transform(Z)
        pd.DataFrame({"side": cells.side, "observation_id": cells.observation_id, "umap1": E[:, 0], "umap2": E[:, 1]}).to_csv(
            emb, sep="\t", index=False)
    e = pd.read_csv(emb, sep="\t")
    assert (e.observation_id.astype(str).to_numpy() == cells.observation_id.astype(str).to_numpy()).all()
    cells["umap1"], cells["umap2"] = e.umap1.to_numpy(), e.umap2.to_numpy()
    meta["embedding"] = "UMAP (umap-learn 0.5.6, 30 neighbours, min_dist 0.3, seed 0) of PCA 30"
    cells["rejected"] = cells.rejected.astype(str).str.lower().isin(["true", "1"])
    cells["should_reject"] = cells.should_reject.astype(str).str.lower().isin(["true", "1"])
    groups = sorted(cells.group.unique())
    gcol = {g: fs.POP[i] for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)
    order = rng.permutation(len(cells))
    c = cells.iloc[order]
    # d1: populations, source vs target
    for j, (side, mk, title) in enumerate((("source", "o", "Source cells"), ("target", "o", "Target cells"))):
        ax = sh.ax(0.25 + j * 1.45, 0.42, 1.30, 1.30); ax.set_xticks([]); ax.set_yticks([])
        for s_ in ("top", "right", "left", "bottom"):
            ax.spines[s_].set_visible(False)
        ax.scatter(cells.umap1, cells.umap2, s=0.6, color=fs.GUIDE, lw=0, rasterized=True)
        t = c[c.side == side]
        ax.scatter(t.umap1, t.umap2, s=0.9, c=[gcol[g] for g in t.group], lw=0, rasterized=True)
        ax.set_title(f"{title} (n = {len(t):,})", fontsize=fs.FS_ANNOT, color=fs.SRC if side == "source" else fs.TGT, pad=2)
        if j == 0:
            ax.set_xlabel("UMAP 1 (display only)", fontsize=fs.FS_SMALL); ax.set_ylabel("UMAP 2", fontsize=fs.FS_SMALL)
    lg = sh.ax(0.25, 0.08, 2.75, 0.16); fs.blank(lg); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    for i, g in enumerate(groups):
        lg.scatter(0.02 + i * 0.19, 0.5, s=10, color=gcol[g]); lg.text(0.045 + i * 0.19, 0.5, g.replace("Group", "Group "), fontsize=fs.FS_SMALL, va="center")
    # d2: decision vs truth on the source
    ax = sh.ax(3.20, 0.42, 1.30, 1.30); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ("top", "right", "left", "bottom"):
        ax.spines[s_].set_visible(False)
    t = c[c.side == "source"]
    cat = np.select([~t.should_reject & ~t.rejected, t.should_reject & t.rejected, ~t.should_reject & t.rejected],
                    ["kept, has counterpart", "rejected, no counterpart", "rejected, has counterpart"], "kept, no counterpart")
    ccol = {"kept, has counterpart": fs.NEUTRAL, "rejected, no counterpart": fs.REJECT,
            "rejected, has counterpart": fs.FALSE_REJ, "kept, no counterpart": fs.FALSE_RET}
    ax.scatter(cells.umap1, cells.umap2, s=0.6, color=fs.GUIDE, lw=0, rasterized=True)
    for k in ccol:
        m = cat == k
        ax.scatter(t.umap1[m], t.umap2[m], s=0.9 if k.startswith("kept, has") else 1.6, color=ccol[k], lw=0, rasterized=True,
                   zorder=3 if k.startswith("kept, has") else 4)
    ax.set_title("Source decision vs truth", fontsize=fs.FS_ANNOT, pad=2)
    vc = pd.Series(cat).value_counts()
    lg2 = sh.ax(3.20, 0.06, 1.40, 0.32); fs.blank(lg2); lg2.set_xlim(0, 1); lg2.set_ylim(0, 1)
    for i, k in enumerate(ccol):
        yy = 0.85 - i * 0.24
        lg2.scatter(0.03, yy, s=8, color=ccol[k]); lg2.text(0.08, yy, f"{k}: {int(vc.get(k, 0)):,}", fontsize=fs.FS_SMALL, va="center")
    # d3: retained coupling by group
    ax = sh.ax(5.42, 0.56, 1.10, 1.12)
    r = gm[gm.block == "retained"].pivot(index="source_group", columns="target_group", values="mass")
    r = r.div(r.sum(1).replace(0, np.nan), axis=0)
    ret_n = cells[(cells.side == "source") & ~cells.rejected].group.value_counts().reindex(r.index).fillna(0).astype(int)
    few = ret_n < 10  # a row share over fewer than 10 retained cells is not shown as a distribution
    rv = r.to_numpy(dtype=float).copy(); rv[few.to_numpy()] = np.nan
    im = ax.imshow(rv, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(r.shape[1]), [g.replace("Group", "") for g in r.columns])
    ax.set_yticks(range(r.shape[0]), [f"{g.replace('Group', '')}  ({n:,})" for g, n in zip(r.index, ret_n)])
    ax.set_xlabel("Target group", labelpad=1); ax.set_ylabel("Source group (retained cells)")
    for i in range(r.shape[0]):
        if few.iloc[i]:
            ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle((-0.5, i - 0.5), r.shape[1], 1, fc=fs.PALE, ec="none"))
            ax.text((r.shape[1] - 1) / 2, i, f"{ret_n.iloc[i]} retained cell (lost group)", ha="center", va="center", fontsize=fs.FS_SMALL, color=fs.INK)
            continue
        for j in range(r.shape[1]):
            v = r.iloc[i, j]
            if np.isfinite(v) and v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs.FS_SMALL, color="white" if v > 0.6 else fs.INK)
    ax.set_title("Retained coupling (row share)", fontsize=fs.FS_ANNOT, pad=2)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4)
    cax = sh.ax(6.60, 0.56, 0.06, 1.12); cb = sh.fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.set_ticks([0, 0.5, 1])
    tg = cells[cells.side == "target"]
    sh.text(4.75, 0.06, f"Target: {int(tg.rejected.sum()):,} of {len(tg):,} rejected, all with a counterpart (false)",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="bottom")
    return cells, gm, meta, vc


def write_text(cells, gm, meta, vc):
    src = cells[cells.side == "source"]; tgt = cells[cells.side == "target"]
    s5 = src[src.should_reject]
    r = gm[gm.block == "retained"]
    diag = r[r.source_group == r.target_group].mass.sum() / r.mass.sum()
    cap = f"""# Figure 1 | ConfidenceOT: why and how

**a**, The problem (schematic). Two samples share some cell populations (A, B) but not others: a population present only in
the source (C) or only in the target (D) has no counterpart, and any match assigned to it is false.

**b**, Standard OT vs ConfidenceOT (schematic, redrawn after the framework figures of CellOT and Waddington-OT). Balanced OT
moves all mass, so source-only and target-only cells are forced into matches. Unbalanced OT (UOT) can move less mass, but
it does not say which cells are unmatched. ConfidenceOT keeps the UOT backbone and adds a keep / reject decision for each
cell; rejected cells are left out of the correspondence. It is not a separate transport formulation.

**c**, Workflow, as implemented (`realdata_benchmark/v3_grid/v3_common.py`, `cancer_metastasis/02_run_pair.py`,
`src/confidenceot`). Integer counts of both samples are rank-encoded per cell (top 256 genes, value 1 - rank/256, mid-rank
ties), reduced to 2,000 top-variance encoded genes, gene-scaled and projected to 50 principal components. A local correction
moves target cells only where credible shared anchors exist (calibration v3, `mnncovlocal`). The cost is the cosine distance
divided by its median pair cost. The UOT coupling (epsilon 0.1, lambda 1) and the per-cell gate are fitted together (M4-E);
the rejection cost is the smallest cost that keeps 99% of cells of a within-sample split null, where no cell can lack a
counterpart. Output: the retained cells with their coupling, and the rejected cells. Schematic.

**d**, A real result: Splatter simulation, level L2 (target depth x 0.5 + dropout), replicate 2, Case 2 (Group 5 removed
from the target), {meta['source_cells']:,} source and {meta['target_cells']:,} target cells, main configuration, M4-E. Left:
display embedding ({meta['embedding']} of log1p CP10k, 2,000 top-variance genes; not the OT input), coloured by group.
Middle: source cells by decision and truth. Group 5 (no counterpart): {int(s5.rejected.sum()):,} of {len(s5):,} rejected;
shared source cells rejected: {int((~src.should_reject & src.rejected).sum()):,} of {int((~src.should_reject).sum()):,}.
Target cells rejected: {meta['target_rejected']:,} of {meta['target_cells']:,} (all have a counterpart). Right: coupling
mass on retained x retained cells by group, as a share of each source group's retained mass; {diag:.1%} of the retained mass
links cells of the same group. This unit is close to the L2 mean (panel 2b: L2 lost-population F1 0.998).
"""
    (OUT / "caption.md").write_text(cap, encoding="utf-8")
    srcs = f"""# Figure 1 sources

| Panel | Source | Notes |
|---|---|---|
| a, b | schematic | redrawn; no data |
| c | `realdata_benchmark/v3_grid/v3_common.py` (representation, correction, fit), `realdata_benchmark/factorial_common.py` (ranknm encoding: top-K per cell, 1 - r/K), `realdata_benchmark/splatter_local/CALIBRATION_V3.md`, `src/confidenceot/api.py` (UOT backbone, M4-E gate, rejection cost) | schematic of the code path |
| d | TACC `{meta['row']}` (`cell_confidence.csv.gz`, `coupling_m4e.npz`) and `{meta['cond']}` (`source.h5ad`, `target.h5ad`) | derived on the TACC login node by `report_20261118/tacc_fig1_example.py` (read-only), copied to `out/Figure1/data/example/` |
"""
    (OUT / "sources.md").write_text(srcs, encoding="utf-8")
    notes = {
        "a": {"title": "Why do we need rejection?", "compared": "Two samples with shared, source-only and target-only populations.",
              "observe": "Some cells have no counterpart in the other sample.", "meaning": "A method that matches every cell must create false matches for them."},
        "b": {"title": "Standard OT vs ConfidenceOT", "compared": "Balanced OT, unbalanced OT and ConfidenceOT on the same toy problem.",
              "observe": "Balanced OT forces all mass; UOT reduces mass but decides nothing; ConfidenceOT keeps UOT and marks unmatched cells as rejected.",
              "meaning": "The new part is the per-cell decision, not a new transport model."},
        "c": {"title": "ConfidenceOT workflow", "compared": "The steps every result in this report went through.",
              "observe": "Counts -> rank encoding -> PCA -> local correction -> cosine cost -> UOT with a keep/reject gate.",
              "meaning": "The rejection threshold comes from a null built from the data itself, not from labels."},
        "d": {"title": "A real example", "compared": f"Splatter L2, Group 5 removed from the target; {meta['source_cells']:,} vs {meta['target_cells']:,} cells.",
              "observe": f"{int(s5.rejected.sum()):,} of {len(s5):,} Group 5 cells rejected; retained cells are matched to their own group ({diag:.0%} of retained mass).",
              "meaning": "When a population is missing, ConfidenceOT rejects it instead of forcing it onto another population."},
    }
    (OUT / "panel_notes.json").write_text(json.dumps(notes, indent=1), encoding="utf-8")


def main():
    fs.apply()
    sh = fs.Sheet(7.2, 5.4, "Figure1")
    panel_a(sh); panel_b(sh); panel_c(sh)
    if not (EX / "cells.tsv").exists():  # layout check only; never shipped
        sh.panel("d", 0.0, 0.0, 7.20, 2.14, "Representative result (awaiting TACC extraction)")
        sh.save(fs.HERE / "out_layoutcheck")
        return
    cells, gm, meta, vc = panel_d(sh)
    sh.save()
    write_text(cells, gm, meta, vc)


if __name__ == "__main__":
    main()

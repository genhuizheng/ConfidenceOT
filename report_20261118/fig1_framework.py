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

OUT = fs.OUT / "Figure1"
EX = OUT / "data" / "example"
POPC = {"A": fs.POP[0], "B": fs.POP[1], "C": fs.POP[2], "D": fs.POP[3]}


def cloud(ax, cx, cy, n, sx, sy, color, rng, s=5, marker="o"):
    x = rng.normal(cx, sx, n); y = rng.normal(cy, sy, n)
    ax.scatter(x, y, s=s, color=color, lw=0, marker=marker, zorder=3)
    return x, y


# ------------------------------------------------------------------ a
def panel_a(sh):
    sh.panel("a", 0.0, 3.50, 2.55, 5.40, "Why do we need rejection?")
    ax = sh.ax(0.08, 3.62, 2.42, 1.48); ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); fs.blank(ax)
    fs.schematic_tag(ax, 1.0, 0.13)
    rng = np.random.default_rng(4)
    for cx, lab, col in ((2.1, "Source", fs.SRC), (7.9, "Target", fs.TGT)):
        ax.add_patch(Ellipse((cx, 3.35), 3.9, 4.6, fc="none", ec=col, lw=0.8))
        ax.text(cx, 5.85, lab, color=col, ha="center", fontsize=fs.FS_TITLE)
    # source: A, B shared; C source-only. target: A, B shared; D target-only
    cloud(ax, 1.55, 4.75, 20, 0.33, 0.25, POPC["A"], rng); cloud(ax, 1.55, 2.05, 20, 0.33, 0.25, POPC["B"], rng)
    cloud(ax, 3.05, 3.5, 14, 0.25, 0.25, POPC["C"], rng)
    cloud(ax, 8.45, 4.75, 20, 0.33, 0.25, POPC["A"], rng); cloud(ax, 8.45, 2.05, 20, 0.33, 0.25, POPC["B"], rng)
    cloud(ax, 6.95, 3.5, 14, 0.25, 0.25, POPC["D"], rng)
    for y in (4.75, 2.05):
        fs.arrow(ax, 2.25, y, 7.75, y, color=fs.RETAIN, lw=0.8, ms=5)
    ax.text(5.0, 4.9, "shared: match", ha="center", va="bottom", fontsize=fs.FS_SMALL, color=fs.RETAIN)
    ax.text(3.05, 3.0, "source-\nonly", ha="center", va="top", fontsize=fs.FS_SMALL, color=fs.INK, linespacing=1.0)
    ax.text(6.95, 3.0, "target-\nonly", ha="center", va="top", fontsize=fs.FS_SMALL, color=fs.INK, linespacing=1.0)
    ax.text(5.0, 3.5, "no counterpart:\nreject", ha="center", va="center", fontsize=fs.FS_SMALL, color=fs.REJECT, linespacing=1.05)
    ax.text(5.0, 0.05, "A population can be lost, gained or changed between two samples", ha="center", va="bottom",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE)


# ------------------------------------------------------------------ b
def mini_plan(ax, kind):
    ax.set_xlim(0, 10); ax.set_ylim(-1.6, 10); fs.blank(ax)
    ys = {"A": [8.6, 7.8, 7.0], "B": [5.4, 4.6, 3.8], "C": [2.2, 1.4], "D": [2.2, 1.4]}
    src = [("A", y) for y in ys["A"]] + [("B", y) for y in ys["B"]] + [("C", y) for y in ys["C"]]
    tgt = [("A", y) for y in ys["A"]] + [("B", y) for y in ys["B"]] + [("D", y) for y in ys["D"]]
    xs, xt = 1.6, 8.4

    def line(y0, y1, w, col, ls="-"):
        ax.plot([xs, xt], [y0, y1], color=col, lw=w, ls=ls, solid_capstyle="round", zorder=1)
    for (g, y) in src:
        if g in ("A", "B"):
            for (h, y1) in tgt:
                if h == g and abs(y1 - y) < 0.1:
                    w = {"balanced": 1.1, "uot": 0.75, "cot": 0.9}[kind]
                    line(y, y1, w, fs.RETAIN)
        else:  # source-only C
            if kind == "balanced":
                line(y, 2.2 if y > 2 else 1.4, 1.1, fs.REJECT); line(y, 4.6, 0.6, fs.REJECT)
            elif kind == "uot":
                line(y, 1.8, 0.45, fs.REJECT); line(y, 4.6, 0.3, fs.REJECT)
    if kind == "balanced":  # target-only D must receive mass
        line(5.4, 2.2, 0.5, fs.REJECT); line(3.8, 1.4, 0.5, fs.REJECT)
    if kind == "uot":
        line(3.8, 1.4, 0.3, fs.REJECT)
    for (g, y) in src:
        rej = kind == "cot" and g == "C"
        ax.scatter(xs, y, s=13, color=POPC[g], zorder=3, edgecolor="white", lw=0.3)
        if rej:
            ax.scatter(xs - 0.75, y, s=14, marker="x", color=fs.REJECT, lw=0.9, zorder=4)
    for (g, y) in tgt:
        rej = kind == "cot" and g == "D"
        ax.scatter(xt, y, s=13, color=POPC[g], zorder=3, edgecolor="white", lw=0.3)
        if rej:
            ax.scatter(xt + 0.75, y, s=14, marker="x", color=fs.REJECT, lw=0.9, zorder=4)
    ax.text(xs, 9.6, "S", color=fs.SRC, ha="center", fontsize=fs.FS_ANNOT)
    ax.text(xt, 9.6, "T", color=fs.TGT, ha="center", fontsize=fs.FS_ANNOT)


def panel_b(sh):
    sh.panel("b", 2.60, 3.50, 7.20, 5.40, "Standard OT vs ConfidenceOT")
    titles = [("balanced", "Balanced OT", "every cell sends and\nreceives all of its mass"),
              ("uot", "Unbalanced OT (UOT)", "mass can shrink, but no cell\nis declared unmatched"),
              ("cot", "ConfidenceOT", "UOT backbone + a keep / reject\ndecision for every cell")]
    for i, (k, t, sub) in enumerate(titles):
        x = 2.78 + i * 1.47
        ax = sh.ax(x, 3.98, 1.25, 1.02); mini_plan(ax, k)
        col = fs.METHOD["ConfidenceOT"] if k == "cot" else (fs.METHOD["UOT"] if k == "uot" else fs.INK)
        sh.text(x + 0.625, 5.08, t, ha="center", va="bottom", fontsize=fs.FS_ANNOT + 0.4, color=col)
        sh.text(x + 0.625, 3.92, sub, ha="center", va="top", fontsize=fs.FS_SMALL, color=fs.INK, linespacing=1.1)
    axt = sh.ax(6.9, 5.12, 0.28, 0.2); fs.blank(axt); fs.schematic_tag(axt, 1.0, 1.0)
    lg = sh.ax(2.78, 3.53, 4.3, 0.12); fs.blank(lg); lg.set_xlim(0, 1); lg.set_ylim(0, 1)
    lg.plot([0.00, 0.04], [0.5, 0.5], color=fs.RETAIN, lw=1.1); lg.text(0.05, 0.5, "correct match", fontsize=fs.FS_SMALL, va="center")
    lg.plot([0.25, 0.29], [0.5, 0.5], color=fs.REJECT, lw=1.1); lg.text(0.30, 0.5, "forced / wrong match", fontsize=fs.FS_SMALL, va="center")
    lg.scatter([0.56], [0.5], marker="x", s=12, color=fs.REJECT, lw=0.9); lg.text(0.58, 0.5, "rejected (no counterpart)", fontsize=fs.FS_SMALL, va="center")
    lg.text(0.86, 0.5, "line width = mass", fontsize=fs.FS_SMALL, va="center", color=fs.SUBTLE)


# ------------------------------------------------------------------ c
def panel_c(sh):
    sh.panel("c", 0.0, 2.18, 7.20, 3.45, "ConfidenceOT workflow (as run for every result in this report)")
    ax = sh.ax(0.05, 2.22, 7.10, 0.98); ax.set_xlim(0, 71); ax.set_ylim(0, 10); fs.blank(ax)
    fs.schematic_tag(ax, 1.0, 1.06)
    steps = [
        ("Count matrices", "source and target\n(integer counts)"),
        ("Rank encoding", "per cell: top 256 genes,\nvalue 1 - rank/256"),
        ("Representation", "2,000 top-variance genes,\ngene scaling, PCA 50"),
        ("Local correction", "shift target only where\ncredible anchors exist"),
        ("Cost matrix", "cosine distance,\n/ median pair cost"),
        ("UOT + gate", "coupling and keep / reject\nfitted together (M4-E)"),
        ("Output", "retained cells + coupling;\nrejected cells listed"),
    ]
    w, gap = 8.85, 1.50
    for i, (head, body) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        hl = head in ("UOT + gate",)
        fs.box(ax, x, 3.2, w, 5.6, "", fc="#FBEDEB" if hl else fs.PALE, ec=fs.REJECT if hl else fs.NEUTRAL, lw=0.7)
        ax.text(x + w / 2, 7.55, head, ha="center", va="center", fontsize=fs.FS_ANNOT + 0.3,
                color=fs.REJECT if hl else fs.INK)
        ax.text(x + w / 2, 5.0, body, ha="center", va="center", fontsize=fs.FS_SMALL, color=fs.INK, linespacing=1.12)
        if i < len(steps) - 1:
            fs.arrow(ax, x + w + 0.15, 6.0, x + w + gap - 0.15, 6.0, color=fs.INK, lw=0.7, ms=5)
    xg = 0.3 + 5 * (w + gap)
    fs.box(ax, xg - 3.2, -0.6, w + 6.4, 2.9, "rejection cost: keeps 99% of cells of a\nwithin-sample split null (no labels)",
           fc="white", ec=fs.RETAIN, tc=fs.RETAIN, fs=fs.FS_SMALL, lw=0.6, linespacing=1.05)
    fs.arrow(ax, xg + w / 2, 2.35, xg + w / 2, 3.15, color=fs.RETAIN, lw=0.7, ms=5)
    ax.text(0.3, 0.8, "Fixed everywhere: ranknm256-midrank|none|scale|cos|pca50; calibration v3; acceptance 0.99; "
            "epsilon 0.1, lambda 1", fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="center")


# ------------------------------------------------------------------ d (real data)
def panel_d(sh):
    sh.panel("d", 0.0, 0.0, 7.20, 2.14, "Representative result (real computation, Splatter N = 5,000)")
    cells = pd.read_csv(EX / "cells.tsv", sep="\t")
    gm = pd.read_csv(EX / "group_mass.tsv", sep="\t")
    meta = json.loads((EX / "meta.json").read_text())
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
    ax = sh.ax(5.15, 0.42, 1.25, 1.25)
    r = gm[gm.block == "retained"].pivot(index="source_group", columns="target_group", values="mass")
    r = r.div(r.sum(1).replace(0, np.nan), axis=0)
    im = ax.imshow(r.to_numpy(dtype=float), cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(r.shape[1]), [g.replace("Group", "") for g in r.columns]); ax.set_yticks(range(r.shape[0]), [g.replace("Group", "") for g in r.index])
    ax.set_xlabel("Target group"); ax.set_ylabel("Source group")
    for i in range(r.shape[0]):
        for j in range(r.shape[1]):
            v = r.iloc[i, j]
            if np.isfinite(v) and v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=fs.FS_SMALL, color="white" if v > 0.6 else fs.INK)
    ax.set_title("Retained coupling (row share)", fontsize=fs.FS_ANNOT, pad=2)
    for s_ in ax.spines.values():
        s_.set_linewidth(0.4)
    cax = sh.ax(6.48, 0.42, 0.06, 1.25); cb = sh.fig.colorbar(im, cax=cax); cb.outline.set_linewidth(0.4); cb.set_ticks([0, 0.5, 1])
    src5 = cells[(cells.side == "source") & cells.should_reject]
    sh.text(5.15, 0.10, f"Group 5 (lost): {int(src5.rejected.sum()):,} of {len(src5):,} rejected;\nrows with no retained cell are empty",
            fontsize=fs.FS_SMALL, color=fs.SUBTLE, va="bottom", linespacing=1.1)
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

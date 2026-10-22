"""Figures of the within-family subclone analysis (one set per classified family / clone).

    python lineage_validation/within_family_subclones/plot_subclones.py --defs DEFS --classification CLASS \\
        --kp-trees DIR [DIR ...] --lineage-export EXPORT --out FIGURES

Reuses the drawing helpers of ../local_lineage_visualization/plot_lineage.py. Per family:
  <name>_1_tree_subclones      the authors' tree (restricted to the family) + primary/met, site, author SubTumor
                               (KP-Tracer), level-1 and level-2 CRISPR subclones
  <name>_2_tree_ot             the tree + true level-1 subclone + per strategy (M4-E) the OT-predicted primary
                               subclone and the outcome (correct / incorrect / ambiguous / abstain / ...)
  <name>_3_heatmap             per strategy and level: true metastatic subclone x primary subclone, mean
                               within-family matching weight over retained metastatic cells; the bar on top
                               is each primary subclone's share of retained source cells (what size alone gives)
  <name>_4_umap                fixed joint UMAP: site, true subclone, OT-predicted subclone (RNA only),
                               retain / reject (RNA only)
  <name>_5_branches            zoom on every shared level-1 subclone (>= 5 primary and >= 5 metastatic cells)
OT predictions are strips beside the tree, never tree edges.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pl = load("plot_lineage", HERE.parent / "local_lineage_visualization" / "plot_lineage.py")
ds = load("define_subclones", HERE / "define_subclones.py")

OUTCOME = {"correct": "#1a9850", "incorrect": "#d73027", "ambiguous": "#fdae61", "abstain": "#bdbdbd",
           "counterpart rejected": "#762a83", "unmatched_observed": "#4575b4", "no CRISPR label": "#ffffff"}
# Subclones: no red or green, which the outcome strip uses for incorrect / correct.
SUB_PALETTE = ["#1f78b4", "#ff7f00", "#6a3d9a", "#b15928", "#a6cee3", "#fdbf6f", "#cab2d6", "#e6ab02", "#8dd3c7",
               "#80b1d3", "#bc80bd", "#666666", "#ffed6f", "#d9b38c", "#5e3c99", "#f4a582", "#92c5de", "#b2abd2",
               "#8c510a", "#35978f", "#c51b7d", "#de77ae", "#7f3b08", "#542788"]


class DrawTree:
    """Adapter: an Induced tree with contracted depth, for plot_lineage.Drawn."""

    def __init__(self, induced):
        self.children = induced.children
        self.name = induced.name
        self.root = induced.root
        depth = {induced.root: 0}
        for v in induced.order:
            step = 1 if len(induced.children[v]) >= 2 else 0
            for c in induced.children[v]:
                depth[c] = depth[v] + step
        self.depth = depth


def tree_for(row, args):
    d1 = lv.tree_module()
    if row["dataset"] == "KPTracer":
        for d in args.kp_trees:
            hits = sorted(Path(d).rglob(row["tree"]))
            if hits:
                return d1.Tree(hits[0])
    else:
        hits = sorted((args.lineage_export / "lineage" / "Quinn2021" / str(row["mouse"])).rglob(row["tree"]))
        if hits:
            return d1.Tree(hits[0])
    raise FileNotFoundError(row["tree"])


def colours_for(labels, offset=0):
    names = sorted(x for x in pd.unique(labels) if x and x != "ambiguous")
    out = {n: SUB_PALETTE[(i + offset) % len(SUB_PALETTE)] for i, n in enumerate(names)}
    out["ambiguous"] = "#e0e0e0"
    return out


def fig_tree_subclones(name, title, drawn, cells, sites, l1c, l2c, out, index):
    fig = plt.figure(figsize=(12, 9))
    extra = ["SubTumor"] if "SubTumor" in cells and cells["SubTumor"].replace("", np.nan).notna().any() else []
    widths = [6, 0.35, 0.35] + [0.35] * len(extra) + [0.45, 0.45]
    grid = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.05)
    ax = fig.add_subplot(grid[0]); drawn.draw(ax, lw=0.3)
    pl.strip(fig.add_subplot(grid[1]), cells["kind"], pl.KIND, "primary / met")
    pl.strip(fig.add_subplot(grid[2]), cells["site"], sites, "site")
    k = 3
    if extra:
        sub = cells["SubTumor"].fillna("")
        pl.strip(fig.add_subplot(grid[k]), sub, {s: pl.SITE_PALETTE[i % len(pl.SITE_PALETTE)] for i, s in enumerate(sorted(set(sub) - {""}))},
                 "author SubTumor\n(tumour piece)")
        k += 1
    pl.strip(fig.add_subplot(grid[k]), cells["L1"], l1c, "CRISPR\nsubclone L1")
    pl.strip(fig.add_subplot(grid[k + 1]), cells["L2"].replace("", "ambiguous"), l2c, "CRISPR\nsubclone L2")
    comp = cells.groupby("L1")["kind"].value_counts().unstack(fill_value=0)
    text = "; ".join(f"{s}: {int(r.get('primary', 0))} primary, {int(r.get('metastasis', 0))} metastatic" for s, r in comp.iterrows())
    fig.suptitle(f"{title}: CRISPR subclones (SUBCLONE_RULES.md), the authors' tree restricted to the family", fontsize=10, y=0.99)
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=7, wrap=True)
    pl.legend(fig, pl.patches(pl.KIND) + pl.patches(sites) + pl.patches(l1c, "L1 ") + pl.patches(l2c, "L2 "), y=-0.09)
    pl.save(fig, out, f"{name}_1_tree_subclones", index, name=name, panel="tree + subclones")


def fig_tree_ot(name, title, drawn, cells, sites, l1c, targets, out, index):
    fig = plt.figure(figsize=(15, 9))
    n = len(lv.STRATEGIES)
    widths = [6, 0.35, 0.35, 0.45] + [0.45, 0.45] * n
    grid = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.05)
    ax = fig.add_subplot(grid[0]); drawn.draw(ax, lw=0.3)
    pl.strip(fig.add_subplot(grid[1]), cells["kind"], pl.KIND, "primary / met")
    pl.strip(fig.add_subplot(grid[2]), cells["site"], sites, "site")
    pl.strip(fig.add_subplot(grid[3]), cells["L1"], l1c, "CRISPR\nsubclone L1")
    tallies = []
    for k, strategy in enumerate(lv.STRATEGIES):
        t = targets.get(strategy)
        if t is None:
            continue
        t1 = t[t["level"] == 1].set_index("cell_id")
        pred = cells["cell_id"].map(t1["predicted"]).fillna("")
        outc = cells["cell_id"].map(t1["outcome"]).fillna("")
        pl.strip(fig.add_subplot(grid[4 + 2 * k]), np.where(cells["kind"].eq("metastasis"), pred, ""), {**l1c, "": "#ffffff"},
                 f"OT predicted\n({pl.NAMES[strategy]})")
        pl.strip(fig.add_subplot(grid[5 + 2 * k]), outc, OUTCOME, "outcome")
        c = t1["outcome"].value_counts()
        tallies.append(f"{pl.NAMES[strategy]}: " + ", ".join(f"{o} {int(c.get(o, 0))}" for o in OUTCOME if c.get(o, 0)))
    fig.suptitle(f"{title}: true CRISPR subclone vs OT-predicted primary subclone (M4-E, level 1)\n"
                 "(the prediction is an expression-based association, drawn beside the tree)", fontsize=10, y=1.0)
    fig.text(0.5, -0.02, "\n".join(tallies), ha="center", va="top", fontsize=7)
    pl.legend(fig, pl.patches(pl.KIND) + pl.patches(sites) + pl.patches(l1c, "subclone ") + pl.patches(OUTCOME), y=-0.13)
    pl.save(fig, out, f"{name}_2_tree_ot", index, name=name, panel="tree + OT classification")


def fig_heatmap(name, title, heat, summary, out, index):
    """Rows: the retained-source share of each primary subclone (what size alone gives), then every true
    metastatic subclone; cells: mean within-family matching weight. One quantity: a share of mass."""
    h = heat[heat["method"] == "M4-E"]
    levels = sorted(h["level"].unique())
    fig, axes = plt.subplots(len(levels), len(lv.STRATEGIES), figsize=(3.9 * len(lv.STRATEGIES), 0.9 + 3.6 * len(levels)),
                             squeeze=False, gridspec_kw={"hspace": 0.75, "wspace": 0.75})
    mesh = None
    for r, level in enumerate(levels):
        for c, strategy in enumerate(lv.STRATEGIES):
            ax = axes[r, c]
            part = h[(h["level"] == level) & (h["strategy"] == strategy)]
            srow = summary[(summary["method"] == "M4-E") & (summary["strategy"] == strategy) & (summary["level"] == level)]
            if part.empty or srow.empty:
                ax.axis("off"); continue
            wide = part.pivot_table(index=["true", "cells"], columns="source", values="mean_weight")
            sources = [x for x in sorted(wide.columns) if x != "family unlabelled"] + ["family unlabelled"]
            wide = wide.reindex(columns=sources)
            # Retained effective transport mass per source group, unlabelled included: the weights' own units.
            ret = json.loads(srow.iloc[0]["source_effective_mass"])
            total = sum(ret.values()) or 1
            share = [ret.get(x, 0) / total for x in sources]
            grid = np.vstack([share, wide.to_numpy(float)])
            mesh = ax.imshow(grid, cmap="Purples", vmin=0, vmax=1, aspect="auto")
            ax.axhline(0.5, color="black", lw=1.2)
            for (i, j), v in np.ndenumerate(grid):
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 else "black")
            ax.set_xticks(range(len(sources)))
            ax.set_xticklabels([x.replace("family unlabelled", "unlab.") for x in sources], rotation=90, fontsize=6)
            ax.set_yticks(range(len(grid)))
            ax.set_yticklabels(["source share"] + [f"{t} (n={n})" for t, n in wide.index], fontsize=6)
            acc, size_acc = srow.iloc[0]["accuracy"], srow.iloc[0]["size_reference_accuracy"]
            ax.set_title(f"{pl.NAMES[strategy]}, level {level}\naccuracy {acc:.2f} (size ref. {size_acc:.2f})"
                         if np.isfinite(acc) else f"{pl.NAMES[strategy]}, level {level}", fontsize=7)
            ax.set_xlabel("primary (source) subclone", fontsize=6)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), fraction=0.012, label="share of within-family matching weight")
    fig.suptitle(f"{title}: mean OT matching weight, true metastatic subclone x primary subclone (M4-E; retained metastatic cells)\n"
                 "top row = each primary group's share of retained effective transport mass = the independent coupling's weight",
                 fontsize=9, y=1.0)
    pl.save(fig, out, f"{name}_3_heatmap", index, name=name, panel="subclone matching heatmap")


def fig_umap(name, title, umap, cells, sites, l1c, targets, sources, out, index):
    u = umap.copy()
    lab = dict(zip(cells["cell_id"], cells["L1"]))
    site = dict(zip(cells["cell_id"], cells["site"]))
    fam_cells = set(cells["cell_id"])
    u = u[u["observation_id"].isin(fam_cells)]
    t = targets.get("rna")
    s = sources.get("rna")
    panels = [("site", u["observation_id"].map(site), sites),
              ("true CRISPR subclone (L1)", u["observation_id"].map(lab).fillna("ambiguous"), l1c)]
    if t is not None:
        t1 = t[t["level"] == 1].set_index("cell_id")
        pred = u["observation_id"].map(t1["predicted"]).fillna("").replace("", "no prediction")
        pred = np.where(u["side"].eq("target"), pred, "source")
        panels.append(("OT-predicted subclone (RNA only, metastatic cells)", pred, {**l1c, "no prediction": "#bdbdbd", "source": "#f0f0f0"}))
        st = pd.concat([t1["state"], s.set_index("cell_id")["state"]]) if s is not None else t1["state"]
        st = st[~st.index.duplicated()]
        panels.append(("RNA only gate", u["observation_id"].map(st).fillna("not in run"), pl.STATE))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.2), squeeze=False)
    x, y = u["umap_1"].to_numpy(float), u["umap_2"].to_numpy(float)
    size = max(1.0, min(8.0, 6000 / max(len(u), 1)))
    for ax, (label, values, cmap) in zip(axes[0], panels):
        values = np.asarray(values, dtype=object)
        for cat, colour in cmap.items():
            hit = values == cat
            if hit.any():
                ax.scatter(x[hit], y[hit], s=size, c=[colour], linewidths=0, rasterized=True, label=f"{pl.short(cat)} ({int(hit.sum())})")
        ax.set_title(label, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=5, frameon=False, markerscale=2, loc="best")
    fig.suptitle(f"{title}: family cells on the fixed joint UMAP of the run's pair", fontsize=9, y=1.03)
    fig.tight_layout()
    pl.save(fig, out, f"{name}_4_umap", index, name=name, panel="UMAP")


def fig_branches(name, title, induced, cells, subs, sites, l1c, targets, out, index):
    shared = subs[(subs["level"].astype(int) == 1) & subs["shared"].astype(str).str.lower().eq("true")]
    if shared.empty:
        return
    t = targets.get("rna")
    t1 = t[t["level"] == 1].set_index("cell_id") if t is not None else None
    fig = plt.figure(figsize=(5.2 * len(shared), 8))
    outer = fig.add_gridspec(1, len(shared), wspace=0.3)
    by_leaf = cells.set_index("leaf")
    for k, (_, sub) in enumerate(shared.iterrows()):
        drawn = pl.Drawn(DrawTree(induced), int(sub["tree_node"]))
        part = by_leaf.reindex([induced.name[v] for v in drawn.leaves])
        g = outer[k].subgridspec(1, 5, width_ratios=[4, 0.4, 0.4, 0.45, 0.45], wspace=0.05)
        ax = fig.add_subplot(g[0]); drawn.draw(ax, lw=0.5)
        ax.set_title(f"{sub['subclone']}: {sub['primary']} primary, {sub['metastatic']} metastatic\n"
                     f"node {sub['tree_node']}, {sub['marker_edits']} marker edits", fontsize=7)
        pl.strip(fig.add_subplot(g[1]), part["kind"].fillna("unknown"), pl.KIND, "primary / met")
        pl.strip(fig.add_subplot(g[2]), part["site"].fillna("unmapped"), sites, "site")
        if t1 is not None:
            pred = part["cell_id"].map(t1["predicted"]).fillna("")
            outc = part["cell_id"].map(t1["outcome"]).fillna("")
            pl.strip(fig.add_subplot(g[3]), np.where(part["kind"].eq("metastasis"), pred, ""), {**l1c, "": "#ffffff"}, "OT predicted\n(RNA only)")
            pl.strip(fig.add_subplot(g[4]), outc, OUTCOME, "outcome")
    fig.suptitle(f"{title}: shared level-1 subclones (>= 5 primary and >= 5 metastatic cells), zoomed; RNA only, M4-E", fontsize=9, y=1.02)
    pl.legend(fig, pl.patches(sites) + pl.patches(l1c, "subclone ") + pl.patches(OUTCOME), y=-0.08)
    pl.save(fig, out, f"{name}_5_branches", index, name=name, panel="representative branches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--defs", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--kp-trees", type=Path, nargs="+", required=True)
    parser.add_argument("--lineage-export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    feas = pd.read_csv(args.defs / "feasibility.tsv", sep="\t", dtype={"mouse": str})
    subs_all = pd.read_csv(args.defs / "subclones.tsv", sep="\t", dtype={"mouse": str})
    index = []
    for _, row in feas[feas["subclone_structure_eligible"].astype(str).str.lower().eq("true")].iterrows():
        name = f"{row['dataset']}_{row['mouse']}_{row['family_or_clone']}"
        cdir = args.classification / name
        if not (cdir / "summary.tsv").exists():
            print(f"{name}: not classified yet"); continue
        cells = pd.read_csv(args.defs / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
        tree = tree_for(row, args)
        leaf_node = {tree.name[v]: v for v in tree.leaves}
        induced = ds.Induced(tree, {leaf_node[x] for x in cells["leaf"] if x in leaf_node})
        cells = cells.set_index("leaf").reindex([induced.name[v] for v in induced.leaves]).reset_index()
        drawn = pl.Drawn(DrawTree(induced), induced.root)
        primary = lambda s, kind=dict(zip(cells["site"], cells["kind"])): kind.get(s) == "primary"
        sites = pl.site_colours(cells["site"], primary)
        l1c = colours_for(cells["L1"])
        l2c = colours_for(cells["L2"].replace("", "ambiguous"), offset=9)  # level 2 in its own colours
        targets, sources = {}, {}
        for strategy in lv.STRATEGIES:
            tp, sp = cdir / f"target_M4E_{strategy}.csv.gz", cdir / f"source_M4E_{strategy}.csv.gz"
            targets[strategy] = pd.read_csv(tp, dtype={"cell_id": str, "true": str, "predicted": str}, keep_default_na=False) if tp.exists() else None
            sources[strategy] = pd.read_csv(sp, dtype={"cell_id": str}, keep_default_na=False) if sp.exists() else None
        heat = pd.read_csv(cdir / "heatmap.tsv", sep="\t")
        summary = pd.read_csv(cdir / "summary.tsv", sep="\t")
        title = f"{'KP-Tracer' if row['dataset'] == 'KPTracer' else 'Quinn 2021'} {row['mouse']} {pl.short(row['family_or_clone'])}"
        subs = subs_all[subs_all["name"] == name]
        fig_tree_subclones(name, title, drawn, cells, sites, l1c, l2c, args.out, index)
        fig_tree_ot(name, title, drawn, cells, sites, l1c, targets, args.out, index)
        fig_heatmap(name, title, heat, summary, args.out, index)
        if (cdir / "umap.csv.gz").exists():
            fig_umap(name, title, pd.read_csv(cdir / "umap.csv.gz", dtype={"observation_id": str}), cells, sites, l1c, targets, sources, args.out, index)
        fig_branches(name, title, induced, cells, subs, sites, l1c, targets, args.out, index)
        print(f"{name}: figures", flush=True)
    pd.DataFrame(index).to_csv(args.out / "figures_index.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

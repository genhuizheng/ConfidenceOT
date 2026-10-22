"""Local figures: real CRISPR lineage (authors' character matrices and trees) beside the existing OT outputs.

    python lineage_validation/local_lineage_visualization/plot_lineage.py \\
        --data lineage_validation/local_lineage_visualization/data/local_lineage_visualization_export

Reads only the downloaded package (tacc_export.py). Trees are the authors' files, never re-inferred;
character states are drawn as deposited. Writes, next to this script:

  annotations/<dataset>_<mouse>_<tree>_cells.csv  one row per tree leaf, in tree leaf order: leaf, cell,
                       site, primary/metastasis, lineage label, matrix row present, missing share, UMAP,
                       and for every strategy (M4-E) the state and OT-predicted source group of the run
                       that holds every cell of the mouse (KP-Tracer D_C3, Quinn D_B)
  figures/<tree>_A_matrix.png/.pdf           character matrix, rows in tree leaf order, + annotation strips
  figures/<tree>_A_tree.png/.pdf             tree overview + two zoomed branches, leaves coloured by site
  figures/<tree>_A_tree_matrix.png/.pdf      tree and matrix aligned (same row order), overview + zooms
  figures/<tree>_B_tree_ot_rejection.png/.pdf  tree + site + retain / reject / exclude of all five strategies
  figures/<tree>_C_tree_ot_coupling.png/.pdf   tree + CRISPR lineage label + OT-predicted source group and weight
  figures/<mouse>_D_umap.png/.pdf            fixed joint UMAP: site, lineage, the five gates, OT prediction
  figures_index.tsv                          every figure with its data, cell counts and coverage

Zoomed branches are chosen by one rule, stated on the figure: clades of 80-400 leaves holding at least
two sites, ranked by their number of metastatic (KP-Tracer) or non-LL (Quinn) leaves, two that do not
overlap. OT associations are predictions from expression; they are drawn as strips beside the tree and
never as tree edges.
"""

from __future__ import annotations

import argparse
import re
import textwrap
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "font.size": 8})
MISSING = {"-", "-1", "nan", "NA", "", "None"}
STATE = {"retained": "#2b8cbe", "rejected": "#d7301f", "excluded": "#8c8c8c", "not in run": "#f0f0f0"}
KIND = {"primary": "#000000", "metastasis": "#c7c7c7", "unknown": "#ffffff"}
NAMES = {"rna": "RNA only", "cnv": "CNV only", "rna_cnv": "CNV + RNA", "cnv_then_rna": "CNV -> RNA", "rna_then_cnv": "RNA -> CNV"}
# Edit states: golden-ratio hues, so that neighbouring state ids differ and the palette is not the site one.
EDIT_PALETTE = [tuple(matplotlib.colors.hsv_to_rgb(((k * 0.618034) % 1.0, 0.55 + 0.3 * (k % 2), 0.75 + 0.2 * (k % 3 == 0))))
                for k in range(97)]
# Sites: primaries in blues, metastases / tissues in a qualitative palette without blue.
PRIMARY_PALETTE = ["#08306b", "#2171b5", "#6baed6", "#9ecae1", "#c6dbef"]
SITE_PALETTE = ["#e41a1c", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#1b9e77", "#d95f02",
                "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#ffd92f", "#b3de69", "#fb8072", "#bc80bd"]


# --- tree geometry -------------------------------------------------------------------------

class Drawn:
    """Rectangular cladogram of the clade under `root`: x = contracted depth (edges, unary nodes merged)."""

    def __init__(self, tree, root: int):
        self.tree = tree
        order, stack = [], [root]
        while stack:
            v = stack.pop()
            order.append(v)
            stack.extend(reversed(tree.children[v]))
        self.order = order
        self.leaves = [v for v in order if not tree.children[v]]
        y = {v: float(i) for i, v in enumerate(self.leaves)}
        for v in reversed(order):
            if tree.children[v]:
                ys = [y[c] for c in tree.children[v]]
                y[v] = (min(ys) + max(ys)) / 2
        base = tree.depth[root]
        self.x = {v: float(tree.depth[v] - base) for v in order}
        self.y = y
        self.root = root

    def segments(self):
        segs = []
        for v in self.order:
            kids = self.tree.children[v]
            if not kids:
                continue
            xv = self.x[v]
            ys = [self.y[c] for c in kids]
            segs.append([(xv, min(ys)), (xv, max(ys))])
            segs += [[(xv, self.y[c]), (self.x[c], self.y[c])] for c in kids]
        return segs

    def draw(self, ax, lw=0.4):
        ax.add_collection(LineCollection(self.segments(), colors="#404040", linewidths=lw))
        xmax = max(self.x.values()) or 1
        ax.set_xlim(-0.02 * xmax, xmax * 1.02)
        ax.set_ylim(len(self.leaves) - 0.5, -0.5)
        ax.set_xlabel("depth (branching events)")
        ax.set_yticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)


def clade_leaves(tree) -> dict:
    count = {}
    for v in reversed(tree.order):
        count[v] = 1 if not tree.children[v] else sum(count[c] for c in tree.children[v])
    return count


def pick_branches(tree, leaf_info: pd.DataFrame, focus: str, n: int = 2, lo: int = 80, hi: int = 400) -> list[int]:
    """Clades of lo-hi leaves with >= 2 sites, ranked by focus leaves; non-overlapping."""
    size = clade_leaves(tree)
    site = dict(zip(leaf_info["node"], leaf_info["site"]))
    hit = dict(zip(leaf_info["node"], leaf_info[focus]))
    below = {}
    for v in reversed(tree.order):
        below[v] = [v] if not tree.children[v] else [x for c in tree.children[v] for x in below[c]]
    candidates = []
    for v in tree.order:
        if tree.children[v] and lo <= size[v] <= hi:
            leaves = below[v]
            if len({site.get(x) for x in leaves}) >= 2:
                candidates.append((sum(bool(hit.get(x)) for x in leaves), size[v], v))
    chosen, used = [], set()
    for _, _, v in sorted(candidates, key=lambda t: (-t[0], -t[1], t[2])):
        leaves = set(below[v])
        if leaves & used:
            continue
        chosen.append(v)
        used |= leaves
        if len(chosen) == n:
            break
    return chosen


# --- strips -------------------------------------------------------------------------------

def strip(ax, values, colours: dict, title: str):
    cats = list(colours)
    index = {c: i for i, c in enumerate(cats)}
    codes = np.array([index.get(v, len(cats)) for v in values], dtype=float)[:, None]
    cmap = ListedColormap([colours[c] for c in cats] + ["#ffffff"])
    ax.imshow(codes, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(cats) + 0.5, interpolation="nearest",
              extent=(0, 1, len(values) - 0.5, -0.5))
    ax.set_xticks([0.5]); ax.set_xticklabels([title], rotation=90, fontsize=7)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def continuous_strip(ax, values, title: str, cmap="viridis", vmin=0.0, vmax=1.0):
    ax.imshow(np.asarray(values, dtype=float)[:, None], aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", extent=(0, 1, len(values) - 0.5, -0.5))
    ax.set_xticks([0.5]); ax.set_xticklabels([title], rotation=90, fontsize=7)
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def matrix_image(ax, matrix: pd.DataFrame | None, rows: list[str]):
    if matrix is None:
        ax.text(0.5, 0.5, "no character matrix", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    sub = matrix.reindex(rows)
    values = sub.to_numpy(dtype=object)
    rgb = np.ones(values.shape + (3,))
    for (i, j), v in np.ndenumerate(values):
        text = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
        if text in MISSING:
            continue  # missing: white
        if text == "0":
            rgb[i, j] = (0.82, 0.82, 0.82)  # unedited
            continue
        try:
            key = int(float(text))
        except ValueError:
            key = sum(map(ord, text))
        rgb[i, j] = EDIT_PALETTE[key % len(EDIT_PALETTE)][:3]
    ax.imshow(rgb, aspect="auto", interpolation="nearest", extent=(-0.5, values.shape[1] - 0.5, len(rows) - 0.5, -0.5))
    step = max(1, values.shape[1] // 15)
    ax.set_xticks(range(0, values.shape[1], step))
    ax.set_xticklabels(list(sub.columns[::step]), rotation=90, fontsize=6)
    ax.set_yticks([])
    ax.set_xlabel("recording sites (character matrix columns)")


def legend(fig, handles, y=0.005, ncol=8):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=7, bbox_to_anchor=(0.5, y))


def patches(colours: dict, prefix=""):
    return [Patch(facecolor=c, edgecolor="#bdbdbd", label=f"{prefix}{short(k)}") for k, c in colours.items()]


def save(fig, out: Path, stem: str, index: list, **info):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=220 if ext == "png" else 300, bbox_inches="tight")
    plt.close(fig)
    index.append({"figure": f"figures/{stem}.png", **info})


# --- data ---------------------------------------------------------------------------------

def read_matrix(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    matrix = pd.read_csv(path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    matrix.index = matrix.index.astype(str)
    return matrix


def site_colours(sites: pd.Series, primary) -> dict:
    values = sorted(s for s in pd.Series(sites).dropna().unique() if s != "unmapped")
    prim = [s for s in values if primary(s)]
    mets = [s for s in values if not primary(s)]
    out = {s: PRIMARY_PALETTE[i % len(PRIMARY_PALETTE)] for i, s in enumerate(prim)}
    out.update({s: SITE_PALETTE[i % len(SITE_PALETTE)] for i, s in enumerate(mets)})
    if "unmapped" in set(sites):
        out["unmapped"] = "#ffffff"
    return out


def short(site) -> str:
    """3508_Apc_N4 -> N4 (mouse and genotype are in the title)."""
    return re.sub(r"^\d+_[A-Za-z0-9]+_", "", str(site))


def ot_tables(data: Path, block: str, pair: str, method="M4E") -> dict:
    out = {}
    for strategy in lv.STRATEGIES:
        states = data / "ot" / "cell_states" / method / strategy / block / f"{pair}.csv.gz"
        match = data / "ot" / "matching" / method / strategy / block / f"{pair}.csv.gz"
        out[strategy] = {"states": pd.read_csv(states, dtype={"cell_id": str}) if states.exists() else None,
                         "matching": pd.read_csv(match, dtype={"cell_id": str, "predicted_group": str, "second_group": str, "top_source_cell": str}, keep_default_na=False) if match.exists() else None}
    return out


def build_leaf_table(tree, leaf_to_cell: dict, cell_info: pd.DataFrame, matrix, ot: dict, umap: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for k, v in enumerate(tree.leaves):
        leaf = tree.name[v]
        rows.append({"leaf_order": k, "node": v, "leaf": leaf, "cell_id": leaf_to_cell.get(leaf, "")})
    table = pd.DataFrame(rows)
    table = table.merge(cell_info, on="cell_id", how="left")
    table["site"] = table["site"].fillna("unmapped")
    table["kind"] = table["kind"].fillna("unknown")
    if matrix is not None:
        values = matrix.reindex(table["leaf"]).to_numpy(dtype=str)
        present = table["leaf"].isin(matrix.index).to_numpy()
        table["in_matrix"] = present
        table["missing_share"] = np.where(present, np.isin(values, list(MISSING)).mean(axis=1), np.nan)
    else:
        table["in_matrix"] = False
        table["missing_share"] = np.nan
    for strategy, parts in ot.items():
        states = parts["states"]
        if states is not None:
            st = dict(zip(states["cell_id"], states["state"]))
            table[f"{strategy}__state"] = [st.get(c, "not in run") for c in table["cell_id"]]
        else:
            table[f"{strategy}__state"] = "not in run"
        match = parts["matching"]
        if match is not None:
            pg = dict(zip(match["cell_id"], match["predicted_group"]))
            pw = dict(zip(match["cell_id"], pd.to_numeric(match["predicted_weight"], errors="coerce")))
            ts = dict(zip(match["cell_id"], match["top_source_cell"]))
            table[f"{strategy}__predicted_group"] = [pg.get(c, "") for c in table["cell_id"]]
            table[f"{strategy}__predicted_weight"] = [pw.get(c, np.nan) for c in table["cell_id"]]
            table[f"{strategy}__top_source_cell"] = [ts.get(c, "") for c in table["cell_id"]]
    if umap is not None:
        u = umap.drop_duplicates("observation_id").set_index("observation_id")
        table["umap_1"] = table["cell_id"].map(u["umap_1"])
        table["umap_2"] = table["cell_id"].map(u["umap_2"])
    return table


# --- figures ------------------------------------------------------------------------------

def figure_matrix(name, title, tree, table, matrix, colours, lineage_colours, out, index, info):
    rows = table["leaf"].tolist()
    fig = plt.figure(figsize=(10, 9))
    grid = fig.add_gridspec(1, 4, width_ratios=[0.25, 0.25, 0.25, 9], wspace=0.04)
    strip(fig.add_subplot(grid[0]), table["kind"], KIND, "primary / met")
    strip(fig.add_subplot(grid[1]), table["site"], colours, "site")
    strip(fig.add_subplot(grid[2]), table["lineage"].fillna(""), lineage_colours, "family / clone")
    ax = fig.add_subplot(grid[3])
    matrix_image(ax, matrix, rows)
    ax.set_title(f"{title}\nrows = {len(rows)} cells in the tree's leaf order; colour = edit state "
                 f"(same colour = same state id within a column); light grey = unedited (0); white = missing", fontsize=8)
    legend(fig, patches(KIND) + patches(colours) + patches(lineage_colours, "family/clone: "), y=-0.04)
    save(fig, out, f"{name}_A_matrix", index, **info, panel="A matrix")


def figure_tree(name, title, tree, table, colours, branches, out, index, info, focus_label):
    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2 + 2 * len(branches), width_ratios=[6, 0.35] + [3, 0.35] * len(branches),
                            height_ratios=[20, 1], wspace=0.06)
    whole = Drawn(tree, tree.root)
    ax = fig.add_subplot(grid[0, 0]); whole.draw(ax, lw=0.25)
    ax.set_title(f"whole tree: {len(whole.leaves)} leaves", fontsize=8)
    fig.suptitle(f"{title}: the authors' tree, leaves coloured by site", fontsize=10, y=0.99)
    order_index = {v: i for i, v in enumerate(whole.leaves)}
    for b, v in enumerate(branches):
        part = Drawn(tree, v)
        lo, hi = order_index[part.leaves[0]], order_index[part.leaves[-1]]
        ax.add_patch(Rectangle((-0.5, lo - 0.5), max(whole.x.values()) + 1, hi - lo + 1, fill=False, edgecolor="#ff7f00", lw=1))
        ax.text(0.3, lo + 0.01 * len(whole.leaves), f"zoom {b + 1}", color="#ff7f00", fontsize=8, va="top", fontweight="bold")
    strip(fig.add_subplot(grid[0, 1]), table["site"], colours, "site")
    for b, v in enumerate(branches):
        part = Drawn(tree, v)
        sub = table.set_index("node").loc[part.leaves]
        axz = fig.add_subplot(grid[0, 2 + 2 * b]); part.draw(axz, lw=0.5)
        counts = sub["site"].value_counts()
        text = ", ".join(f"{short(k)} {v}" for k, v in counts.items())
        axz.set_title(f"zoom {b + 1}: {len(part.leaves)} leaves\n" + "\n".join(textwrap.wrap(text, 48)), fontsize=7)
        strip(fig.add_subplot(grid[0, 3 + 2 * b]), sub["site"], colours, "site")
    fig.text(0.5, 0.06, f"zoomed branches: clades of 80-400 leaves with >= 2 sites, the two with most {focus_label} leaves "
             "that do not overlap; the authors' tree, not re-inferred", ha="center", fontsize=7)
    legend(fig, patches(colours), y=0.0)
    save(fig, out, f"{name}_A_tree", index, **info, panel="A tree")


def figure_tree_matrix(name, title, tree, table, matrix, colours, branches, out, index, info):
    views = [tree.root] + list(branches)
    fig = plt.figure(figsize=(14, 5.5 * len(views)))
    outer = fig.add_gridspec(len(views), 1, hspace=0.25)
    for r, v in enumerate(views):
        part = Drawn(tree, v)
        sub = table.set_index("node").loc[part.leaves].reset_index()
        g = outer[r].subgridspec(1, 4, width_ratios=[4, 0.3, 0.3, 8], wspace=0.04)
        ax = fig.add_subplot(g[0]); part.draw(ax, lw=0.25 if r == 0 else 0.5)
        ax.set_title(("whole tree" if r == 0 else f"zoom {r}") + f": {len(part.leaves)} leaves", fontsize=8)
        strip(fig.add_subplot(g[1]), sub["kind"], KIND, "primary / met")
        strip(fig.add_subplot(g[2]), sub["site"], colours, "site")
        axm = fig.add_subplot(g[3]); matrix_image(axm, matrix, sub["leaf"].tolist())
    fig.suptitle(f"{title}: tree and character matrix share one row order (leaf order); white = missing, light grey = unedited",
                 fontsize=9, y=0.995)
    legend(fig, patches(KIND) + patches(colours), y=-0.01)
    save(fig, out, f"{name}_A_tree_matrix", index, **info, panel="A tree + matrix")


def figure_rejection(name, title, tree, table, colours, run_label, out, index, info):
    part = Drawn(tree, tree.root)
    fig = plt.figure(figsize=(12, 9))
    n = len(lv.STRATEGIES)
    grid = fig.add_gridspec(1, 3 + n, width_ratios=[6, 0.35, 0.35] + [0.45] * n, wspace=0.05)
    ax = fig.add_subplot(grid[0]); part.draw(ax, lw=0.25)
    strip(fig.add_subplot(grid[1]), table["kind"], KIND, "primary / met")
    strip(fig.add_subplot(grid[2]), table["site"], colours, "site")
    for k, strategy in enumerate(lv.STRATEGIES):
        strip(fig.add_subplot(grid[3 + k]), table[f"{strategy}__state"], STATE, NAMES[strategy])
    tallies = []
    for strategy in lv.STRATEGIES:
        c = table[f"{strategy}__state"].value_counts()
        tallies.append(f"{NAMES[strategy]}: " + ", ".join(f"{s} {int(c.get(s, 0))}" for s in STATE))
    fig.suptitle(f"{title}: M4-E gate of {run_label} on the CRISPR tree", fontsize=10, y=0.99)
    fig.text(0.5, -0.03, "\n".join(tallies), ha="center", va="top", fontsize=6.5)
    legend(fig, patches(KIND) + patches(colours) + patches(STATE), y=-0.14)
    save(fig, out, f"{name}_B_tree_ot_rejection", index, **info, panel="B tree + OT rejection")


def figure_coupling(name, title, tree, table, colours, lineage_colours, group_colours, run_label, target_mask, truth_label,
                    out, index, info):
    part = Drawn(tree, tree.root)
    fig = plt.figure(figsize=(14, 9))
    n = len(lv.STRATEGIES)
    grid = fig.add_gridspec(1, 4 + 2 * n, width_ratios=[6, 0.35, 0.35, 0.35] + [0.45, 0.3] * n, wspace=0.05)
    ax = fig.add_subplot(grid[0]); part.draw(ax, lw=0.25)
    strip(fig.add_subplot(grid[1]), table["kind"], KIND, "primary / met")
    strip(fig.add_subplot(grid[2]), table["site"], colours, "site")
    strip(fig.add_subplot(grid[3]), table["lineage"].fillna(""), lineage_colours, f"CRISPR / author\n{truth_label}")
    summary = []
    for k, strategy in enumerate(lv.STRATEGIES):
        key = f"{strategy}__predicted_group"
        if key not in table:
            continue
        pred = np.where(target_mask, table[key].replace("", "no prediction"), "")
        strip(fig.add_subplot(grid[4 + 2 * k]), pred, {**group_colours, "no prediction": "#f0f0f0"}, f"OT predicted\n({NAMES[strategy]})")
        weight = np.where(target_mask, table[f"{strategy}__predicted_weight"], np.nan)
        continuous_strip(fig.add_subplot(grid[5 + 2 * k]), weight, "weight", cmap="Greys")
        has = target_mask & (table[key] != "")
        agree = (table.loc[has, key] == table.loc[has, "lineage_group"]).sum()
        summary.append(f"{NAMES[strategy]}: {int(has.sum())} target leaves with a prediction, {int(agree)} equal to the CRISPR-supported group")
    fig.suptitle(f"{title}: OT-predicted source group of {run_label}\n"
                 "(an expression-based association drawn beside the tree, not a tree branch)", fontsize=10, y=1.0)
    fig.text(0.5, -0.03, "\n".join(summary), ha="center", va="top", fontsize=6.5)
    legend(fig, patches(KIND) + patches(colours) + patches(group_colours, "OT: "), y=-0.14)
    save(fig, out, f"{name}_C_tree_ot_coupling", index, **info, panel="C tree + OT coupling")


def figure_umap(name, title, umap: pd.DataFrame, cells: pd.DataFrame, colours, lineage_colours, group_colours, ot, out, index, info):
    u = umap.copy()
    u["site"] = u["observation_id"].map(cells.set_index("cell_id")["site"]).fillna("?")
    u["lineage"] = u["observation_id"].map(cells.set_index("cell_id")["lineage"]).fillna("")
    panels = [("site", u["site"], colours), ("clone (LineageGroup)" if "Quinn" in title else "family (MetFamily)", u["lineage"], lineage_colours)]
    for strategy in lv.STRATEGIES:
        st = ot[strategy]["states"]
        if st is not None:
            key = st["side"] + ":" + st["cell_id"]
            state = pd.Series(st["state"].to_numpy(), index=key)
            values = (u["side"] + ":" + u["observation_id"]).map(state).fillna("not in run")
            panels.append((f"{NAMES[strategy]} gate", values, STATE))
    m = ot["rna"]["matching"]
    if m is not None:
        pred = dict(zip(m["cell_id"], m["predicted_group"]))
        values = np.where(u["side"].eq("target"), [pred.get(c, "") or "no prediction" for c in u["observation_id"]], "source")
        panels.append(("OT predicted source group (RNA only)", values, {**group_colours, "no prediction": "#bdbdbd", "source": "#f0f0f0"}))
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    x, y = u["umap_1"].to_numpy(float), u["umap_2"].to_numpy(float)
    size = max(0.6, min(6.0, 8000 / max(len(u), 1)))
    for ax, (label, values, cmap) in zip(axes.ravel(), panels):
        values = np.asarray(values, dtype=object)
        ax.scatter(x, y, s=size * 0.5, c="#f2f2f2", linewidths=0, rasterized=True)
        for cat, colour in cmap.items():
            hit = values == cat
            ax.scatter(x[hit], y[hit], s=size, c=[colour], linewidths=0, rasterized=True, label=f"{cat} ({int(hit.sum())})")
        ax.set_title(label, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        if len(cmap) <= 18:
            ax.legend(fontsize=5, frameon=False, markerscale=2, loc="best")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{title}: fixed joint UMAP (48_reference_umap.py, counts only), same coordinates in every panel", fontsize=9, y=1.02)
    fig.tight_layout()
    save(fig, out, f"{name}_D_umap", index, **info, panel="D UMAP")


# --- per dataset --------------------------------------------------------------------------

def kp_mouse(data: Path, mouse: str, out: Path, ann: Path, index: list) -> None:
    d1 = lv.tree_module()
    stems = sorted(p.name[: -len("_tree.nwk")] for p in (data / "lineage" / "KPTracer").glob(f"{mouse}_*_All_tree.nwk"))
    if not stems:
        print(f"KP {mouse}: no _All tree"); return
    stem = stems[0]
    tree = d1.Tree(data / "lineage" / "KPTracer" / f"{stem}_tree.nwk")
    matrix = read_matrix(data / "lineage" / "KPTracer" / f"{stem}_character_matrix.txt")
    meta = pd.read_csv(data / "meta" / f"KPTracer_meta_{mouse}.csv.gz", dtype=str)
    primary = lambda s: bool(re.search(r"(^|_)T\d+$", str(s)))  # README rule: the tumour id is T<n>
    info_cells = pd.DataFrame({"cell_id": meta["cell_id"], "site": meta["Tumor"],
                               "kind": ["primary" if primary(t) else "metastasis" for t in meta["Tumor"]],
                               "lineage": meta["MetFamily"], "is_normal": meta.get("is_normal")})
    pair = f"{mouse}__D_C3__primaries__metastases"
    ot = ot_tables(data, "kp_D_C3", pair)
    umap_path = data / "umap" / "kp_D_C3" / pair / "reference_umap.csv.gz"
    umap = pd.read_csv(umap_path, dtype={"observation_id": str}) if umap_path.exists() else None
    leaf_to_cell = {tree.name[v]: tree.name[v] for v in tree.leaves}
    table = build_leaf_table(tree, leaf_to_cell, info_cells, matrix, ot, umap)
    # The OT groups are source tumours; the family of a primary tumour is named after it.
    family_of_primary = info_cells[info_cells["kind"].eq("primary")].drop_duplicates("site").set_index("site")["lineage"]
    for strategy in lv.STRATEGIES:
        key = f"{strategy}__predicted_group"
        if key in table:
            table[f"{strategy}__predicted_family"] = table[key].map(family_of_primary).fillna("")
    sites = site_colours(table["site"], primary)
    # A family is named after its primary tumour, so it takes that primary's colour; a family with no
    # primary on this tree (or _Unk) gets grey.
    families = {f: sites.get(re.sub(r"_Fam$", "", str(f)), "#7f7f7f") for f in sorted(table["lineage"].dropna().unique())}
    primaries = sorted(s for s in table["site"].unique() if primary(s))
    groups = {p: sites[p] for p in primaries}
    ann.mkdir(parents=True, exist_ok=True)
    table.drop(columns=["node"]).to_csv(ann / f"KPTracer_{mouse}_{stem}_cells.csv", index=False)
    mapped = int((table["cell_id"].isin(meta["cell_id"])).sum())
    info = {"dataset": "KPTracer", "mouse": mouse, "tree": f"{stem}_tree.nwk", "leaves": len(table),
            "leaves_with_metadata": mapped, "leaves_in_matrix": int(table["in_matrix"].sum()),
            "matrix_missing_share": float(np.nanmean(table["missing_share"])) if matrix is not None else np.nan,
            "ot_run": pair}
    title = f"KP-Tracer mouse {mouse}, {stem}"
    table["met"] = table["kind"].eq("metastasis")
    branches = pick_branches(tree, table, "met")
    figure_matrix(stem, title, tree, table, matrix, sites, families, out, index, info)
    figure_tree(stem, title, tree, table, sites, branches, out, index, info, "metastatic")
    figure_tree_matrix(stem, title, tree, table, matrix, sites, branches, out, index, info)
    if all(ot[s]["states"] is not None for s in ot):
        figure_rejection(stem, title, tree, table, sites, "D_C3 (pooled primaries -> pooled metastases)", out, index, info)
        # CRISPR-supported group of a metastatic leaf: the primary tumour its family is named after.
        primary_of_family = {v: k for k, v in family_of_primary.items()}
        table["lineage_group"] = table["lineage"].map(primary_of_family).fillna("")
        figure_coupling(stem, title, tree, table, sites, families, groups, "D_C3", table["met"].to_numpy(),
                        "family (MetFamily)", out, index, info)
    if umap is not None:
        figure_umap(f"KPTracer_{mouse}", f"KP-Tracer mouse {mouse}, D_C3 cells", umap, info_cells, sites, families, groups, ot, out, index, info)
    print(f"KP {mouse}: {len(table)} leaves, {len(branches)} zoomed branches")


def quinn_mouse(data: Path, mouse: str, clones: list[str], out: Path, ann: Path, index: list) -> None:
    d1 = lv.tree_module()
    trees = pd.read_csv(data / "labels" / "trees.tsv", sep="\t", dtype=str, keep_default_na=False)
    leaves = pd.read_csv(data / "labels" / "tree_leaves.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    labels = pd.read_csv(data / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    labels = labels[(labels["dataset"] == "Quinn2021") & (labels["mouse"] == mouse)]
    primary = lambda s: re.sub(r"^\d+k", "", str(s)) == "LL"
    cells = pd.DataFrame({"cell_id": labels["cell"], "site": labels["site"],
                          "kind": ["primary" if primary(s) else "metastasis" for s in labels["site"]],
                          "lineage": labels["lineage"]})
    cells["lineage_group"] = cells["lineage"]
    pair = f"{mouse}__D_B__LL__metastases"
    ot = ot_tables(data, "quinn_D_B", pair)
    umap_path = data / "umap" / "quinn_D_B" / pair / "reference_umap.csv.gz"
    umap = pd.read_csv(umap_path, dtype={"observation_id": str}) if umap_path.exists() else None
    sites = site_colours(cells["site"], primary)
    clone_order = cells["lineage"].value_counts().index.tolist()
    # Clones get their own palette (tab20, the 20 largest; the rest grey), distinct from the site colours.
    clone_colours = {c: plt.get_cmap("tab20").colors[i] for i, c in enumerate(clone_order[:20])}
    clone_colours.update({c: "#9e9e9e" for c in clone_order[20:]})
    if umap is not None:
        figure_umap(f"Quinn2021_{mouse}", f"Quinn 2021 mouse {mouse}, LL -> pooled tissues", umap, cells, sites,
                    clone_colours, clone_colours, ot, out, index, {"dataset": "Quinn2021", "mouse": mouse, "ot_run": pair})
    for clone in clones:
        row = trees[(trees["dataset"] == "Quinn2021") & (trees["mouse"] == mouse) & (trees["lineage"] == clone)]
        if row.empty or row.iloc[0]["status"] != "tree":
            print(f"Quinn {mouse} LG{clone}: no tree ({'' if row.empty else row.iloc[0]['status']})"); continue
        rel = re.split(r"/lineage/" + re.escape(mouse) + "/", row.iloc[0]["tree"])[-1]
        tree_path = data / "lineage" / "Quinn2021" / mouse / rel
        tree = d1.Tree(tree_path)
        candidates = sorted(p for p in (data / "lineage" / "Quinn2021" / mouse / "character_matrices").rglob("*")
                            if p.is_file() and re.search(rf"(?i)lg{clone}[_.]", p.name))
        preferred = [p for p in candidates if "alleleThresh" in p.name] or candidates
        matrix = read_matrix(preferred[0]) if len(preferred) == 1 else None
        mapping = leaves[(leaves["mouse"] == mouse) & (leaves["lineage"] == clone)]
        leaf_to_cell = dict(zip(mapping["leaf"], mapping["cell"]))
        table = build_leaf_table(tree, leaf_to_cell, cells, matrix, ot, umap)
        name = f"Quinn2021_{mouse}_LG{clone}"
        ann.mkdir(parents=True, exist_ok=True)
        table.drop(columns=["node"]).to_csv(ann / f"{name}_cells.csv", index=False)
        info = {"dataset": "Quinn2021", "mouse": mouse, "tree": tree_path.name, "leaves": len(table),
                "leaves_with_metadata": int((table["cell_id"] != "").sum()), "leaves_in_matrix": int(table["in_matrix"].sum()),
                "matrix": preferred[0].name if len(preferred) == 1 else f"{len(candidates)} candidates",
                "matrix_missing_share": float(np.nanmean(table["missing_share"])) if matrix is not None else np.nan, "ot_run": pair}
        title = f"Quinn 2021 mouse {mouse}, LineageGroup {clone}"
        table["met"] = table["kind"].eq("metastasis")
        branches = pick_branches(tree, table, "met")
        lineage_colours = {clone: clone_colours.get(clone, "#636363")}
        figure_matrix(name, title, tree, table, matrix, sites, lineage_colours, out, index, info)
        figure_tree(name, title, tree, table, sites, branches, out, index, info, "non-LL")
        figure_tree_matrix(name, title, tree, table, matrix, sites, branches, out, index, info)
        if all(ot[s]["states"] is not None for s in ot):
            figure_rejection(name, title, tree, table, sites, "D_B (LL -> pooled tissues)", out, index, info)
            figure_coupling(name, title, tree, table, sites, lineage_colours, clone_colours, "D_B", table["met"].to_numpy(),
                            "clone (LineageGroup)", out, index, info)
        print(f"Quinn {mouse} LG{clone}: {len(table)} leaves, matrix {info['matrix']}, {len(branches)} zoomed branches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kp-mice", nargs="*", default=["3508", "3515"])
    parser.add_argument("--quinn", nargs="*", default=["10k:1,2"], help="MOUSE:CLONE,CLONE")
    parser.add_argument("--out", type=Path, default=HERE / "figures")
    parser.add_argument("--annotations", type=Path, default=HERE / "annotations")
    args = parser.parse_args()
    index: list = []
    for mouse in args.kp_mice:
        kp_mouse(args.data, mouse, args.out, args.annotations, index)
    for spec in args.quinn:
        mouse, clones = spec.split(":")
        quinn_mouse(args.data, mouse, clones.split(","), args.out, args.annotations, index)
    pd.DataFrame(index).to_csv(HERE / "figures_index.tsv", sep="\t", index=False, lineterminator="\n")
    print(f"{len(index)} figures")


if __name__ == "__main__":
    main()

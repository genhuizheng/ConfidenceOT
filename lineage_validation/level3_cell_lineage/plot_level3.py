"""Level 3 figures (PLAN.md, Figures; the selection rules are fixed there, nothing is picked by eye).

    python lineage_validation/level3_cell_lineage/plot_level3.py --runs LEVEL2_ROOT --scores SCORES --summary SUMMARY --out FIGDIR

A  one file per family with evaluable cells and per strategy: the clade of the frozen level-1 subclone holding the
   most evaluable metastatic cells, at most 40 primary and 40 metastatic leaves thinned every n-th in leaf order,
   and each drawn metastatic leaf linked to its top-1 OT primary (comparable sources, q~) under A (left) and B (right);
   the link colour is the tree distance. A top-1 primary outside the thinned set is added and drawn hollow.
B  per family, mean delta_D_norm with its 95% bootstrap interval, A and B side by side, one panel per strategy;
   mouse means as diamonds.
C  per strategy and method, the evaluable cells at the 90th and 10th percentile of delta_D_norm (pooled; ties to the
   smaller cell id): mass on sources at each tree distance, OT (q~) against independent (pi~), with the number of
   comparable sources at each distance, annotated with d_min, top-1 distance and weight, c_ot and delta_D_norm.
Couplings are read one at a time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_core as c  # noqa: E402
from ground_truth import load_cells  # noqa: E402

COL = {"A_standard_OT": "#2166ac", "B_our_OT": "#d6604d"}  # blue / orange-red: separable under common CVD
LAB = {"A_standard_OT": "A  Standard OT (BOT)", "B_our_OT": "B  Our OT (UOT, budget 0)"}
STRAT = {"rna": "RNA-only", "cnv": "CNV-only", "rna_cnv": "RNA+CNV"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def coupling(runs: Path, strategy: str, case_id: str, method: str):
    with np.load(runs / c.STRATEGY_FOLDERS[strategy] / case_id / method / "coupling.npz") as z:
        P = z["coupling"]
    return P


def short(fam: str) -> str:
    return fam.replace("KPTracer_", "KP ").replace("Quinn2021_", "Quinn ").replace("_", " ")


# ---------------------------------------------------------------- Figure A
def clade_leaves(ft: c.FamilyTree, root: int) -> set:
    out, stack = set(), [root]
    while stack:
        v = stack.pop()
        kids = ft.tree.children[v]
        if not kids:
            out.add(v)
        stack.extend(kids)
    return out


def lca(ft: c.FamilyTree, nodes) -> int:
    paths = []
    for v in nodes:
        p = []
        while v >= 0:
            p.append(v); v = ft.parent[v]
        paths.append(p[::-1])
    k = 0
    while all(len(p) > k for p in paths) and len({p[k] for p in paths}) == 1:
        k += 1
    return paths[0][k - 1]


def thin(items: list, cap: int) -> list:
    if len(items) <= cap:
        return items
    step = int(np.ceil(len(items) / cap))
    return items[::step]


def draw_tree(ax, ft: c.FamilyTree, root: int, leaves: list, y_of: dict):
    keep = set()
    for v in leaves:
        while v >= 0 and v not in keep:
            keep.add(v)
            if v == root:
                break
            v = ft.parent[v]
    x = {v: float(ft.depth[v] - ft.depth[root]) for v in keep}
    y = dict(y_of)
    for v in reversed(ft.tree.order):
        if v in keep and v not in y:
            kids = [k for k in ft.tree.children[v] if k in keep]
            y[v] = float(np.mean([y[k] for k in kids])) if kids else 0.0
    for v in keep:
        if v == root:
            continue
        p = ft.parent[v]
        while p not in keep:
            p = ft.parent[p]
        ax.plot([x[p], x[p]], [y[p], y[v]], color="#9a9a9a", lw=0.5)
        ax.plot([x[p], x[v]], [y[v], y[v]], color="#9a9a9a", lw=0.5)
    return x, y


def figure_a(runs: Path, scores: pd.DataFrame, cases: pd.DataFrame, trees: dict, out: Path):
    for _, case in cases.iterrows():
        cells = load_cells(runs, case["case_id"])
        fams, allt, n_src = c.ground_truth(case, cells, trees)
        sel = []
        for fam, (ft, s_rows, ftab) in fams.items():
            fev = ftab[ftab["status"].eq("evaluable")]
            l1 = fev[fev["L1"].astype(str).str.startswith("L")]["L1"].value_counts() if len(fev) else pd.Series(dtype=int)
            if l1.empty:
                continue
            sub = l1.index[0]  # the level-1 subclone with the most evaluable metastatic cells
            table = c.family_cells(fam)
            members = [ft.leaf_of[x] for x in table[table["L1"].eq(sub)]["leaf"] if x in ft.leaf_of]
            root = lca(ft, members)
            in_clade = clade_leaves(ft, root)
            s_nodes = ft.s_nodes  # comparable sources in s_rows order
            prim = sorted([k for k, v in enumerate(s_nodes) if v in in_clade], key=lambda k: ft.pos[s_nodes[k]])
            mets = fev[fev["target_node"].astype(int).isin(in_clade)].copy()
            mets = mets.sort_values("target_node", key=lambda s: ft.pos[s.astype(int).to_numpy()])
            if mets.empty:
                continue
            sel.append(dict(fam=fam, ft=ft, s_rows=s_rows, sub=sub, root=root, in_clade=in_clade, prim=prim, mets=mets,
                            prim_show=thin(prim, 40), met_show=thin(list(mets.index), 40)))
        for strategy in c.STRATEGY_FOLDERS:
            tops = {s["fam"]: {} for s in sel}
            for method in c.METHODS:  # one coupling in memory, shared by every family of the case
                P = coupling(runs, strategy, case["case_id"], method)
                for s in sel:
                    t = {}
                    for idx in s["met_show"]:
                        j = int(s["mets"].at[idx, "target_col"])
                        q = P[s["s_rows"], j].astype(float)
                        k = int(np.argmax(q))
                        t[idx] = (k, float(q[k] / q.sum()), int(s["ft"].distances(int(s["mets"].at[idx, "target_node"]))[k]))
                    tops[s["fam"]][method] = t
                del P
            for s in sel:
                draw_a(s, tops[s["fam"]], strategy, out)


def draw_a(s: dict, tops: dict, strategy: str, out: Path):
    ft, root, mets, s_nodes = s["ft"], s["root"], s["mets"], s["ft"].s_nodes
    prim_show, met_show = s["prim_show"], s["met_show"]
    tops_k = {k for t in tops.values() for (k, _, _) in t.values()}
    inside = {k for k in tops_k if s_nodes[k] in s["in_clade"]}
    extra = sorted(inside - set(prim_show), key=lambda k: ft.pos[s_nodes[k]])
    leaf_list = sorted([("p", k) for k in prim_show + extra] + [("m", i) for i in met_show],
                       key=lambda e: ft.pos[s_nodes[e[1]]] if e[0] == "p" else ft.pos[int(mets.at[e[1], "target_node"])])
    y_of, node_of = {}, {}
    for yy, (kind, key) in enumerate(leaf_list):
        node = int(s_nodes[key]) if kind == "p" else int(mets.at[key, "target_node"])
        y_of[node] = float(yy); node_of[(kind, key)] = node
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.0, 0.13 * len(leaf_list) + 1.8)), sharey=True)
    dmax = max(1, max(d for t in tops.values() for (_, _, d) in t.values()))
    cmap = plt.get_cmap("viridis_r")
    y_out = -2.0  # where links to a top-1 primary outside the clade end
    for ax, method in zip(axes, c.METHODS):
        x, y = draw_tree(ax, ft, root, list(y_of), y_of)
        xl = max(x.values()) + 0.6
        for (kind, key), node in node_of.items():
            if kind == "p":
                ax.scatter(x[node], y[node], s=14, marker="o", facecolor="white" if key in extra else "#4d4d4d",
                           edgecolor="#4d4d4d", lw=0.6, zorder=3)
            else:
                ax.scatter(x[node], y[node], s=18, marker="^", color="#1b7837", zorder=3)
        n_out = 0
        for idx, (k, w, d) in tops[method].items():
            y0 = y[node_of[("m", idx)]]
            if ("p", k) in node_of:
                y1 = y[node_of[("p", k)]]
            else:
                y1 = y_out; n_out += 1
            xs = xl + 0.5 * abs(y1 - y0) ** 0.5 * np.sin(np.linspace(0, np.pi, 30))
            ax.plot(xs, np.linspace(y0, y1, 30), color=cmap(d / dmax), lw=0.4 + 1.6 * w, alpha=0.85)
        ax.scatter([xl], [y_out], s=30, marker="s", color="#b2182b", zorder=3)
        ax.text(xl + 0.3, y_out, "top-1 primary outside this clade", fontsize=6, va="center", color="#b2182b")
        hit = np.mean([d == int(mets.at[idx, "d_min"]) for idx, (k, w, d) in tops[method].items()])
        ax.set_title(f"{LAB[method]}\ntop-1 in the tree-nearest set: {hit:.0%}\ntop-1 outside the clade: {n_out} "
                     f"of {len(tops[method])} drawn metastatic cells", fontsize=8, color=COL[method])
        ax.set_xlabel("contracted tree depth below the clade root")
        ax.set_yticks([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, dmax))
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01, label="tree distance of the top-1 OT primary")
    handles = [Line2D([], [], marker="o", ls="", color="#4d4d4d", label="primary leaf"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", color="#4d4d4d", label="primary leaf added (a top-1 outside the thinned set)"),
               Line2D([], [], marker="^", ls="", color="#1b7837", label="metastatic leaf (evaluable)")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, frameon=False, bbox_to_anchor=(0, -0.08))
    fig.suptitle(f"{short(s['fam'])}, clade of level-1 subclone {s['sub']} (frozen definition): {len(s['prim'])} primary and {len(mets)} evaluable "
                 f"metastatic leaves, {len(prim_show)} + {len(met_show)} drawn (every n-th in leaf order); {STRAT[strategy]}", fontsize=9)
    fig.savefig(out / f"A_lineage_matching_{strategy}_{s['fam']}.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure B
def figure_b(summary: Path, out: Path):
    fam = pd.read_csv(summary / "family_summary.tsv", sep="\t")
    mouse = pd.read_csv(summary / "mouse_summary.tsv", sep="\t")
    order = fam.drop_duplicates("family").sort_values(["dataset", "mouse", "family"])["family"].tolist()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for ax, strategy in zip(axes, c.STRATEGY_FOLDERS):
        for off, method in ((-0.17, "A_standard_OT"), (0.17, "B_our_OT")):
            g = fam[fam["strategy"].eq(strategy) & fam["method"].eq(method)].set_index("family").reindex(order)
            xs = np.arange(len(order)) + off
            ax.errorbar(xs, g["delta_D_norm"], yerr=[g["delta_D_norm"] - g["delta_D_norm_lo"], g["delta_D_norm_hi"] - g["delta_D_norm"]],
                        fmt="o", ms=3.5, color=COL[method], lw=0.8, capsize=0, label=LAB[method])
            mm = mouse[mouse["strategy"].eq(strategy) & mouse["method"].eq(method)]
            for _, r in mm.iterrows():
                idx = [i for i, f in enumerate(order) if f.split("_")[1] == str(r["mouse"]) and f.split("_")[0] == r["dataset"]]
                ax.plot([min(idx) - 0.4 + off * 0.3, max(idx) + 0.4 + off * 0.3], [r["delta_D_norm"]] * 2, color=COL[method], lw=1.6, alpha=0.5)
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_ylabel("ΔD normalised\n(0 = independent, 1 = nearest relatives)")
        ax.set_title(STRAT[strategy], loc="left", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False, loc="upper right")
    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels([short(f) for f in order], rotation=60, ha="right", fontsize=7)
    fig.suptitle("Weighted CRISPR tree distance: OT against the independent coupling with the same marginals, per family "
                 "(mean over evaluable metastatic cells, 95% bootstrap interval; bars = mouse means)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "B_ot_vs_independent.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure C
def figure_c(runs: Path, scores: pd.DataFrame, cases: pd.DataFrame, trees: dict, out: Path):
    ev = scores[scores["status"].eq("evaluable")].dropna(subset=["delta_D_norm"])
    picks = []
    for strategy in c.STRATEGY_FOLDERS:
        for method in c.METHODS:
            g = ev[ev["strategy"].eq(strategy) & ev["method"].eq(method)].sort_values(["delta_D_norm", "cell"])
            for label, qtl in (("success (90th percentile)", 0.9), ("failure (10th percentile)", 0.1)):
                target = g["delta_D_norm"].quantile(qtl, interpolation="lower")
                row = g[g["delta_D_norm"].eq(target)].sort_values("cell").iloc[0]
                picks.append((strategy, method, label, row))
    gt_cache = {}
    for strategy in c.STRATEGY_FOLDERS:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for col, method in enumerate(c.METHODS):
            for rowi, label in enumerate(("success (90th percentile)", "failure (10th percentile)")):
                ax = axes[rowi, col]
                _, _, _, r = next(p for p in picks if p[0] == strategy and p[1] == method and p[2] == label)
                case = cases[cases["case_id"].eq(r["case_id"])].iloc[0]
                if r["case_id"] not in gt_cache:
                    gt_cache[r["case_id"]] = c.ground_truth(case, load_cells(runs, r["case_id"]), trees)
                fams, allt, n_src = gt_cache[r["case_id"]]
                ft, s_rows, ftab = fams[r["family"]]
                t = ftab[ftab["cell"].eq(r["cell"])].iloc[0]
                d = ft.distances(int(t["target_node"]))
                P = coupling(runs, strategy, r["case_id"], method)
                m = P.sum(1, dtype=np.float64)
                q = P[:, int(t["target_col"])].astype(np.float64)
                del P
                qt = q[s_rows] / q[s_rows].sum()
                pit = m[s_rows] / m[s_rows].sum()
                ds = np.arange(d.min(), d.max() + 1)
                mass_ot = np.array([qt[d == k].sum() for k in ds]); mass_ind = np.array([pit[d == k].sum() for k in ds])
                n_at = np.array([(d == k).sum() for k in ds])
                ax.bar(ds - 0.2, mass_ind, 0.4, color="#bdbdbd", label="independent coupling (π̃, same marginals)")
                ax.bar(ds + 0.2, mass_ot, 0.4, color=COL[method], label="OT (q̃)")
                for k, n in zip(ds, n_at):
                    if n:
                        ax.text(k, max(mass_ot.max(), mass_ind.max()) * 1.04, str(n), ha="center", fontsize=6, color="#555555")
                ax.axvline(d.min(), color="#1b7837", lw=0.8, ls="--")
                ax.set_xlabel("tree distance to the metastatic cell (numbers above: primaries at that distance)")
                ax.set_ylabel("share of the cell's matched mass")
                ax.set_title(f"{LAB[method]}: {label}\n{short(r['family'])}, site {r['site']}, cell {r['cell']}", fontsize=8, color=COL[method])
                ax.text(0.98, 0.72, f"d_min {int(d.min())} (dashed)\ntop-1: distance {r['top1_d']:.0f}, weight {r['top1_weight']:.3f}\n"
                        f"D_OT {r['D_ot']:.2f} vs D_ind {r['D_ind']:.2f}\nΔD_norm {r['delta_D_norm']:.2f}\n"
                        f"c_OT {r['c_ot']:.2f} (mass on comparable primaries)", transform=ax.transAxes, ha="right", fontsize=7)
                if rowi == 0 and col == 0:
                    ax.legend(fontsize=7, frameon=False, loc="upper left")
        fig.suptitle(f"{STRAT[strategy]}: cell-level examples chosen by rule (pooled evaluable cells, 90th and 10th percentile of ΔD_norm per method)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"C_examples_{strategy}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="ABC")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(c.L2_INPUTS / "cases.tsv", sep="\t", dtype=str)
    scores = pd.concat([pd.read_csv(p, sep="\t", low_memory=False) for p in sorted((args.scores / "cells").glob("*.tsv.gz"))
                        if not p.name.endswith("__targets.tsv.gz")], ignore_index=True)
    trees = c.tree_files()
    if "B" in args.only:
        figure_b(args.summary, args.out)
    if "C" in args.only:
        figure_c(args.runs, scores, cases, trees, args.out)
    if "A" in args.only:
        figure_a(args.runs, scores, cases, trees, args.out)


if __name__ == "__main__":
    main()

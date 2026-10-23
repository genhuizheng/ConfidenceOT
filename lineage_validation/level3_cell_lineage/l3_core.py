"""Shared pieces of Level 3 (PLAN.md): trees, leaf mapping, tree distances, per-cell status and matching quantities."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LV = HERE.parent
REPO = LV.parent
sys.path.insert(0, str(LV))
import lv_common as lv  # noqa: E402

DEFS = LV / "within_family_subclones" / "output"
L2_INPUTS = LV / "level2_pairwise_ot" / "inputs"
TREE_ROOTS = [REPO / "results" / "kp_tracer_lineage_validation" / "export" / "trees" / "cassiopeia",
              LV / "local_lineage_visualization" / "data" / "local_lineage_visualization_export" / "lineage" / "Quinn2021"]
STRATEGY_FOLDERS = {"rna": "full", "cnv": "full_cnv", "rna_cnv": "full_rna_cnv"}
METHODS = ("A_standard_OT", "B_our_OT")
MIN_SOURCES, AMBIGUOUS_SHARE, TOPK = 5, 0.5, (1, 5, 10)


def tree_files(roots=None) -> dict:
    found = {}
    for root in roots or TREE_ROOTS:
        for p in Path(root).rglob("*"):
            if p.is_file():
                found.setdefault(p.name, p)
    return found


class FamilyTree:
    """One authors' tree, with preorder positions, clade sizes and the contracted depth of the project Tree."""

    def __init__(self, path: Path):
        tree = lv.tree_module().Tree(Path(path))
        self.tree = tree
        n = len(tree.parent)
        self.parent = np.asarray(tree.parent)
        self.depth = np.asarray(tree.depth, dtype=np.int64)
        self.pos = np.full(n, -1, dtype=np.int64)
        self.pos[np.asarray(tree.order)] = np.arange(len(tree.order))
        self.size = np.ones(n, dtype=np.int64)
        for v in reversed(tree.order):
            for c in tree.children[v]:
                self.size[v] += self.size[c]
        self.leaf_of = tree.leaf_of

    def prepare(self, source_nodes: np.ndarray):
        """Sort the comparable sources by preorder position once; distances() then costs O(sources + depth)."""
        self.s_nodes = np.asarray(source_nodes, dtype=np.int64)
        self.s_order = np.argsort(self.pos[self.s_nodes], kind="stable")
        self.s_sorted_pos = self.pos[self.s_nodes][self.s_order]
        self.s_depth_sorted = self.depth[self.s_nodes][self.s_order]

    def distances(self, target_node: int) -> np.ndarray:
        """d(i, j) = depth(i) + depth(j) - 2 depth(LCA) for every prepared source i, in the order given to prepare()."""
        sp = self.s_sorted_pos
        lca = np.empty(len(sp), dtype=np.int64)
        v = int(target_node)
        a = int(np.searchsorted(sp, self.pos[v])); b = int(np.searchsorted(sp, self.pos[v] + self.size[v]))
        lca[a:b] = self.depth[v]  # only another cell on the same leaf (never: one leaf is one cell)
        v = self.parent[v]
        while v >= 0:
            A = int(np.searchsorted(sp, self.pos[v])); B = int(np.searchsorted(sp, self.pos[v] + self.size[v]))
            lca[A:a] = self.depth[v]
            lca[b:B] = self.depth[v]
            a, b = A, B
            v = self.parent[v]
        if a != 0 or b != len(sp):
            raise ValueError("a source leaf is not under the tree root")
        d_sorted = self.s_depth_sorted + self.depth[target_node] - 2 * lca
        out = np.empty_like(d_sorted)
        out[self.s_order] = d_sorted
        return out


def family_cells(name: str) -> pd.DataFrame:
    table = pd.read_csv(DEFS / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
    if table["cell_id"].duplicated().any():
        raise ValueError(f"{name}: duplicated cell ids")
    table["reliable"] = table["reliable"].eq("True")
    table["missing_share"] = pd.to_numeric(table["missing_share"], errors="coerce")
    return table.set_index("cell_id")


def family_tree_name(name: str) -> str:
    feas = pd.read_csv(DEFS / "feasibility.tsv", sep="\t", dtype=str)
    feas["name"] = feas["dataset"] + "_" + feas["mouse"] + "_" + feas["family_or_clone"]
    row = feas[feas["name"].eq(name)]
    if len(row) != 1:
        raise ValueError(f"{name}: {len(row)} rows in feasibility.tsv")
    return row["tree"].iloc[0]


def ground_truth(case: pd.Series, cells: pd.DataFrame, trees: dict):
    """Per family of the case: the tree prepared on its comparable sources, and every metastatic cell's status.

    cells: the Level 2 run's cells.tsv.gz (side, cell, site) in plan row / column order.
    Returns {family: (FamilyTree, source_rows, target table)} and the table of every metastatic cell.
    """
    src = cells[cells["side"].eq("source")].reset_index(drop=True)
    tgt = cells[cells["side"].eq("target")].reset_index(drop=True)
    out, rows = {}, []
    claimed = pd.Series("", index=tgt["cell"].astype(str))
    for fam in case["families"].split(";"):
        table = family_cells(fam)
        path = trees[family_tree_name(fam)]
        ft = FamilyTree(path)
        leaf_node = lambda leaf: ft.leaf_of.get(leaf, -1)
        s_in = src["cell"].astype(str).map(lambda c: c in table.index and bool(table.at[c, "reliable"]))
        s_rows = np.flatnonzero(s_in.to_numpy())
        s_nodes = np.array([leaf_node(table.at[c, "leaf"]) for c in src["cell"].astype(str).iloc[s_rows]], dtype=np.int64)
        if (s_nodes < 0).any():
            raise ValueError(f"{fam}: {int((s_nodes < 0).sum())} reliable source cells have no leaf in {path.name}")
        ft.prepare(s_nodes)
        fam_rows = []
        for j, (cell, site) in enumerate(zip(tgt["cell"].astype(str), tgt["site"].astype(str))):
            if cell not in table.index:
                continue
            claimed[cell] = fam
            r = table.loc[cell]
            rec = {"family": fam, "target_col": j, "cell": cell, "site": site, "leaf": r["leaf"], "reliable": bool(r["reliable"]),
                   "missing_share": r["missing_share"], "L1": r["L1"], "L2": r["L2"], "n_sources_comparable": len(s_rows)}
            if not r["reliable"]:
                rec.update(status="not_evaluable", reason="unreliable leaf (missing share > 0.5)")
            elif len(s_rows) < MIN_SOURCES:
                rec.update(status="not_evaluable", reason=f"fewer than {MIN_SOURCES} comparable primary cells")
            else:
                node = leaf_node(r["leaf"])
                if node < 0:
                    raise ValueError(f"{fam}: leaf {r['leaf']} not in {path.name}")
                d = ft.distances(node)
                dmin, dmax = int(d.min()), int(d.max())
                n_near = int((d == dmin).sum())
                rec.update(target_node=node, d_min=dmin, d_max=dmax, n_nearest=n_near, share_nearest=n_near / len(d),
                           d_mean_sources=float(d.mean()))
                if dmin == dmax:
                    rec.update(status="not_informative", reason="every comparable primary cell at the same distance")
                elif n_near / len(d) >= AMBIGUOUS_SHARE:
                    rec.update(status="ambiguous", reason=f"tree-nearest set holds >= {AMBIGUOUS_SHARE:.0%} of the comparable primaries")
                else:
                    rec.update(status="evaluable", reason="")
            fam_rows.append(rec)
        ftab = pd.DataFrame(fam_rows)
        out[fam] = (ft, s_rows, ftab)
        rows.append(ftab)
    for j, (cell, site) in enumerate(zip(tgt["cell"].astype(str), tgt["site"].astype(str))):
        if claimed[cell] == "":
            rows.append(pd.DataFrame([{"family": "", "target_col": j, "cell": cell, "site": site, "status": "not_evaluable",
                                       "reason": "no leaf in a family / clone tree of the case"}]))
    allt = pd.concat(rows, ignore_index=True)
    allt.insert(0, "case_id", case["case_id"])
    return out, allt, len(src)


def cell_metrics(qcol: np.ndarray, m: np.ndarray, s_rows: np.ndarray, d: np.ndarray) -> dict:
    """Matching quantities of one metastatic cell (PLAN.md): q over all sources, m the method's row masses."""
    c_ot = float(qcol[s_rows].sum())
    c_ind = float(m[s_rows].sum() / m.sum())
    out = {"c_ot": c_ot, "c_ind": c_ind}
    order_all = np.argsort(-qcol, kind="stable")
    comparable = np.zeros(len(qcol), dtype=bool); comparable[s_rows] = True
    for k in TOPK:
        out[f"top{k}_share_comparable"] = float(comparable[order_all[:k]].mean())
    if c_ot <= 0:
        out["note"] = "no mass on comparable sources"
        return out
    qt = qcol[s_rows] / c_ot
    pit = m[s_rows] / m[s_rows].sum()
    dmin = d.min()
    near = d == dmin
    D_ot, D_ind = float(qt @ d), float(pit @ d)
    out.update(D_ot=D_ot, D_ind=D_ind, delta_D=D_ind - D_ot, d_min=float(dmin),
               delta_D_norm=(D_ind - D_ot) / (D_ind - dmin) if D_ind > dmin else np.nan,
               M_ot=float(qt[near].sum()), M_ind=float(pit[near].sum()),
               M_ot_total=float(qcol[s_rows][near].sum()), M_ind_total=float(m[s_rows][near].sum() / m.sum()))
    out["log2_close_enrichment"] = float(np.log2(out["M_ot"] / out["M_ind"])) if out["M_ot"] > 0 and out["M_ind"] > 0 else np.nan
    order = np.argsort(-qt, kind="stable")
    for k in TOPK:
        kk = min(k, len(order))
        out[f"top{k}_mean_d"] = float(d[order[:kk]].mean())
    out["top1_d"] = float(d[order[0]]); out["top1_weight"] = float(qt[order[0]]); out["top1_in_nearest"] = bool(near[order[0]])
    out["top1_in_nearest_ind_rate"] = out["M_ind"]
    return out

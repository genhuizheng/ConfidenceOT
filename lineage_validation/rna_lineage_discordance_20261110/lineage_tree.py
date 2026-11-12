"""One clade tree per KP family, shared by every figure (92_, 95_).

Clades: the all-eligible-cell A2 clades (state_lineage_ext/all_cells__original/cells.tsv.gz; UPGMA on 1 - s within
the family, cut 0.5, clusters < 20 cells pooled as "minor"). Between clades: the mean 1 - s over up to 300 cells per
clade (seed 20261111, evaluable pairs), and average linkage on that matrix. The tree summarises the raw edits; it is
not the authors' Cassiopeia tree.

family_tree(...) computes it once and caches ROOT/explore/trees/<family>.npz. A figure that shows only some clades
draws the tree pruned to them: absent leaves are removed and unary nodes collapsed, so topology and heights are those
of the full family tree and every figure of a family orders its clades the same way.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.spatial.distance import squareform

SEED, PER_CLADE = 20261111, 300
SPECIAL = ("minor", "Other", "unassigned")


def family_tree(fam, cells, t, prior, mc, cache: Path):
    """cells: DataFrame (cell, clade) of the family. Returns dict(names, D, Z) or None (< 2 named clades)."""
    f = cache / f"{fam}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        return {"names": list(z["names"]), "D": z["D"], "Z": z["Z"]}
    named = sorted(c for c in cells.clade.unique() if c not in SPECIAL)
    if len(named) < 2:
        return None
    rng = np.random.default_rng(SEED)
    pick = []
    for c in named:
        ids = cells.cell[cells.clade == c].to_numpy(); pick.append(rng.choice(ids, min(PER_CLADE, len(ids)), replace=False))
    ids = np.concatenate(pick); grp = np.repeat(np.arange(len(named)), [len(x) for x in pick])
    L = mc.Lineage(t[(t.lineage == fam) & t.cell.isin(set(ids))], ids, np.repeat("source", len(ids)), prior)
    s, ok = L.agreement(np.arange(len(ids)), np.arange(len(ids))); s = np.asarray(s)
    D = np.zeros((len(named), len(named)))
    for i in range(len(named)):
        for j in range(len(named)):
            m = ok[np.ix_(grp == i, grp == j)]; v = s[np.ix_(grp == i, grp == j)][m]
            D[i, j] = 1 - v.mean() if v.size else 1.0
    D = (D + D.T) / 2; np.fill_diagonal(D, 0)
    Z = linkage(squareform(D, checks=False), "average")
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, names=np.array(named, dtype=object), D=D, Z=Z)
    return {"names": named, "D": D, "Z": Z}


def _prune(node, keep):
    """Return (pruned node as nested tuple) or None. Leaf = ('leaf', name); internal = ('node', height, left, right)."""
    if node.is_leaf():
        return ("leaf", node.id) if node.id in keep else None
    l, r = _prune(node.get_left(), keep), _prune(node.get_right(), keep)
    if l is None:
        return r
    if r is None:
        return l
    return ("node", node.dist, l, r)


def _leaves(n):
    return [n[1]] if n[0] == "leaf" else _leaves(n[2]) + _leaves(n[3])


def pruned_order(tree, present):
    """Clades of `present` that are in the tree, in the full tree's leaf order; the rest (special or untreed) after."""
    if tree is None:
        return [c for c in present if c not in SPECIAL] , None
    idx = {c: i for i, c in enumerate(tree["names"])}
    keep = {idx[c] for c in present if c in idx}
    root = to_tree(tree["Z"])
    pr = _prune(root, keep) if keep else None
    order = [tree["names"][i] for i in _leaves(pr)] if pr is not None else []
    return order, pr


def draw_pruned(ax, pr, names, y_of, x_left, x_right, y_bar=None, hmax=None, color="#333333", lw=0.6, fontsize=5.0):
    """Elbow tree growing left from x_right (leaf tips) to x_left (root height). y_of: clade -> y of its row centre.
    y_bar: where to draw the distance scale bar (1 - s); None = no bar."""
    if pr is None or pr[0] == "leaf":
        return
    hmax = hmax or pr[1] or 1.0
    xs = lambda h: x_right - (h / hmax) * (x_right - x_left)  # noqa: E731

    def rec(n):
        if n[0] == "leaf":
            return y_of[names[n[1]]], 0.0
        yl, hl = rec(n[2]); yr, hr = rec(n[3]); h = n[1]
        for y_, h_ in ((yl, hl), (yr, hr)):
            ax.plot([xs(h_), xs(h)], [y_, y_], color=color, lw=lw, solid_capstyle="butt")
        ax.plot([xs(h), xs(h)], [yl, yr], color=color, lw=lw, solid_capstyle="butt")
        return (yl + yr) / 2, h
    rec(pr)
    if y_bar is not None:
        step = 0.1 if hmax <= 0.6 else 0.2
        x1 = x_left + (x_right - x_left) * step / hmax
        ax.plot([x_left, x1], [y_bar, y_bar], color=color, lw=0.8, solid_capstyle="butt")
        ax.text(x1 + (x_right - x_left) * 0.04, y_bar, f"{step:.1f} (1 - s)", fontsize=fontsize, va="center", ha="left", color=color)

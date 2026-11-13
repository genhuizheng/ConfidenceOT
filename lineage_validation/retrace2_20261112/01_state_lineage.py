"""RETrace2 PLAN.md: cells, lineage / methylation distances, the authors' clades, lineage-state association and
lineage vs methylation concordance.

    python -I lineage_validation/retrace2_20261112/01_state_lineage.py --data results/retrace2_20261112/extracted/2025_RETrace2_MSH2_Female_Combined \
        --out results/retrace2_20261112

Read only. The distance pickle is read with a restricted unpickler (builtins containers and numbers only).
Writes OUT/prep/{cells.tsv, D_lineage.npz, D_meth.npz, charmat.npz} and OUT/analysis/{association*.tsv, concordance*.tsv,
mantel.tsv, attrition.tsv}.
"""

from __future__ import annotations

import argparse
import builtins
import collections
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

SEED, NPERM, NPERM_C, K = 20261112, 999, 99, 10


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins" and name in ("dict", "list", "tuple", "set", "float", "int", "str"):
            return getattr(builtins, name)
        if module == "collections" and name in ("OrderedDict", "defaultdict"):
            return getattr(collections, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


def authors_clades(tree, lo=10, hi=20):
    """The authors' rule (13_AnnotatePhylotree_CladeBias.ipynb): non-overlapping subtrees of lo-hi leaves, from the root."""
    out, used = [], set()

    def rec(cl):
        leaves = [x.name for x in cl.get_terminals()]
        if lo <= len(leaves) <= hi and not set(leaves) & used:
            out.append(leaves); used.update(leaves); return
        if len(leaves) > hi and not cl.is_terminal():
            for c in cl.clades:
                rec(c)
    rec(tree.root)
    out.sort(key=len, reverse=True)
    return out


def entropy(c):
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


def cramer(T):
    T = T[T.sum(1) > 0][:, T.sum(0) > 0]
    n = T.sum()
    if n == 0 or min(T.shape) < 2:
        return np.nan
    e = np.outer(T.sum(1), T.sum(0)) / n
    return float(np.sqrt(((T - e) ** 2 / np.where(e > 0, e, 1)).sum() / (n * (min(T.shape) - 1))))


def permute_within(lab, grp, rng):
    out = lab.copy()
    for g in np.unique(grp):
        i = np.flatnonzero(grp == g); out[i] = lab[rng.permutation(i)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    prep, ana = a.out / "prep", a.out / "analysis"; prep.mkdir(parents=True, exist_ok=True); ana.mkdir(parents=True, exist_ok=True)
    R = a.data / "results_10000targets_3000cpgs"
    ss = pd.read_csv(a.data / "phylotree_annotation" / "2025_MSH2_female_combined_samplesheet_with_celltype.csv")
    ms = pd.read_excel(a.data / "ms_stats" / "Supplementary_Table_MSH2_Per_Cell_Statistics.xlsx")
    meth = pd.read_csv(a.data / "methylation_stats" / "Supplementary_Table_MSH2_Per_Cell_Methylation_Statistics.csv")
    dd = SafeUnpickler(open(R / "phylo" / "2025_MSH2_female_combined.buildPhylo.distDict.pkl", "rb")).load()
    cells = list(dd["samples"]); n = len(cells); ix = {c: i for i, c in enumerate(cells)}
    DL = np.zeros((n, n)); NT = np.zeros((n, n))
    for (s1, s2), v in dd["sampleComp"].items():
        i, j = ix[s1], ix[s2]; DL[i, j] = DL[j, i] = v["dist"]; NT[i, j] = NT[j, i] = v["num_targets"]
    np.savez_compressed(prep / "D_lineage.npz", cells=np.array(cells), D=DL, shared_targets=NT)
    # cells table
    C = pd.DataFrame({"cell": cells}).merge(ss[["sample_id", "group", "cell_type_assignment"]], left_on="cell", right_on="sample_id", how="left")
    C["tissue"] = C.group; C["organ"] = C.tissue.str.split("_").str[0]
    C = C.merge(ms.rename(columns={"Cell_ID": "cell"})[["cell", "MS_Total_Reads", "MS_Targets_10X"]], on="cell", how="left")
    C = C.merge(meth.rename(columns={"sample_id": "cell"})[["cell", "CpG_Count_1X", "CG_Methylation_Rate_Percent"]], on="cell", how="left")
    att = ss.groupby("group").size().rename("cells_all").to_frame().join(C.groupby("tissue").size().rename("cells_in_tree"))
    att.to_csv(ana / "attrition.tsv", sep="\t")
    # methylation profile distance
    pdm = pd.read_csv(R / "infer_celltype" / "pd_matrix" / "pairwise_dissimilarity_matrix.csv", index_col=0).reindex(cells)
    shs = pd.read_csv(R / "infer_celltype" / "pd_matrix" / "shared_sites_matrix.csv", index_col=0).reindex(cells)
    Z = (pdm - pdm.mean()) / pdm.std(ddof=0)
    Zc = Z.sub(Z.mean(1), axis=0); Zc = Zc.div(np.sqrt((Zc ** 2).sum(1)), axis=0)
    DM = 1 - Zc.to_numpy() @ Zc.to_numpy().T
    np.fill_diagonal(DM, 0)
    np.savez_compressed(prep / "D_meth.npz", cells=np.array(cells), D=DM, profile=pdm.to_numpy(), shared_sites=shs.to_numpy(), refs=np.array(pdm.columns))
    C["meth_shared_sites_median"] = shs.median(1).to_numpy()
    # clades (authors' rule on their bootstrap tree)
    from Bio import Phylo
    tree = Phylo.read(R / "phylo" / "2025_MSH2_female_combined.buildPhylo.newick-bootstrap.txt", "newick")
    cl = authors_clades(tree)
    clade = {c: f"clade{k + 1}" for k, leaves in enumerate(cl) for c in leaves}
    C["clade"] = C.cell.map(clade).fillna("unclustered")
    C.to_csv(prep / "cells.tsv", sep="\t", index=False)
    print(f"{n} cells, {len(cl)} clades of 10-20 cells, unclustered {int((C.clade == 'unclustered').sum())}")
    print(att.to_string())
    # ---- association ----
    rng = np.random.default_rng(SEED)
    rows, srows = [], []
    for name, lab_col, within, subset in (("organ", "organ", None, None), ("tissue", "tissue", "organ", None),
                                          ("brain_celltype", "cell_type_assignment", None, "Brain")):
        D = C[C.clade != "unclustered"]
        if subset:
            D = D[D.organ == subset]
        if D.clade.nunique() < 2 or D[lab_col].nunique() < 2:
            continue
        ci = pd.Index(sorted(D.clade.unique())).get_indexer(D.clade); lv = pd.Index(sorted(D[lab_col].unique())); li = lv.get_indexer(D[lab_col])
        grp = D[within].to_numpy() if within else np.zeros(len(D))

        def tab(l):
            T = np.zeros((ci.max() + 1, len(lv))); np.add.at(T, (ci, l), 1); return T
        T = tab(li); V = cramer(T)
        nulls = [tab(permute_within(li, grp, rng)) for _ in range(NPERM)]
        Vn = np.array([cramer(x) for x in nulls])
        rows.append({"label": name, "null": f"within {within}" if within else "global", "cells": len(D), "clades": ci.max() + 1, "levels": len(lv),
                     "cramer_v": V, "null_mean": Vn.mean(), "excess": V - Vn.mean(), "z": (V - Vn.mean()) / Vn.std() if Vn.std() > 0 else np.nan,
                     "p": float((1 + (Vn >= V).sum()) / (NPERM + 1))})
        for k, x in enumerate(lv):
            col = T[:, k]; nc = np.array([nn[:, k] for nn in nulls])
            hill = np.exp(entropy(col)); hn = np.array([np.exp(entropy(c)) for c in nc])
            ms_ = col.max() / col.sum(); mn = np.array([c.max() / c.sum() for c in nc])
            srows.append({"label": name, "level": x, "cells": int(col.sum()), "clades_ge3": int((col >= 3).sum()), "hill": hill, "hill_null": hn.mean(),
                          "hill_z": (hill - hn.mean()) / hn.std() if hn.std() > 0 else np.nan, "max_share": ms_, "max_share_null": mn.mean()})
        pd.DataFrame(T, index=sorted(D.clade.unique()), columns=lv).to_csv(ana / f"table_clade_x_{name}.tsv", sep="\t")
    pd.DataFrame(rows).to_csv(ana / "association.tsv", sep="\t", index=False)
    pd.DataFrame(srows).to_csv(ana / "association_levels.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    # ---- concordance (view C analogue) ----
    tis = C.tissue.to_numpy()
    off = ~np.eye(n, dtype=bool)
    m25 = np.percentile(DM[off], 75)          # bottom quartile of methylation similarity = top quartile of D_M
    l75 = np.percentile(DL[off], 75)          # lineage-far = top quartile of D_L

    def conc(DMx):
        LN = np.argsort(np.where(np.eye(n, dtype=bool), np.inf, DL), 1)[:, :K]
        MN = np.argsort(np.where(np.eye(n, dtype=bool), np.inf, DMx), 1)[:, :K]
        div = (np.take_along_axis(DMx, LN, 1) >= m25).mean(1)
        cv = (np.take_along_axis(DL, MN, 1) >= l75).mean(1)
        return div, cv
    div, cv = conc(DM)
    dn, cn = np.zeros(n), np.zeros(n)
    for _ in range(NPERM_C):
        pi = permute_within(np.arange(n), tis, rng)
        d_, c_ = conc(DM[np.ix_(pi, pi)]); dn += d_; cn += c_
    dn /= NPERM_C; cn /= NPERM_C
    C2 = C.assign(div=div, div_null=dn, conv=cv, conv_null=cn)
    C2.to_csv(ana / "concordance_cells.tsv", sep="\t", index=False)
    cs = C2.groupby("tissue")[["div", "div_null", "conv", "conv_null"]].mean()
    cs.loc["all"] = C2[["div", "div_null", "conv", "conv_null"]].mean()
    cs.to_csv(ana / "concordance.tsv", sep="\t"); print(cs.round(3).to_string())
    # ---- Mantel-type Spearman of D_L vs D_M ----
    from scipy.stats import spearmanr
    mrow = []
    org = C.organ.to_numpy()
    for scope, mask in (("all pairs", off), ("same organ", off & (org[:, None] == org[None, :])), ("different organ", org[:, None] != org[None, :]),
                        ("same tissue", off & (tis[:, None] == tis[None, :]))):
        iu = np.triu(mask, 1)
        r = spearmanr(DL[iu], DM[iu]).correlation
        perm = []
        for _ in range(NPERM_C):
            pi = permute_within(np.arange(n), tis, rng) if scope != "different organ" else rng.permutation(n)
            perm.append(spearmanr(DL[iu], DM[np.ix_(pi, pi)][iu]).correlation)
        perm = np.array(perm)
        mrow.append({"scope": scope, "pairs": int(iu.sum()), "spearman": r, "null_mean": perm.mean(), "null_sd": perm.std(),
                     "p": float((1 + (np.abs(perm) >= abs(r)).sum()) / (NPERM_C + 1))})
    pd.DataFrame(mrow).to_csv(ana / "mantel.tsv", sep="\t", index=False); print(pd.DataFrame(mrow).round(3).to_string(index=False))
    # lineage distance by organ pair (descriptive)
    lv = sorted(set(org)); M = pd.DataFrame(index=lv, columns=lv, dtype=float)
    for x in lv:
        for y in lv:
            mk = (org[:, None] == x) & (org[None, :] == y) & off
            M.loc[x, y] = DL[mk].mean()
    M.to_csv(ana / "lineage_distance_by_organ.tsv", sep="\t"); print(M.round(4).to_string())


if __name__ == "__main__":
    main()

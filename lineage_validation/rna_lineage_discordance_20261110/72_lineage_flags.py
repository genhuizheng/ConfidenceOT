"""PLAN_v2.md amendment A1: one flag row per lineage (every lineage, none dropped).

    python lineage_validation/rna_lineage_discordance_20261110/72_lineage_flags.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs

Per lineage (KP MetFamily / Quinn LineageGroup within one mouse or experiment; the cells of lineages/cells.tsv.gz):
  from lineages/resolution.tsv: cells (clone size), sites, informative events, median informative events per cell,
    distinct event sets / cells, IQR of s, resolved and the reason
  missing_share  mean over cells of the share of the lineage's intBCs not detected in the cell
  median_depth   median pre-equalisation total counts (RNA obs of the units)
  polytomy_share tree resolution: share of the lineage's cells whose nearest non-unary ancestor in the authors' tree
                 has > 50 children. Quinn: the hybrid trees (m<exp>_lg<N>_tree_hybrid_priors.alleleThresh.processed.txt).
                 KP: NA, the KP-Tracer trees are not in the local inputs (only their priors)
  tree_leaves_matched  share of the lineage's cells found among the tree's leaves
Writes ROOT/lineages/flags.tsv.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"))
import matrix_core as mc  # noqa: E402

POLY = 50


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def newick_leaf_parent_degree(text: str) -> dict:
    """leaf name -> number of children of its nearest ancestor with >= 2 children (unary nodes suppressed)."""
    text = text.strip().rstrip(";")
    children, parent = {}, {}
    nid, n = 0, len(text)
    # iterative parser: '(' opens a node, ',' separates, ')' closes; names follow ')' or stand alone
    tokens_children = [[]]
    i = 0
    while i < n:
        c = text[i]
        if c == "(":
            tokens_children.append([]); i += 1
        elif c == ",":
            i += 1
        elif c == ")":
            kids = tokens_children.pop(); j = i + 1
            while j < n and text[j] not in ",()": j += 1
            nid += 1; me = ("n", nid); children[me] = kids
            for k in kids:
                parent[k] = me
            tokens_children[-1].append(me); i = j
        else:
            j = i
            while j < n and text[j] not in ",()": j += 1
            name = text[i:j].split(":")[0]
            leaf = ("l", name); tokens_children[-1].append(leaf); i = j
    out = {}
    for k in parent:
        if k[0] != "l":
            continue
        p = parent[k]
        while len(children[p]) < 2 and p in parent:
            p = parent[p]
        out[k[1]] = len(children[p])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    a = p.parse_args()
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); tabs = bb.load_tables(a.inputs)
    res = pd.read_csv(a.root / "lineages" / "resolution.tsv", sep="\t", dtype={"mouse": str})
    cells = pd.read_csv(a.root / "lineages" / "cells.tsv.gz", sep="\t", dtype={"mouse": str, "cell": str, "table_cell": str})
    depth = []
    for f in glob.glob(str(a.root / "inputs_A" / "rna_lineage_discordance_20261110" / "rna_coords" / "*" / "obs.tsv.gz")):
        depth.append(pd.read_csv(f, sep="\t", usecols=["cell", "total_counts_raw"], dtype={"cell": str}))
    depth = pd.concat(depth).drop_duplicates("cell").set_index("cell").total_counts_raw
    trees = a.inputs / "data" / "quinn2021_GSE161363" / "converted" / "lineage"
    rows = []
    for (ds, mouse, lin), g in cells.dropna(subset=["lineage"]).groupby(["dataset", "mouse", "lineage"]):
        key = "KPTracer" if ds == "KPTracer" else f"Quinn2021:{mouse}"
        t, prior = tabs[key]
        sub = t[t.lineage == lin]
        L = mc.Lineage(sub[sub.cell.isin(set(g.table_cell))], g.table_cell.to_numpy(), np.repeat("source", len(g)), prior)
        miss = 1 - np.asarray(L.observed.sum(1)).ravel() / max(len(L.intbcs), 1)
        rec = {"dataset": ds, "mouse": mouse, "lineage": lin, "missing_share": float(miss.mean()),
               "median_depth": float(g.cell.map(depth).median()), "polytomy_share": np.nan, "tree_leaves_matched": np.nan}
        if ds != "KPTracer":
            lg = str(lin).lower()
            f = trees / mouse / "trees" / "trees" / f"m{mouse}_{lg}_tree_hybrid_priors.alleleThresh.processed.txt"
            if f.exists():
                deg = newick_leaf_parent_degree(f.read_text())
                d = g.table_cell.map(deg)
                rec["tree_leaves_matched"] = float(d.notna().mean())
                if d.notna().any():
                    rec["polytomy_share"] = float((d.dropna() > POLY).mean())
        rows.append(rec)
    F = res.merge(pd.DataFrame(rows), on=["dataset", "mouse", "lineage"], how="left")
    F.to_csv(a.root / "lineages" / "flags.tsv", sep="\t", index=False)
    print(F.groupby("dataset")[["cells", "missing_share", "median_depth", "polytomy_share", "tree_leaves_matched"]].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()

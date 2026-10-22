"""Feasibility audit and CRISPR-defined subclones of every KP-Tracer family and Quinn clone tree.

    python lineage_validation/within_family_subclones/define_subclones.py --out OUT \\
        --kp-trees DIR [--kp-trees DIR ...] --kp-meta FILE [--kp-meta FILE ...] --lineage-export EXPORT

Implements SUBCLONE_RULES.md §1-§5 (fixed at a545f1c) on the authors' trees and character matrices.
Reads no OT output. Writes:
  OUT/feasibility.tsv            one row per family (KP-Tracer) or clone (Quinn) on a tree, eligible or not,
                                 with the reason
  OUT/subclones.tsv              every level-1 and level-2 subclone: tree node id, cells, primary/metastatic
                                 counts, site composition, shared flag, marker edits
  OUT/cells/<name>.csv.gz        one row per family leaf in tree leaf order: leaf, cell, site, primary/met,
                                 missing share, reliable, L1 and L2 subclone (or the reason it has none),
                                 author SubTumor (KP-Tracer)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402

MIN_MAJOR = 20
MAX_MISSING = 0.5
MARKER_OBSERVED, MARKER_IN, MARKER_OUT = 0.5, 0.8, 0.2
SHARED = 5
MISSING = {"-", "-1", "nan", "NA", "", "None"}


class Induced:
    """The authors' tree restricted to a set of leaves (nodes with no kept leaf below are dropped)."""

    def __init__(self, tree, keep_leaves: set[int]):
        below = {}
        for v in reversed(tree.order):
            kids = tree.children[v]
            below[v] = (v in keep_leaves) if not kids else any(below[c] for c in kids)
        self.name = tree.name
        self.children = {v: [c for c in tree.children[v] if below[c]] for v in tree.order if below[v]}
        root = tree.root
        self.root = root
        order, stack = [], [root]
        while stack:
            v = stack.pop()
            order.append(v)
            stack.extend(reversed(self.children[v]))
        self.order = order
        self.leaves = [v for v in order if not self.children[v]]
        self.parent = {c: v for v in order for c in self.children[v]}
        self.leaf_below = {}
        for v in reversed(order):
            self.leaf_below[v] = [v] if not self.children[v] else [x for c in self.children[v] for x in self.leaf_below[c]]


def matrix_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    m = pd.read_csv(path, sep="\t", index_col=0, dtype=str, keep_default_na=False)
    m.index = m.index.astype(str)
    return m


def markers(matrix: pd.DataFrame, inside: list[str], context: list[str]) -> list[dict]:
    """Marker edits of `inside` against `context \\ inside` (SUBCLONE_RULES.md §3)."""
    outside = sorted(set(context) - set(inside))
    a = matrix.reindex(inside).to_numpy(dtype=object)
    b = matrix.reindex(outside).to_numpy(dtype=object)
    found = []
    for k, column in enumerate(matrix.columns):
        ain = np.array([str(x) for x in a[:, k]], dtype=object)
        bin_ = np.array([str(x) for x in b[:, k]], dtype=object) if len(outside) else np.array([], dtype=object)
        a_obs = ~np.isin(ain, list(MISSING))
        if a_obs.mean() < MARKER_OBSERVED or a_obs.sum() == 0:
            continue
        states, counts = np.unique(ain[a_obs], return_counts=True)
        for s, c in zip(states, counts):
            if s == "0" or c / a_obs.sum() < MARKER_IN:
                continue
            b_obs = ~np.isin(bin_, list(MISSING))
            out_freq = float((bin_[b_obs] == s).mean()) if b_obs.any() else 0.0
            if out_freq <= MARKER_OUT:
                found.append({"character": str(column), "state": str(s), "in_freq": round(float(c / a_obs.sum()), 3),
                              "in_observed": round(float(a_obs.mean()), 3), "out_freq": round(out_freq, 3)})
    return found


def partition(t: Induced, root: int, reliable: dict, matrix: pd.DataFrame | None):
    """Level partition below `root` (SUBCLONE_RULES.md §4). Returns (split node, [(child, cells, markers)], {leaf: reason})."""
    rel = lambda v: [x for x in t.leaf_below[v] if reliable.get(x)]
    reasons = {}
    v = root
    while True:
        kids = t.children[v]
        major = [c for c in kids if len(rel(c)) >= MIN_MAJOR]
        if len(major) == 1:
            for c in kids:
                if c != major[0]:
                    for x in t.leaf_below[c]:
                        reasons.setdefault(x, "minor branch")
            v = major[0]
            continue
        if len(major) == 0:
            for x in t.leaf_below[v]:
                reasons.setdefault(x, "no split with >= 2 major branches")
            return None, [], reasons
        break
    context = [t.name[x] for x in rel(v)]
    chosen = []
    for c in kids:
        if c not in major:
            for x in t.leaf_below[c]:
                reasons.setdefault(x, "minor branch")
            continue
        inside = [t.name[x] for x in rel(c)]
        found = markers(matrix, inside, context) if matrix is not None else []
        if found:
            chosen.append((c, rel(c), found))
        else:
            for x in t.leaf_below[c]:
                reasons.setdefault(x, "branch without a marker edit")
    return v, chosen, reasons


def depth_of(t: Induced, v: int) -> int:
    d = 0
    while v in t.parent:
        v = t.parent[v]
        if len(t.children[v]) >= 2:
            d += 1
    return d


def analyse(name: str, dataset: str, mouse: str, family: str, tree_path: Path, matrix: pd.DataFrame | None,
            leaf_cells: pd.DataFrame, out: Path, rows: list, subs: list) -> None:
    """leaf_cells: leaf, cell_id, site, kind (primary/metastasis), plus optional SubTumor."""
    d1 = lv.tree_module()
    tree = d1.Tree(tree_path)
    leaf_node = {tree.name[v]: v for v in tree.leaves}
    fam = leaf_cells[leaf_cells["leaf"].isin(leaf_node)].drop_duplicates("leaf")
    t = Induced(tree, {leaf_node[x] for x in fam["leaf"]})
    info = fam.set_index("leaf")
    miss = {}
    if matrix is not None:
        values = matrix.reindex(fam["leaf"]).to_numpy(dtype=str)
        present = fam["leaf"].isin(matrix.index).to_numpy()
        share = np.where(present, np.isin(values, list(MISSING)).mean(axis=1), np.nan)
        miss = dict(zip(fam["leaf"], share))
    reliable = {leaf_node[x]: (matrix is not None and np.isfinite(miss.get(x, np.nan)) and miss[x] <= MAX_MISSING)
                for x in fam["leaf"]}
    n_rel = lambda kind: int(sum(1 for x in fam["leaf"] if reliable[leaf_node[x]] and info.loc[x, "kind"] == kind))
    row = {"dataset": dataset, "mouse": mouse, "family_or_clone": family, "tree": tree_path.name,
           "tree_leaves_total": len(tree.leaves), "family_leaves_mapped": len(fam),
           "matrix": "yes" if matrix is not None else "no", "reliable_cells": int(sum(reliable.values())),
           "reliable_primary": n_rel("primary"), "reliable_metastatic": n_rel("metastasis"),
           "unreliable_missing_gt_0.5": int(sum(1 for v in reliable.values() if not v)),
           "sites": json.dumps(fam.groupby("site").size().to_dict()),
           "author_subclone_annotation": "KP-Tracer SubTumor = physical tumour piece, not a lineage subclone" if dataset == "KPTracer"
           else "none deposited"}
    labels = {x: {"L1": "", "L2": "", "reason": ""} for x in fam["leaf"]}
    for x in fam["leaf"]:
        if not reliable[leaf_node[x]]:
            labels[x]["reason"] = "missing share > 0.5" if matrix is not None else "no character matrix"
    split, chosen, reasons = partition(t, t.root, reliable, matrix) if matrix is not None else (None, [], {})
    for v, reason in reasons.items():
        x = t.name[v]
        if x in labels and not labels[x]["reason"]:
            labels[x]["reason"] = reason
    chosen.sort(key=lambda c: -len(c[1]))
    l1 = []
    for k, (node, cells, found) in enumerate(chosen, 1):
        sid = f"L1.{k}"
        l1.append((sid, node, cells))
        for v in cells:
            labels[t.name[v]]["L1"] = sid
            labels[t.name[v]]["reason"] = ""
        subs.append(sub_row(name, dataset, mouse, family, 1, sid, "", node, cells, found, t, info, miss, depth_of(t, node)))
    n_l2 = 0
    for sid, node, cells in l1:
        split2, chosen2, _ = partition(t, node, reliable, matrix)
        chosen2.sort(key=lambda c: -len(c[1]))
        for k, (node2, cells2, found2) in enumerate(chosen2, 1):
            sid2 = f"{sid}.{k}"
            n_l2 += 1
            for v in cells2:
                labels[t.name[v]]["L2"] = sid2
            subs.append(sub_row(name, dataset, mouse, family, 2, sid2, sid, node2, cells2, found2, t, info, miss, depth_of(t, node2)))
    shared = [s for s in subs if s["name"] == name and s["level"] == 1 and s["shared"]]
    row.update(split_node=split if split is not None else "", split_depth=depth_of(t, split) if split is not None else "",
               level1_subclones=len(l1), level1_shared=len(shared), level2_subclones=n_l2,
               cells_in_level1=int(sum(len(c) for _, _, c in l1)))
    reasons_out = []
    if matrix is None:
        reasons_out.append("no character matrix")
    if len(l1) < 2:
        reasons_out.append(f"{len(l1)} level-1 subclone(s)")
    if row["reliable_primary"] < MIN_MAJOR:
        reasons_out.append(f"{row['reliable_primary']} reliable primary cells (< {MIN_MAJOR})")
    if row["reliable_metastatic"] < MIN_MAJOR:
        reasons_out.append(f"{row['reliable_metastatic']} reliable metastatic cells (< {MIN_MAJOR})")
    row["subclone_structure_eligible"] = not reasons_out
    row["reason"] = "; ".join(reasons_out)
    rows.append(row)
    order = {t.name[v]: i for i, v in enumerate(t.leaves)}
    cells_table = fam.copy()
    cells_table["leaf_order"] = cells_table["leaf"].map(order)
    cells_table["tree_node"] = cells_table["leaf"].map(leaf_node)
    cells_table["missing_share"] = cells_table["leaf"].map(miss)
    cells_table["reliable"] = cells_table["leaf"].map(lambda x: reliable[leaf_node[x]])
    for key in ("L1", "L2", "reason"):
        cells_table[key] = cells_table["leaf"].map(lambda x: labels[x][key])
    cells_table["L1"] = cells_table["L1"].replace("", "ambiguous")
    (out / "cells").mkdir(parents=True, exist_ok=True)
    cells_table.sort_values("leaf_order").to_csv(out / "cells" / f"{name}.csv.gz", index=False, compression="gzip")


def sub_row(name, dataset, mouse, family, level, sid, parent, node, cells, found, t, info, miss, depth):
    leaves = [t.name[v] for v in cells]
    sub = info.loc[leaves]
    prim = int((sub["kind"] == "primary").sum())
    met = int((sub["kind"] == "metastasis").sum())
    return {"name": name, "dataset": dataset, "mouse": mouse, "family_or_clone": family, "level": level, "subclone": sid,
            "parent": parent, "tree_node": int(node), "branch_depth": depth, "cells": len(leaves), "primary": prim,
            "metastatic": met, "shared": prim >= SHARED and met >= SHARED,
            "sites": json.dumps(sub.groupby("site").size().to_dict()),
            "mean_missing_share": round(float(np.nanmean([miss.get(x, np.nan) for x in leaves])), 3),
            "marker_edits": len(found), "markers": json.dumps(found[:20])}


def kp_families(args, rows, subs):
    frames = []
    for p in args.kp_meta:
        frame = pd.read_csv(p, dtype=str, low_memory=False)
        if "cell_id" not in frame:  # KPTracer_meta.csv as deposited: the cell id is the unnamed first column
            frame = frame.rename(columns={frame.columns[0]: "cell_id"})
        frames.append(frame)
    meta = pd.concat(frames, ignore_index=True).drop_duplicates("cell_id")
    meta = meta[meta["is_normal"].astype(str).str.lower().ne("true")]
    trees = {}
    for d in args.kp_trees:
        for p in sorted(Path(d).rglob("*_tree.nwk")):
            if p.name.endswith(("_All_tree.nwk", "_Fam_tree.nwk")):
                trees.setdefault(p.name[: -len("_tree.nwk")], p)
    mats = {}
    for d in args.kp_trees:
        for p in sorted(Path(d).rglob("*_character_matrix.txt")):
            mats.setdefault(p.name[: -len("_character_matrix.txt")], p)
    d1 = lv.tree_module()
    for stem, path in sorted(trees.items()):
        mouse = stem.split("_")[0]
        tree = d1.Tree(path)
        names = {tree.name[v] for v in tree.leaves}
        m = meta[meta["cell_id"].isin(names)]
        fams = [stem[: -len("_Fam")]] if stem.endswith("_Fam") else sorted(m["MetFamily"].dropna().unique())
        for family in fams:
            if stem.endswith("_All") and f"{family}_Fam" in trees:
                continue  # protocol rule: the family's own _Fam tree is used
            part = m[m["MetFamily"] == family]
            name = f"KPTracer_{mouse}_{family}"
            if family.endswith("_Unk"):
                rows.append({"dataset": "KPTracer", "mouse": mouse, "family_or_clone": family, "tree": path.name,
                             "subclone_structure_eligible": False, "reason": "family unknown (_Unk)"})
                continue
            primary = family  # a family is named after its primary tumour
            leaf_cells = pd.DataFrame({"leaf": part["cell_id"], "cell_id": part["cell_id"], "site": part["Tumor"],
                                       "kind": np.where(part["Tumor"] == primary, "primary", "metastasis"),
                                       "SubTumor": part.get("SubTumor")})
            if not (part["Tumor"] == primary).any():
                rows.append({"dataset": "KPTracer", "mouse": mouse, "family_or_clone": family, "tree": path.name,
                             "family_leaves_mapped": len(part), "subclone_structure_eligible": False,
                             "reason": "no primary tumour of this family on the tree"})
                continue
            if (part["Tumor"] != primary).sum() == 0:
                rows.append({"dataset": "KPTracer", "mouse": mouse, "family_or_clone": family, "tree": path.name,
                             "family_leaves_mapped": len(part), "reliable_metastatic": 0, "subclone_structure_eligible": False,
                             "reason": "no metastatic cell of this family (primary only)"})
                continue
            analyse(name, "KPTracer", mouse, family, path, matrix_frame(mats.get(stem)), leaf_cells, args.out, rows, subs)
            print(f"{name}: done", flush=True)


def quinn_clones(args, rows, subs):
    export = args.lineage_export
    trees = pd.read_csv(export / "labels" / "trees.tsv", sep="\t", dtype=str, keep_default_na=False)
    leaves = pd.read_csv(export / "labels" / "tree_leaves.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    cells = pd.read_csv(export / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    for _, info in trees[trees["dataset"].eq("Quinn2021")].iterrows():
        mouse, clone = info["mouse"], info["lineage"]
        if info["status"] != "tree":
            rows.append({"dataset": "Quinn2021", "mouse": mouse, "family_or_clone": f"LG{clone}", "tree": "",
                         "family_leaves_mapped": 0, "subclone_structure_eligible": False, "reason": info["status"]})
            continue
        rel = re.split(r"[\\/]lineage[\\/]" + re.escape(mouse) + r"[\\/]", info["tree"])[-1]
        path = export / "lineage" / "Quinn2021" / mouse / rel
        cands = sorted(p for p in (export / "lineage" / "Quinn2021" / mouse / "character_matrices").rglob("*")
                       if p.is_file() and re.search(rf"(?i)lg{clone}[_.]", p.name) and "alleleThresh" in p.name)
        mp = leaves[(leaves["mouse"] == mouse) & (leaves["lineage"] == clone)]
        c = cells[(cells["dataset"] == "Quinn2021") & (cells["mouse"] == mouse)].set_index("cell")
        leaf_cells = pd.DataFrame({"leaf": mp["leaf"], "cell_id": mp["cell"], "site": mp["cell"].map(c["site"]).to_numpy()})
        leaf_cells["kind"] = np.where(leaf_cells["site"].str.replace(r"^\d+k", "", regex=True) == "LL", "primary", "metastasis")
        analyse(f"Quinn2021_{mouse}_LG{clone}", "Quinn2021", mouse, f"LG{clone}", path,
                matrix_frame(cands[0]) if len(cands) == 1 else None, leaf_cells, args.out, rows, subs)
    print("Quinn: done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kp-trees", type=Path, nargs="+", required=True)
    parser.add_argument("--kp-meta", type=Path, nargs="+", required=True)
    parser.add_argument("--lineage-export", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows, subs = [], []
    kp_families(args, rows, subs)
    quinn_clones(args, rows, subs)
    pd.DataFrame(rows).to_csv(args.out / "feasibility.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame(subs).to_csv(args.out / "subclones.tsv", sep="\t", index=False, lineterminator="\n")
    print(f"{len(rows)} families/clones, {len(subs)} subclones", flush=True)


if __name__ == "__main__":
    main()

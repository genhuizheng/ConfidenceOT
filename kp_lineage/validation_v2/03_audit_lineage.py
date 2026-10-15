"""Phase-0 audit for the KP-Tracer lineage validation (validation_v2). Read only.

    python kp_lineage/validation_v2/03_audit_lineage.py \\
        --tracer KPTracer-Data --prepared PREPARED --ot-block OT/kptracer --out AUDIT_DIR

It establishes what the cell-level analysis could stand on, and computes no
association between lineage and any gate outcome:

* **Trees.** Every newick under trees/ (Cassiopeia trees and the
  neighbor-joining ones): leaves, internal nodes, whether branch lengths are
  present and non-trivial, polytomies, depth, leaf-name format, and which
  tumours (by the authors' obs['Tumor']) each tree's leaves come from.
* **Character matrices.** Rows, characters, the missing-state fraction, and
  overlap with the tree of the same name.
* **Identifiers.** Whether tree leaves match the expression cell ids used in
  the prepared h5ads and in the OT outputs (KPTracer_meta.csv index), exactly or
  after a stated normalisation.
* **Coverage.** For every eligible same-family pair: which family tree, if any,
  holds both its source and its target tumour, and how many of the pair's gated
  source cells and target cells are leaves of it -- counted separately for
  retained and rejected source cells, because a coverage difference between the
  two groups would bias any comparison. Counts only.
* **Gate semantics, observed.** Per label: the spread of source retained
  fractions across pairs, active rejection bounds and budget overrides
  (whether retention is structurally constrained), and the agreement of the
  retained flag with signed_rejection_margin < 0.

Outputs: audit.json, trees.tsv, tree_tumours.tsv, character_matrices.tsv,
pair_coverage.tsv, gate_semantics.tsv, audit.txt.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

METHOD = "M4-E"


# --- newick ---------------------------------------------------------------

def parse_newick(text: str):
    """Parse one newick string into parent/children/name/length arrays.

    Iterative, so deep caterpillar trees cannot overflow the stack. Names may be
    quoted; comments in square brackets are dropped.
    """
    text = re.sub(r"\[[^\]]*\]", "", text.strip())
    if text.endswith(";"):
        text = text[:-1]
    parent, children, name, length = [-1], [[]], [""], [None]
    current = 0
    i, n = 0, len(text)

    def read_label(position):
        if position < n and text[position] in "'\"":
            quote = text[position]
            end = text.index(quote, position + 1)
            return text[position + 1:end], end + 1
        end = position
        while end < n and text[end] not in ",():;":
            end += 1
        return text[position:end].strip(), end

    def read_length(position):
        if position < n and text[position] == ":":
            end = position + 1
            while end < n and text[end] not in ",();":
                end += 1
            value = text[position + 1:end].strip()
            return (float(value) if value else None), end
        return None, position

    while i < n:
        char = text[i]
        if char == "(":
            parent.append(current); children.append([]); name.append(""); length.append(None)
            node = len(parent) - 1
            children[current].append(node)
            current = node
            i += 1
        elif char == ",":
            current = parent[current]
            parent.append(current); children.append([]); name.append(""); length.append(None)
            node = len(parent) - 1
            children[current].append(node)
            current = node
            i += 1
        elif char == ")":
            current = parent[current]
            i += 1
            label, i = read_label(i)
            name[current] = label
            length[current], i = read_length(i)
        elif char.isspace():
            i += 1
        else:
            label, i = read_label(i)
            name[current] = label
            length[current], i = read_length(i)
    # The first "(" opens a child of the placeholder root; collapse it.
    if len(children[0]) == 1 and not name[0]:
        root = children[0][0]
    else:
        root = 0
    return {"parent": parent, "children": children, "name": name, "length": length, "root": root}


def tree_summary(tree) -> dict:
    children, length, root = tree["children"], tree["length"], tree["root"]
    nodes = [root]
    depth = {root: 0}
    order = []
    while nodes:
        node = nodes.pop()
        order.append(node)
        for child in children[node]:
            depth[child] = depth[node] + 1
            nodes.append(child)
    leaves = [v for v in order if not children[v]]
    internal = [v for v in order if children[v]]
    lengths = [length[v] for v in order if v != root and length[v] is not None]
    degree = [len(children[v]) for v in internal]
    leaf_depth = np.asarray([depth[v] for v in leaves], dtype=float)
    return {
        "leaves": len(leaves),
        "internal_nodes": len(internal),
        "edges_with_length": len(lengths),
        "edges": len(order) - 1,
        "distinct_branch_lengths": len(set(lengths)),
        "branch_length_min": float(min(lengths)) if lengths else None,
        "branch_length_max": float(max(lengths)) if lengths else None,
        "internal_with_more_than_two_children": int(sum(d > 2 for d in degree)),
        "max_children": int(max(degree)) if degree else 0,
        "unary_internal_nodes": int(sum(d == 1 for d in degree)),
        "leaf_depth_median": float(np.median(leaf_depth)) if len(leaf_depth) else None,
        "leaf_depth_max": int(leaf_depth.max()) if len(leaf_depth) else None,
        "leaf_names": [tree["name"][v] for v in leaves],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--ot-block", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["ranknm256_noscale_ds_cos", "raw"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    meta = pd.read_csv(args.tracer / "KPTracer_meta.csv", index_col=0, low_memory=False)
    meta.index = meta.index.astype(str)
    tumour_of = meta["Tumor"].astype(str)
    say(f"KPTracer_meta.csv: {len(meta)} cells, index examples {list(meta.index[:3])}")
    say(f"  TS_Present True {int(meta['TS_Present'].sum())} of {len(meta)}")

    # Trees.
    tree_rows, composition_rows, leaf_sets = [], [], {}
    tree_files = sorted((args.tracer / "trees").glob("*.nwk")) + \
        sorted((args.tracer / "trees" / "neighbor_joining").glob("*.tree"))
    exact = set(meta.index)
    for path in tree_files:
        kind = "neighbor_joining" if path.parent.name == "neighbor_joining" else "cassiopeia"
        stem = path.name.replace("_tree.nwk", "").replace("_tree_nj.processed.tree", "")
        try:
            tree = parse_newick(path.read_text())
            summary = tree_summary(tree)
        except Exception as error:  # noqa: BLE001 - every tree is reported, readable or not
            tree_rows.append({"tree": stem, "kind": kind, "file": str(path), "error": f"{type(error).__name__}: {error}"})
            continue
        leaves = summary.pop("leaf_names")
        mapped = [leaf for leaf in leaves if leaf in exact]
        leaf_sets[(stem, kind)] = set(mapped)
        tumours = Counter(tumour_of.loc[mapped]) if mapped else Counter()
        tree_rows.append({
            "tree": stem, "kind": kind, "file": str(path), **summary,
            "leaves_matching_meta_exactly": len(mapped),
            "leaf_examples": "|".join(leaves[:3]),
            "tumours": len(tumours),
        })
        for tumour, count in sorted(tumours.items()):
            composition_rows.append({"tree": stem, "kind": kind, "tumor": tumour, "leaves": count})
    trees = pd.DataFrame(tree_rows)
    trees.to_csv(args.out / "trees.tsv", sep="\t", index=False)
    composition = pd.DataFrame(composition_rows)
    composition.to_csv(args.out / "tree_tumours.tsv", sep="\t", index=False)
    say(f"\ntrees: {len(trees)} files")
    with pd.option_context("display.width", 250, "display.max_rows", 300, "display.max_columns", 30):
        say(trees.drop(columns=["file"]).to_string(index=False))

    # Character matrices.
    matrix_rows = []
    for path in sorted((args.tracer / "trees").glob("*_character_matrix.txt")):
        stem = path.name.replace("_character_matrix.txt", "")
        try:
            matrix = pd.read_csv(path, sep="\t", index_col=0)
        except Exception as error:  # noqa: BLE001
            matrix_rows.append({"matrix": stem, "error": f"{type(error).__name__}: {error}"})
            continue
        values = matrix.astype(str).to_numpy()
        missing = np.isin(values, ["-", "-1", "nan", "NA", ""]).mean() if values.size else np.nan
        tree_leaves = leaf_sets.get((stem, "cassiopeia"), set())
        rows = set(matrix.index.astype(str))
        matrix_rows.append({
            "matrix": stem, "cells": len(matrix), "characters": matrix.shape[1],
            "missing_fraction": float(missing),
            "state_examples": "|".join(sorted(set(values.ravel()[:200]))[:8]),
            "rows_matching_meta": len(rows & exact),
            "rows_in_same_name_tree": len(rows & tree_leaves),
            "tree_leaves_not_in_matrix": len(tree_leaves - rows),
        })
    matrices = pd.DataFrame(matrix_rows)
    matrices.to_csv(args.out / "character_matrices.tsv", sep="\t", index=False)
    say(f"\ncharacter matrices: {len(matrices)}")
    with pd.option_context("display.width", 250, "display.max_rows", 300):
        say(matrices.to_string(index=False))

    # Coverage of every eligible same-family pair, per label.
    manifest = pd.read_csv(args.prepared / "pair_manifest_all.csv")
    manifest["same_met_family"] = manifest["same_met_family"].fillna(False).astype(bool)
    manifest["eligible"] = manifest["eligible"].astype(bool)
    family_trees = {key: leaves for key, leaves in leaf_sets.items()
                    if re.search(r"_(Fam|All|Mets)$", key[0])}
    coverage_rows, semantics_rows = [], []
    for label in args.labels:
        label_dir = args.ot_block / label
        retained_fractions = []
        for row in manifest[manifest["eligible"]].to_dict("records"):
            outputs = sorted((label_dir / str(row["pair_id"])).glob("scope_malignant/*/SUCCESS"))
            if len(outputs) != 1:
                coverage_rows.append({"label": label, "pair_id": row["pair_id"], "status": f"{len(outputs)} outputs"})
                continue
            folder = outputs[0].parent
            cells = pd.read_csv(folder / "cell_confidence.csv")
            cells = cells[cells["method"].astype(str).eq(METHOD)]
            source = cells[cells["side"].eq("source")]
            target = cells[cells["side"].eq("target")]
            metrics = pd.read_csv(folder / "pair_metrics.csv")
            metric = metrics[metrics["method"].astype(str).eq(METHOD)].iloc[0]
            retained = source["retained"].astype(bool)
            margin_rule = source["signed_rejection_margin"].to_numpy() < 0
            retained_fractions.append(float(retained.mean()))
            semantics_rows.append({
                "label": label, "pair_id": row["pair_id"], "same_met_family": row["same_met_family"],
                "source_cells": len(source), "target_cells": len(target),
                "source_retained_fraction": float(retained.mean()),
                "target_retained_fraction": float(target["retained"].astype(bool).mean()),
                "source_retained_equals_margin_below_0": float(np.mean(retained.to_numpy() == margin_rule)),
                "source_budget_overridden_cells": int(source["budget_overridden"].astype(bool).sum()),
                "source_lower_bound_active": metric.get("source_lower_bound_active"),
                "source_upper_bound_active": metric.get("source_upper_bound_active"),
                "rejection_cost": float(metric["rejection_cost"]),
                "observation_ids_in_meta": int(source["observation_id"].astype(str).isin(exact).sum()),
            })
            if not row["same_met_family"]:
                continue
            ids_source = set(source["observation_id"].astype(str))
            ids_target = set(target["observation_id"].astype(str))
            ids_retained = set(source.loc[retained.to_numpy(), "observation_id"].astype(str))
            ids_rejected = ids_source - ids_retained
            holders = []
            for (stem, kind), leaves in family_trees.items():
                if leaves & ids_source and leaves & ids_target:
                    holders.append((stem, kind, leaves))
            if not holders:
                coverage_rows.append({"label": label, "pair_id": row["pair_id"], "status": "no family tree holds both sides",
                                      "source_cells": len(ids_source), "target_cells": len(ids_target)})
                continue
            for stem, kind, leaves in holders:
                coverage_rows.append({
                    "label": label, "pair_id": row["pair_id"], "status": "covered",
                    "tree": stem, "kind": kind,
                    "source_cells": len(ids_source), "target_cells": len(ids_target),
                    "source_in_tree": len(ids_source & leaves), "target_in_tree": len(ids_target & leaves),
                    "retained_cells": len(ids_retained), "retained_in_tree": len(ids_retained & leaves),
                    "rejected_cells": len(ids_rejected), "rejected_in_tree": len(ids_rejected & leaves),
                })
        fractions = np.asarray(retained_fractions)
        if fractions.size:
            say(f"\n{label}: source retained fraction across {fractions.size} pairs: "
                f"min {fractions.min():.3f}, q25 {np.quantile(fractions, .25):.3f}, median {np.median(fractions):.3f}, "
                f"q75 {np.quantile(fractions, .75):.3f}, max {fractions.max():.3f}")
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(args.out / "pair_coverage.tsv", sep="\t", index=False)
    semantics = pd.DataFrame(semantics_rows)
    semantics.to_csv(args.out / "gate_semantics.tsv", sep="\t", index=False)
    say("\nsame-family pair coverage:")
    with pd.option_context("display.width", 250, "display.max_rows", 400, "display.max_columns", 30):
        say(coverage.to_string(index=False) if len(coverage) else "(none)")
    if len(semantics):
        say("\ngate semantics:")
        say(f"  retained flag agrees with margin < 0: min over pairs "
            f"{semantics['source_retained_equals_margin_below_0'].min():.4f}")
        say(f"  pairs with any budget-overridden source cell: "
            f"{int((semantics['source_budget_overridden_cells'] > 0).sum())}")
        say(f"  pairs with an active source bound: lower "
            f"{int(semantics['source_lower_bound_active'].astype(str).eq('True').sum())}, upper "
            f"{int(semantics['source_upper_bound_active'].astype(str).eq('True').sum())}")
        say(f"  OT observation ids found in KPTracer_meta: "
            f"{int(semantics['observation_ids_in_meta'].sum())} of {int(semantics['source_cells'].sum())} source cells")

    report = {
        "meta_cells": int(len(meta)),
        "trees": int(len(trees)),
        "trees_with_branch_lengths": int((trees.get("edges_with_length", pd.Series(dtype=float)).fillna(0) > 0).sum()),
        "family_trees": sorted(f"{s} ({k})" for s, k in family_trees),
        "same_family_pairs_covered": {
            label: int(coverage[(coverage["label"].eq(label)) & coverage["status"].eq("covered")]["pair_id"].nunique())
            if len(coverage) else 0 for label in args.labels},
        "same_family_pairs_uncovered": {
            label: sorted(coverage[(coverage["label"].eq(label)) & ~coverage["status"].eq("covered")]["pair_id"].unique().tolist())
            if len(coverage) else [] for label in args.labels},
    }
    (args.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()

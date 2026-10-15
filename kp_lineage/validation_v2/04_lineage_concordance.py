"""E1 / E1b cell-level lineage concordance (validation_v2, step D1).

    python kp_lineage/validation_v2/04_lineage_concordance.py \\
        --tracer KPTracer-Data --prepared PREPARED --ot-block OT/kptracer \\
        --depth DOWNSAMPLED/predownsample_depth.csv.gz --out LINEAGE_DIR

Implements VALIDATION_V2_PREREGISTRATION.md (frozen 2026-10-15, text of commit
b0817f1) sections 1.1-1.6 exactly; nothing here is tuned on any outcome. It
reads the existing gates and the authors' trees and writes only under --out,
which must not already hold a SUCCESS file.

* **Pairs (1.1).** The eligible same-family pairs. Primary tree: the Cassiopeia
  ``<target MetFamily>_Fam`` if that file exists, otherwise
  ``<mouse>_<genotype>_All``; a pair that neither holds on both sides is
  "not lineage-covered". Source cells are the pair's M4-E source cells that are
  leaves of that tree; target cells likewise. Evaluable: at least 5 mapped
  retained, 5 mapped rejected and 1 mapped target cell.
* **s (1.2).** For a source cell, the root depth, in edges, of its deepest
  ancestor whose clade holds a mapped target cell, after contracting unary
  internal nodes.
* **delta (1.3).** Cliff's delta of s, retained against rejected; mean per
  metastasis, mean per mouse.
* **Test (1.4).** One-sided Wilcoxon signed-rank of the mouse means (H1 > 0);
  mice with mean above 0; 95% percentile bootstrap of the mean of mouse means,
  10,000 resamples of mice, seed 20261015. Supported if that mean is above 0
  and p < 0.05; otherwise inconclusive.
* **E1b (1.5).** Spearman rho between r = -signed_rejection_margin and s, per
  evaluable pair, aggregated and tested the same way.
* **Sensitivity (1.6).** Minimum groups of 10 and 20; the other family tree and
  the neighbor-joining tree; patristic distance only if a tree has branch
  lengths; coverage balance; leave one mouse out; s against pre-equalisation
  depth and detected genes.

Diagnostics written alongside: group sizes, ties of s (within and between
groups), tree resolution, and per-pair differences between trees. Figures: one
descriptive drawing of the actual primary tree per evaluable pair of
--figure-label, and per-pair delta by mouse. They describe; the test is E1.

Outputs: pair_trees.tsv (one row per label, pair and covering tree),
pairs.tsv (primary tree only), metastases.tsv, mice.tsv, cell_scores.csv.gz,
summary.json, report.txt, provenance.json, figures/, SUCCESS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHOD = "M4-E"
MINIMUM_GROUP = 5
SENSITIVITY_GROUPS = (10, 20)
BOOTSTRAP = 10_000
SEED = 20261015
PREREGISTRATION = "VALIDATION_V2_PREREGISTRATION.md"
FROZEN_MARK = "Status: **FROZEN 2026-10-15 b0817f1.**"

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit03", HERE / "03_audit_lineage.py")
audit03 = importlib.util.module_from_spec(spec)
sys.modules["audit03"] = audit03
spec.loader.exec_module(audit03)


# --- trees ------------------------------------------------------------------

class Tree:
    """One parsed tree with the arrays s needs."""

    def __init__(self, path: Path):
        parsed = audit03.parse_newick(path.read_text())
        self.path = path
        self.children = parsed["children"]
        self.parent = parsed["parent"]
        self.name = parsed["name"]
        self.root = parsed["root"]
        order, stack = [], [self.root]
        while stack:
            node = stack.pop()
            order.append(node)
            stack.extend(reversed(self.children[node]))
        self.order = order  # preorder, children left to right
        size = len(self.parent)
        raw_depth = np.zeros(size, dtype=int)
        contracted = np.zeros(size, dtype=int)
        for node in order:
            # A unary node is merged with its only child, so passing through it adds no edge.
            step = 1 if len(self.children[node]) >= 2 else 0
            for child in self.children[node]:
                raw_depth[child] = raw_depth[node] + 1
                contracted[child] = contracted[node] + step
        self.raw_depth = raw_depth
        self.depth = contracted
        self.leaves = [v for v in order if not self.children[v]]
        self.leaf_of = {self.name[v]: v for v in self.leaves}
        internal = [v for v in order if self.children[v]]
        degree = np.asarray([len(self.children[v]) for v in internal])
        self.resolution = {
            "tree_leaves": len(self.leaves),
            "tree_internal_nodes": len(internal),
            "tree_internal_nodes_after_contraction": int((degree >= 2).sum()),
            "tree_unary_nodes": int((degree == 1).sum()),
            "tree_max_children": int(degree.max()) if degree.size else 0,
            "tree_polytomies": int((degree > 2).sum()),
            "tree_has_branch_lengths": any(x is not None for i, x in enumerate(parsed["length"]) if i != self.root),
        }

    def shared_depth(self, target_leaves: np.ndarray, source_leaves: np.ndarray) -> np.ndarray:
        """s for each source leaf: contracted depth of its deepest ancestor holding a target leaf."""
        marked = np.zeros(len(self.parent), dtype=bool)
        marked[target_leaves] = True
        for node in reversed(self.order):
            if marked[node] and node != self.root:
                marked[self.parent[node]] = True
        best = np.full(len(self.parent), -1, dtype=int)
        for node in self.order:
            if node == self.root:
                best[node] = self.depth[node] if marked[node] else -1
            else:
                best[node] = self.depth[node] if marked[node] else best[self.parent[node]]
        return best[source_leaves]


def tree_path(tracer: Path, stem: str, kind: str) -> Path:
    if kind == "cassiopeia":
        return tracer / "trees" / f"{stem}_tree.nwk"
    return tracer / "trees" / "neighbor_joining" / f"{stem}_tree_nj.processed.tree"


# --- statistics -----------------------------------------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Cliff's delta of a against b, and the between-group tie probability."""
    b = np.sort(b)
    below = np.searchsorted(b, a, side="left")
    not_above = np.searchsorted(b, a, side="right")
    greater = below.sum()                       # pairs with a > b
    less = (len(b) - not_above).sum()           # pairs with a < b
    total = len(a) * len(b)
    return float((greater - less) / total), float((total - greater - less) / total)


def aggregate(pairs: pd.DataFrame, column: str) -> dict:
    """Mean per metastasis, mean per mouse, one-sided Wilcoxon and bootstrap (1.3-1.4)."""
    pairs = pairs[pairs[column].notna()]
    if pairs.empty:
        return {"pairs": 0, "metastases": 0, "mice": 0, "verdict": "no evaluable pair"}
    metastases = pairs.groupby(["patient_id", "target_sample"], sort=True)[column].mean().rename(column).reset_index()
    mice = metastases.groupby("patient_id", sort=True)[column].mean()
    values = mice.to_numpy(dtype=float)
    result = {
        "pairs": int(len(pairs)), "metastases": int(len(metastases)), "mice": int(len(mice)),
        "mouse_means": {str(k): float(v) for k, v in mice.items()},
        "mean_of_mouse_means": float(values.mean()),
        "median_of_mouse_means": float(np.median(values)),
        "mice_above_0": int((values > 0).sum()),
        "mice_below_0": int((values < 0).sum()),
        "mice_equal_0": int((values == 0).sum()),
    }
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    result["bootstrap_95"] = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    if len(values) >= 2 and np.any(values != 0):
        test = stats.wilcoxon(values, alternative="greater")
        result["wilcoxon_statistic"] = float(test.statistic)
        result["wilcoxon_p_one_sided"] = float(test.pvalue)
    else:
        result["wilcoxon_statistic"] = None
        result["wilcoxon_p_one_sided"] = None
    p = result["wilcoxon_p_one_sided"]
    result["verdict"] = "supported" if (p is not None and result["mean_of_mouse_means"] > 0 and p < 0.05) else "inconclusive"
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(stats.spearmanr(x[keep], y[keep])[0])


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# --- per-pair computation -------------------------------------------------------

def detected_genes(path: str, cache: dict) -> pd.Series:
    """Detected genes per cell of one prepared tumour h5ad (raw counts)."""
    if path not in cache:
        import anndata
        data = anndata.read_h5ad(path)
        matrix = data.X
        counts = np.asarray((matrix > 0).sum(axis=1)).ravel()
        cache[path] = pd.Series(counts, index=data.obs_names.astype(str))
    return cache[path]


def pair_on_tree(tree: Tree, source: pd.DataFrame, target_ids: list[str]) -> dict:
    """Counts, s, delta and rho for one pair on one tree."""
    mapped = source[source["observation_id"].isin(tree.leaf_of)]
    targets = np.asarray([tree.leaf_of[c] for c in target_ids if c in tree.leaf_of], dtype=int)
    retained = mapped["retained"].to_numpy(dtype=bool)
    row = {
        "source_cells": int(len(source)),
        "retained_cells": int(source["retained"].sum()),
        "rejected_cells": int((~source["retained"]).sum()),
        "target_cells": int(len(target_ids)),
        "mapped_retained": int(retained.sum()),
        "mapped_rejected": int((~retained).sum()),
        "mapped_target": int(len(targets)),
        **tree.resolution,
    }
    row["mapped_retained_fraction"] = row["mapped_retained"] / row["retained_cells"] if row["retained_cells"] else np.nan
    row["mapped_rejected_fraction"] = row["mapped_rejected"] / row["rejected_cells"] if row["rejected_cells"] else np.nan
    if len(targets) == 0 or len(mapped) == 0:
        return {**row, "cells": None}
    leaves = np.asarray([tree.leaf_of[c] for c in mapped["observation_id"]], dtype=int)
    s = tree.shared_depth(targets, leaves)
    r = -mapped["signed_rejection_margin"].to_numpy(dtype=float)
    s_ret, s_rej = s[retained], s[~retained]
    row.update({
        "s_distinct": int(np.unique(s).size),
        "s_zero_fraction": float(np.mean(s == 0)),
        "s_modal_fraction": float(np.bincount(s - s.min()).max() / len(s)),
        "s_retained_median": float(np.median(s_ret)) if len(s_ret) else np.nan,
        "s_rejected_median": float(np.median(s_rej)) if len(s_rej) else np.nan,
        "s_retained_mean": float(s_ret.mean()) if len(s_ret) else np.nan,
        "s_rejected_mean": float(s_rej.mean()) if len(s_rej) else np.nan,
        "source_leaf_depth_median": float(np.median(tree.depth[leaves])),
    })
    if len(s_ret) and len(s_rej):
        row["delta"], row["tie_probability"] = cliffs_delta(s_ret.astype(float), s_rej.astype(float))
    else:
        row["delta"], row["tie_probability"] = np.nan, np.nan
    row["rho"] = spearman(r, s.astype(float))
    cells = pd.DataFrame({"observation_id": mapped["observation_id"].to_numpy(), "retained": retained,
                          "r": r, "s": s})
    return {**row, "cells": cells}


# --- figures --------------------------------------------------------------------

COLOURS = {"metastasis": "#000000", "retained": "#2b8cbe", "rejected": "#d7301f"}


def draw_tree(tree: Tree, source: pd.DataFrame, target_ids: list[str], title: str, path: Path) -> None:
    """The actual tree as parsed (unit edges: the trees carry no branch lengths), leaves marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    y = np.zeros(len(tree.parent), dtype=float)
    for rank, leaf in enumerate(tree.leaves):
        y[leaf] = rank
    for node in reversed(tree.order):
        if tree.children[node]:
            values = [y[c] for c in tree.children[node]]
            y[node] = (min(values) + max(values)) / 2
    top = max(int(tree.raw_depth.max()), 1)
    x = tree.raw_depth / top
    segments = []
    for node in tree.order:
        kids = tree.children[node]
        if not kids:
            continue
        span = [y[c] for c in kids]
        segments.append(((x[node], min(span)), (x[node], max(span))))
        for child in kids:
            segments.append(((x[node], y[child]), (x[child], y[child])))
    n = len(tree.leaves)
    height = float(np.clip(n / 350.0, 4.0, 14.0))
    fig, ax = plt.subplots(figsize=(9.0, height))
    ax.add_collection(LineCollection(segments, colors="#4d4d4d", linewidths=0.35, rasterized=True))
    status = dict(zip(source["observation_id"], np.where(source["retained"], "retained", "rejected")))
    targets = set(target_ids)
    columns = (("metastasis", 1.06), ("retained", 1.22), ("rejected", 1.38))
    counts = {}
    for key, left in columns:
        if key == "metastasis":
            rows = [y[tree.leaf_of[c]] for c in targets if c in tree.leaf_of]
        else:
            rows = [y[v] for v in tree.leaves if status.get(tree.name[v]) == key]
        counts[key] = len(rows)
        ax.add_collection(LineCollection([((left, r), (left + 0.12, r)) for r in rows],
                                         colors=COLOURS[key], linewidths=0.6, rasterized=True))
    labels = {"metastasis": "Metastasis", "retained": "Retained\nprimary", "rejected": "Rejected\nprimary"}
    for key, left in columns:
        ax.text(left + 0.06, -0.015 * n - 1, f"{labels[key]}\n{counts[key]} cells", ha="center", va="bottom",
                fontsize=8, color=COLOURS[key], linespacing=1.1)
    ax.set_xlim(-0.01, 1.52)
    ax.set_ylim(n + 1, -0.13 * n - 2)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9, loc="left")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def draw_delta(pairs: pd.DataFrame, mice: dict, path: Path) -> None:
    """Per-pair delta by mouse, with the mouse mean."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(mice)
    fig, ax = plt.subplots(figsize=(max(3.5, 0.75 * len(order) + 1.5), 3.6))
    for position, mouse in enumerate(order):
        values = pairs.loc[pairs["patient_id"].astype(str).eq(mouse), "delta"].to_numpy(dtype=float)
        jitter = np.linspace(-0.18, 0.18, len(values)) if len(values) > 1 else np.zeros(1)
        ax.scatter(position + jitter, values, s=18, color="#7f7f7f", zorder=2)
        ax.scatter([position], [mice[mouse]], s=70, marker="_", color="black", linewidths=2.2, zorder=3)
    ax.axhline(0, color="#999999", linewidth=0.8, zorder=1)
    ax.set_xticks(range(len(order)), order)
    ax.set_xlabel("Mouse")
    ax.set_ylabel("Cliff's delta of s (retained vs rejected)")
    ax.set_ylim(-1, 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# --- main -----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--ot-block", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True, help="27_'s predownsample_depth.csv.gz")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["ranknm256_noscale_ds_cos", "raw"])
    parser.add_argument("--figure-label", default="ranknm256_noscale_ds_cos")
    args = parser.parse_args()

    if (args.out / "SUCCESS").exists():
        raise SystemExit(f"{args.out}/SUCCESS exists; D1 is not rerun into the same directory")
    prereg = (HERE / PREREGISTRATION).read_text(encoding="utf-8")
    if FROZEN_MARK not in prereg:
        raise SystemExit(f"{PREREGISTRATION} is not frozen; D1 runs only under the frozen protocol")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    manifest = pd.read_csv(args.prepared / "pair_manifest_all.csv")
    manifest["same_met_family"] = manifest["same_met_family"].fillna(False).astype(bool)
    manifest["eligible"] = manifest["eligible"].astype(bool)
    family = manifest[manifest["eligible"] & manifest["same_met_family"]].copy()
    family["patient_id"] = family["patient_id"].astype(str)
    say(f"eligible same-family pairs: {len(family)}")

    depth = pd.read_csv(args.depth)
    depth = depth.drop_duplicates(["sample_id", "observation_id"]).set_index(["sample_id", "observation_id"])["predownsample_total_counts"]

    trees: dict[tuple[str, str], Tree] = {}

    def load(stem: str, kind: str):
        key = (stem, kind)
        if key not in trees:
            path = tree_path(args.tracer, stem, kind)
            trees[key] = Tree(path) if path.exists() else None
        return trees[key]

    family_stems = sorted({p.name.replace("_tree.nwk", "") for p in (args.tracer / "trees").glob("*_tree.nwk")
                           if p.name.replace("_tree.nwk", "").endswith(("_Fam", "_All", "_Mets"))} |
                          {p.name.replace("_tree_nj.processed.tree", "") for p in (args.tracer / "trees" / "neighbor_joining").glob("*.tree")
                           if p.name.replace("_tree_nj.processed.tree", "").endswith(("_Fam", "_All", "_Mets"))})

    tree_rows, cell_tables, inputs = [], [], {}
    gene_cache: dict = {}
    for label in args.labels:
        for row in family.to_dict("records"):
            pair_id = row["pair_id"]
            base = {"label": label, "pair_id": pair_id, "patient_id": row["patient_id"],
                    "source_sample": row["source_sample"], "target_sample": row["target_sample"],
                    "target_met_family": row["target_met_family"]}
            outputs = sorted((args.ot_block / label / pair_id).glob("scope_malignant/*/SUCCESS"))
            if len(outputs) != 1:
                tree_rows.append({**base, "role": "primary", "status": f"{len(outputs)} completed outputs"})
                continue
            confidence = outputs[0].parent / "cell_confidence.csv"
            inputs[f"{label}/{pair_id}/cell_confidence.csv"] = md5(confidence)
            cells = pd.read_csv(confidence)
            cells = cells[cells["method"].astype(str).eq(METHOD)]
            cells["observation_id"] = cells["observation_id"].astype(str)
            cells["retained"] = cells["retained"].astype(bool)
            source = cells[cells["side"].eq("source")].reset_index(drop=True)
            target_ids = cells.loc[cells["side"].eq("target"), "observation_id"].tolist()

            # Tree rule (1.1): <target MetFamily>_Fam if it exists, else <mouse>_<genotype>_All.
            fam_stem = f"{row['target_met_family']}_Fam"
            all_stem = f"{row['target_sample'].rsplit('_', 1)[0]}_All"
            primary = None
            for stem in (fam_stem, all_stem):
                tree = load(stem, "cassiopeia")
                if tree is None:
                    continue
                if any(c in tree.leaf_of for c in source["observation_id"]) and any(c in tree.leaf_of for c in target_ids):
                    primary = stem
                    break
            rule_note = ("_Fam" if primary == fam_stem else "_All" if primary == all_stem else "none")
            fam_exists = load(fam_stem, "cassiopeia") is not None
            candidates = []
            for stem in family_stems:
                for kind in ("cassiopeia", "neighbor_joining"):
                    tree = load(stem, kind)
                    if tree is None:
                        continue
                    if any(c in tree.leaf_of for c in source["observation_id"]) and any(c in tree.leaf_of for c in target_ids):
                        candidates.append((stem, kind))
            if primary is None:
                tree_rows.append({**base, "role": "primary", "status": "not lineage-covered",
                                  "fam_tree_exists": fam_exists, "source_cells": len(source), "target_cells": len(target_ids)})
                continue
            for stem, kind in [(primary, "cassiopeia")] + [c for c in candidates if c != (primary, "cassiopeia")]:
                tree = load(stem, kind)
                result = pair_on_tree(tree, source, target_ids)
                table = result.pop("cells")
                role = "primary" if (stem, kind) == (primary, "cassiopeia") else (
                    "neighbor_joining_same_tree" if (stem == primary and kind == "neighbor_joining") else
                    "other_family_tree_cassiopeia" if kind == "cassiopeia" else "other_family_tree_neighbor_joining")
                evaluable = (result["mapped_retained"] >= MINIMUM_GROUP and result["mapped_rejected"] >= MINIMUM_GROUP
                             and result["mapped_target"] >= 1)
                record = {**base, "role": role, "status": "evaluable" if evaluable else "not evaluable",
                          "tree": stem, "kind": kind, "tree_rule": rule_note, "fam_tree_exists": fam_exists, **result}
                if role == "primary" and table is not None:
                    sample = row["source_sample"]
                    table["predownsample_total_counts"] = [depth.get((sample, c), np.nan) for c in table["observation_id"]]
                    genes = detected_genes(row["source_h5ad"], gene_cache) if isinstance(row.get("source_h5ad"), str) else pd.Series(dtype=float)
                    table["detected_genes"] = genes.reindex(table["observation_id"]).to_numpy(dtype=float)
                    record["rho_s_depth"] = spearman(table["s"].to_numpy(float), table["predownsample_total_counts"].to_numpy(float))
                    record["rho_s_detected_genes"] = spearman(table["s"].to_numpy(float), table["detected_genes"].to_numpy(float))
                    record["depth_missing"] = int(table["predownsample_total_counts"].isna().sum())
                    record["genes_missing"] = int(table["detected_genes"].isna().sum())
                    cell_tables.append(table.assign(label=label, pair_id=pair_id, tree=stem))
                    if label == args.figure_label and evaluable:
                        draw_tree(tree, source[source["observation_id"].isin(tree.leaf_of)], target_ids,
                                  f"{pair_id}   tree {stem} (Cassiopeia)   label {label}",
                                  args.out / "figures" / f"tree_{pair_id}")
                tree_rows.append(record)
            say(f"{label} {pair_id}: primary {primary}, {len(candidates)} covering trees")

    pair_trees = pd.DataFrame(tree_rows)
    pair_trees.to_csv(args.out / "pair_trees.tsv", sep="\t", index=False)
    if cell_tables:
        pd.concat(cell_tables, ignore_index=True).to_csv(args.out / "cell_scores.csv.gz", index=False, compression="gzip")
    primary_rows = pair_trees[pair_trees["role"].eq("primary")].copy()
    primary_rows.to_csv(args.out / "pairs.tsv", sep="\t", index=False)

    summary = {"preregistration": "VALIDATION_V2_PREREGISTRATION.md, frozen 2026-10-15, text of commit b0817f1",
               "method": METHOD, "minimum_group": MINIMUM_GROUP, "bootstrap": BOOTSTRAP, "seed": SEED,
               "primary_label": args.labels[0], "labels": {}}
    metastasis_tables, mouse_tables = [], []
    for label in args.labels:
        rows = primary_rows[primary_rows["label"].eq(label)]
        evaluable = rows[rows["status"].eq("evaluable")]
        entry = {
            "same_family_pairs": int(len(rows)),
            "covered": int(rows["status"].isin(["evaluable", "not evaluable"]).sum()),
            "not_lineage_covered": rows.loc[rows["status"].eq("not lineage-covered"), "pair_id"].tolist(),
            "missing_outputs": rows.loc[~rows["status"].isin(["evaluable", "not evaluable", "not lineage-covered"]), "pair_id"].tolist(),
            "not_evaluable": rows.loc[rows["status"].eq("not evaluable"),
                                      ["pair_id", "mapped_retained", "mapped_rejected", "mapped_target"]].to_dict("records"),
            "tree_rule_used": rows["tree_rule"].value_counts().to_dict() if "tree_rule" in rows else {},
            "E1": aggregate(evaluable, "delta"),
            "E1b": aggregate(evaluable, "rho"),
        }
        sensitivity = {}
        covered = rows[rows["status"].isin(["evaluable", "not evaluable"])]
        for minimum in SENSITIVITY_GROUPS:
            keep = covered[(covered["mapped_retained"] >= minimum) & (covered["mapped_rejected"] >= minimum) & (covered["mapped_target"] >= 1)]
            sensitivity[f"minimum_group_{minimum}"] = {"E1": aggregate(keep, "delta"), "E1b": aggregate(keep, "rho")}
        alternatives = pair_trees[pair_trees["label"].eq(label) & ~pair_trees["role"].eq("primary") & pair_trees["status"].eq("evaluable")]
        for role_name, roles in (("other_family_tree", ["other_family_tree_cassiopeia"]),
                                 ("neighbor_joining", ["neighbor_joining_same_tree"])):
            alt = alternatives[alternatives["role"].isin(roles)].drop_duplicates("pair_id")
            swapped = evaluable.set_index("pair_id").copy()
            replaced = [p for p in alt["pair_id"] if p in swapped.index]
            for p in replaced:
                swapped.loc[p, "delta"] = float(alt.set_index("pair_id").loc[p, "delta"])
                swapped.loc[p, "rho"] = float(alt.set_index("pair_id").loc[p, "rho"])
            swapped = swapped.reset_index()
            diffs = (alt.set_index("pair_id")["delta"].reindex(replaced) - evaluable.set_index("pair_id")["delta"].reindex(replaced))
            sensitivity[role_name] = {"pairs_replaced": len(replaced),
                                      "delta_difference_alternative_minus_primary": {k: float(v) for k, v in diffs.items()},
                                      "E1": aggregate(swapped, "delta"), "E1b": aggregate(swapped, "rho")}
        has_lengths = bool(pair_trees["tree_has_branch_lengths"].eq(True).any()) if "tree_has_branch_lengths" in pair_trees else False
        sensitivity["patristic_distance"] = ("not applicable: no covering tree has branch lengths (1.6 item 3)"
                                             if not has_lengths else "branch lengths present: not implemented, report")
        balance = (evaluable["mapped_retained_fraction"] - evaluable["mapped_rejected_fraction"]).abs() < 0.1
        sensitivity["coverage_balanced"] = {"pairs_kept": int(balance.sum()), "E1": aggregate(evaluable[balance], "delta")}
        loo = {}
        e1_mice = entry["E1"].get("mouse_means", {})
        for mouse in e1_mice:
            rest = [v for k, v in e1_mice.items() if k != mouse]
            loo[mouse] = float(np.mean(rest)) if rest else None
        sensitivity["leave_one_mouse_out_mean_of_mouse_means"] = loo
        sensitivity["s_vs_predownsample_depth"] = aggregate(evaluable, "rho_s_depth")
        sensitivity["s_vs_detected_genes"] = aggregate(evaluable, "rho_s_detected_genes")
        entry["sensitivity"] = sensitivity
        entry["diagnostics"] = {
            "tie_probability_median": float(evaluable["tie_probability"].median()) if len(evaluable) else None,
            "tie_probability_max": float(evaluable["tie_probability"].max()) if len(evaluable) else None,
            "s_distinct_median": float(evaluable["s_distinct"].median()) if len(evaluable) else None,
            "s_modal_fraction_median": float(evaluable["s_modal_fraction"].median()) if len(evaluable) else None,
        }
        summary["labels"][label] = entry
        if len(evaluable):
            met = evaluable.groupby(["patient_id", "target_sample"], sort=True)[["delta", "rho"]].mean().reset_index()
            metastasis_tables.append(met.assign(label=label))
            mouse = met.groupby("patient_id", sort=True)[["delta", "rho"]].mean().reset_index()
            mouse_tables.append(mouse.assign(label=label))
        if label == args.figure_label and len(evaluable):
            draw_delta(evaluable, entry["E1"]["mouse_means"], args.out / "figures" / f"delta_by_mouse_{label}")
    if metastasis_tables:
        pd.concat(metastasis_tables).to_csv(args.out / "metastases.tsv", sep="\t", index=False)
        pd.concat(mouse_tables).to_csv(args.out / "mice.tsv", sep="\t", index=False)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    for label, entry in summary["labels"].items():
        e1, e1b = entry["E1"], entry["E1b"]
        say(f"\n{label}: covered {entry['covered']}/{entry['same_family_pairs']}, evaluable {e1.get('pairs')} pairs, "
            f"{e1.get('metastases')} metastases, {e1.get('mice')} mice")
        say(f"  E1  delta: mean of mouse means {e1.get('mean_of_mouse_means')}, 95% {e1.get('bootstrap_95')}, "
            f"mice > 0 {e1.get('mice_above_0')}/{e1.get('mice')}, Wilcoxon p {e1.get('wilcoxon_p_one_sided')} -> {e1.get('verdict')}")
        say(f"  E1b rho:   mean of mouse means {e1b.get('mean_of_mouse_means')}, 95% {e1b.get('bootstrap_95')}, "
            f"mice > 0 {e1b.get('mice_above_0')}/{e1b.get('mice')}, Wilcoxon p {e1b.get('wilcoxon_p_one_sided')} -> {e1b.get('verdict')}")
    columns = ["label", "pair_id", "status", "tree", "mapped_retained", "mapped_rejected", "mapped_target",
               "delta", "tie_probability", "rho", "s_distinct", "s_modal_fraction", "s_retained_median",
               "s_rejected_median", "tree_max_children", "rho_s_depth", "rho_s_detected_genes"]
    with pd.option_context("display.width", 250, "display.max_rows", 200, "display.max_columns", 30):
        say("\n" + primary_rows[[c for c in columns if c in primary_rows]].to_string(index=False))
    (args.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def git(*command):
        try:
            return subprocess.run(["git", "-C", str(HERE), *command], capture_output=True, text=True).stdout.strip()
        except OSError:
            return "unknown"
    inputs["pair_manifest_all.csv"] = md5(args.prepared / "pair_manifest_all.csv")
    inputs["predownsample_depth.csv.gz"] = md5(args.depth)
    for (stem, kind), tree in trees.items():
        if tree is not None:
            inputs[f"tree/{kind}/{stem}"] = md5(tree.path)
    provenance = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--", str(HERE))),
                  "preregistration_last_commit": git("log", "-1", "--format=%H", "--", str(HERE / PREREGISTRATION)),
                  "arguments": {k: str(v) for k, v in vars(args).items()},
                  "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
                  "scipy": __import__("scipy").__version__, "inputs_md5": inputs}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (args.out / "SUCCESS").write_text("done\n", encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()

"""Phase B audit for the KP-Tracer Level 2/3 validation (validation_v3). Read only.

    python kp_lineage/validation_v3/01_audit_v3.py \\
        --tracer KPTracer-Data --prepared PREPARED --ot-block OT/kptracer --out AUDIT_DIR

Computes no association between lineage and any gate or coupling outcome. It
establishes, per primary-metastasis pair that shares a lineage tree, whether the
lineage reference can carry a Level 3 comparison at all:

* **Inventory.** Every file under KPTracer-Data (path, size); the heads of any
  prior, expansion and fitness files, which the README names but does not
  specify.
* **Pairs.** The 28 lineage-covered same-family pairs (validation_v2 rule: the
  Cassiopeia ``<target MetFamily>_Fam`` tree, else ``<mouse>_<genotype>_All``)
  and every eligible different-family pair whose source and target tumours sit in
  one family tree (Cassiopeia ``<mouse>_<genotype>_All``, else the
  neighbor-joining one). Cells are the original M4-E gated cells, all of them
  (retained and rejected): the coupling covers every source cell.
* **Lineage informativeness of L.** L_ij = root depth, in edges after contracting
  unary nodes, of the MRCA of source cell i and target cell j. Its distribution
  under the uniform product u x v is computed exactly on the tree (the mass of
  cell pairs whose MRCA is node v is A(v)B(v) - sum over children of A(c)B(c)),
  never by enumerating cell pairs: mean, variance, number of distinct values,
  the share of cell pairs at the modal value (ties), and the share whose MRCA is
  the root. A pair with zero variance cannot distinguish any coupling from its
  independent null.
* **Tree robustness.** Where a neighbor-joining tree of the same name holds both
  sides, the Spearman correlation of L between the two reconstructions on a
  seeded sample of 20,000 cell pairs, and the same informativeness numbers.
* **Edits.** For the character matrix of the pair's tree: characters, missing and
  unedited shares, distinct states per character, how many characters carry a
  state seen in more than one tumour of the family, and the share of source
  cells and of target cells with at least one informative (edited, observed)
  state. Under u x v, the share of cell pairs that share at least one identical
  informative state.

Outputs: inventory.tsv, file_heads.txt, LINEAGE_PAIR_ELIGIBILITY.tsv,
lineage_value_histograms.tsv, character_stats.tsv, audit_v3.json, report.txt,
SUCCESS.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("d1", HERE.parent / "validation_v2" / "04_lineage_concordance.py")
d1 = importlib.util.module_from_spec(_spec)
sys.modules["d1"] = d1
_spec.loader.exec_module(d1)

LABEL = "ranknm256_noscale_ds_cos"
METHOD = "M4-E"
SAMPLE = 20_000
SEED = 20261016


def node_masses(tree: "d1.Tree", weights: dict[int, float]) -> np.ndarray:
    """Sum of leaf weights in every node's subtree."""
    mass = np.zeros(len(tree.parent))
    for leaf, w in weights.items():
        mass[leaf] = w
    for node in reversed(tree.order):
        if node != tree.root:
            mass[tree.parent[node]] += mass[node]
    return mass


def lca_distribution(tree: "d1.Tree", source_leaves: np.ndarray, target_leaves: np.ndarray) -> pd.Series:
    """Mass of cell pairs (uniform over source x target) by MRCA contracted depth."""
    a = node_masses(tree, {int(v): 1.0 / len(source_leaves) for v in source_leaves})
    b = node_masses(tree, {int(v): 1.0 / len(target_leaves) for v in target_leaves})
    at = a * b
    for node in tree.order:
        for child in tree.children[node]:
            at[node] -= a[child] * b[child]
    # A unary node and its child share a clade: the pair mass sits on the deepest of them,
    # and their contracted depths are equal, so the histogram is unaffected.
    keep = at > 1e-15
    frame = pd.DataFrame({"depth": tree.depth[keep], "mass": at[keep]})
    return frame.groupby("depth")["mass"].sum()


def informativeness(hist: pd.Series) -> dict:
    values, mass = hist.index.to_numpy(float), hist.to_numpy(float)
    mean = float((values * mass).sum())
    var = float(((values - mean) ** 2 * mass).sum())
    return {"L_mean": mean, "L_variance": var, "L_distinct": int(len(values)),
            "L_modal_share": float(mass.max()), "L_root_share": float(mass[values == 0].sum()) if (values == 0).any() else 0.0,
            "L_total_mass_check": float(mass.sum())}


def sampled_lca_depths(tree: "d1.Tree", pairs: list[tuple[int, int]]) -> np.ndarray:
    ancestors = {}

    def path(v):
        if v not in ancestors:
            chain, x = [v], v
            while x != tree.root:
                x = tree.parent[x]
                chain.append(x)
            ancestors[v] = chain
        return ancestors[v]

    out = np.empty(len(pairs))
    for k, (i, j) in enumerate(pairs):
        pi = set(path(i))
        lca = next(x for x in path(j) if x in pi)
        out[k] = tree.depth[lca]
    return out


def gated_cells(ot_block: Path, label: str, pair_id: str) -> tuple[list[str], list[str]] | None:
    done = sorted((ot_block / label / pair_id).glob("scope_malignant/*/SUCCESS"))
    if len(done) != 1:
        return None
    cells = pd.read_csv(done[0].parent / "cell_confidence.csv", usecols=["method", "side", "observation_id"])
    cells = cells[cells["method"].astype(str).eq(METHOD)]
    cells["observation_id"] = cells["observation_id"].astype(str)
    return (cells.loc[cells["side"].eq("source"), "observation_id"].tolist(),
            cells.loc[cells["side"].eq("target"), "observation_id"].tolist())


def read_matrix(tracer: Path, stem: str) -> pd.DataFrame | None:
    path = tracer / "trees" / f"{stem}_character_matrix.txt"
    if not path.exists():
        return None
    matrix = pd.read_csv(path, sep="\t", index_col=0, dtype=str)
    matrix.index = matrix.index.astype(str)
    return matrix


MISSING = {"-", "-1", "nan", "NA", ""}


def character_stats(matrix: pd.DataFrame, source: list[str], target: list[str], tumour_of: pd.Series) -> dict:
    values = matrix.to_numpy(dtype=str)
    missing = np.isin(values, list(MISSING))
    unedited = values == "0"
    informative = ~missing & ~unedited
    row = {"characters": int(matrix.shape[1]), "matrix_cells": int(matrix.shape[0]),
           "missing_share": float(missing.mean()), "unedited_share": float(unedited.mean()),
           "informative_share": float(informative.mean()),
           "distinct_states_per_character_median": float(np.median(
               [len(set(values[informative[:, c], c])) for c in range(values.shape[1])]))}
    # A state carried by cells of more than one tumour of the family: a shared (inherited
    # across tumours, or recurrent) edit. It does not separate inheritance from homoplasy.
    tumours = tumour_of.reindex(matrix.index).astype(str).to_numpy()
    shared = 0
    for c in range(values.shape[1]):
        keep = informative[:, c]
        frame = pd.DataFrame({"state": values[keep, c], "tumour": tumours[keep]})
        shared += int((frame.groupby("state")["tumour"].nunique() > 1).any())
    row["characters_with_cross_tumour_state"] = shared
    for side, ids in (("source", source), ("target", target)):
        present = [c for c in ids if c in matrix.index]
        sub = informative[matrix.index.get_indexer(present)]
        row[f"{side}_in_matrix"] = len(present)
        row[f"{side}_with_informative_edit"] = float(sub.any(axis=1).mean()) if len(present) else np.nan
    # Share of (source, target) cell pairs that share at least one identical informative state.
    src = [c for c in source if c in matrix.index]
    tgt = [c for c in target if c in matrix.index]
    if src and tgt:
        features = {}
        rows, cols = [], []
        for r, cell in enumerate(matrix.index):
            for c in np.flatnonzero(informative[r]):
                key = (c, values[r, c])
                cols.append(features.setdefault(key, len(features)))
                rows.append(r)
        onehot = sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                                   shape=(matrix.shape[0], len(features)))
        s = onehot[matrix.index.get_indexer(src)]
        t = onehot[matrix.index.get_indexer(tgt)]
        shared_counts = (s @ t.T)
        row["pairs_sharing_an_edit_share"] = float((shared_counts > 0).sum() / (len(src) * len(tgt)))
        row["shared_edits_mean"] = float(shared_counts.sum() / (len(src) * len(tgt)))
    return row


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--ot-block", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "SUCCESS").exists():
        raise SystemExit(f"{args.out}/SUCCESS exists; the audit is not rerun into the same directory")
    args.out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def say(text=""):
        lines.append(str(text))
        print(text, flush=True)

    # Inventory.
    inventory = [{"path": str(p.relative_to(args.tracer)), "bytes": p.stat().st_size}
                 for p in sorted(args.tracer.rglob("*")) if p.is_file()]
    pd.DataFrame(inventory).to_csv(args.out / "inventory.tsv", sep="\t", index=False)
    heads = []
    for item in inventory:
        name = item["path"].lower()
        if any(word in name for word in ("prior", "expansion", "fitness", "allele")) and item["bytes"] < 2e9:
            path = args.tracer / item["path"]
            try:
                with open(path, "rb") as handle:
                    head = handle.read(1500)
                text = head.decode("utf-8", errors="replace") if b"\x00" not in head else f"<binary, {len(head)} bytes read>"
            except OSError as error:
                text = f"<unreadable: {error}>"
            heads.append(f"===== {item['path']} ({item['bytes']} bytes)\n{text}\n")
    (args.out / "file_heads.txt").write_text("\n".join(heads), encoding="utf-8")
    kinds = pd.Series([p["path"].split("/")[0] for p in inventory]).value_counts()
    say(f"inventory: {len(inventory)} files; top-level: {kinds.to_dict()}")
    say(f"prior/expansion/fitness/allele files: {len(heads)}")

    meta = pd.read_csv(args.tracer / "KPTracer_meta.csv", index_col=0, low_memory=False)
    meta.index = meta.index.astype(str)
    tumour_of = meta["Tumor"].astype(str)

    manifest = pd.read_csv(args.prepared / "pair_manifest_all.csv")
    manifest["same_met_family"] = manifest["same_met_family"].fillna(False).astype(bool)
    eligible = manifest[manifest["eligible"].astype(bool)].copy()

    trees: dict = {}

    def load(stem, kind):
        key = (stem, kind)
        if key not in trees:
            path = d1.tree_path(args.tracer, stem, kind)
            trees[key] = d1.Tree(path) if path.exists() else None
        return trees[key]

    rng = np.random.default_rng(SEED)
    rows, hist_rows, char_rows = [], [], []
    for row in eligible.to_dict("records"):
        pair_id = row["pair_id"]
        same = bool(row["same_met_family"])
        prefix = str(row["target_sample"]).rsplit("_", 1)[0]
        candidates = ([(f"{row['target_met_family']}_Fam", "cassiopeia"), (f"{prefix}_All", "cassiopeia")] if same
                      else [(f"{prefix}_All", "cassiopeia"), (f"{prefix}_All", "neighbor_joining")])
        base = {"pair_id": pair_id, "mouse": str(row["patient_id"]), "source_tumor": row["source_sample"],
                "target_tumor": row["target_sample"], "same_met_family": same}
        cells = gated_cells(args.ot_block, LABEL, pair_id)
        cells_raw = gated_cells(args.ot_block, "raw", pair_id)
        if cells is None:
            rows.append({**base, "status": "no completed OT output"})
            continue
        source, target = cells
        base["source_cells"], base["target_cells"] = len(source), len(target)
        base["raw_arm_same_cells"] = (cells_raw is not None and set(cells_raw[0]) == set(source)
                                      and set(cells_raw[1]) == set(target))
        chosen = None
        for stem, kind in candidates:
            tree = load(stem, kind)
            if tree is not None and any(c in tree.leaf_of for c in source) and any(c in tree.leaf_of for c in target):
                chosen = (stem, kind, tree)
                break
        if chosen is None:
            rows.append({**base, "status": "unsupported: no family tree holds both sides"})
            continue
        stem, kind, tree = chosen
        s_leaves = np.asarray([tree.leaf_of[c] for c in source if c in tree.leaf_of], dtype=int)
        t_leaves = np.asarray([tree.leaf_of[c] for c in target if c in tree.leaf_of], dtype=int)
        hist = lca_distribution(tree, s_leaves, t_leaves)
        info = informativeness(hist)
        for depth, mass in hist.items():
            hist_rows.append({"pair_id": pair_id, "tree": stem, "kind": kind, "mrca_depth": int(depth), "mass": float(mass)})
        record = {**base, "status": "shared tree", "tree": stem, "kind": kind,
                  "source_mapped": len(s_leaves), "target_mapped": len(t_leaves),
                  "source_mapped_share": len(s_leaves) / len(source), "target_mapped_share": len(t_leaves) / len(target),
                  **tree.resolution, **info}
        # Neighbor-joining reconstruction of the same name, when it holds both sides.
        nj = load(stem, "neighbor_joining") if kind == "cassiopeia" else None
        if nj is not None and all(c in nj.leaf_of for c in source) and all(c in nj.leaf_of for c in target):
            nj_info = informativeness(lca_distribution(
                nj, np.asarray([nj.leaf_of[c] for c in source]), np.asarray([nj.leaf_of[c] for c in target])))
            record.update({f"nj_{k}": v for k, v in nj_info.items()})
            take = min(SAMPLE, len(source) * len(target))
            si = rng.integers(0, len(source), take)
            ti = rng.integers(0, len(target), take)
            cass = sampled_lca_depths(tree, [(tree.leaf_of[source[i]], tree.leaf_of[target[j]]) for i, j in zip(si, ti)])
            njd = sampled_lca_depths(nj, [(nj.leaf_of[source[i]], nj.leaf_of[target[j]]) for i, j in zip(si, ti)])
            record["cass_vs_nj_spearman_L"] = float(stats.spearmanr(cass, njd)[0]) if np.ptp(cass) and np.ptp(njd) else np.nan
            record["cass_vs_nj_pairs_sampled"] = int(take)
        else:
            record["cass_vs_nj_spearman_L"] = np.nan
        matrix = read_matrix(args.tracer, stem) if kind == "cassiopeia" else None
        if matrix is not None:
            stats_row = character_stats(matrix, source, target, tumour_of)
            char_rows.append({"pair_id": pair_id, "matrix": stem, **stats_row})
            record["character_matrix"] = stem
            record["pairs_sharing_an_edit_share"] = stats_row.get("pairs_sharing_an_edit_share")
        else:
            record["character_matrix"] = ""
        rows.append(record)
        say(f"{pair_id} [{'same' if same else 'other'}] {stem} ({kind}): L var {info['L_variance']:.3f}, "
            f"modal share {info['L_modal_share']:.3f}, distinct {info['L_distinct']}")

    eligibility = pd.DataFrame(rows)
    eligibility.to_csv(args.out / "LINEAGE_PAIR_ELIGIBILITY.tsv", sep="\t", index=False)
    pd.DataFrame(hist_rows).to_csv(args.out / "lineage_value_histograms.tsv", sep="\t", index=False)
    pd.DataFrame(char_rows).to_csv(args.out / "character_stats.tsv", sep="\t", index=False)
    shared = eligibility[eligibility["status"].eq("shared tree")]
    summary = {
        "eligible_pairs": int(len(eligibility)),
        "shared_tree_pairs": {"same_family": int(shared["same_met_family"].sum()),
                              "different_family": int((~shared["same_met_family"]).sum())},
        "unsupported_pairs": int(eligibility["status"].str.startswith("unsupported").sum()),
        "zero_variance_pairs": shared.loc[shared["L_variance"] <= 0, "pair_id"].tolist(),
        "raw_arm_cell_sets_identical": bool(eligibility["raw_arm_same_cells"].fillna(False).all()),
        "files": {"inventory": len(inventory), "prior_expansion_fitness_allele": len(heads)},
    }
    (args.out / "audit_v3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    say(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_rows", 200):
        cols = ["pair_id", "same_met_family", "tree", "kind", "source_mapped", "target_mapped", "L_mean", "L_variance",
                "L_distinct", "L_modal_share", "L_root_share", "cass_vs_nj_spearman_L", "pairs_sharing_an_edit_share"]
        say(shared[[c for c in cols if c in shared]].round(3).to_string(index=False))
    (args.out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    provenance = {"script": str(Path(__file__).resolve()), "md5": md5(Path(__file__)), "seed": SEED,
                  "label_for_cells": LABEL, "arguments": {k: str(v) for k, v in vars(args).items()}}
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (args.out / "SUCCESS").write_text("done\n", encoding="utf-8")
    print("DONE")


if __name__ == "__main__":
    main()

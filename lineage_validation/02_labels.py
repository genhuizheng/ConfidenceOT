"""Lineage labels and tree leaves of every cell in the lineage-validation sides (VALIDATION_PROTOCOL.md §5).

    python lineage_validation/02_labels.py --root ROOT

Reads ROOT/inputs/cell_sites.tsv.gz (written by 01_build_inputs.py), the KP-Tracer tumour table
and trees, and the Quinn converted H5AD obs and per-clone trees. Writes ROOT/labels/:
  cells.tsv.gz        dataset, mouse, cell, site, lineage ('' = unknown; KP-Tracer '_Unk' kept as written)
  trees.tsv           one row per (dataset, mouse, lineage): the tree file used (§5.4), or why none,
                      its leaves, the lineage's cells, mapped cells, leaves or cells left unmapped
                      because they matched more than once, other files listed for the clone
  tree_leaves.tsv.gz  dataset, mouse, lineage, tree, cell, leaf (mapped cells only)
  provenance.json, SUCCESS
Refuses if SUCCESS exists. Labels never go back into an OT input.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lv_common as lv  # noqa: E402

TRACER = Path("/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data")
KP_ROOT = Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot")
QUINN = Path("/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363/converted")
QUINN_TREE = re.compile(r"_lg(\d+)_tree_hybrid_priors\.alleleThresh\.processed\.txt$", re.IGNORECASE)


def clone_name(value) -> str:
    """LineageGroup as one string: '3', 3 and 3.0 are the same clone; missing is unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value) in ("", "nan", "NaN", "None"):
        return lv.UNKNOWN
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except ValueError:
        pass
    return text


def clone_number(name: str) -> str | None:
    found = re.findall(r"\d+", name)
    return str(int(found[-1])) if found else None


def map_leaves(leaf_names: list[str], cells: list[str], key) -> tuple[dict, int, int]:
    """Exact matching (§5.4): {cell: leaf}; a leaf or a cell that matches more than once is unmapped."""
    leaf_count = Counter(leaf_names)
    by_key: dict[str, list[str]] = {}
    for cell in cells:
        k = key(cell)
        if k is not None:
            by_key.setdefault(k, []).append(cell)
    mapping, dup_leaves, dup_cells = {}, 0, 0
    for leaf, count in leaf_count.items():
        hits = by_key.get(leaf, [])
        if not hits:
            continue
        if count > 1:
            dup_leaves += 1
            continue
        if len(hits) > 1:
            dup_cells += len(hits)
            continue
        mapping[hits[0]] = leaf
    return mapping, dup_leaves, dup_cells


def kp_labels(sites: pd.DataFrame, tracer: Path, kp_root: Path, say) -> tuple[pd.DataFrame, list, list]:
    tumours = pd.read_csv(kp_root / "prepared" / "kptracer_tumours.csv", dtype=str)
    family = {(m, t): f for m, t, f in zip(tumours["mouse"], tumours["tumor"], tumours["met_family"].fillna(lv.UNKNOWN))}
    cells = sites[sites["dataset"].eq("KPTracer")].copy()
    cells["lineage"] = [family.get((m, s), lv.UNKNOWN) for m, s in zip(cells["mouse"], cells["site"])]
    d1 = lv.tree_module()
    tree_rows, leaf_rows = [], []
    for (mouse, lineage), part in cells.groupby(["mouse", "lineage"]):
        row = {"dataset": "KPTracer", "mouse": mouse, "lineage": lineage, "lineage_cells": len(part)}
        if lineage == lv.UNKNOWN or lineage.endswith("_Unk"):
            tree_rows.append({**row, "status": "unknown lineage"}); continue
        fam = d1.tree_path(tracer, f"{lineage}_Fam", "cassiopeia")
        alls = sorted((tracer / "trees").glob(f"{mouse}_*_All_tree.nwk"))
        if fam.exists():
            path = fam
        elif len(alls) == 1:
            path = alls[0]
        else:
            tree_rows.append({**row, "status": "no tree" if not alls else f"{len(alls)} _All trees"}); continue
        tree = d1.Tree(path)
        names = [tree.name[v] for v in tree.leaves]
        mapping, dup_leaves, dup_cells = map_leaves(names, part["cell"].tolist(), lambda c: c)
        tree_rows.append({**row, "status": "tree", "tree": str(path), "leaves": len(names), "mapped_cells": len(mapping),
                          "duplicate_leaves": dup_leaves, "cells_matching_twice": dup_cells, "other_files": ""})
        leaf_rows += [{"dataset": "KPTracer", "mouse": mouse, "lineage": lineage, "tree": str(path), "cell": c, "leaf": l}
                      for c, l in mapping.items()]
        say(f"KP {mouse} {lineage}: {path.name}, {len(mapping)} of {len(part)} cells mapped")
    return cells, tree_rows, leaf_rows


def quinn_labels(sites: pd.DataFrame, quinn: Path, say) -> tuple[pd.DataFrame, list, list]:
    import anndata as ad
    cells = sites[sites["dataset"].eq("Quinn2021")].copy()
    parts, tree_rows, leaf_rows = [], [], []
    d1 = lv.tree_module()
    for mouse, part in cells.groupby("mouse"):
        data = ad.read_h5ad(quinn / "h5ad" / f"{mouse}.h5ad", backed="r")
        obs = data.obs.copy()
        obs.index = obs.index.astype(str)
        data.file.close()
        missing = sorted(set(part["cell"]) - set(obs.index))
        if missing:
            raise KeyError(f"Quinn {mouse}: {len(missing)} side cells absent from the converted obs, e.g. {missing[:3]}")
        part = part.copy()
        part["lineage"] = [clone_name(v) for v in obs.loc[part["cell"], "LineageGroup"].to_numpy(object)]
        sample = obs.loc[part["cell"], "sampleID"].astype(str).to_numpy()
        if not np.array_equal(sample, part["site"].astype(str).to_numpy()):
            raise ValueError(f"Quinn {mouse}: sampleID differs from the side's site")
        parts.append(part)
        files = sorted(p for p in (quinn / "lineage" / mouse / "trees").rglob("*") if p.is_file())
        by_clone: dict[str, list[Path]] = {}
        for p in files:
            hit = QUINN_TREE.search(p.name)
            if hit:
                by_clone.setdefault(str(int(hit.group(1))), []).append(p)
        site_of = dict(zip(part["cell"], part["site"].astype(str)))
        for lineage, members in part.groupby("lineage"):
            row = {"dataset": "Quinn2021", "mouse": mouse, "lineage": lineage, "lineage_cells": len(members)}
            if lineage == lv.UNKNOWN:
                tree_rows.append({**row, "status": "unknown lineage"}); continue
            number = clone_number(lineage)
            chosen = by_clone.get(number, []) if number else []
            others = [str(p.relative_to(quinn)) for p in files if number and re.search(rf"(?i)_lg{number}_", p.name) and p not in chosen]
            if len(chosen) != 1:
                tree_rows.append({**row, "status": "no tree file" if not chosen else f"{len(chosen)} tree files",
                                  "other_files": ";".join(others)}); continue
            path = chosen[0]
            try:
                tree = d1.Tree(path)
            except Exception as error:  # noqa: BLE001 - listed, not substituted
                tree_rows.append({**row, "status": f"unreadable: {type(error).__name__}", "tree": str(path)}); continue
            names = [tree.name[v] for v in tree.leaves]
            if mouse == "10k":
                # §5.4: in 10k a cell is <sampleID>.<leaf name>.
                def key(cell, site_of=site_of):
                    prefix = site_of[cell] + "."
                    return cell[len(prefix):] if cell.startswith(prefix) else None
            else:
                def key(cell):
                    return cell
            mapping, dup_leaves, dup_cells = map_leaves(names, members["cell"].tolist(), key)
            tree_rows.append({**row, "status": "tree", "tree": str(path), "leaves": len(names), "mapped_cells": len(mapping),
                              "duplicate_leaves": dup_leaves, "cells_matching_twice": dup_cells, "other_files": ";".join(others)})
            leaf_rows += [{"dataset": "Quinn2021", "mouse": mouse, "lineage": lineage, "tree": str(path), "cell": c, "leaf": l}
                          for c, l in mapping.items()]
        say(f"Quinn {mouse}: {part['lineage'].ne(lv.UNKNOWN).sum()} of {len(part)} cells with a clone; "
            f"{sum(1 for r in tree_rows if r['mouse'] == mouse and r.get('status') == 'tree')} clone trees used")
    return (pd.concat(parts, ignore_index=True) if parts else cells.assign(lineage=[])), tree_rows, leaf_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tracer", type=Path, default=TRACER)
    parser.add_argument("--kp-root", type=Path, default=KP_ROOT)
    parser.add_argument("--quinn", type=Path, default=QUINN)
    args = parser.parse_args()
    lv.check_frozen()
    out = args.root / "labels"
    if (out / "SUCCESS").exists():
        sys.exit(f"{out}/SUCCESS exists; labels are not rebuilt")
    out.mkdir(parents=True, exist_ok=True)
    sites_path = args.root / "inputs" / "cell_sites.tsv.gz"
    sites = pd.read_csv(sites_path, sep="\t", dtype=str).drop_duplicates(["dataset", "mouse", "cell"])
    lines = []

    def say(text):
        print(text, flush=True)
        lines.append(text)

    kp, kp_trees, kp_leaves = kp_labels(sites, args.tracer, args.kp_root, say)
    quinn, q_trees, q_leaves = quinn_labels(sites, args.quinn, say)
    cells = pd.concat([kp, quinn], ignore_index=True)[["dataset", "mouse", "cell", "site", "lineage"]]
    if cells.duplicated(["dataset", "mouse", "cell"]).any():
        raise ValueError("a cell with two sites")
    cells.to_csv(out / "cells.tsv.gz", sep="\t", index=False, compression="gzip", lineterminator="\n")
    pd.DataFrame(kp_trees + q_trees).to_csv(out / "trees.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame(kp_leaves + q_leaves, columns=["dataset", "mouse", "lineage", "tree", "cell", "leaf"]).to_csv(
        out / "tree_leaves.tsv.gz", sep="\t", index=False, compression="gzip", lineterminator="\n")
    lv.write_json(out / "provenance.json", {"commit": lv.commit(), "protocol": lv.FROZEN, "cell_sites_md5": lv.md5(sites_path),
                                            "tracer": str(args.tracer), "kp_root": str(args.kp_root),
                                            "quinn": str(args.quinn), "log": lines})
    (out / "SUCCESS").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()

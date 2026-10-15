"""Visualization-ready export of the KP-Tracer validation (validation_v2, Phase 2).

    python kp_lineage/validation_v2/05_export_package.py \\
        --d1 LINEAGE_DIR --readout READOUT_DIR --audit AUDIT_DIR \\
        --tracer KPTracer-Data --prepared PREPARED --ot-block OT/kptracer --out EXPORT_DIR

Copies and joins finished results; computes no new statistic. Everything is
written under --out, which must not already hold MANIFEST.tsv.

    tables/cells.csv.gz          every M4-E cell of every eligible pair, both labels:
                                 ids, side, retained, gate score, and s on the primary tree
    tables/cell_lineage.csv.gz   s and r of every mapped source cell on every covering tree
    tables/pair_gates.csv        per pair and label: retained fractions and the M4-E metrics
    tables/leaf_map.csv.gz       every leaf of every exported tree -> cell id, tumour, mouse
    lineage/                     D1 outputs as written (pairs, metastases, mice, summary, ...)
    readout_tumour_level/        the original tumour-level readout, as written
    audit/                       the lineage data audit outputs, as written
    trees/                       the covering family trees, byte-identical newick, plus
                                 Cassiopeia character matrices where present
    metadata/                    KPTracer_meta rows of every exported cell and leaf
    config/                      preregistration, readout prespecification, audit document,
                                 pair manifest, per-pair run.json and pair_metrics.csv
    code/                        the scripts that produced these files
    MANIFEST.tsv, MD5SUMS, README.md
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

METHOD = "M4-E"
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("audit03", HERE / "03_audit_lineage.py")
audit03 = importlib.util.module_from_spec(spec)
sys.modules["audit03"] = audit03
spec.loader.exec_module(audit03)
REPO_DOCS = [
    HERE / "VALIDATION_V2_PREREGISTRATION.md",
    HERE / "LINEAGE_DATA_AUDIT.md",
    HERE / "VALIDATION_V2_EXECUTION_PLAN.md",
    HERE.parent / "READOUT_PRESPECIFICATION_2026-10-14.md",
]
CODE = [
    HERE.parent / "01_prepare_kptracer_pairs.py",
    HERE.parent / "02_kptracer_readout.py",
    HERE / "03_audit_lineage.py",
    HERE / "04_lineage_concordance.py",
    HERE / "05_export_package.py",
    HERE / "submit_v2.sh",
]
DESCRIPTIONS = {
    "tables/cells.csv.gz": "One row per label, pair and M4-E cell of every eligible pair: cell_id, mouse_id, tumor_id, pair_id, "
                           "side, same_met_family, retained, raw_retained, signed_rejection_margin, gate_score_r "
                           "(= -signed_rejection_margin), decision_cost, rejection_cost, budget_overridden; for same-family "
                           "covered pairs also s_primary (shared-ancestry depth on the primary tree), primary_tree, in_primary_tree.",
    "tables/cell_lineage.csv.gz": "One row per label, pair, covering tree and lineage-mapped source cell: s, r, retained, "
                                  "tree, kind, role (primary or sensitivity tree), predownsample_total_counts, detected_genes.",
    "tables/pair_gates.csv": "One row per label and eligible pair: cell counts and retained fractions per side, and the M4-E row of pair_metrics.csv.",
    "tables/leaf_map.csv.gz": "One row per leaf of every exported tree: tree, kind, leaf_name, cell_id (leaf names equal cell ids exactly), "
                              "in_meta, tumor_id, mouse_id, met_family.",
    "README.md": "Description of the package.",
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if md5(source) != md5(destination):
        raise RuntimeError(f"copy of {source} does not match")


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            copy(path, destination / path.relative_to(source))


def tree_file(tracer: Path, stem: str, kind: str) -> Path:
    if kind == "cassiopeia":
        return tracer / "trees" / f"{stem}_tree.nwk"
    return tracer / "trees" / "neighbor_joining" / f"{stem}_tree_nj.processed.tree"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--ot-block", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["ranknm256_noscale_ds_cos", "raw"])
    args = parser.parse_args()

    if (args.out / "MANIFEST.tsv").exists():
        raise SystemExit(f"{args.out}/MANIFEST.tsv exists; the export is not rewritten")
    for needed in (args.d1 / "SUCCESS", args.readout / "readout_summary.json", args.audit / "audit.json"):
        if not needed.exists():
            raise SystemExit(f"{needed} is missing; export only finished results")
    args.out.mkdir(parents=True, exist_ok=True)
    tables = args.out / "tables"
    tables.mkdir(exist_ok=True)

    # Finished outputs, as written.
    copy_tree(args.d1, args.out / "lineage")
    copy_tree(args.readout, args.out / "readout_tumour_level")
    copy_tree(args.audit, args.out / "audit")
    for doc in REPO_DOCS:
        copy(doc, args.out / "config" / doc.name)
    for script in CODE:
        copy(script, args.out / "code" / script.name)
    copy(args.prepared / "pair_manifest_all.csv", args.out / "config" / "pair_manifest_all.csv")

    manifest = pd.read_csv(args.prepared / "pair_manifest_all.csv")
    manifest["same_met_family"] = manifest["same_met_family"].fillna(False).astype(bool)
    eligible = manifest[manifest["eligible"].astype(bool)].copy()
    meta = pd.read_csv(args.tracer / "KPTracer_meta.csv", index_col=0, low_memory=False)
    meta.index = meta.index.astype(str)

    lineage = pd.read_csv(args.d1 / "cell_scores.csv.gz")
    lineage["observation_id"] = lineage["observation_id"].astype(str)
    lineage = lineage.rename(columns={"observation_id": "cell_id"})
    lineage.to_csv(tables / "cell_lineage.csv.gz", index=False, compression="gzip")
    primary = lineage[lineage["role"].eq("primary")][["label", "pair_id", "cell_id", "s", "tree"]].rename(
        columns={"s": "s_primary", "tree": "primary_tree"})
    pair_trees = pd.read_csv(args.d1 / "pairs.tsv", sep="\t")
    primary_tree_of = pair_trees.dropna(subset=["tree"]).set_index(["label", "pair_id"])["tree"].to_dict()

    cell_tables, gate_rows = [], []
    for label in args.labels:
        for row in eligible.to_dict("records"):
            outputs = sorted((args.ot_block / label / str(row["pair_id"])).glob("scope_malignant/*/SUCCESS"))
            if len(outputs) != 1:
                gate_rows.append({"label": label, "pair_id": row["pair_id"], "status": f"{len(outputs)} completed outputs"})
                continue
            folder = outputs[0].parent
            for name in ("run.json", "pair_metrics.csv"):
                if (folder / name).exists():
                    copy(folder / name, args.out / "config" / "ot_runs" / label / str(row["pair_id"]) / name)
            cells = pd.read_csv(folder / "cell_confidence.csv")
            cells = cells[cells["method"].astype(str).eq(METHOD)].copy()
            cells["cell_id"] = cells["observation_id"].astype(str)
            cells["tumor_id"] = [row["source_sample"] if side == "source" else row["target_sample"] for side in cells["side"]]
            cells["gate_score_r"] = -cells["signed_rejection_margin"].astype(float)
            keep = ["cell_id", "tumor_id", "side", "retained", "raw_retained", "signed_rejection_margin", "gate_score_r",
                    "decision_cost", "rejection_cost", "budget_overridden"]
            table = cells[[c for c in keep if c in cells]].assign(
                label=label, pair_id=row["pair_id"], mouse_id=str(row["patient_id"]),
                same_met_family=bool(row["same_met_family"]))
            cell_tables.append(table)
            metrics = pd.read_csv(folder / "pair_metrics.csv")
            metric = metrics[metrics["method"].astype(str).eq(METHOD)].iloc[0].to_dict()
            source, target = table[table["side"].eq("source")], table[table["side"].eq("target")]
            gate_rows.append({
                "label": label, "pair_id": row["pair_id"], "mouse_id": str(row["patient_id"]),
                "source_tumor": row["source_sample"], "target_tumor": row["target_sample"],
                "same_met_family": bool(row["same_met_family"]), "status": "completed",
                "source_cells": len(source), "target_cells": len(target),
                "source_retained_fraction": float(source["retained"].astype(bool).mean()),
                "target_retained_fraction": float(target["retained"].astype(bool).mean()),
                **{f"m4e_{k}": v for k, v in metric.items() if k != "method"},
            })
    pd.DataFrame(gate_rows).to_csv(tables / "pair_gates.csv", index=False)

    # Trees that cover any pair, byte-identical, and their leaves.
    all_trees = pd.read_csv(args.d1 / "pair_trees.tsv", sep="\t").dropna(subset=["tree"])
    leaf_rows = []
    for stem, kind in sorted(set(zip(all_trees["tree"], all_trees["kind"]))):
        path = tree_file(args.tracer, stem, kind)
        copy(path, args.out / "trees" / kind / path.name)
        matrix = args.tracer / "trees" / f"{stem}_character_matrix.txt"
        if kind == "cassiopeia" and matrix.exists():
            copy(matrix, args.out / "trees" / "character_matrices" / matrix.name)
        parsed = audit03.parse_newick(path.read_text())
        names = [parsed["name"][v] for v in range(len(parsed["parent"])) if not parsed["children"][v]]
        for name in names:
            leaf_rows.append({"tree": stem, "kind": kind, "tree_file": f"trees/{kind}/{path.name}", "leaf_name": name,
                              "cell_id": name, "in_meta": name in meta.index})
    leaves = pd.DataFrame(leaf_rows)
    mapped = leaves["in_meta"]
    leaves.loc[mapped, "tumor_id"] = meta.loc[leaves.loc[mapped, "cell_id"], "Tumor"].astype(str).to_numpy()
    leaves.loc[mapped, "mouse_id"] = meta.loc[leaves.loc[mapped, "cell_id"], "Mouse"].astype(str).to_numpy()
    leaves.loc[mapped, "met_family"] = meta.loc[leaves.loc[mapped, "cell_id"], "MetFamily"].astype(str).to_numpy()
    leaves.to_csv(tables / "leaf_map.csv.gz", index=False, compression="gzip")

    cells = pd.concat(cell_tables, ignore_index=True)
    cells = cells.merge(primary, on=["label", "pair_id", "cell_id"], how="left")
    cells["primary_tree"] = [primary_tree_of.get((l, p)) for l, p in zip(cells["label"], cells["pair_id"])]
    cassiopeia_leaves = set(zip(leaves.loc[leaves["kind"].eq("cassiopeia"), "tree"], leaves.loc[leaves["kind"].eq("cassiopeia"), "cell_id"]))
    cells["in_primary_tree"] = [(t, c) in cassiopeia_leaves for t, c in zip(cells["primary_tree"], cells["cell_id"])]
    order = ["label", "pair_id", "mouse_id", "tumor_id", "cell_id", "side", "same_met_family", "retained", "raw_retained",
             "signed_rejection_margin", "gate_score_r", "decision_cost", "rejection_cost", "budget_overridden",
             "primary_tree", "in_primary_tree", "s_primary"]
    cells = cells[[c for c in order if c in cells]]
    cells.to_csv(tables / "cells.csv.gz", index=False, compression="gzip")

    wanted = sorted(set(cells["cell_id"]) | set(leaves.loc[mapped, "cell_id"]))
    present = [c for c in wanted if c in meta.index]
    (args.out / "metadata").mkdir(exist_ok=True)
    meta.loc[present].rename_axis("cell_id").to_csv(args.out / "metadata" / "kptracer_meta_subset.csv.gz", compression="gzip")

    def git(*command):
        try:
            return subprocess.run(["git", "-C", str(HERE), *command], capture_output=True, text=True).stdout.strip()
        except OSError:
            return "unknown"
    readme = [
        "# KP-Tracer lineage validation: export package",
        "",
        f"Written by `code/05_export_package.py` at commit {git('rev-parse', 'HEAD')}.",
        "",
        "- Primary representation: `ranknm256_noscale_ds_cos`. Comparison: `raw`. Column `label` names it.",
        "- Gate: M4-E, acceptance 0.99. `retained` is True when `signed_rejection_margin` < 0; `gate_score_r` = -margin.",
        "- `s_primary`: shared-ancestry depth on the pair's primary tree, preregistration 1.2 (unary nodes contracted).",
        "- Truth: the authors' MetFamily, Tumor and is_normal only.",
        "- Statistics: `lineage/summary.json` (E1, E1b, sensitivity) and `readout_tumour_level/readout_summary.json`.",
        "- Trees are the authors' files, unmodified (md5 in MANIFEST.tsv). Leaf names equal cell ids exactly.",
        "- Not included yet: the cross-mouse negative control (steps D2-D4) and baselines (E3); they have not been run.",
        "",
        "Every file is listed in MANIFEST.tsv with its size, md5 and, for tables, its row count.",
    ]
    (args.out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    rows = []
    for path in sorted(args.out.rglob("*")):
        if not path.is_file() or path.name in ("MANIFEST.tsv", "MD5SUMS"):
            continue
        relative = path.relative_to(args.out).as_posix()
        count = None
        if path.suffix in (".csv", ".tsv") or path.name.endswith((".csv.gz", ".tsv.gz")):
            count = int(len(pd.read_csv(path, sep="\t" if ".tsv" in path.name else ",", low_memory=False)))
        top = relative.split("/")[0]
        description = DESCRIPTIONS.get(relative, {
            "lineage": "D1 output, as written by 04_lineage_concordance.py",
            "readout_tumour_level": "original tumour-level readout, as written by 02_kptracer_readout.py",
            "audit": "lineage data audit output, as written by 03_audit_lineage.py",
            "trees": "authors' tree or character matrix, byte-identical",
            "metadata": "KPTracer_meta.csv rows of every exported cell and leaf, all original columns",
            "config": "configuration or document, copied",
            "code": "script that produced the package",
        }.get(top, ""))
        rows.append({"path": relative, "bytes": path.stat().st_size, "md5": md5(path), "rows": count, "description": description})
    listing = pd.DataFrame(rows)
    listing["rows"] = listing["rows"].astype("Int64")
    listing.to_csv(args.out / "MANIFEST.tsv", sep="\t", index=False)
    with open(args.out / "MD5SUMS", "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['md5']}  {row['path']}\n")
    (args.out / "manifest_md5.txt").write_text(md5(args.out / "MANIFEST.tsv") + "  MANIFEST.tsv\n", encoding="utf-8")
    print(f"files {len(rows)} (plus MANIFEST.tsv, MD5SUMS, manifest_md5.txt)")
    print("DONE")


if __name__ == "__main__":
    main()

"""Export real lineage data and the existing lineage-validation OT outputs of a few mice, for local plotting.

    python lineage_validation/local_lineage_visualization/tacc_export.py --out EXPORT_DIR

Read only. Nothing is re-fitted and no validation metric is recomputed; the only derived table is
the per-cell display of each saved coupling (``matching/``, below), from the protocol's own q_ij
(VALIDATION_PROTOCOL.md §6). Everything goes under EXPORT_DIR, which must not exist.

  lineage/KPTracer/          the authors' Cassiopeia _All and _Fam trees and character matrices of the
                             chosen mice, byte-identical copies
  lineage/Quinn2021/<m>/     the authors' character_matrices/ and trees/ folders as converted (files
                             above --max-file-mb are listed in skipped_files.tsv, not copied)
  meta/                      KPTracer_meta.csv rows of the chosen mice; the Quinn converted obs (every
                             column, every cell) of the chosen mice
  labels/                    the frozen pipeline's lineage labels, tree list and leaf -> cell mapping,
                             restricted to the chosen mice
  ot/gates/                  cell_confidence.csv, pair_metrics.csv, run.json of every run of the chosen
                             mice (5 strategies x designs), as written by 02_run_pair.py
  ot/cell_states/            per run and method: every analysed cell with state retained / rejected /
                             excluded (two-step: removed at step 1) and its gate margin
  ot/matching/               per run and method, per target cell: retained-source mass D_j, the source
                             group (KP-Tracer primary tumour, Quinn LL clone) with the largest W_j(g),
                             its weight, ties, the runner-up, the single source cell with the largest
                             q_ij, and every W_j(g) when there are at most 12 groups
  ot/scores/                the existing scores (03_score_runs.py outputs) of these runs, as written
  summary/                   the existing summary tables (05_aggregate.py), as written
  umap/                      the fixed joint UMAP coordinates (48_reference_umap.py) of these pairs
  figures_existing/          existing 06_figures.py images of these mice
  couplings/                 coupling_m4e.npz / coupling_m4r.npz, byte-identical, for --coupling-mice only
  audit/                     cell_id_mapping.tsv (tree leaves, matrix rows, metadata, OT cells),
                             deg_gsea_inventory.tsv, skipped_files.tsv
  MANIFEST.tsv (path, bytes, md5), provenance.json, SUCCESS
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402

LV_ROOT = Path("/scratch/10119/ghzheng/OT_project/lineage_validation")
TRACER = Path("/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data")
KP_ROOT = Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot")
QUINN = Path("/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363/converted")
MAX_GROUP_COLUMNS = 12


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def mouse_of(pair: str) -> str:
    return pair.split("__")[0]


def export_lineage(args, out: Path, say) -> list[dict]:
    skipped = []
    for mouse in args.kp_mice:
        stems = sorted(p.name[: -len("_tree.nwk")] for p in (args.tracer / "trees").glob(f"{mouse}_*_tree.nwk")
                       if p.name.endswith(("_All_tree.nwk", "_Fam_tree.nwk")))
        for stem in stems:
            for name in (f"{stem}_tree.nwk", f"{stem}_character_matrix.txt"):
                src = args.tracer / "trees" / name
                if src.exists():
                    copy(src, out / "lineage" / "KPTracer" / name)
                else:
                    skipped.append({"file": str(src), "reason": "absent"})
        say(f"KP {mouse}: trees {stems}")
    meta = pd.read_csv(args.tracer / "KPTracer_meta.csv", index_col=0, low_memory=False)
    meta.index.name = "cell_id"
    for mouse in args.kp_mice:
        part = meta[meta["Mouse"].astype(str).eq(mouse)]
        (out / "meta").mkdir(parents=True, exist_ok=True)
        part.to_csv(out / "meta" / f"KPTracer_meta_{mouse}.csv.gz", compression="gzip")
        say(f"KP {mouse}: {len(part)} metadata rows")
    import anndata as ad
    for mouse in args.quinn_mice:
        base = args.quinn / "lineage" / mouse
        for kind in ("character_matrices", "trees"):
            for src in sorted(p for p in (base / kind).rglob("*") if p.is_file()):
                size = src.stat().st_size
                if size > args.max_file_mb * 1e6:
                    skipped.append({"file": str(src), "reason": f"{size / 1e6:.1f} MB > {args.max_file_mb} MB"})
                    continue
                copy(src, out / "lineage" / "Quinn2021" / mouse / src.relative_to(base))
        for src in sorted(p for p in base.iterdir() if p.is_file()):
            skipped.append({"file": str(src), "reason": f"allele table, {src.stat().st_size / 1e6:.1f} MB, not needed for the figures"})
        data = ad.read_h5ad(args.quinn / "h5ad" / f"{mouse}.h5ad", backed="r")
        obs = data.obs.copy()
        data.file.close()
        obs.index = obs.index.astype(str)
        obs.index.name = "cell_id"
        obs.to_csv(out / "meta" / f"Quinn2021_obs_{mouse}.csv.gz", compression="gzip")
        say(f"Quinn {mouse}: {len(obs)} obs rows, lineage folders copied")
    return skipped


def export_labels(args, out: Path, mice: dict) -> None:
    labels = args.lv_root / "labels"
    cells = pd.read_csv(labels / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    trees = pd.read_csv(labels / "trees.tsv", sep="\t", dtype=str, keep_default_na=False)
    leaves = pd.read_csv(labels / "tree_leaves.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    keep = lambda t: t[[(d, m) in mice for d, m in zip(t["dataset"], t["mouse"])]]
    (out / "labels").mkdir(parents=True, exist_ok=True)
    keep(cells).to_csv(out / "labels" / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    keep(trees).to_csv(out / "labels" / "trees.tsv", sep="\t", index=False)
    keep(leaves).to_csv(out / "labels" / "tree_leaves.tsv.gz", sep="\t", index=False, compression="gzip")


def run_list(args, mice: dict) -> list[dict]:
    items = []
    for block, dataset in lv.BLOCKS.items():
        table = pd.read_csv(args.lv_root / "inputs" / "manifests" / f"{block}_raw.csv", dtype=str)
        for pair, mouse in zip(table["pair_id"], table["patient_id"]):
            if (dataset, mouse) in mice:
                items += [{"block": block, "dataset": dataset, "pair": pair, "mouse": mouse, "strategy": s}
                          for s in lv.STRATEGIES]
    return items


def export_runs(args, out: Path, items: list[dict], say) -> None:
    runs = args.lv_root / "runs" / "acc0.99"
    labels = pd.read_csv(out / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    lineage = {k: g.set_index("cell")["lineage"] for k, g in labels.groupby(["dataset", "mouse"])}
    for item in items:
        block, pair, strategy, dataset = item["block"], item["pair"], item["strategy"], item["dataset"]
        try:
            directory = lv.run_directory(runs, strategy, block, pair)
        except lv.RunMissing as error:
            say(f"no run: {error}")
            continue
        gate_dir = out / "ot" / "gates" / block / pair / strategy
        for name in ("cell_confidence.csv", "pair_metrics.csv", "run.json"):
            copy(directory / name, gate_dir / name)
        (gate_dir / "SOURCE").write_text(str(directory) + "\n", encoding="utf-8")
        if item["mouse"] in args.coupling_mice:
            for name in ("coupling_m4e.npz", "coupling_m4r.npz"):
                copy(directory / name, out / "couplings" / block / pair / strategy / name)
        for method in lv.METHODS:
            run = lv.load_run(runs, strategy, block, pair, method)
            states = []
            for side, ids, sites, kept, margin in (("source", run["source"], run["source_site"], run["rho"], run["source_margin"]),
                                                   ("target", run["target"], run["target_site"], run["tau"], run["target_margin"])):
                state = np.where(kept, "retained", np.where(np.isinf(margin), "excluded", "rejected"))
                states.append(pd.DataFrame({"side": side, "cell_id": ids, "site": sites, "state": state,
                                            "signed_rejection_margin": np.where(np.isinf(margin), np.nan, margin)}))
            tag = f"{method.replace('-', '')}"
            dest = out / "ot" / "cell_states" / tag / strategy / block
            dest.mkdir(parents=True, exist_ok=True)
            pd.concat(states, ignore_index=True).to_csv(dest / f"{pair}.csv.gz", index=False, compression="gzip")

            # Display of the saved coupling: W_j(g) of the protocol (§6), groups = source primary
            # tumour (KP-Tracer) or LL clone (Quinn), rejected cells weigh 0.
            if dataset == "KPTracer":
                groups_s = run["source_site"].astype(str)
            else:
                groups_s = lineage[(dataset, item["mouse"])].reindex(run["source"]).fillna(lv.UNKNOWN).to_numpy(str)
            names, gs = np.unique(groups_s, return_inverse=True)
            P, rho, tau = run["P"], run["rho"], run["tau"]
            GP = lv.onehot(gs, len(names), rho.astype(float)) @ P
            D = GP.sum(axis=0)
            valid = tau & (D > 0)
            W = np.zeros_like(GP)
            W[:, valid] = GP[:, valid] / D[valid]
            order = np.argsort(-W, axis=0, kind="stable")
            top, second = order[0], order[1] if len(names) > 1 else order[0]
            best = W[top, np.arange(W.shape[1])]
            ties = (np.abs(W - best) <= lv.TIE).sum(axis=0)
            Pr = P * rho[:, None]
            top_cell = Pr.argmax(axis=0)
            top_q = np.where(valid, Pr[top_cell, np.arange(P.shape[1])] / np.where(D > 0, D, 1), 0.0)
            table = pd.DataFrame({
                "cell_id": run["target"], "site": run["target_site"],
                "state": np.where(tau, "retained", np.where(np.isinf(run["target_margin"]), "excluded", "rejected")),
                "retained_source_mass": D,
                "predicted_group": np.where(valid, names[top], ""),
                "predicted_weight": np.where(valid, best, np.nan),
                "tied_groups": np.where(valid, ties, 0),
                "second_group": np.where(valid & (len(names) > 1), names[second], ""),
                "second_weight": np.where(valid & (len(names) > 1), W[second, np.arange(W.shape[1])], np.nan),
                "top_source_cell": np.where(valid, run["source"][top_cell], ""),
                "top_source_q": np.where(valid, top_q, np.nan)})
            if len(names) <= MAX_GROUP_COLUMNS:
                for g, name in enumerate(names):
                    table[f"W__{name}"] = W[g]
            dest = out / "ot" / "matching" / tag / strategy / block
            dest.mkdir(parents=True, exist_ok=True)
            table.to_csv(dest / f"{pair}.csv.gz", index=False, compression="gzip")
            del run, P, Pr, GP, W
            src_scores = args.lv_root / "scores" / method / strategy / block / pair
            if src_scores.exists():
                shutil.copytree(src_scores, out / "ot" / "scores" / method / strategy / block / pair)
        say(f"run {strategy} {pair}: exported")


def export_umap_and_figures(args, out: Path, items: list[dict], mice: dict) -> None:
    for block, pair in sorted({(i["block"], i["pair"]) for i in items}):
        src = args.lv_root / "umap" / block / "pairs" / pair
        for name in ("reference_umap.csv.gz", "reference.json"):
            if (src / name).exists():
                copy(src / name, out / "umap" / block / pair / name)
    figures = args.lv_root / "figures"
    names = {m for _, m in mice}
    pairs = {i["pair"] for i in items}
    for path in sorted(p for p in figures.rglob("*.png")):
        stem = path.stem
        if stem in pairs or any(stem.endswith(f"__{m}") for m in names) or path.parent.name == "strategies":
            copy(path, out / "figures_existing" / path.relative_to(figures))
    summary = args.lv_root / "summary"
    for path in sorted(summary.glob("*")):
        if path.is_file():
            copy(path, out / "summary" / path.name)


def deg_inventory(args, out: Path, mice: dict) -> None:
    rows = []
    pattern = re.compile(r"(?i)(deg|gsea|deseq|hallmark)")
    names = [m for _, m in mice]
    for root in (args.lv_root, args.kp_root):
        for path in root.rglob("*"):
            text = str(path.relative_to(root))
            if path.is_file() and pattern.search(text):
                rows.append({"root": str(root), "path": text, "bytes": path.stat().st_size,
                             "mentions_chosen_mouse": any(m in text for m in names)})
            if len(rows) > 5000:
                break
    (out / "audit").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["root", "path", "bytes", "mentions_chosen_mouse"]).to_csv(
        out / "audit" / "deg_gsea_inventory.tsv", sep="\t", index=False)


def mapping_audit(out: Path, quinn: Path) -> None:
    """Tree leaves vs matrix rows vs metadata vs OT-analysed cells, per tree file."""
    d1 = lv.tree_module()
    leaves_map = pd.read_csv(out / "labels" / "tree_leaves.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    trees = pd.read_csv(out / "labels" / "trees.tsv", sep="\t", dtype=str, keep_default_na=False)
    ot_cells = {}
    for path in (out / "ot" / "cell_states" / "M4E").rglob("*.csv.gz"):
        mouse = mouse_of(path.name[: -len(".csv.gz")])
        ot_cells.setdefault(mouse, set()).update(pd.read_csv(path, usecols=["cell_id"], dtype=str)["cell_id"])
    rows = []
    kp_files = sorted((out / "lineage" / "KPTracer").glob("*_tree.nwk"))
    for tree_path in kp_files:
        stem = tree_path.name[: -len("_tree.nwk")]
        mouse = stem.split("_")[0]
        meta = pd.read_csv(out / "meta" / f"KPTracer_meta_{mouse}.csv.gz", usecols=["cell_id", "Tumor"], dtype=str)
        tree = d1.Tree(tree_path)
        names = [tree.name[v] for v in tree.leaves]
        matrix_path = out / "lineage" / "KPTracer" / f"{stem}_character_matrix.txt"
        rows_m = pd.read_csv(matrix_path, sep="\t", index_col=0, dtype=str).index.astype(str) if matrix_path.exists() else pd.Index([])
        rows.append({"dataset": "KPTracer", "mouse": mouse, "tree": tree_path.name, "matrix": matrix_path.name if matrix_path.exists() else "",
                     "leaves": len(names), "unique_leaves": len(set(names)), "matrix_rows": len(rows_m),
                     "unique_matrix_rows": rows_m.nunique(), "leaves_in_matrix": len(set(names) & set(rows_m)),
                     "matrix_rows_not_leaves": len(set(rows_m) - set(names)),
                     "leaves_in_metadata": len(set(names) & set(meta["cell_id"])),
                     "leaves_mapped_to_cells": len(set(names) & set(meta["cell_id"])),
                     "mapped_cells_in_ot_runs": len(set(names) & ot_cells.get(mouse, set())),
                     "leaf_to_cell_rule": "leaf name == cell id"})
    for _, info in trees[trees["dataset"].eq("Quinn2021") & trees["status"].eq("tree")].iterrows():
        mouse, clone = info["mouse"], info["lineage"]
        # labels/trees.tsv holds the TACC path; the part after lineage/<mouse>/ is the copied file.
        rel = re.split(r"[\\/]lineage[\\/]" + re.escape(mouse) + r"[\\/]", info["tree"])[-1]
        local_tree = out / "lineage" / "Quinn2021" / mouse / rel
        tree = d1.Tree(local_tree)
        names = [tree.name[v] for v in tree.leaves]
        candidates = sorted(p for p in (out / "lineage" / "Quinn2021" / mouse / "character_matrices").rglob("*")
                            if p.is_file() and re.search(rf"(?i)lg{clone}[_.]", p.name))
        preferred = [p for p in candidates if "alleleThresh" in p.name] or candidates
        rows_m = pd.Index([])
        if len(preferred) == 1:
            rows_m = pd.read_csv(preferred[0], sep="\t", index_col=0, dtype=str).index.astype(str)
        mapped = leaves_map[(leaves_map["mouse"] == mouse) & (leaves_map["lineage"] == clone)]
        rows.append({"dataset": "Quinn2021", "mouse": mouse, "tree": local_tree.name, "clone": clone,
                     "matrix": preferred[0].name if len(preferred) == 1 else f"{len(candidates)} candidates: " + ";".join(p.name for p in candidates),
                     "leaves": len(names), "unique_leaves": len(set(names)), "matrix_rows": len(rows_m),
                     "unique_matrix_rows": rows_m.nunique(), "leaves_in_matrix": len(set(names) & set(rows_m)),
                     "matrix_rows_not_leaves": len(set(rows_m) - set(names)),
                     "leaves_mapped_to_cells": len(mapped), "clone_cells": int(info["lineage_cells"]),
                     "mapped_cells_in_ot_runs": len(set(mapped["cell"]) & ot_cells.get(mouse, set())),
                     "leaf_to_cell_rule": "cell == <sampleID>.<leaf>" if mouse == "10k" else "leaf name == cell id"})
    pd.DataFrame(rows).to_csv(out / "audit" / "cell_id_mapping.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lv-root", type=Path, default=LV_ROOT)
    parser.add_argument("--tracer", type=Path, default=TRACER)
    parser.add_argument("--kp-root", type=Path, default=KP_ROOT)
    parser.add_argument("--quinn", type=Path, default=QUINN)
    parser.add_argument("--kp-mice", nargs="+", default=["3508", "3515"])
    parser.add_argument("--quinn-mice", nargs="+", default=["5k", "10k", "30k", "100k"])
    parser.add_argument("--coupling-mice", nargs="+", default=["3508", "3515", "10k"])
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    args = parser.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists; nothing is overwritten")
    args.out.mkdir(parents=True)
    started = time.time()
    lines = []

    def say(text):
        print(text, flush=True)
        lines.append(text)

    mice = {("KPTracer", m) for m in args.kp_mice} | {("Quinn2021", m) for m in args.quinn_mice}
    skipped = export_lineage(args, args.out, say)
    export_labels(args, args.out, mice)
    items = run_list(args, mice)
    say(f"{len(items)} runs (pair x strategy) of the chosen mice")
    export_runs(args, args.out, items, say)
    export_umap_and_figures(args, args.out, items, mice)
    deg_inventory(args, args.out, mice)
    pd.DataFrame(skipped, columns=["file", "reason"]).to_csv(args.out / "audit" / "skipped_files.tsv", sep="\t", index=False)
    mapping_audit(args.out, args.quinn)
    manifest = [{"path": str(p.relative_to(args.out)).replace("\\", "/"), "bytes": p.stat().st_size, "md5": lv.md5(p)}
                for p in sorted(args.out.rglob("*")) if p.is_file()]
    pd.DataFrame(manifest).to_csv(args.out / "MANIFEST.tsv", sep="\t", index=False)
    lv.write_json(args.out / "provenance.json", {"commit": lv.commit(), "protocol": lv.FROZEN, "args": {k: str(v) for k, v in vars(args).items()},
                                                 "seconds": time.time() - started, "log": lines})
    (args.out / "SUCCESS").write_text("done\n", encoding="utf-8")
    say(f"done: {len(manifest)} files, {sum(r['bytes'] for r in manifest) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

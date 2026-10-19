"""One frozen reference UMAP per primary-metastasis pair, with every gate's labels on it.

Same cells, same positions, different gate decisions.

    # 1. coordinates, one chunk of pairs per array task
    python cancer_metastasis/48_reference_umap.py pairs \\
        --manifest UNIFORM.csv --manifest PROSTATE.csv --datasets GSE180661 GSE225857 GSE271675 \\
        --gates-root CYCLING_ROOT --out OUT --task-index 0 --task-count 10

    # 2. one coordinate table, one annotation table per configuration, the changes
    python cancer_metastasis/48_reference_umap.py export \\
        --manifest ... --datasets ... --gates-root CYCLING_ROOT --out OUT

**Cells.** For each pair, the reference cells of a side are the union of the
cells any configuration gated on that side (M4-E rows of every
``cell_confidence.csv`` found for the pair). A configuration that gated fewer
cells -- arm B/D drop malignant cells outside G1, a CNV arm needs a CNV
profile -- leaves the others *excluded* on its table, never *rejected*.

**Coordinates.** Computed once per pair on both sides stacked, from the
original counts (the manifests' ``*_h5ads_json``, not the equalised copies):
normalize_total 1e4, log1p, 2,000 Seurat HVGs, scale (max 10), PCA 40 (arpack),
20 neighbours, UMAP, one fixed seed. Conventional on purpose: independent of
every gate representation. Nothing is aligned between pairs; coordinates
compare only within one pair. They are written once and never recomputed.

**CNV embedding.** With --cnv-root, a second frozen embedding of each pair from
the inferCNV profiles the CNV gates read (profile - 1, PCA 40, then the same
neighbours and UMAP), for the reference cells that have a profile, and each
cell's mean (profile - 1) per chromosome. It sits beside the RNA reference and
never replaces it; a patient without profiles (prostate, so far) is recorded.

**Configurations.** Every ``<modality>/arm_<X>/<label>`` under --gates-root,
crossed with the methods in --methods; a configuration reads the block
directory ``<modality>/arm_<X>/<block>/<label>`` of each pair's dataset. Block is
``prostate`` for GSE271675 and ``uniform`` otherwise (--block overrides).

**Statuses on an annotation table.** ``retained`` / ``rejected``: gated by that
configuration. ``excluded``: a reference cell the configuration did not gate,
although it completed the pair. A pair a configuration did not complete, or
was never run on, has no rows; ``annotation_coverage.csv`` says which.

**Changes.** For every pair, side and two configurations: cells gated by both,
how many change label, in which direction, as counts and percentages of the
cells gated by both only. Summaries per dataset add pooled counts and the
per-pair median, minimum and maximum.

UMAP separation is not biological validation; this shows how the gates label
the same cells.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from common import expression_matrix, load_exact_side

SEED = 20260904
N_HVG, N_PCS, N_NEIGHBORS = 2000, 40, 20
DEFAULT_BLOCKS = {"GSE271675": "prostate"}
ARMS = {
    "A": "no cycling filter",
    "B": "malignant cells outside G1 removed on both sides",
    "C": "the 97 cell-cycle genes removed",
    "D": "B and C",
}
COLOURS = {"metastasis": "#4d4d4d", "retained": "#2b8cbe", "rejected": "#d7301f",
           "excluded": "#d9d9d9", "unavailable": "#d9d9d9"}
GATE_COLUMNS = ["method", "side", "observation_id", "retained", "signed_rejection_margin",
                "normalized_rejection_score"]


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def read_manifests(paths: list[Path], datasets: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    table = pd.concat(frames, ignore_index=True)
    table = table[table["dataset_id"].astype(str).isin(datasets)]
    if "eligible" in table:
        table = table[table["eligible"].astype(bool)]
    table = table.drop_duplicates(["dataset_id", "pair_id"])
    if table["pair_id"].duplicated().any():
        raise SystemExit(f"pair ids shared between datasets: {sorted(table.loc[table['pair_id'].duplicated(), 'pair_id'])[:5]}")
    return table.sort_values(["dataset_id", "pair_id"], kind="stable").reset_index(drop=True)


def configurations(root: Path, methods: list[str]) -> list[dict]:
    """One entry per modality, arm, label and method; ``roots`` maps each block to its gate directory."""
    found: dict[str, dict] = {}
    for directory in sorted(root.glob("*/arm_*/*/*")):
        if not directory.is_dir():
            continue
        label, block = directory.name, directory.parent.name
        arm, modality = directory.parent.parent.name, directory.parent.parent.parent.name
        for method in methods:
            name = f"{modality}|{arm}|{label}|{method}"
            entry = found.setdefault(name, {
                "configuration": name, "file_stem": safe(name.replace("|", "__")), "modality": modality,
                "arm": arm, "arm_meaning": ARMS.get(arm.replace("arm_", ""), ""), "label": label,
                "method": method, "roots": {}})
            entry["roots"][block] = str(directory)
    return list(found.values())


def gate_file(config_root: Path, pair_id: str) -> Path | None:
    done = sorted((config_root / pair_id).glob("scope_malignant/*/SUCCESS"))
    if len(done) != 1:
        return None
    path = done[0].parent / "cell_confidence.csv"
    return path if path.exists() else None


def pair_gates(row: dict, configs: list[dict], blocks: dict) -> dict:
    """{configuration: M4-E/M4-R rows of that gate for this pair} for completed gates."""
    block = blocks.get(str(row["dataset_id"]), "uniform")
    by_root: dict[str, pd.DataFrame] = {}
    result = {}
    for config in configs:
        root = config["roots"].get(block)
        if root is None:
            continue
        path = gate_file(Path(root), str(row["pair_id"]))
        if path is None:
            continue
        if root not in by_root:
            table = pd.read_csv(path, usecols=lambda c: c in GATE_COLUMNS)
            table["observation_id"] = table["observation_id"].astype(str)
            table["retained"] = table["retained"].astype(bool)
            by_root[root] = table
        table = by_root[root]
        result[config["configuration"]] = table[table["method"].astype(str).eq(config["method"])]
    return result


def cell_uid(dataset: str, patient: str, sample: str, observation: str) -> str:
    return f"{dataset}:{patient}:{sample}:{observation}"


# --- pairs ---------------------------------------------------------------------

def reference_umap(row: dict, gates: dict, out: Path) -> dict:
    import anndata as ad
    import scanpy as sc

    pieces, counts = [], {}
    for side, sample_column, paths_column in (("source", "source_sample", "source_h5ads_json"),
                                              ("target", "target_sample", "target_h5ads_json")):
        wanted = sorted(set().union(*[set(g.loc[g["side"].eq(side), "observation_id"]) for g in gates.values()]))
        sample = str(row[sample_column])
        data = load_exact_side(json.loads(str(row[paths_column])), sample)
        data.obs_names = data.obs_names.astype(str)
        missing = sorted(set(wanted) - set(data.obs_names))
        if missing:
            raise KeyError(f"{side}: {len(missing)} gated cells are absent from the h5ad, e.g. {missing[:3]}")
        data = data[wanted].copy()
        matrix = sparse.csr_matrix(expression_matrix(data), dtype=np.float64)
        if matrix.data.size and (matrix.data.min() < 0 or not np.allclose(matrix.data, np.round(matrix.data))):
            raise ValueError(f"{side}: the expression matrix is not raw counts")
        piece = ad.AnnData(X=matrix.astype(np.float32), var=pd.DataFrame(index=data.var_names.astype(str)))
        piece.obs_names = [cell_uid(str(row["dataset_id"]), str(row["patient_id"]), sample, c) for c in wanted]
        piece.obs["side"] = side
        piece.obs["sample"] = sample
        piece.obs["observation_id"] = wanted
        piece.obs["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel()
        piece.obs["detected_genes"] = np.asarray((matrix > 0).sum(axis=1)).ravel()
        pieces.append(piece)
        counts[side] = len(wanted)
    data = ad.concat(pieces, join="inner", merge="same", index_unique=None)
    if data.obs_names.has_duplicates:
        raise RuntimeError("duplicate cell ids across the two sides")
    sc.pp.normalize_total(data, target_sum=1e4)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes=N_HVG, flavor="seurat", subset=True)
    hvg = list(data.var_names)
    sc.pp.scale(data, max_value=10)
    n_pcs = min(N_PCS, data.n_vars - 1, data.n_obs - 1)
    sc.tl.pca(data, n_comps=n_pcs, svd_solver="arpack", random_state=SEED)
    sc.pp.neighbors(data, n_neighbors=N_NEIGHBORS, n_pcs=n_pcs, random_state=SEED)
    sc.tl.umap(data, random_state=SEED)
    table = pd.DataFrame({
        "pair_id": row["pair_id"], "dataset_id": row["dataset_id"], "patient_id": str(row["patient_id"]),
        "cell_uid": data.obs_names.astype(str), "side": data.obs["side"].to_numpy(),
        "sample": data.obs["sample"].to_numpy(), "observation_id": data.obs["observation_id"].to_numpy(),
        "umap_1": data.obsm["X_umap"][:, 0], "umap_2": data.obsm["X_umap"][:, 1],
        "total_counts": data.obs["total_counts"].to_numpy(), "detected_genes": data.obs["detected_genes"].to_numpy(),
    })
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "reference_umap.csv.gz", index=False, compression="gzip")
    record = {"pair_id": row["pair_id"], "dataset_id": row["dataset_id"], "cells": counts,
              "genes_shared": int(len(pieces[0].var_names.intersection(pieces[1].var_names))),
              "n_hvg": N_HVG, "n_pcs": n_pcs, "n_neighbors": N_NEIGHBORS, "seed": SEED,
              "hvg_md5": hashlib.md5("\n".join(hvg).encode()).hexdigest(),
              "configurations_contributing_cells": sorted(gates),
              "scanpy": sc.__version__}
    try:
        import umap
        record["umap_learn"] = umap.__version__
    except Exception:  # noqa: BLE001
        pass
    (out / "reference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "hvg.txt").write_text("\n".join(hvg) + "\n", encoding="utf-8")
    return record


def cnv_reference(row: dict, cells: pd.DataFrame, cnv_root: Path, out: Path) -> dict:
    """A second frozen embedding of the same pair, from each cell's own inferCNV profile.

    The profiles are the ones the CNV gates read (44_export_cnv_cells.R:
    cnv.f32, cells.tsv, genes.tsv per patient, 1 = copy-neutral). Reference cells
    without a profile are left out of this embedding and counted. Also writes
    each cell's mean (profile - 1) over the genes of every chromosome.
    """
    import anndata as ad
    import scanpy as sc
    from sklearn.decomposition import PCA

    unit = cnv_root / str(row["dataset_id"]) / str(row["patient_id"])
    if not (unit / "cnv.f32").exists():
        return {"cnv_status": "no CNV profiles for this patient", "cnv_unit": str(unit)}
    names = pd.read_csv(unit / "cells.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(unit / "genes.tsv", sep="\t", dtype=str)
    if names["barcode"].duplicated().any():
        raise ValueError(f"{unit}: a cell name repeats in cells.tsv")
    values = np.memmap(unit / "cnv.f32", dtype=np.float32, mode="r", shape=(len(names), len(genes)))
    position = pd.Index(names["barcode"]).get_indexer(pd.Index(cells["observation_id"].astype(str)))
    have = position >= 0
    use = cells[have].reset_index(drop=True)
    joint = np.asarray(values[position[have]], dtype=np.float64) - 1.0
    components = min(N_PCS, joint.shape[0] - 1, joint.shape[1])
    pcs = PCA(n_components=components, random_state=SEED).fit_transform(joint).astype(np.float32)
    data = ad.AnnData(X=np.zeros((len(use), 1), dtype=np.float32))
    data.obs_names = use["cell_uid"].astype(str).to_numpy()
    data.obsm["X_pca"] = pcs
    sc.pp.neighbors(data, n_neighbors=N_NEIGHBORS, use_rep="X_pca", random_state=SEED)
    sc.tl.umap(data, random_state=SEED)
    table = use[["pair_id", "dataset_id", "patient_id", "cell_uid", "side"]].copy()
    table["umap_1"], table["umap_2"] = data.obsm["X_umap"][:, 0], data.obsm["X_umap"][:, 1]
    table.to_csv(out / "reference_umap_cnv.csv.gz", index=False, compression="gzip")
    chromosome = genes["chr"].astype(str).to_numpy()
    means = use[["pair_id", "cell_uid", "side"]].copy()
    for name in pd.unique(chromosome):
        means[f"cnv_mean_{name}"] = joint[:, chromosome == name].mean(axis=1)
    means.to_csv(out / "cnv_chromosome_means.csv.gz", index=False, compression="gzip")
    return {"cnv_status": "done", "cnv_unit": str(unit), "cnv_genes": int(len(genes)), "cnv_pcs": int(components),
            "cnv_cells": {side: int((use["side"] == side).sum()) for side in ("source", "target")},
            "cnv_cells_without_profile": {side: int(((cells["side"] == side) & ~have).sum()) for side in ("source", "target")}}


def run_pairs(args) -> int:
    manifest = read_manifests(args.manifest, args.datasets)
    configs = configurations(args.gates_root, args.methods)
    blocks = {**DEFAULT_BLOCKS, **dict(x.split("=", 1) for x in args.block)}
    failures = 0
    for index, row in enumerate(manifest.to_dict("records")):
        if index % args.task_count != args.task_index:
            continue
        out = args.out / "pairs" / safe(str(row["pair_id"]))
        if (out / "DONE").exists():
            print(f"skip {row['pair_id']}: DONE", flush=True)
            continue
        try:
            gates = pair_gates(row, configs, blocks)
            if not gates:
                # In the manifest, but no gate ever ran on it: nothing to overlay, and not a failure.
                out.mkdir(parents=True, exist_ok=True)
                (out / "NOT_GATED").write_text("no configuration completed this pair\n", encoding="utf-8")
                (out / "FAILED").unlink(missing_ok=True)
                print(f"not gated {row['pair_id']}", flush=True)
                continue
            record = reference_umap(row, gates, out)
            if args.cnv_root is not None:
                cells = pd.read_csv(out / "reference_umap.csv.gz", usecols=["pair_id", "dataset_id", "patient_id",
                                                                             "cell_uid", "side", "observation_id"])
                try:
                    record.update(cnv_reference(row, cells, args.cnv_root, out))
                except Exception as error:  # noqa: BLE001 - the RNA reference stands; the CNV one is reported
                    record.update({"cnv_status": f"error: {type(error).__name__}: {error}"})
                (out / "reference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (out / "DONE").write_text("done\n", encoding="utf-8")
            (out / "FAILED").unlink(missing_ok=True)
            print(f"{row['pair_id']}: {record['cells']}", flush=True)
        except Exception as error:  # noqa: BLE001 - a failed pair is recorded and reported, never hidden
            failures += 1
            out.mkdir(parents=True, exist_ok=True)
            (out / "FAILED").write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}", encoding="utf-8")
            print(f"FAILED {row['pair_id']}: {type(error).__name__}: {error}", flush=True)
    return 1 if failures else 0


def run_cnv(args) -> int:
    """Add the CNV embedding to pairs that are already DONE, leaving their RNA reference as it is.

    For pairs finished before the CNV embedding existed (or run without --cnv-root):
    reads the pair's own reference_umap.csv.gz cells, writes reference_umap_cnv.csv.gz
    and cnv_chromosome_means.csv.gz, and adds the cnv_* fields to reference.json. A pair
    that already has its CNV embedding, or is not DONE, is skipped. reference_umap.csv.gz
    is never rewritten.
    """
    if args.cnv_root is None:
        raise SystemExit("cnv mode needs --cnv-root")
    manifest = read_manifests(args.manifest, args.datasets)
    failures = 0
    for index, row in enumerate(manifest.to_dict("records")):
        if index % args.task_count != args.task_index:
            continue
        out = args.out / "pairs" / safe(str(row["pair_id"]))
        if not (out / "DONE").exists() or (out / "reference_umap_cnv.csv.gz").exists():
            continue
        try:
            cells = pd.read_csv(out / "reference_umap.csv.gz", usecols=["pair_id", "dataset_id", "patient_id",
                                                                         "cell_uid", "side", "observation_id"])
            record = json.loads((out / "reference.json").read_text())
            record.update(cnv_reference(row, cells, args.cnv_root, out))
            (out / "reference.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            (out / "CNV_FAILED").unlink(missing_ok=True)
            print(f"{row['pair_id']}: {record['cnv_status']}", flush=True)
        except Exception as error:  # noqa: BLE001 - recorded and reported, never hidden
            failures += 1
            (out / "CNV_FAILED").write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
                                            encoding="utf-8")
            print(f"CNV FAILED {row['pair_id']}: {type(error).__name__}: {error}", flush=True)
    return 1 if failures else 0


# --- export --------------------------------------------------------------------

def run_export(args) -> int:
    export = args.out / args.export_name
    if (export / "MANIFEST.tsv").exists():
        raise SystemExit(f"{export}/MANIFEST.tsv exists; the export is not rewritten")
    manifest = read_manifests(args.manifest, args.datasets)
    configs = configurations(args.gates_root, args.methods)
    blocks = {**DEFAULT_BLOCKS, **dict(x.split("=", 1) for x in args.block)}
    (export / "annotations").mkdir(parents=True, exist_ok=True)

    coordinates, pair_status, cnv_coordinates, cnv_means = [], [], [], []
    for row in manifest.to_dict("records"):
        folder = args.out / "pairs" / safe(str(row["pair_id"]))
        if (folder / "DONE").exists():
            coordinates.append(pd.read_csv(folder / "reference_umap.csv.gz"))
            if (folder / "reference_umap_cnv.csv.gz").exists():
                cnv_coordinates.append(pd.read_csv(folder / "reference_umap_cnv.csv.gz"))
                cnv_means.append(pd.read_csv(folder / "cnv_chromosome_means.csv.gz"))
            info = json.loads((folder / "reference.json").read_text())
            pair_status.append({"pair_id": row["pair_id"], "dataset_id": row["dataset_id"], "status": "done",
                                "source_cells": info["cells"]["source"], "target_cells": info["cells"]["target"],
                                "hvg_md5": info["hvg_md5"], "cnv_status": info.get("cnv_status", "not requested"),
                                "cnv_source_cells": info.get("cnv_cells", {}).get("source"),
                                "cnv_target_cells": info.get("cnv_cells", {}).get("target"),
                                "cnv_source_cells_without_profile": info.get("cnv_cells_without_profile", {}).get("source"),
                                "cnv_target_cells_without_profile": info.get("cnv_cells_without_profile", {}).get("target")})
        elif (folder / "NOT_GATED").exists():
            pair_status.append({"pair_id": row["pair_id"], "dataset_id": row["dataset_id"], "status": "not gated",
                                "error": "no configuration completed this pair"})
        else:
            message = (folder / "FAILED").read_text().splitlines()[0] if (folder / "FAILED").exists() else "not run"
            pair_status.append({"pair_id": row["pair_id"], "dataset_id": row["dataset_id"], "status": "failed", "error": message})
    reference = pd.concat(coordinates, ignore_index=True)
    reference.to_csv(export / "reference_coordinates.csv.gz", index=False, compression="gzip")
    if cnv_coordinates:
        pd.concat(cnv_coordinates, ignore_index=True).to_csv(export / "reference_coordinates_cnv.csv.gz", index=False, compression="gzip")
        pd.concat(cnv_means, ignore_index=True).to_csv(export / "cnv_chromosome_means.csv.gz", index=False, compression="gzip")
    status_table = pd.DataFrame(pair_status)
    for column in [c for c in status_table if c.endswith("_cells") or c.endswith("_without_profile")]:
        status_table[column] = status_table[column].astype("Int64")
    status_table.to_csv(export / "pairs.csv", index=False)
    done = set(reference["pair_id"])

    writers = {c["configuration"]: export / "annotations" / f"{c['file_stem']}.csv.gz" for c in configs}
    first = {name: True for name in writers}
    coverage, change_rows = [], []
    for row in manifest.to_dict("records"):
        pair_id = str(row["pair_id"])
        block = blocks.get(str(row["dataset_id"]), "uniform")
        gates = pair_gates(row, configs, blocks)
        for config in configs:
            name = config["configuration"]
            state = ("completed" if name in gates else
                     "not run for this dataset" if block not in config["roots"] else "not completed")
            coverage.append({"configuration": name, "pair_id": pair_id, "dataset_id": row["dataset_id"],
                             "gate": state, "reference_umap": pair_id in done})
        if pair_id not in done:
            continue
        cells = reference[reference["pair_id"].eq(pair_id)][["cell_uid", "side", "sample", "observation_id"]].reset_index(drop=True)
        key = pd.MultiIndex.from_frame(cells[["side", "observation_id"]])
        codes = {}
        for name, gate in gates.items():
            gate = gate.drop_duplicates(["side", "observation_id"]).set_index(["side", "observation_id"])
            outside = gate.index.difference(key)
            if len(outside):
                raise RuntimeError(f"{pair_id} {name}: {len(outside)} gated cells are not reference cells")
            joined = gate.reindex(key)
            present = joined["retained"].notna().to_numpy()
            retained = joined["retained"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            status = np.where(~present, "excluded", np.where(retained, "retained", "rejected"))
            table = pd.DataFrame({
                "pair_id": pair_id, "cell_uid": cells["cell_uid"], "side": cells["side"], "status": status,
                "retained": pd.array(np.where(present, retained, pd.NA), dtype="boolean"),
                "gate_score_r": -joined["signed_rejection_margin"].to_numpy(dtype=float),
                "signed_rejection_margin": joined["signed_rejection_margin"].to_numpy(dtype=float),
                "normalized_rejection_score": joined["normalized_rejection_score"].to_numpy(dtype=float),
            })
            table.to_csv(writers[name], mode="w" if first[name] else "a", header=first[name], index=False, compression="gzip")
            first[name] = False
            codes[name] = np.where(~present, -1, retained.astype(int))
        sides = cells["side"].to_numpy()
        for a, b in itertools.combinations(sorted(codes), 2):
            for side in ("source", "target"):
                mask = sides == side
                x, y = codes[a][mask], codes[b][mask]
                both = (x >= 0) & (y >= 0)
                n = int(both.sum())
                to_rejected = int((both & (x == 1) & (y == 0)).sum())
                to_retained = int((both & (x == 0) & (y == 1)).sum())
                change_rows.append({
                    "pair_id": pair_id, "dataset_id": row["dataset_id"], "side": side,
                    "configuration_a": a, "configuration_b": b,
                    "cells_a": int((x >= 0).sum()), "cells_b": int((y >= 0).sum()), "cells_in_both": n,
                    "changed": to_rejected + to_retained,
                    "changed_percent": 100.0 * (to_rejected + to_retained) / n if n else np.nan,
                    "retained_in_a_rejected_in_b": to_rejected, "rejected_in_a_retained_in_b": to_retained,
                    "retained_percent_a": 100.0 * (x[both] == 1).mean() if n else np.nan,
                    "retained_percent_b": 100.0 * (y[both] == 1).mean() if n else np.nan,
                })
        print(f"{pair_id}: {len(gates)} configurations", flush=True)
    for name, path in writers.items():
        if first[name]:
            path.unlink(missing_ok=True)
    pd.DataFrame(coverage).to_csv(export / "annotation_coverage.csv", index=False)
    changes = pd.DataFrame(change_rows)
    changes.to_csv(export / "classification_changes_by_pair.csv.gz", index=False, compression="gzip")
    if len(changes):
        summary = changes.groupby(["dataset_id", "side", "configuration_a", "configuration_b"], sort=True).agg(
            pairs=("pair_id", "nunique"), cells_in_both=("cells_in_both", "sum"), changed=("changed", "sum"),
            retained_in_a_rejected_in_b=("retained_in_a_rejected_in_b", "sum"),
            rejected_in_a_retained_in_b=("rejected_in_a_retained_in_b", "sum"),
            pair_median_changed_percent=("changed_percent", "median"),
            pair_min_changed_percent=("changed_percent", "min"),
            pair_max_changed_percent=("changed_percent", "max")).reset_index()
        summary.insert(summary.columns.get_loc("changed") + 1, "pooled_changed_percent",
                       100.0 * summary["changed"] / summary["cells_in_both"].where(summary["cells_in_both"] > 0))
        summary.to_csv(export / "classification_changes_summary.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in c.items() if k != "roots"}, "gate_roots": json.dumps(c["roots"]),
                   "annotation_file": f"annotations/{c['file_stem']}.csv.gz" if writers[c["configuration"]].exists() else ""}
                  for c in configs]).to_csv(export / "configurations.csv", index=False)
    (export / "colours.json").write_text(json.dumps(COLOURS, indent=2), encoding="utf-8")
    readme = [
        "# Reference UMAP export",
        "",
        "Same cells, same positions, different gate decisions.",
        "",
        "- `reference_coordinates.csv.gz`: one row per pair and reference cell; `umap_1`, `umap_2` are that pair's frozen",
        "  coordinates (log-normalised RNA, 2,000 HVGs, PCA 40, 20 neighbours, seed 20260904). Compare positions only within one pair.",
        "- `reference_coordinates_cnv.csv.gz`: a second frozen embedding of the same pairs from each cell's inferCNV profile",
        "  (profile - 1, PCA 40, 20 neighbours, same seed), for cells with a profile; prostate has none yet. The RNA one is the reference;",
        "  the two are never mixed. `cnv_chromosome_means.csv.gz`: each cell's mean (profile - 1) per chromosome.",
        "- `annotations/<configuration>.csv.gz`: that gate's label of every reference cell of every pair it completed;",
        "  `status` is retained, rejected or excluded (a reference cell this configuration did not gate). Join on `pair_id` + `cell_uid`.",
        "- `annotation_coverage.csv`: per configuration and pair, completed / not completed / not run for this dataset.",
        "- `classification_changes_by_pair.csv.gz` and `classification_changes_summary.csv`: label changes between two",
        "  configurations, counted on cells gated by both only.",
        "- `configurations.csv`: modality, arm (and what it removes), label, method, gate root.",
        "- `colours.json`: metastasis dark grey, retained blue, rejected red, excluded or unavailable light grey.",
        "- `cell_uid` = dataset:patient:sample:observation_id; the same primary cell keeps its uid in every pair.",
        "",
        "UMAP separation is not biological validation; these files show how different gates label the same cells.",
    ]
    (export / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    rows = []
    for path in sorted(export.rglob("*")):
        if path.is_file() and path.name not in ("MANIFEST.tsv", "MD5SUMS"):
            rows.append({"path": path.relative_to(export).as_posix(), "bytes": path.stat().st_size, "md5": md5(path)})
    pd.DataFrame(rows).to_csv(export / "MANIFEST.tsv", sep="\t", index=False)
    with open(export / "MD5SUMS", "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(f"{row['md5']}  {row['path']}\n")
    failed = [p for p in pair_status if p["status"] == "failed"]
    not_gated = sum(p["status"] == "not gated" for p in pair_status)
    print(f"pairs {len(pair_status)}, reference UMAP done {len(done)}, not gated {not_gated}, failed {len(failed)}; "
          f"configurations {len(configs)}; files {len(rows)}")
    for p in failed:
        print(f"  FAILED {p['pair_id']}: {p.get('error')}")
    print("DONE")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["pairs", "cnv", "export"])
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--gates-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["M4-E", "M4-R"])
    parser.add_argument("--block", action="append", default=[], help="DATASET=BLOCK, overriding the default")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--cnv-root", type=Path, help="44_export_cnv_cells.R output; adds the CNV embedding")
    parser.add_argument("--export-name", default="export",
                        help="Folder under --out for the export; a new name keeps an earlier export intact")
    args = parser.parse_args()
    modes = {"pairs": run_pairs, "cnv": run_cnv, "export": run_export}
    sys.exit(modes[args.mode](args))


if __name__ == "__main__":
    main()

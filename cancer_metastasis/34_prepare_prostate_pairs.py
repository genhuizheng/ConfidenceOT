"""Split the prostate multiome object into per-specimen H5ADs and pair them.

Faming Zhao's ``AllPatients_Malignant_harmony.h5ad`` holds 239,430 cells across
36,601 genes from GSE271675, a 10x Multiome series with matched primary prostate
tumours and lymph-node metastases.  The file is 15 GB, of which half is a
normalised ``layers['data']`` copy of the counts and 110 MB is the Harmony and
PCA embeddings.  This writes one slim H5AD per specimen carrying raw counts
only, plus a pair manifest with the columns ``01_build_pair_manifest.py``
emits, so the existing array, pseudobulk, differential expression and scoring
stages run on it unmodified.

**The embeddings are deliberately discarded.**  Depth and batch correction stay
outside the algorithm in this project, and ``prepare_joint_representation``
builds its own HVG selection and PCA. Feeding it a Harmony space would layer a
second correction inside the representation, and no result could then be
attributed to either. What is wanted from the upstream work is its
*annotation*, not its coordinates.

**One lymph node per patient.**  The specimens are many-to-many: Patient2 alone
has five nodes and six tumours, which is 30 possible pairs sharing the same
cells. Pairing every combination made a single primary cell appear in up to
eight fits in the ovarian analysis, and collapsing that afterwards cost most of
the retained set. The largest node per patient is named instead, every primary
specimen is paired once against it, and each primary cell is therefore gated
exactly once. Node size is fixed before any gate is fitted, so the choice
cannot be steered by the result.

**Patient3 has no metastatic cells in this object** -- 11,406 primary and zero
lymph node -- so it yields no pair and the paired design rests on four
patients, not the five the upstream summary describes. The report states this
rather than letting a silently smaller *n* propagate.

``AuthorLabel`` is written to ``cell_type`` so the existing
``--include-annotations`` filter selects on it. The labels are not all
epithelial: 2,714 cells carry immune or stromal labels, including 585 B cells,
and 209 are ``Unknown``. ``Fibroblast`` and ``Fibroblasts`` are two spellings
of one label. Those are dropped here by default. 585 B cells is 0.24% of the
object, far too few to explain a B-cell signal in a pseudobulk contrast, so a
residual of ambient immunoglobulin or of doublets labelled ``Epithelial`` is
expected to remain and is what ``33_score_lineage_contamination.py`` looks for.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# Immune, stromal and unassigned labels. Kept out of the default: Epithelial,
# Basal Epithelial and Neuroendocrine are epithelial lineages, and
# neuroendocrine prostate cancer is a malignant state rather than a
# contaminant.
DEFAULT_EXCLUDED = (
    "BCell", "TCell", "Lymphoid", "Macrophage", "Monocyte", "Mast",
    "Erythroblast", "Endothelial", "Fibroblast", "Fibroblasts", "Unknown",
)

CHUNK = 64_000_000


def read_obs_column(handle: h5py.File, name: str) -> pd.Series:
    node = handle["obs"][name]
    if isinstance(node, h5py.Group) and "categories" in node:
        categories = np.asarray([
            value.decode() if isinstance(value, bytes) else str(value)
            for value in node["categories"][:]
        ], dtype=object)
        codes = np.asarray(node["codes"][:])
        values = np.where(codes >= 0, categories[np.clip(codes, 0, None)], "")
        return pd.Series(values.astype(str), name=name)
    values = node[:]
    if h5py.check_string_dtype(node.dtype) or node.dtype.kind in "OS":
        values = [v.decode() if isinstance(v, bytes) else str(v) for v in values]
    return pd.Series(values, name=name)


def read_index(handle: h5py.File, axis: str) -> np.ndarray:
    group = handle[axis]
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    values = group[key][:]
    return np.asarray(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=str
    )


def read_counts(handle: h5py.File) -> sparse.csr_matrix:
    """Load X as CSR with 32-bit payloads, reading the arrays in chunks.

    The stored data are int64, which for 646 million nonzeros is 5.2 GB before
    anything else is allocated. Counts fit in int32, so casting chunk by chunk
    halves the footprint and avoids holding both widths at once.
    """
    node = handle["X"]
    if not isinstance(node, h5py.Group):
        raise RuntimeError("X is dense; this object was expected to be sparse CSR")
    shape = tuple(int(value) for value in node.attrs["shape"])
    nnz = node["data"].shape[0]
    data = np.empty(nnz, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    for start in range(0, nnz, CHUNK):
        stop = min(start + CHUNK, nnz)
        data[start:stop] = node["data"][start:stop]
        indices[start:stop] = node["indices"][start:stop]
    indptr = np.asarray(node["indptr"][:], dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dataset-id", default="GSE271675")
    parser.add_argument(
        "--exclude-label", action="append", dest="excluded", default=None,
        help=f"AuthorLabel value to drop. Default: {', '.join(DEFAULT_EXCLUDED)}",
    )
    parser.add_argument("--patient-column", default="Patient")
    parser.add_argument("--site-column", default="Site")
    parser.add_argument("--specimen-column", default="Specimen")
    parser.add_argument("--label-column", default="AuthorLabel")
    parser.add_argument("--metastasis-value", default="LN_Metastasis")
    parser.add_argument("--primary-value", default="Primary_Tumor")
    parser.add_argument("--minimum-cells-per-side", type=int, default=100)
    args = parser.parse_args()
    excluded = set(args.excluded if args.excluded is not None else DEFAULT_EXCLUDED)

    destination = args.output_root
    (destination / "h5ad").mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5ad, "r") as handle:
        obs = pd.DataFrame({
            "patient_id": read_obs_column(handle, args.patient_column),
            "site": read_obs_column(handle, args.site_column),
            "sample_id": read_obs_column(handle, args.specimen_column),
            "cell_type": read_obs_column(handle, args.label_column),
        })
        obs["observation_id"] = read_index(handle, "obs")
        genes = read_index(handle, "var")
        label_counts = obs["cell_type"].value_counts()
        keep = ~obs["cell_type"].isin(excluded)
        matrix = read_counts(handle)

    if matrix.shape[0] != len(obs):
        raise RuntimeError(f"X has {matrix.shape[0]} rows for {len(obs)} obs rows")
    if matrix.shape[1] != len(genes):
        raise RuntimeError(f"X has {matrix.shape[1]} columns for {len(genes)} var rows")

    var = pd.DataFrame({"gene_symbol": genes}, index=pd.Index(genes, name=None))
    written: dict[str, dict] = {}
    for sample, group in obs[keep.to_numpy()].groupby("sample_id", sort=True):
        if group.empty:
            continue
        rows = np.asarray(group.index, dtype=np.int64)
        piece = ad.AnnData(
            X=matrix[rows].tocsr(),
            obs=group[["patient_id", "site", "sample_id", "cell_type"]]
            .set_index(group["observation_id"].to_numpy()),
            var=var.copy(),
        )
        piece.obs_names = pd.Index(group["observation_id"].to_numpy(), dtype=str)
        path = destination / "h5ad" / f"{sample}.h5ad"
        piece.write_h5ad(path, compression="lzf")
        written[str(sample)] = {
            "path": str(path),
            "patient_id": str(group["patient_id"].iloc[0]),
            "site": str(group["site"].iloc[0]),
            "n": int(len(group)),
        }
        del piece

    # One node per patient, the largest after filtering, so a primary cell is
    # gated exactly once. Ties break on the specimen name for determinism.
    frame = pd.DataFrame(written).T.reset_index(names="sample_id")
    frame["n"] = frame["n"].astype(int)
    nodes = frame[frame["site"].eq(args.metastasis_value)]
    chosen = (
        nodes.sort_values(["patient_id", "n", "sample_id"],
                          ascending=[True, False, True], kind="stable")
        .drop_duplicates("patient_id", keep="first")
        .set_index("patient_id")
    )
    primaries = frame[frame["site"].eq(args.primary_value)]

    rows, skipped = [], []
    for patient, group in primaries.groupby("patient_id", sort=True):
        if patient not in chosen.index:
            skipped.append({"patient_id": patient, "reason": "no metastatic specimen"})
            continue
        node = chosen.loc[patient]
        for record in group.sort_values("sample_id").to_dict("records"):
            reasons = []
            if int(record["n"]) < args.minimum_cells_per_side:
                reasons.append(f"source_cells<{args.minimum_cells_per_side}")
            if int(node["n"]) < args.minimum_cells_per_side:
                reasons.append(f"target_cells<{args.minimum_cells_per_side}")
            rows.append({
                "pair_id": f"{patient}__{record['sample_id']}__{node['sample_id']}",
                "dataset_id": args.dataset_id,
                "patient_id": patient,
                "source_h5ad": record["path"],
                "target_h5ad": node["path"],
                "source_h5ads_json": json.dumps([record["path"]]),
                "target_h5ads_json": json.dumps([node["path"]]),
                "source_sample": record["sample_id"],
                "target_sample": node["sample_id"],
                "source_n": int(record["n"]),
                "target_n": int(node["n"]),
                "source_file_n": 1,
                "target_file_n": 1,
                "source_file_kind": "specimen",
                "target_file_kind": "specimen",
                "has_cell_type": True,
                "eligible": not reasons,
                "skip_reason": ";".join(reasons),
            })

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("No pair could be formed")
    manifest.insert(0, "pair_index", np.arange(len(manifest), dtype=int))
    manifest["eligible_index"] = -1
    eligible = manifest.index[manifest["eligible"]]
    manifest.loc[eligible, "eligible_index"] = np.arange(len(eligible), dtype=int)
    manifest.to_csv(destination / "pair_manifest_all.csv", index=False)
    manifest[manifest["eligible"]].to_csv(
        destination / "pair_manifest_eligible.csv", index=False
    )

    report = {
        "source_h5ad": str(args.h5ad),
        "dataset_id": args.dataset_id,
        "cells_in": int(len(obs)),
        "cells_kept": int(keep.sum()),
        "excluded_labels": sorted(excluded),
        "excluded_cell_n": int((~keep).sum()),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "specimen_n": len(written),
        "designated_metastasis": {
            str(patient): str(row["sample_id"]) for patient, row in chosen.iterrows()
        },
        "patients_without_metastasis": skipped,
        "pairs_total": int(len(manifest)),
        "pairs_eligible": int(manifest["eligible"].sum()),
        "embeddings_discarded": ["X_harmony", "X_pca", "X_umap_harmony", "X_umap_pca"],
        "note": (
            "cell_type carries AuthorLabel so --include-annotations selects on "
            "it. Depth is not equalised here; run 27_downsample_counts.py on "
            "the emitted manifest and set --target-depth from this object's own "
            "distribution, since multiome nuclei do not match the whole-cell "
            "3' depth the ovarian target was chosen from."
        ),
    }
    (destination / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    pd.set_option("display.width", 200)
    print(manifest[[
        "pair_id", "patient_id", "source_sample", "target_sample",
        "source_n", "target_n", "eligible",
    ]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

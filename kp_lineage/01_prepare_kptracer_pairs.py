"""Split KP-Tracer into one H5AD per tumour and pair every metastasis with
every primary tumour of its mouse, labelled by the authors' MetFamily.

    python kp_lineage/01_prepare_kptracer_pairs.py \\
        KPTracer-Data/expression/adata_processed.combined.h5ad OUTPUT_ROOT [--plan]

Yang, Jones et al. (Cell 2022) profiled 104,197 cells from KP, KPL and KPA
mouse lung tumours and their metastases, with Cas9 lineage recording. The
processed object stores integer counts for 27,998 genes in ``raw/X``; the
top-level ``X`` is 4,000 highly variable genes and is not used, because
``prepare_joint_representation`` makes its own gene selection.

Every label here is the authors', none is derived:

* **Tumour** is ``obs['Tumor']``, named ``<mouse>_<genotype>_<tumour id>``.
  The KPTracer-Data README defines a tumour id that does not begin with "T"
  as a metastasis; that rule is applied as written, and tumour ids
  containing "Normal" or missing ones are not tumours and pair with nothing.
* **Metastatic family** is ``obs['MetFamily']``, the README's "relationships
  of tumors due to metastasis". A metastasis and a primary tumour are in the
  same family when their MetFamily values are equal. This is the truth the
  analysis is scored against; nothing finer (a seeding clade within a
  tumour) is used, because the authors do not provide one per cell.
* **Malignant** is ``not obs['is_normal']``, the authors' normal-cell
  annotation, written to ``obs['malignant']`` as malignant / non_malignant
  so ``02_run_pair.py --analysis-scope malignant --malignant-column
  malignant`` selects on it. Normal cells are kept in the files, labelled.
* **cell_type** is ``obs['Cluster-Name']``, the authors' transferred labels.

Pairs: each metastasis is paired with every primary tumour of the same
mouse, so each pair asks one question -- does this primary tumour's
malignant population have support in this metastasis -- and the column
``same_met_family`` records the authors' answer. A primary cell therefore
enters as many fits as its mouse has metastases; the readout is per pair,
not a per-cell consensus, so nothing is collapsed across fits. Pairs whose
smaller side has fewer than ``--minimum-cells-per-side`` malignant cells
are listed with ``eligible`` False and a reason, never dropped.

``--plan`` writes the tumour table and the manifests without the H5ADs, so
the design can be read before anything large is written. The manifest has
the columns ``01_build_pair_manifest.py`` emits plus the family columns,
so the existing array, depth equalisation (``27_downsample_counts.py``),
pseudobulk and DEG stages run on it unmodified.
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

CHUNK = 64_000_000
OBS_COLUMNS = {
    "Tumor": "tumor",
    "MetFamily": "met_family",
    "Mouse": "mouse",
    "genotype": "genotype",
    "is_normal": "is_normal",
    "Cluster-Name": "cluster_name",
    "SubTumor": "sub_tumor",
    "Lane": "lane",
}


def text(values) -> np.ndarray:
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def read_obs_column(group: h5py.Group, name: str) -> np.ndarray:
    node = group[name]
    if isinstance(node, h5py.Group) and "categories" in node:
        categories = text(node["categories"][:])
        codes = np.asarray(node["codes"][:])
        out = np.full(codes.shape, "nan", dtype=object)
        out[codes >= 0] = categories[codes[codes >= 0]]
        return out
    values = node[:]
    if values.dtype.kind in "OS" or h5py.check_string_dtype(node.dtype):
        return text(values)
    return values


def read_index(group: h5py.Group) -> np.ndarray:
    key = group.attrs.get("_index", "_index")
    key = key.decode() if isinstance(key, bytes) else str(key)
    node = group[key]
    if isinstance(node, h5py.Group):  # an index stored as a categorical
        return read_obs_column(group, key).astype(str)
    return text(node[:]).astype(str)


def read_csr(node: h5py.Group) -> sparse.csr_matrix:
    shape = tuple(int(v) for v in node.attrs["shape"])
    nnz = node["data"].shape[0]
    data = np.empty(nnz, dtype=np.int32)
    indices = np.empty(nnz, dtype=np.int32)
    for start in range(0, nnz, CHUNK):
        stop = min(start + CHUNK, nnz)
        chunk = node["data"][start:stop]
        if not np.all(np.equal(np.mod(chunk, 1), 0)):
            raise RuntimeError("raw/X holds non-integer values; expected raw counts")
        data[start:stop] = chunk
        indices[start:stop] = node["indices"][start:stop]
    indptr = np.asarray(node["indptr"][:], dtype=np.int64)
    return sparse.csr_matrix((data, indices, indptr), shape=shape)


def tumour_kind(name: str) -> str:
    """The README's rule on the last token of <mouse>_<genotype>_<tumour id>."""
    if name in ("nan", "", "None") or "Normal" in name:
        return "not_a_tumour"
    tumour_id = name.split("_")[-1]
    return "primary" if tumour_id.startswith("T") else "metastasis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("h5ad", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--dataset-id", default="KPTracer")
    parser.add_argument("--minimum-cells-per-side", type=int, default=20,
                        help="Malignant cells a side needs for the pair to be eligible "
                             "(01_build_pair_manifest.py's default)")
    parser.add_argument("--plan", action="store_true", help="Write the tables, not the H5ADs")
    args = parser.parse_args()
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5ad, "r") as handle:
        obs = pd.DataFrame({new: read_obs_column(handle["obs"], old) for old, new in OBS_COLUMNS.items()})
        obs["observation_id"] = read_index(handle["obs"])
        raw_genes = read_index(handle["raw"]["var"])
        raw_shape = tuple(int(v) for v in handle["raw"]["X"].attrs["shape"])
        matrix = None if args.plan else read_csr(handle["raw"]["X"])

    if raw_shape[0] != len(obs):
        raise RuntimeError(f"raw/X has {raw_shape[0]} rows for {len(obs)} obs rows")
    if raw_shape[1] != len(raw_genes):
        raise RuntimeError(f"raw/X has {raw_shape[1]} columns for {len(raw_genes)} raw/var rows")
    obs["tumor"] = obs["tumor"].astype(str)
    obs["met_family"] = obs["met_family"].astype(str)
    obs["mouse"] = obs["mouse"].astype(str)
    normal = obs["is_normal"]
    if normal.dtype == object:  # stored as text: map the two spellings, refuse anything else
        mapping = {"True": True, "False": False, "true": True, "false": False}
        unknown = sorted(set(normal.astype(str)) - set(mapping))
        if unknown:
            raise RuntimeError(f"is_normal has values other than True/False: {unknown}")
        normal = normal.astype(str).map(mapping)
    obs["is_normal"] = normal.astype(bool)
    obs["malignant"] = np.where(obs["is_normal"], "non_malignant", "malignant")
    obs["kind"] = obs["tumor"].map(tumour_kind)

    # One row per tumour. MetFamily is a tumour-level label; any tumour whose
    # cells carry more than one value is reported, and its family is the value
    # of the majority of its malignant cells.
    rows = []
    for tumour, group in obs.groupby("tumor", sort=True):
        malignant = group[group["malignant"].eq("malignant")]
        families = (malignant if len(malignant) else group)["met_family"].value_counts()
        rows.append({
            "tumor": tumour, "kind": tumour_kind(tumour), "mouse": group["mouse"].iloc[0],
            "genotype": group["genotype"].iloc[0], "cells": len(group), "malignant_cells": len(malignant),
            "met_family": families.index[0], "met_family_values": int(len(families)),
            "met_family_counts": json.dumps({str(k): int(v) for k, v in families.items()}),
            "sub_tumors": int(group["sub_tumor"].nunique()),
        })
    tumours = pd.DataFrame(rows)
    tumours.to_csv(out / "kptracer_tumours.csv", index=False)

    written: dict[str, str] = {}
    if not args.plan:
        (out / "h5ad").mkdir(exist_ok=True)
        genes = pd.Index(raw_genes)
        duplicated = int(genes.duplicated().sum())
        var = pd.DataFrame({"gene_symbol": raw_genes}, index=pd.Index(raw_genes))
        for tumour in tumours.loc[tumours["kind"].ne("not_a_tumour"), "tumor"]:
            group = obs[obs["tumor"].eq(tumour)]
            rows_index = np.asarray(group.index, dtype=np.int64)
            piece_obs = pd.DataFrame({
                "patient_id": group["mouse"].to_numpy(),
                "site": group["kind"].to_numpy(),
                "sample_id": group["tumor"].to_numpy(),
                "cell_type": group["cluster_name"].astype(str).to_numpy(),
                "malignant": group["malignant"].to_numpy(),
                "met_family": group["met_family"].to_numpy(),
                "sub_tumor": group["sub_tumor"].astype(str).to_numpy(),
                "genotype": group["genotype"].astype(str).to_numpy(),
                "lane": group["lane"].astype(str).to_numpy(),
            }, index=pd.Index(group["observation_id"].to_numpy(), dtype=str))
            piece = ad.AnnData(X=matrix[rows_index].tocsr(), obs=piece_obs, var=var.copy())
            if duplicated:
                piece.var_names_make_unique()
            path = out / "h5ad" / f"{tumour}.h5ad"
            piece.write_h5ad(path, compression="lzf")
            written[tumour] = str(path)
            del piece
    else:
        duplicated = int(pd.Index(raw_genes).duplicated().sum())

    def path_of(tumour: str) -> str:
        return written.get(tumour, str(out / "h5ad" / f"{tumour}.h5ad"))

    table = tumours.set_index("tumor")
    pairs = []
    for metastasis in tumours.loc[tumours["kind"].eq("metastasis")].itertuples(index=False):
        primaries = tumours[tumours["kind"].eq("primary") & tumours["mouse"].eq(metastasis.mouse)]
        if primaries.empty:
            pairs.append({"pair_id": f"{metastasis.mouse}__none__{metastasis.tumor}",
                          "patient_id": metastasis.mouse, "source_sample": "", "target_sample": metastasis.tumor,
                          "eligible": False, "skip_reason": "no primary tumour in this mouse"})
            continue
        for primary in primaries.itertuples(index=False):
            reasons = []
            if primary.malignant_cells < args.minimum_cells_per_side:
                reasons.append(f"source_malignant_cells<{args.minimum_cells_per_side}")
            if metastasis.malignant_cells < args.minimum_cells_per_side:
                reasons.append(f"target_malignant_cells<{args.minimum_cells_per_side}")
            pairs.append({
                "pair_id": f"{metastasis.mouse}__{primary.tumor}__{metastasis.tumor}",
                "dataset_id": args.dataset_id,
                "patient_id": metastasis.mouse,
                "source_h5ad": path_of(primary.tumor),
                "target_h5ad": path_of(metastasis.tumor),
                "source_h5ads_json": json.dumps([path_of(primary.tumor)]),
                "target_h5ads_json": json.dumps([path_of(metastasis.tumor)]),
                "source_sample": primary.tumor,
                "target_sample": metastasis.tumor,
                "source_n": int(primary.cells),
                "target_n": int(metastasis.cells),
                "source_malignant_n": int(primary.malignant_cells),
                "target_malignant_n": int(metastasis.malignant_cells),
                "source_file_n": 1, "target_file_n": 1,
                "source_file_kind": "tumour", "target_file_kind": "tumour",
                "has_cell_type": True,
                "genotype": metastasis.genotype,
                "source_met_family": primary.met_family,
                "target_met_family": metastasis.met_family,
                "same_met_family": bool(primary.met_family == metastasis.met_family),
                "eligible": not reasons,
                "skip_reason": ";".join(reasons),
            })
    manifest = pd.DataFrame(pairs)
    manifest.insert(0, "pair_index", np.arange(len(manifest), dtype=int))
    manifest["eligible_index"] = -1
    eligible = manifest.index[manifest["eligible"].astype(bool)]
    manifest.loc[eligible, "eligible_index"] = np.arange(len(eligible), dtype=int)
    manifest.to_csv(out / "pair_manifest_all.csv", index=False)
    manifest[manifest["eligible"].astype(bool)].to_csv(out / "pair_manifest_eligible.csv", index=False)

    mice_with_metastasis = sorted(tumours.loc[tumours["kind"].eq("metastasis"), "mouse"].unique())
    report = {
        "source_h5ad": str(args.h5ad),
        "plan_only": bool(args.plan),
        "cells": int(len(obs)),
        "malignant_cells": int(obs["malignant"].eq("malignant").sum()),
        "genes": int(len(raw_genes)), "duplicated_gene_names": duplicated,
        "tumours_by_kind": {k: int(v) for k, v in tumours["kind"].value_counts().items()},
        "tumours_with_several_met_family_values": tumours.loc[tumours["met_family_values"] > 1, "tumor"].tolist(),
        "mice_with_a_metastasis": mice_with_metastasis,
        "pairs": int(len(manifest)),
        "pairs_eligible": int(manifest["eligible"].astype(bool).sum()),
        "pairs_eligible_same_family": int((manifest["eligible"].astype(bool)
                                           & manifest.get("same_met_family", False).astype(bool)).sum()),
        "h5ads_written": len(written),
        "minimum_cells_per_side": args.minimum_cells_per_side,
    }
    (out / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(tumours.loc[tumours["mouse"].isin(mice_with_metastasis),
                      ["tumor", "kind", "genotype", "cells", "malignant_cells", "met_family",
                       "met_family_values", "sub_tumors"]].to_string(index=False), flush=True)
    columns = ["pair_id", "source_malignant_n", "target_malignant_n", "same_met_family", "eligible", "skip_reason"]
    print(manifest[[c for c in columns if c in manifest.columns]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

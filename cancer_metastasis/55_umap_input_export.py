"""Counts for local UMAP work: one h5ad per dataset with exactly the cells of the strategy export.

    python cancer_metastasis/55_umap_input_export.py --cells EXPORT/cells \\
        --manifest PANCANCER_ELIGIBLE.csv --manifest PROSTATE_ELIGIBLE.csv --out OUT \\
        --datasets GSE225857 [--ds-manifest UNIFORM_DS.csv --ds-manifest PROSTATE_DS.csv]

Cells are 53_patient_umap.py's: the union by cell_uid of every reference cell of the dataset's
pairs, both sides. Writes OUT/<dataset>_raw.h5ad (original counts, as the raw/log1p and replication
UMAPs read them) and, with --ds-manifest, OUT/<dataset>_ds.h5ad (the depth-equalised counts the
strategy OT read; cells absent from them are listed in OUT/<dataset>_ds_missing.tsv). X is the
integer count matrix (CSR); obs is cell_uid, dataset_id, patient_id, side, sample, observation_id,
pairs; var is the gene key. The OT labels are not copied: join the export's cells/ tables on
cell_uid. Nothing is computed from the counts here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_spec = importlib.util.spec_from_file_location("patient53", HERE / "53_patient_umap.py")
patient53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patient53)


def write(cells: pd.DataFrame, sources: dict, path: Path) -> list:
    import anndata as ad
    from scipy import sparse

    from common import expression_matrix

    missing: list = []
    sides = []
    for side in ("primary", "metastasis"):
        data = patient53.load_side(cells[cells["side"] == side], sources, missing)
        sides.append(ad.AnnData(X=sparse.csr_matrix(expression_matrix(data)),
                                var=pd.DataFrame(index=data.var_names.astype(str)),
                                obs=pd.DataFrame(index=data.obs_names.astype(str))))
    data = ad.concat(sides, join="inner", index_unique=None)
    obs = cells.set_index("cell_uid").loc[data.obs_names].rename_axis("cell_uid").reset_index()
    data.obs = obs.set_index("cell_uid", drop=False).astype(str)
    data.obs.index.name = None
    data.uns["source"] = {"script": "55_umap_input_export.py", "cells": int(data.n_obs), "genes": int(data.n_vars)}
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(path, compression="gzip")
    return missing


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--ds-manifest", type=Path, action="append", default=[])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = patient53.patient_cells(args.cells, args.datasets)
    raw_sources = patient53.sample_sources(args.manifest)
    ds_sources = patient53.sample_sources(args.ds_manifest) if args.ds_manifest else None
    report = {}
    for dataset in args.datasets:
        part = cells[cells["dataset_id"] == dataset]
        raw_path = args.out / f"{dataset}_raw.h5ad"
        missing = write(part, raw_sources, raw_path)
        if missing:
            raise SystemExit(f"{dataset}: {len(missing)} reference cells are absent from the original h5ads")
        report[dataset] = {"cells": part["side"].value_counts().to_dict(), "raw": str(raw_path.name),
                           "raw_md5": md5(raw_path)}
        if ds_sources is not None:
            ds_path = args.out / f"{dataset}_ds.h5ad"
            missing = write(part, ds_sources, ds_path)
            pd.DataFrame(missing, columns=["cell_uid", "side", "sample", "observation_id"]).to_csv(
                args.out / f"{dataset}_ds_missing.tsv", sep="\t", index=False)
            report[dataset].update({"ds": ds_path.name, "ds_md5": md5(ds_path), "ds_missing": len(missing)})
        print(dataset, json.dumps(report[dataset]), flush=True)
        # one manifest per dataset, so jobs exporting different datasets into one folder never collide
        (args.out / f"{dataset}_MANIFEST.json").write_text(json.dumps(report[dataset], indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

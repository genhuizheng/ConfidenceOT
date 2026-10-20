"""One infercnvpy unit per KP-Tracer mouse, from the per-tumour H5ADs already prepared.

    python kp_lineage/cnv/01_build_infercnvpy_units.py --prepared PREPARED_ROOT \\
        --gtf gencode.vM23.annotation.gtf.gz --out UNITS_ROOT

Reads only. For every mouse with an eligible pair (prepared/pair_manifest_eligible.csv),
every tumour H5AD of that mouse (prepared/h5ad/<tumour>.h5ad, as written by
01_prepare_kptracer_pairs.py; raw counts, the authors' labels) is put into one unit, so the
mouse's primary tumours and metastases are inferred against one reference on one scale:

* cells: every cell of those H5ADs, unchanged;
* group: ``ref_normal`` for the authors' normal cells (malignant == non_malignant, i.e.
  is_normal), ``malignant`` for the rest; nothing else decides the reference;
* source_file, barcode: the tumour H5AD and the cell's obs_name there, the name
  02_run_pair.py looks a cell up by;
* genes: the H5AD's genes that GENCODE vM23 (GRCm38/mm10) names exactly once, with that
  gene's chromosome, start and end. A symbol with no GENCODE gene, or with more than one,
  has no position and is left out (infercnvpy skips such genes anyway); both are counted.

Writes UNITS_ROOT/<mouse>/unit.h5ad (X = raw counts, obs group/source_file/barcode/tumour,
var chr/start/stop) and unit.json, and UNITS_ROOT/units.tsv (KPTracer, mouse, path), the
list cancer_metastasis/49_infercnvpy_unit.py --unit-h5ad runs on. A mouse without a normal
cell is still written and listed; 49_ stops on it with "no ref_ group", which is reported.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def gencode_genes(path: Path) -> tuple[pd.DataFrame, set]:
    """gene_name -> chr, start, end, from the GTF's gene records; names seen more than once are returned apart."""
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = re.search(r'gene_name "([^"]+)"', fields[8])
            if match:
                rows.append((match.group(1), fields[0], int(fields[3]), int(fields[4])))
    table = pd.DataFrame(rows, columns=["gene", "chr", "start", "stop"])
    repeated = set(table.loc[table["gene"].duplicated(keep=False), "gene"])
    return table[~table["gene"].isin(repeated)].set_index("gene"), repeated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--gtf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    positions, repeated = gencode_genes(args.gtf)
    manifest = pd.read_csv(args.prepared / "pair_manifest_eligible.csv", dtype=str)
    tumours = pd.read_csv(args.prepared / "kptracer_tumours.csv", dtype=str)
    mice = sorted(manifest["patient_id"].unique(), key=lambda m: (len(m), m))
    args.out.mkdir(parents=True, exist_ok=True)
    listing = []
    for mouse in mice:
        names = tumours.loc[tumours["mouse"].eq(mouse) & tumours["kind"].ne("not_a_tumour"), "tumor"].tolist()
        pieces, sources = [], []
        for name in sorted(names):
            path = args.prepared / "h5ad" / f"{name}.h5ad"
            if not path.exists():
                raise FileNotFoundError(path)
            piece = ad.read_h5ad(path)
            piece.obs = pd.DataFrame({
                "group": np.where(piece.obs["malignant"].astype(str).eq("non_malignant"), "ref_normal", "malignant"),
                "source_file": path.name, "barcode": piece.obs_names.astype(str), "tumour": name,
            }, index=piece.obs_names.astype(str))
            pieces.append(piece)
            sources.append(path.name)
        unit = ad.concat(pieces, join="inner", merge="same")
        if unit.obs_names.duplicated().any():
            raise ValueError(f"mouse {mouse}: a cell name repeats across its tumours")
        genes = pd.Index(unit.var_names.astype(str))
        known = genes.isin(positions.index)
        unit = unit[:, known].copy()
        unit.var = positions.loc[unit.var_names.astype(str), ["chr", "start", "stop"]].copy()
        unit.X = unit.X.tocsr() if sp.issparse(unit.X) else sp.csr_matrix(unit.X)
        folder = args.out / mouse
        folder.mkdir(parents=True, exist_ok=True)
        unit.write_h5ad(folder / "unit.h5ad", compression="lzf")
        record = {
            "mouse": mouse, "tumours": sources, "cells": int(unit.n_obs),
            "cells_by_group": {str(k): int(v) for k, v in unit.obs["group"].value_counts().items()},
            "cells_by_tumour": {str(k): int(v) for k, v in unit.obs["tumour"].value_counts().items()},
            "genes_in_h5ad": int(len(genes)), "genes_with_position": int(known.sum()),
            "genes_without_gencode_gene": int((~genes.isin(positions.index) & ~genes.isin(repeated)).sum()),
            "genes_with_more_than_one_gencode_gene": int(genes.isin(repeated).sum()),
            "gtf": str(args.gtf),
        }
        (folder / "unit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        listing.append(("KPTracer", mouse, str(folder / "unit.h5ad")))
        print(mouse, record["cells_by_group"], f"genes {record['genes_with_position']}/{record['genes_in_h5ad']}",
              flush=True)
    pd.DataFrame(listing).to_csv(args.out / "units.tsv", sep="\t", header=False, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()

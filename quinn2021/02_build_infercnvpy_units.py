"""One infercnvpy unit per Quinn et al. 2021 experiment, from the converted H5ADs.

    python quinn2021/02_build_infercnvpy_units.py --converted CONVERTED_DIR \\
        --gtf-grch38 gencode.v28.annotation.gtf.gz --gtf-grch37 gencode.v19.annotation.gtf.gz \\
        --out UNITS_ROOT

Reads only. Every cell of an experiment's H5AD (converted/h5ad/<tag>.h5ad, raw counts as
deposited) goes into its unit unchanged. All cells are A549 tumour cells, so there is no
reference group: 49_infercnvpy_unit.py runs these units with --no-reference, infercnvpy's
default of the mean of all cells.

Gene positions come from the GENCODE release matching the reference each matrix was
counted against, joined on the Ensembl gene id (version suffix dropped):
  5k, 30k, LM0  (33,694 genes, Cell Ranger GRCh38-3.0.0, Ensembl 93)  -> GENCODE v28, GRCh38
  10k, 100k     (32,738 genes, Cell Ranger hg19-1.2.0)                -> GENCODE v19, GRCh37
A gene id the GTF lacks has no position and is left out (infercnvpy skips such genes
anyway); the number is recorded per unit.

Writes UNITS_ROOT/<tag>/unit.h5ad (X raw counts; obs group = observation, source_file,
barcode; var chr/start/stop, gene_id, gene_symbol), unit.json, and UNITS_ROOT/units.tsv
(Quinn2021, tag, path) for 49_ --unit-h5ad.
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

ASSEMBLY = {"5k": "grch38", "30k": "grch38", "LM0": "grch38", "10k": "grch37", "100k": "grch37"}
EXPECTED_GENES = {"grch38": 33694, "grch37": 32738}


def gene_positions(path: Path) -> pd.DataFrame:
    rows = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            match = re.search(r'gene_id "([^"]+)"', fields[8])
            if match:
                rows.append((match.group(1).split(".")[0], fields[0], int(fields[3]), int(fields[4])))
    table = pd.DataFrame(rows, columns=["gene_id", "chr", "start", "stop"])
    # PAR genes appear twice in GENCODE (chrX and chrY, the chrY copy tagged _PAR_Y); keep the first.
    return table.drop_duplicates("gene_id", keep="first").set_index("gene_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--gtf-grch38", type=Path, required=True)
    parser.add_argument("--gtf-grch37", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    positions = {"grch38": gene_positions(args.gtf_grch38), "grch37": gene_positions(args.gtf_grch37)}
    args.out.mkdir(parents=True, exist_ok=True)
    listing = []
    for path in sorted((args.converted / "h5ad").glob("*.h5ad")):
        tag = path.stem
        if tag not in ASSEMBLY:
            raise ValueError(f"{path}: no assembly recorded for experiment {tag}")
        assembly = ASSEMBLY[tag]
        data = ad.read_h5ad(path)
        if data.n_vars != EXPECTED_GENES[assembly]:
            raise ValueError(f"{tag}: {data.n_vars} genes, expected {EXPECTED_GENES[assembly]} for {assembly}")
        ids = data.var["gene_id"].astype(str).str.split(".").str[0]
        table = positions[assembly]
        known = ids.isin(table.index).to_numpy()
        unit = ad.AnnData(
            X=(data.X.tocsr() if sp.issparse(data.X) else sp.csr_matrix(data.X))[:, known],
            obs=pd.DataFrame({"group": "observation", "source_file": path.name,
                              "barcode": data.obs_names.astype(str)}, index=data.obs_names.astype(str)),
            var=pd.DataFrame({"chr": table.loc[ids[known], "chr"].to_numpy(),
                              "start": table.loc[ids[known], "start"].to_numpy(),
                              "stop": table.loc[ids[known], "stop"].to_numpy(),
                              "gene_id": ids[known].to_numpy(),
                              "gene_symbol": data.var["gene_symbol"].astype(str).to_numpy()[known]},
                             index=data.var_names[known]))
        folder = args.out / tag
        folder.mkdir(parents=True, exist_ok=True)
        unit.write_h5ad(folder / "unit.h5ad", compression="lzf")
        record = {"experiment": tag, "assembly": assembly, "cells": int(unit.n_obs),
                  "genes_in_h5ad": int(data.n_vars), "genes_with_position": int(known.sum()),
                  "genes_without_position": int((~known).sum()),
                  "gtf": str(args.gtf_grch38 if assembly == "grch38" else args.gtf_grch37)}
        (folder / "unit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        listing.append(("Quinn2021", tag, str(folder / "unit.h5ad")))
        print(tag, assembly, unit.shape, f"genes {record['genes_with_position']}/{record['genes_in_h5ad']}", flush=True)
    pd.DataFrame(listing).to_csv(args.out / "units.tsv", sep="\t", header=False, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()

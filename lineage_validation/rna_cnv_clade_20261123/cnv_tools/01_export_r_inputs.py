"""The infercnvpy units as plain files the R CNV tools read (inferCNV, CopyKAT, SCEVAN, CONICSmat): same cells, same
genes, same reference as the existing infercnvpy profiles; only the method changes.

    python lineage_validation/rna_cnv_clade_20261123/cnv_tools/01_export_r_inputs.py \
        --units KP_UNITS.tsv QUINN_UNITS.tsv --out INPUTS_ROOT [--skip LM0]

Each units.tsv row: dataset, patient, unit.h5ad (kp_lineage/cnv/01_build_infercnvpy_units.py,
quinn2021/02_build_infercnvpy_units.py): X = raw counts (cells x genes), obs group / source_file / barcode, var
chr / start / stop. Written to INPUTS_ROOT/<dataset>/<patient>/:
  counts.mtx      raw counts, genes x cells (MatrixMarket, integer), in the h5ad's cell and gene order
  genes.tsv       gene, chr, start, stop (no header row repeated per gene; header line present)
  cells.tsv       cell, group, is_reference, barcode, source_file; is_reference = group starts with "ref_" (the
                  authors' normal cells in KP-Tracer; Quinn has none, so every cell is an observation)
  SUCCESS         with the counts of cells, reference cells and genes
Counts are checked to be non-negative integers. Refuses a unit with SUCCESS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--units", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip", nargs="*", default=["LM0"], help="patients not used by any OT unit (Quinn LM0: pre-injection)")
    a = ap.parse_args()
    for table in a.units:
        for line in table.read_text().splitlines():
            if not line.strip():
                continue
            dataset, patient, h5ad = line.split("\t")
            if patient in a.skip:
                print("skip", dataset, patient); continue
            out = a.out / dataset / patient
            if (out / "SUCCESS").exists():
                print("done", dataset, patient); continue
            out.mkdir(parents=True, exist_ok=True)
            d = ad.read_h5ad(h5ad)
            X = sp.csr_matrix(d.X)
            if X.nnz and (X.data.min() < 0 or not np.allclose(X.data, np.round(X.data))):
                raise SystemExit(f"{h5ad}: X is not non-negative integer counts")
            for col in ("group", "source_file", "barcode"):
                if col not in d.obs:
                    raise SystemExit(f"{h5ad}: obs has no {col}")
            for col in ("chr", "start", "stop"):
                if col not in d.var:
                    raise SystemExit(f"{h5ad}: var has no {col}")
            if not d.obs_names.is_unique or not d.var_names.is_unique:
                raise SystemExit(f"{h5ad}: cell or gene names repeat")
            scipy.io.mmwrite(str(out / "counts.mtx"), sp.coo_matrix(X.T.astype(np.int64)), field="integer")
            pd.DataFrame({"gene": d.var_names.astype(str), "chr": d.var["chr"].astype(str).to_numpy(),
                          "start": d.var["start"].astype(int).to_numpy(), "stop": d.var["stop"].astype(int).to_numpy()}
                         ).to_csv(out / "genes.tsv", sep="\t", index=False)
            group = d.obs["group"].astype(str).to_numpy()
            cells = pd.DataFrame({"cell": d.obs_names.astype(str), "group": group, "is_reference": np.char.startswith(group.astype(str), "ref_"),
                                  "barcode": d.obs["barcode"].astype(str).to_numpy(), "source_file": d.obs["source_file"].astype(str).to_numpy()})
            cells.to_csv(out / "cells.tsv", sep="\t", index=False)
            rec = {"dataset": dataset, "patient": patient, "unit_h5ad": h5ad, "cells": int(d.n_obs),
                   "reference_cells": int(cells.is_reference.sum()), "genes": int(d.n_vars)}
            (out / "SUCCESS").write_text(json.dumps(rec, indent=1))
            print(rec, flush=True)


if __name__ == "__main__":
    main()

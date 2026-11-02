"""The regenerated frozen levels must be the frozen benchmark: compare counts, cells and groups of a 20_generate_n5000.R
directory (L0 / L1 / L2) with the native generator's directory for the same N, replicate and level.

    python realdata_benchmark/splatter_n5000/25_check_native_identity.py --new GEN_DIR --native NATIVE_GEN_DIR

Writes GEN_DIR/NATIVE_IDENTITY.json (identical: true / false and the first difference) and exits 1 when not
identical. If it fails, L0-L2 rows must be refitted on the new data instead of reusing the frozen fits.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--new", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    a = p.parse_args()
    rec = {"new": str(a.new), "native": str(a.native)}
    cn, co = pd.read_csv(a.new / "cells.csv"), pd.read_csv(a.native / "cells.csv")
    rec["cells_identical"] = cn.equals(co)
    gn, go = pd.read_csv(a.new / "genes.csv"), pd.read_csv(a.native / "genes.csv")
    rec["genes_identical"] = list(gn.gene_id) == list(go.gene_id) and np.allclose(gn.gene_mean, go.gene_mean, rtol=0, atol=0)
    if rec["cells_identical"] and rec["genes_identical"]:
        A = sp.csr_matrix(scipy.io.mmread(str(a.new / "counts.mtx"))); B = sp.csr_matrix(scipy.io.mmread(str(a.native / "counts.mtx")))
        d = (A != B)
        rec["counts_identical"] = A.shape == B.shape and d.nnz == 0
        rec["counts_differing_entries"] = int(d.nnz) if A.shape == B.shape else None
    else:
        rec["counts_identical"] = False
    rec["identical"] = bool(rec["cells_identical"] and rec["genes_identical"] and rec["counts_identical"])
    (a.new / "NATIVE_IDENTITY.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))
    sys.exit(0 if rec["identical"] else 1)


if __name__ == "__main__":
    main()

"""Pilot 3, Cook & Vanderhyden 2020 (GSE147405): A549 lung adenocarcinoma cells under TGF-beta1, a time course of EMT
(0 d, 8 h, 1 d, 3 d, 7 d) followed by withdrawal of TGF-beta1 after 7 d (8 h, 1 d, 3 d after removal: '_rm'). The same
data are used by TIGON and DiffusionOT. Raw CellRanger counts, the authors' metadata (Time, Cluster, Pseudotime, Phase).

Units = adjacent time points, every cell (no cap; sides of 129-885 cells):
  E1 0d -> 8h, E2 8h -> 1d, E3 1d -> 3d, E4 3d -> 7d                  (EMT induction)
  W1 7d -> 8h_rm, W2 8h_rm -> 1d_rm, W3 1d_rm -> 3d_rm               (withdrawal)
  R0 0d -> 3d_rm                                                      (does withdrawal return to the untreated state?)
There is no lineage or per-cell truth; evaluation uses the authors' pseudotime and EMT / epithelial marker programs.

    python -I lineage_validation/app_search_20261117/03_prepare_cook.py --data results/app_search_20261117/data/cook_gse147405 \
        --out results/app_search_20261117/cook_pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

UNITS = {"E1_0d_to_8h": ("0d", "8h"), "E2_8h_to_1d": ("8h", "1d"), "E3_1d_to_3d": ("1d", "3d"), "E4_3d_to_7d": ("3d", "7d"),
         "W1_7d_to_8h_rm": ("7d", "8h_rm"), "W2_8h_rm_to_1d_rm": ("8h_rm", "1d_rm"), "W3_1d_rm_to_3d_rm": ("1d_rm", "3d_rm"),
         "R0_0d_to_3d_rm": ("0d", "3d_rm")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import anndata as ad
    meta = pd.read_csv(a.data / "GSE147405_A549_TGFB1_TimeCourse_metadata.csv.gz", index_col=0)
    chunks, genes = [], []
    for ch in pd.read_csv(a.data / "GSE147405_A549_TGFB1_TimeCourse_UMI_matrix.csv.gz", index_col=0, chunksize=2000):
        genes += list(ch.index.astype(str)); chunks.append(sp.csr_matrix(ch.to_numpy(dtype=np.float32)))
        cells = ch.columns.astype(str)
    X = sp.vstack(chunks).T.tocsr()                       # cells x genes
    assert np.allclose(X.data, np.round(X.data)), "counts are not integers"
    A = ad.AnnData(X=X, obs=meta.reindex(cells), var=pd.DataFrame(index=ad.utils.make_index_unique(pd.Index(genes))))
    print(A.shape, A.obs.Time.value_counts().to_dict(), "metadata missing:", int(A.obs.Time.isna().sum()), flush=True)
    keep_cols = ["Time", "Cluster", "Pseudotime", "Phase", "nCount_RNA", "nFeature_RNA", "percent.mito"]
    (a.out / "inputs").mkdir(parents=True, exist_ok=True); rows = []
    for u, (t0, t1) in UNITS.items():
        rec = {"pair_id": u, "dataset_id": "Cook2020_A549_TGFB1", "patient_id": "A549", "eligible": True}
        for side, t in (("source", t0), ("target", t1)):
            B = A[(A.obs.Time == t).to_numpy()].copy(); B.obs = B.obs[keep_cols].copy()
            B.obs["sample_id"] = f"{u}__{side}"; B.obs["malignant"] = "malignant"; B.obs.index = [f"{u}_{side}_{i}" for i in B.obs.index]
            f = a.out / "inputs" / f"{u}__{side}.h5ad"; B.write_h5ad(f, compression="gzip")
            rec[f"{side}_sample"] = f"{u}__{side}"; rec[f"{side}_h5ad"] = str(f.resolve()); rec[f"{side}_cells"] = B.n_obs; rec[f"{side}_time"] = t
        rows.append(rec); print(u, rec["source_cells"], "->", rec["target_cells"], flush=True)
    pd.DataFrame(rows).to_csv(a.out / "manifest.csv", index=False)


if __name__ == "__main__":
    main()

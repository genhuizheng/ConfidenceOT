"""Pilot 2, Kang et al. 2018 (GSE96583 batch 2): PBMC control -> IFN-beta stimulated (6 h), the perturbation dataset used
by CellOT and scGen. Raw CellRanger counts of the two lanes, singlets with an authors' cell type. Cell identity does not
change under 6 h of IFN-beta, so the cell type is a per-cell ground truth for the matching (a correct coupling keeps
mass within a cell type), and no population is truly one-sided: rejection there is false rejection.

Units (at most CAP cells per side, random, seed 20261117):
  K0_ctrl_to_stim          every cell type on both sides (negative control for rejection)
  K1_noB_in_stim           B cells removed from the stimulated side: control B cells have no counterpart (source-side truth)
  K2_noFCGR3A_in_ctrl      FCGR3A+ monocytes removed from the control side: stimulated FCGR3A+ monocytes have no
                           counterpart, while a close population (CD14+ monocytes) remains (target-side truth)
The removal is a constructed one-sided population on real data (as in uniPort's held-out cell types); it says what
the gate does when a population is truly absent, not what IFN-beta does.

    python -I lineage_validation/app_search_20261117/02_prepare_kang.py --data results/app_search_20261117/data/kang_gse96583 \
        --out results/app_search_20261117/kang_pilot
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

CAP, SEED = 3000, 20261117


def read_lane(d, gsm, mtx):
    X = scipy.io.mmread(gzip.open(d / mtx)).tocsr().T.tocsr()      # cells x genes
    bc = pd.read_csv(d / f"{gsm}_barcodes.tsv.gz", header=None)[0].astype(str).to_numpy()
    return X, bc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    import anndata as ad
    genes = pd.read_csv(a.data / "GSE96583_batch2.genes.tsv.gz", sep="\t", header=None)
    sym = pd.Index(ad.utils.make_index_unique(pd.Index(genes[1].astype(str).to_numpy())), name=None)
    lab = pd.read_csv(a.data / "GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    Xc, bc_c = read_lane(a.data, "GSM2560248", "GSM2560248_2.1.mtx.gz")
    Xs, bc_s = read_lane(a.data, "GSM2560249", "GSM2560249_2.2.mtx.gz")
    # stimulated barcodes that collide with control barcodes carry '-11' in the label table
    key_s = np.array([b.replace("-1", "-11") if b in set(bc_c) else b for b in bc_s])
    sides = {}
    for name, X, keys, stim in (("ctrl", Xc, bc_c, "ctrl"), ("stim", Xs, key_s, "stim")):
        L = lab.reindex(keys)
        ok = (L.stim == stim).to_numpy() & (L.multiplets == "singlet").to_numpy() & L.cell.notna().to_numpy()
        A = ad.AnnData(X=sp.csr_matrix(X[ok], dtype=np.float32), obs=pd.DataFrame({"cell_type": L.cell[ok].astype(str).to_numpy(),
                       "donor": L.ind[ok].astype(str).to_numpy(), "condition": stim}, index=[f"{name}_{k}" for k in keys[ok]]),
                       var=pd.DataFrame(index=sym))
        sides[name] = A
        print(name, A.n_obs, "singlets with a cell type;", A.obs.cell_type.value_counts().to_dict(), flush=True)
    assert all(float(s.X.max()) >= 1 and np.allclose(s.X.data, np.round(s.X.data)) for s in sides.values()), "counts are not integers"
    units = {"K0_ctrl_to_stim": (None, None), "K1_noB_in_stim": (None, "B cells"), "K2_noFCGR3A_in_ctrl": ("FCGR3A+ Monocytes", None)}
    (a.out / "inputs").mkdir(parents=True, exist_ok=True)
    rows = []
    for u, (drop_src, drop_tgt) in units.items():
        rng = np.random.default_rng(SEED)
        rec = {"pair_id": u, "dataset_id": "Kang2018", "patient_id": "pooled_8_donors", "eligible": True,
               "removed_from_source": drop_src or "", "removed_from_target": drop_tgt or ""}
        for side, nm, drop in (("source", "ctrl", drop_src), ("target", "stim", drop_tgt)):
            A = sides[nm]; idx = np.flatnonzero((A.obs.cell_type != drop).to_numpy()) if drop else np.arange(A.n_obs)
            keep = np.sort(rng.choice(idx, min(CAP, len(idx)), replace=False))
            B = A[keep].copy(); B.obs["sample_id"] = f"{u}__{side}"; B.obs["malignant"] = "malignant"
            f = a.out / "inputs" / f"{u}__{side}.h5ad"; B.write_h5ad(f, compression="gzip")
            rec[f"{side}_sample"] = f"{u}__{side}"; rec[f"{side}_h5ad"] = str(f.resolve()); rec[f"{side}_cells"] = len(keep)
        rows.append(rec); print(rec, flush=True)
    pd.DataFrame(rows).to_csv(a.out / "manifest.csv", index=False)


if __name__ == "__main__":
    main()

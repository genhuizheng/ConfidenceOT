"""PLAN_v3_deg.md on TACC: raw counts of every eligible cell (E2 rule, minus the allele-table check, which the local
groups already applied) of the given KP mice, one h5ad per mouse, from the authors' processed object (raw/X).

    python lineage_validation/rna_lineage_discordance_20261110/101_export_kp_counts.py \
        --h5ad .../KPTracer-Data/expression/adata_processed.combined.h5ad --mice 3451 3454 3457 3460 3465 3466 3508 3513 3515 3724 \
        --out OUT

Cells: is_normal False, Cluster-Name present, a tumour (README rule). obs keeps tumor, cluster_name, met_family,
total_counts and detected_genes; var = raw/var gene symbols. Writes OUT/KPTracer_<mouse>.h5ad (the unit_id key of
deg_v3/groups.tsv, read by 60_deg_gsea.py --counts). Refuses an existing OUT.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--mice", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists")
    a.out.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location("prep", REPO / "kp_lineage" / "01_prepare_kptracer_pairs.py")
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    import anndata as ad
    with h5py.File(a.h5ad, "r") as f:
        cells = prep.read_index(f["obs"])
        obs = pd.DataFrame({"cell": cells})
        for col, name in prep.OBS_COLUMNS.items():
            obs[name] = prep.read_obs_column(f["obs"], col) if col in f["obs"] else "nan"
        X = prep.read_csr(f["raw/X"])
        genes = prep.read_index(f["raw/var"]).astype(str)
    obs["tumour_kind"] = [prep.tumour_kind(str(t)) for t in obs.tumor]
    norm = obs.is_normal.astype(str).str.lower().isin(["true", "1", "1.0"])
    lab = obs.cluster_name.astype(str)
    ok = ~norm & ~lab.isin(["nan", "", "None"]) & obs.tumour_kind.isin(["primary", "metastasis"])
    for m in a.mice:
        keep = (ok & (obs.mouse.astype(str) == str(m))).to_numpy()
        o = obs[keep].reset_index(drop=True)
        Xm = X[keep].tocsr()
        o["total_counts"] = np.asarray(Xm.sum(1)).ravel(); o["detected_genes"] = np.diff(Xm.indptr)
        A = ad.AnnData(Xm.astype(np.float32), obs=o.set_index("cell")[["tumor", "met_family", "cluster_name", "tumour_kind", "total_counts", "detected_genes"]],
                       var=pd.DataFrame(index=genes))
        A.obs.index = A.obs.index.astype(str)
        A.write_h5ad(a.out / f"KPTracer_{m}.h5ad", compression="gzip")
        print(f"{m}: {A.n_obs} cells x {A.n_vars} genes", flush=True)


if __name__ == "__main__":
    main()

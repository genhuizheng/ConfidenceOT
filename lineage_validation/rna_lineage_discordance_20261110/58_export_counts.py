"""On TACC: export the raw counts of the cells in GROUPS.tsv (one h5ad per unit), read with the v3 worker's own Inputs,
so that DEG / GSEA can run locally on exactly the counts the RNA-OT fits used. No computation beyond the export.

    python lineage_validation/rna_lineage_discordance_20261110/58_export_counts.py --groups GROUPS.tsv \
        --worklist /scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/WORKLIST.tsv --out OUT

Writes OUT/<unit>.h5ad (raw integer counts of the unit's grouped cells, var = the unit's genes, obs side and the
input obs columns). Refuses a unit whose file exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    import anndata as ad
    sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid"))
    spec = importlib.util.spec_from_file_location("v3worker", REPO / "realdata_benchmark" / "v3_grid" / "worker.py")
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    wl = pd.read_csv(a.worklist, sep="\t", dtype=str)
    G = pd.read_csv(a.groups, sep="\t", dtype=str)
    inputs = w.Inputs()
    for u, g in G.groupby("unit_id", sort=False):
        f = a.out / f"{u}.h5ad"
        if f.exists():
            print(f"{u}: exists"); continue
        row = wl[(wl.unit_id == u) & (wl.key == "logcpm|none")].iloc[0]
        v = inputs.get(row)
        S, T = v["S"].copy(), v["T"].copy()
        S.obs["side"] = "source"; T.obs["side"] = "target"
        A = ad.concat([S, T], join="inner", index_unique=None)
        A.obs_names = A.obs_names.astype(str)
        cells = pd.Index(g.cell.unique())
        missing = cells.difference(A.obs_names)
        if len(missing):
            raise SystemExit(f"{u}: {len(missing)} grouped cells not in the inputs")
        A = A[cells].copy()
        for c in A.obs.columns:
            if A.obs[c].dtype == object:
                A.obs[c] = A.obs[c].astype(str)
        A.write_h5ad(f, compression="gzip")
        print(f"{u}: {A.n_obs} cells x {A.n_vars} genes", flush=True)


if __name__ == "__main__":
    main()

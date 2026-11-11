"""Quinn RNA states (PLAN_v2.md section 3.5): Leiden clusters (resolution 1.0, 15 neighbours, seed 20261110) on the
v3 RNA coordinates of each Quinn unit and configuration (cos configurations on unit rows).

    python lineage_validation/rna_lineage_discordance_20261110/35_quinn_states.py --root results/rna_lineage_discordance_20261110

Writes ROOT/states/<unit>__<tag>.tsv.gz (cell, side, state).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

CONFIGS = {"l5best": "ranknm256-midrank__none__scale__cos__pca50", "logcpm_pca": "logcpm__none__noscale__sqeuc__pca30"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "states"; out.mkdir(parents=True, exist_ok=True)
    base = a.root / "inputs_A" / "rna_lineage_discordance_20261110" / "rna_coords"
    for d in sorted(glob.glob(str(base / "L23__Quinn2021_*"))):
        u = Path(d).name
        for tag, f in CONFIGS.items():
            o = out / f"{u}__{tag}.tsv.gz"
            if o.exists():
                continue
            z = np.load(Path(d) / f"{f}.npz")
            Z = np.vstack([z["X"], z["Y"]]).astype(np.float32)
            if bool(z["cos"]):
                Z /= np.linalg.norm(Z, axis=1, keepdims=True)
            A = ad.AnnData(np.zeros((len(Z), 1), np.float32)); A.obsm["X_rna"] = Z
            sc.pp.neighbors(A, n_neighbors=15, use_rep="X_rna", random_state=20261110)
            sc.tl.leiden(A, resolution=1.0, random_state=20261110, flavor="igraph", n_iterations=2, directed=False)
            ids = np.r_[z["source_ids"].astype(str), z["target_ids"].astype(str)]
            side = np.r_[np.repeat("source", len(z["X"])), np.repeat("target", len(z["Y"]))]
            pd.DataFrame({"cell": ids, "side": side, "state": "S" + A.obs.leiden.astype(str).to_numpy()}).to_csv(o, sep="\t", index=False, compression="gzip")
            print(f"{u} {tag}: {A.obs.leiden.nunique()} states", flush=True)


if __name__ == "__main__":
    main()

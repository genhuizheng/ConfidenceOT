"""LARRY PLAN.md: the 4 unit inputs (source = day 2, target = day 4 or 6, per starting population) for the v3 worker.

    python lineage_validation/larry_20261112/02_prepare.py --root results/larry_20261112 \
        --tacc-root /scratch/10119/ghzheng/OT_project/larry_20261112

At most 10,000 cells per side: every clonally informative cell first (day-2 cells whose clone has cells on the target
day; target cells whose clone has cells on day 2), then random other cells of that side (seed 20261112). Clone labels
stay in obs for scoring and are never part of the OT input.
Writes ROOT/upload/inputs/<unit>__{source,target}.h5ad (recovered integer counts; obs: sample_id, malignant =
"malignant" so every cell is in scope, time, population, cell_type, well, clone) and ROOT/upload/manifest.csv
(02_run_pair.py columns, paths under --tacc-root), and ROOT/prep/units.tsv (cells per side, informative share).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED, CAP = 20261112, 10_000


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tacc-root", required=True)
    a = p.parse_args()
    import anndata as ad
    A = ad.read_h5ad(a.root / "prep" / "counts_recovered.h5ad")
    o = A.obs
    tcol, pcol, ccol = "Time point", "Starting population", "Cell type annotation"
    o["time"] = o[tcol].astype(float).astype(int); o["population"] = np.where(o[pcol].astype(str).str.contains("Sca1+", regex=False), "LSK", "LK")
    o["clone"] = o["clone"].astype(int)
    up = a.root / "upload" / "inputs"; up.mkdir(parents=True, exist_ok=True)
    rows, man = [], []
    for pop in ("LSK", "LK"):
        for day in (4, 6):
            u = f"LARRY_{pop}_d2_to_d{day}"
            src = np.flatnonzero(((o.population == pop) & (o.time == 2)).to_numpy())
            tgt = np.flatnonzero(((o.population == pop) & (o.time == day)).to_numpy())
            cs = set(o.clone.to_numpy()[src]) - {-1}; ct = set(o.clone.to_numpy()[tgt]) - {-1}
            shared = cs & ct
            rng = np.random.default_rng(SEED)
            picked = {}
            for side, idx in (("source", src), ("target", tgt)):
                inf = idx[np.isin(o.clone.to_numpy()[idx], list(shared))]
                rest = np.setdiff1d(idx, inf)
                if len(inf) >= CAP:
                    keep = np.sort(rng.choice(inf, CAP, replace=False))
                else:
                    keep = np.sort(np.r_[inf, rng.choice(rest, min(CAP - len(inf), len(rest)), replace=False)])
                picked[side] = (keep, len(inf))
                B = A[keep].copy()
                B.obs = pd.DataFrame({"sample_id": f"{u}__{side}", "malignant": "malignant", "time": o.time.to_numpy()[keep],
                                      "population": pop, "cell_type": o[ccol].astype(str).to_numpy()[keep], "well": o["Well"].astype(str).to_numpy()[keep],
                                      "clone": o.clone.to_numpy()[keep], "library": o["Library"].astype(str).to_numpy()[keep]},
                                     index=[f"{u}_{side}_{i}" for i in keep])
                B.obs.index = B.obs.index.astype(str); B.var_names = B.var_names.astype(str)
                B.write_h5ad(up / f"{u}__{side}.h5ad", compression="gzip")
            rows.append({"unit_id": u, "population": pop, "target_day": day, "source_all": len(src), "target_all": len(tgt), "shared_clones": len(shared),
                         "source_cells": len(picked["source"][0]), "source_informative": picked["source"][1],
                         "target_cells": len(picked["target"][0]), "target_informative": picked["target"][1]})
            man.append({"pair_id": u, "dataset_id": "LARRY", "patient_id": pop, "source_sample": f"{u}__source", "target_sample": f"{u}__target",
                        "source_h5ad": f"{a.tacc_root}/inputs/{u}__source.h5ad", "target_h5ad": f"{a.tacc_root}/inputs/{u}__target.h5ad", "eligible": True})
            print(rows[-1], flush=True)
    pd.DataFrame(man).to_csv(a.root / "upload" / "manifest.csv", index=False)
    pd.DataFrame(rows).to_csv(a.root / "prep" / "units.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

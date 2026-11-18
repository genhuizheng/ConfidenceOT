"""Application-search pilots: subsample prepared source / target h5ad pairs to at most CAP cells per side (cells with an
informative label column first, then random others, seed 20261117) and write a manifest for the v3 worker ('larry'
manifest route: any pair of count h5ad files). Labels stay in obs for scoring only.

    python lineage_validation/app_search_20261117/01_subsample_units.py --manifest IN_MANIFEST.csv --out OUT_ROOT \
        --cap 3000 [--informative-col clone --informative-min 0] [--pairs PAIR ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", type=Path, required=True, help="columns pair_id, source_h5ad, target_h5ad (+ any others kept)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cap", type=int, default=3000)
    p.add_argument("--informative-col", default=None, help="obs column; for LARRY 'clone' (shared clones are kept first)")
    p.add_argument("--pairs", nargs="*")
    p.add_argument("--local-root", type=Path, default=None, help="replace the manifest's directory of the h5ad paths with this one")
    a = p.parse_args()
    import anndata as ad
    m = pd.read_csv(a.manifest)
    if a.pairs:
        m = m[m.pair_id.isin(a.pairs)]
    (a.out / "inputs").mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in m.iterrows():
        paths = {s: Path(r[f"{s}_h5ad"]) for s in ("source", "target")}
        if a.local_root is not None:
            paths = {s: a.local_root / v.name for s, v in paths.items()}
        A = {s: ad.read_h5ad(v) for s, v in paths.items()}
        shared = None
        if a.informative_col:
            cs = set(A["source"].obs[a.informative_col]) - {-1}; ct = set(A["target"].obs[a.informative_col]) - {-1}
            shared = cs & ct
        rng = np.random.default_rng(20261117)
        rec = r.to_dict()
        for s in ("source", "target"):
            idx = np.arange(A[s].n_obs)
            inf = idx[A[s].obs[a.informative_col].isin(shared).to_numpy()] if shared is not None else np.array([], int)
            rest = np.setdiff1d(idx, inf)
            keep = np.sort(rng.choice(inf, a.cap, replace=False)) if len(inf) >= a.cap else \
                np.sort(np.r_[inf, rng.choice(rest, min(a.cap - len(inf), len(rest)), replace=False)])
            B = A[s][keep].copy()
            f = a.out / "inputs" / f"{r.pair_id}__{s}.h5ad"; B.write_h5ad(f, compression="gzip")
            rec[f"{s}_h5ad"] = str(f.resolve()); rec[f"{s}_cells"] = len(keep); rec[f"{s}_informative"] = len(inf) if len(inf) < a.cap else a.cap
        rows.append(rec); print(rec, flush=True)
    pd.DataFrame(rows).to_csv(a.out / "manifest.csv", index=False)


if __name__ == "__main__":
    main()

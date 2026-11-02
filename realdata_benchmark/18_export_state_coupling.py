"""Aggregate every LTL331 coupling to the authors' 18 clusters, so state-to-state transport can be read locally without
the couplings (runs on TACC for the benchmark worklists; locally for the frozen run).

    python realdata_benchmark/18_export_state_coupling.py --converted CONVERTED --out OUT.tsv.gz \
        (--root ROOT [--worklist worklist_ltl331.tsv] | --frozen-run-root RUN_ROOT --labels LABEL ...)

For each fit (benchmark rows: ROOT/<row out>/fits/pair*/*/out/*/scope_malignant/budget_*/; frozen run:
RUN_ROOT/runs/acc0.99/ltl331/<label>/<pair>/scope_malignant/budget_*/) and each of M4-E / M4-R: the coupling
(coupling_m4e.npz / coupling_m4r.npz, rows = the source cells and columns = the target cells in cell_confidence.csv
order) summed over source cluster x target cluster (seurat_clusters of the converted obs, the authors' clustering),
over all cells (block all) and over retained x retained cells (block retained). Long table: config_id, pair_index,
pair_id, method, block, source_cluster, target_cluster, mass; plus, per side and cluster, cells and retained cells
(source_cluster / target_cluster = "cells" rows: mass column holds the count, block = side_cells / side_retained).
Normalisation (enrichment against the marginals) is done downstream; nothing is filtered. Refuses if OUT exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def aggregate(d: Path, cluster: pd.Series, key: dict) -> list[dict]:
    cc = pd.read_csv(d / "cell_confidence.csv", usecols=["method", "side", "observation_id", "rejected"])
    rows = []
    for method in ("M4-E", "M4-R"):
        f = d / f"coupling_{method.lower().replace('-', '')}.npz"
        if not f.exists():
            continue
        P = np.load(f)["coupling"]
        g = cc[cc.method == method]; s, t = g[g.side == "source"], g[g.side == "target"]
        if P.shape != (len(s), len(t)):
            raise ValueError(f"{d}: coupling {P.shape} vs cells {(len(s), len(t))}")
        sc, tc = cluster.reindex(s.observation_id).to_numpy(), cluster.reindex(t.observation_id).to_numpy()
        rs, rt = ~s.rejected.to_numpy(bool), ~t.rejected.to_numpy(bool)
        for block, Q, a, b in (("all", P, sc, tc), ("retained", P[np.ix_(rs, rt)], sc[rs], tc[rt])):
            us, ut = pd.unique(a), pd.unique(b)
            A = (a[:, None] == us[None, :]).astype(float); B = (b[:, None] == ut[None, :]).astype(float)
            M = A.T @ Q @ B
            for i, x in enumerate(us):
                for j, y in enumerate(ut):
                    rows.append({**key, "method": method, "block": block, "source_cluster": x, "target_cluster": y, "mass": float(M[i, j])})
        for side, cl, keep in (("source", sc, rs), ("target", tc, rt)):
            for x in pd.unique(cl):
                rows.append({**key, "method": method, "block": f"{side}_cells", "source_cluster": x if side == "source" else "cells",
                             "target_cluster": x if side == "target" else "cells", "mass": float((cl == x).sum())})
                rows.append({**key, "method": method, "block": f"{side}_retained", "source_cluster": x if side == "source" else "cells",
                             "target_cluster": x if side == "target" else "cells", "mass": float(((cl == x) & keep).sum())})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--worklist", default="worklist_ltl331.tsv")
    p.add_argument("--frozen-run-root", type=Path)
    p.add_argument("--labels", nargs="*", default=[])
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    cluster = pd.concat([ad.read_h5ad(f, backed="r").obs["seurat_clusters"].astype(str) for f in sorted((a.converted / "h5ad").glob("*.h5ad"))])
    jobs = []
    if a.root is not None:
        w = pd.read_csv(a.root / a.worklist, sep="\t", dtype=str)
        for _, r in w.iterrows():
            jobs.append(({"config_id": r.config_id, "pair_index": int(r.pair_index), "pair_id": r.pair_id},
                         sorted(Path(r.out).glob("fits/pair*/*/out/*/scope_malignant/budget_*"))))
    if a.frozen_run_root is not None:
        pairs = pd.read_csv(a.frozen_run_root / "inputs" / "ds" / "pair_manifest_downsampled.csv")
        pairs = pairs[pairs.eligible.astype(bool)].reset_index(drop=True) if "eligible" in pairs else pairs
        for label in a.labels:
            for i, r in pairs.iterrows():
                jobs.append(({"config_id": f"label:{label}", "pair_index": i, "pair_id": r.pair_id},
                             sorted((a.frozen_run_root / "runs" / "acc0.99" / "ltl331" / label / r.pair_id / "scope_malignant").glob("budget_*"))))
    rows, status = [], []
    for key, dirs in jobs:
        st = {**key, "fit_dirs": len(dirs)}
        if len(dirs) == 1:
            try:
                rows += aggregate(dirs[0], cluster, key); st["ok"] = True
            except Exception as e:
                st["ok"] = False; st["error"] = repr(e)
        status.append(st)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, sep="\t", index=False, compression="gzip")
    s = pd.DataFrame(status); s.to_csv(str(a.out).replace(".tsv.gz", "") + "_STATUS.tsv", sep="\t", index=False)
    print(s.groupby(["fit_dirs"] + (["ok"] if "ok" in s else [])).size().to_string())
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()

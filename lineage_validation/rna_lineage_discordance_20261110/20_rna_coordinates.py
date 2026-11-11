"""View A of PLAN_v2.md on TACC: the exact joint RNA coordinates the v3 RNA-OT fits used, for two configurations.

    python lineage_validation/rna_lineage_discordance_20261110/20_rna_coordinates.py \
        --worklist /scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/WORKLIST.tsv --units rld_units.txt --out OUT

For every unit in --units and each encoding key (ranknm256-midrank|none, logcpm|none), the worklist row is loaded
with the v3 worker's own Inputs (realdata_benchmark/v3_grid/worker.py: the same cells, counts and seed as the fit) and
v3_common.representations yields the corrected coordinates of the key's 8 configurations. The two configurations of
PLAN_v2.md are kept:
  ranknm256-midrank|none|scale|cos|pca50     (Splatter L5 best)
  logcpm|none|noscale|sqeuc|pca30            (logCPM + PCA)
No OT is fitted. Writes OUT/<unit>/<config>.npz (X source, Y target float32, cos flag, source_ids, target_ids) and
OUT/<unit>/obs.tsv.gz (side, cell id, total raw counts, detected genes, every obs column the inputs carry).
Refuses a unit whose output exists.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WANTED = {"ranknm256-midrank|none": "ranknm256-midrank|none|scale|cos|pca50", "logcpm|none": "logcpm|none|noscale|sqeuc|pca30"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--units", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid"))
    spec = importlib.util.spec_from_file_location("v3worker", REPO / "realdata_benchmark" / "v3_grid" / "worker.py")
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    v3 = w.v3
    wl = pd.read_csv(a.worklist, sep="\t", dtype=str)
    units = [x.strip() for x in a.units.read_text().splitlines() if x.strip()]
    inputs = w.Inputs()
    for u in units:
        o = a.out / u
        if (o / "obs.tsv.gz").exists():
            print(f"{u}: exists, kept"); continue
        o.mkdir(parents=True, exist_ok=True)
        for key, cid in WANTED.items():
            row = wl[(wl.unit_id == u) & (wl.key == key)]
            if len(row) != 1:
                raise SystemExit(f"{u} {key}: {len(row)} worklist rows")
            row = row.iloc[0]
            v = inputs.get(row)
            S, T = v["S"], v["T"]
            for c, X, Y, cos, diag in v3.representations(json.loads(row.spec), S, T, v["seed"]):
                if c == cid:
                    np.savez_compressed(o / f"{cid.replace('|', '__')}.npz", X=np.asarray(X, np.float32), Y=np.asarray(Y, np.float32),
                                        cos=bool(cos), source_ids=np.asarray(S.obs_names, dtype=str), target_ids=np.asarray(T.obs_names, dtype=str),
                                        diag=json.dumps(diag, default=float))
            print(f"{u} {cid}: {S.n_obs} x {T.n_obs}", flush=True)
        frames = []
        for side, A in (("source", S), ("target", T)):
            X = A.X
            obs = A.obs.copy()
            obs.insert(0, "total_counts_raw", np.asarray(X.sum(1)).ravel())
            obs.insert(1, "detected_genes", np.asarray((X > 0).sum(1)).ravel())
            obs.insert(0, "cell", A.obs_names.astype(str)); obs.insert(0, "side", side)
            frames.append(obs.reset_index(drop=True))
        pd.concat(frames).to_csv(o / "obs.tsv.gz", sep="\t", index=False, compression="gzip")


if __name__ == "__main__":
    main()

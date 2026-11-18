"""Check that otdb_common's direct ConfidenceOT call equals the production path (v3_common.fit -> 02_run_pair.py) on the
C1 coordinates: same gates and the same M4-E coupling. Splatter local pairs given on the command line.

    python ot_depth_benchmark_20261117/check_against_02.py --pairs L1_depth:1:0 L2_depth_dropout:2:1 L0_matched:1:3
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import otdb_common as oc  # noqa: E402

ROOT = oc.REPO / "results" / "splatter_local_20261103"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", nargs="+", default=["L1_depth:1:0", "L2_depth_dropout:2:1", "L0_matched:1:3"])
    a = p.parse_args()
    import anndata as ad
    sys.path.insert(0, str(oc.REPO / "realdata_benchmark" / "v3_grid"))
    import v3_common as v3
    import run_local_benchmark as lb
    args = [x for x in lb.FIT_ARGS if x != "--allow-precomputed"] + ["--calibration-null", "within_side_split"]
    rows = []
    for spec in a.pairs:
        level, rep, idx = spec.split(":"); rep, idx = int(rep), int(idx)
        cond = ROOT / "conditions" / f"rep{rep}" / level
        man = pd.read_csv(cond / "manifest.csv"); r = man.iloc[idx]
        S, T = ad.read_h5ad(oc.REPO / r.source_h5ad), ad.read_h5ad(oc.REPO / r.target_h5ad)
        man = man.copy()
        for side in ("source", "target"):
            man[f"{side}_h5ad"] = [str(oc.REPO / x) for x in man[f"{side}_h5ad"]]
        X, Y, _ = oc.c1_coordinates(S, T, 20260925 + idx)
        work = oc.REPO / "results" / "ot_depth_benchmark_20261117" / "check_work"
        code, log, run = v3.fit(X, Y, True, {"source": S.obs.copy(), "target": T.obs.copy()}, man, idx, args, work)
        if code:
            print(log[-2000:]); raise SystemExit(code)
        cells = pd.read_csv(run / "cell_confidence.csv"); cells = cells[cells.method == "M4-E"]
        P02 = np.load(run / "coupling_m4e.npz")["coupling"]
        C, Xc, Yc, scale, rng = oc.cost_matrix(X, Y, True, idx)
        P, sg, tg, rc = oc.confidenceot_fit(Xc, Yc, C, scale, rng, idx)
        g02s = ~cells[cells.side == "source"].rejected.to_numpy(bool); g02t = ~cells[cells.side == "target"].rejected.to_numpy(bool)
        rows.append({"pair": spec, "gate_equal_source": bool((g02s == sg).all()), "gate_equal_target": bool((g02t == tg).all()),
                     "coupling_max_abs_diff": float(np.abs(P02 - P).max()), "coupling_max": float(P.max()),
                     "rejected_source": int((~sg).sum()), "rejected_target": int((~tg).sum())})
        shutil.rmtree(work, ignore_errors=True)
        print(rows[-1], flush=True)
    out = oc.REPO / "results" / "ot_depth_benchmark_20261117" / "CHECK_AGAINST_02.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False); print("->", out)


if __name__ == "__main__":
    main()

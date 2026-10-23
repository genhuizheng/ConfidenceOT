"""Level 3 matching quantities from the saved Level 2 couplings (PLAN.md), one coupling in memory at a time.

    python lineage_validation/level3_cell_lineage/score_level3.py --runs LEVEL2_ROOT --out OUT [--cases 0,4]

LEVEL2_ROOT holds full/, full_cnv/, full_rna_cnv/ with <case_id>/<method>/coupling.npz (the TACC layout or a
download of it). For every case, family and metastatic cell that is evaluable, ambiguous or not informative
(PLAN.md status, method-independent), and for every strategy x method: coverage c_ot / c_ind, D_ot, D_ind,
delta_D and delta_D_norm, close-relative masses and log2 enrichment, top-k (k = 1, 5, 10) shares and distances.
The independent baseline uses the coupling's own row masses (uniform for BOT, realised for UOT).
Writes OUT/cells/<case_id>.tsv.gz (one row per cell x strategy x method) and OUT/run_status.tsv (every case x
strategy x method: ok, or missing with the absent file). A missing coupling is reported, never substituted.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_core as c  # noqa: E402
from ground_truth import load_cells  # noqa: E402

SCORED = ("evaluable", "ambiguous", "not_informative")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", default=None, help="comma-separated case indices (default: all)")
    parser.add_argument("--case-table", type=Path, default=c.L2_INPUTS / "cases.tsv")
    args = parser.parse_args()
    cases = pd.read_csv(args.case_table, sep="\t", dtype=str)
    if args.cases:
        cases = cases[cases["index"].isin(args.cases.split(","))]
    trees = c.tree_files()
    (args.out / "cells").mkdir(parents=True, exist_ok=True)
    status = []
    for _, case in cases.iterrows():
        started = time.perf_counter()
        cells = load_cells(args.runs, case["case_id"])
        fams, allt, n_src = c.ground_truth(case, cells, trees)
        # distances once per family, reused by all six couplings (int16: tree distances are small integers)
        dist = {}
        for fam, (ft, s_rows, ftab) in fams.items():
            sc = ftab[ftab["status"].isin(SCORED)]
            D = np.empty((len(sc), len(s_rows)), dtype=np.int16)
            for k, node in enumerate(sc["target_node"].astype(int)):
                D[k] = ft.distances(node)
            dist[fam] = (sc.reset_index(drop=True), s_rows, D)
        rows = []
        for strategy, folder in c.STRATEGY_FOLDERS.items():
            for method in c.METHODS:
                path = args.runs / folder / case["case_id"] / method / "coupling.npz"
                if not path.exists():
                    status.append({"index": case["index"], "case_id": case["case_id"], "strategy": strategy, "method": method,
                                   "status": "missing", "file": str(path)})
                    continue
                with np.load(path) as z:
                    P = z["coupling"]  # float32 as saved; sums in float64
                if P.shape != (n_src, len(cells) - n_src):
                    raise ValueError(f"{path}: shape {P.shape}, cells.tsv.gz has {n_src} x {len(cells) - n_src}")
                m = P.sum(1, dtype=np.float64)
                col = P.sum(0, dtype=np.float64)
                Pt = np.ascontiguousarray(P.T)  # one metastatic cell's column contiguous
                del P
                for fam, (sc, s_rows, D) in dist.items():
                    for k, r in sc.iterrows():
                        j = int(r["target_col"])
                        q = Pt[j].astype(np.float64) / col[j]
                        rec = {"strategy": strategy, "method": method, "family": fam, "cell": r["cell"], "site": r["site"],
                               "status": r["status"], "n_sources_comparable": len(s_rows), "n_nearest": int(r["n_nearest"]),
                               "missing_share": r["missing_share"], "L1": r["L1"], "L2": r["L2"]}
                        rec.update(c.cell_metrics(q, m, s_rows, D[k].astype(np.float64)))
                        rows.append(rec)
                del Pt
                status.append({"index": case["index"], "case_id": case["case_id"], "strategy": strategy, "method": method,
                               "status": "ok", "file": str(path)})
        table = pd.DataFrame(rows)
        table.insert(0, "case_id", case["case_id"])
        table.to_csv(args.out / "cells" / f"{case['case_id']}.tsv.gz", sep="\t", index=False, compression="gzip")
        allt.to_csv(args.out / "cells" / f"{case['case_id']}__targets.tsv.gz", sep="\t", index=False, compression="gzip")
        print(f"{case['case_id']}: {len(table)} cell x run rows, {time.perf_counter() - started:.0f} s", flush=True)
    pd.DataFrame(status).to_csv(args.out / "run_status.tsv", sep="\t", index=False)
    print(pd.DataFrame(status).groupby("status").size().to_string())


if __name__ == "__main__":
    main()

"""Post-hoc within-subclone decomposition (PLAN.md, dated section): is there lineage signal beyond the subclone?

    python lineage_validation/level3_cell_lineage/within_subclone.py --runs LEVEL2_ROOT --out OUT

For every evaluable metastatic cell j with a frozen level-L subclone label g (L = 1, 2), the comparable primaries
are restricted to the same subclone (S_j^g) and l3_core.cell_metrics is computed on them:
  c_ot / c_ind   share of j's matched mass on its own subclone's primaries vs the independent coupling
                 (the subclone-level signal)
  delta_D_norm   OT vs independent within the subclone (the cell-level signal beyond the subclone)
Cells with |S_j^g| < 5 or equal distances to all of S_j^g are counted, not scored. One coupling in memory at a time.
Writes OUT/cells/<case_id>.tsv.gz, OUT/family_summary.tsv (strategy x method x family x level) and
OUT/overall.tsv (strategy x method x level: families and mice with delta_D_norm > 0, medians).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_core as c  # noqa: E402
from ground_truth import load_cells  # noqa: E402
from summarize_level3 import summarise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = pd.read_csv(c.L2_INPUTS / "cases.tsv", sep="\t", dtype=str)
    subs = pd.read_csv(c.L2_INPUTS / "subclones.tsv", sep="\t", dtype=str)
    trees = c.tree_files()
    (args.out / "cells").mkdir(parents=True, exist_ok=True)
    allrows = []
    for _, case in cases.iterrows():
        cells = load_cells(args.runs, case["case_id"])
        src_ids = cells[cells["side"].eq("source")]["cell"].astype(str).to_numpy()
        fams, allt, n_src = c.ground_truth(case, cells, trees)
        work = []  # (family, level, target row, rows of S_j^g, distances)
        skipped = 0
        for fam, (ft, s_rows, ftab) in fams.items():
            table = c.family_cells(fam)
            for level in ("1", "2"):
                allowed = set(subs.loc[subs["name"].eq(fam) & subs["level"].eq(level), "subclone"])
                if not allowed:
                    continue
                s_lab = np.array([table.at[x, f"L{level}"] for x in src_ids[s_rows]], dtype=object)
                for _, t in ftab[ftab["status"].eq("evaluable")].iterrows():
                    g = t[f"L{level}"]
                    if g not in allowed:
                        continue
                    keep = s_lab == g
                    if keep.sum() < c.MIN_SOURCES:
                        skipped += 1
                        continue
                    d = ft.distances(int(t["target_node"]))[keep]
                    if d.min() == d.max():
                        skipped += 1
                        continue
                    work.append((fam, level, t, s_rows[keep], d.astype(np.float64)))
        rows = []
        for strategy, folder in c.STRATEGY_FOLDERS.items():
            for method in c.METHODS:
                with np.load(args.runs / folder / case["case_id"] / method / "coupling.npz") as z:
                    P = z["coupling"]
                m = P.sum(1, dtype=np.float64)
                col = P.sum(0, dtype=np.float64)
                Pt = np.ascontiguousarray(P.T)
                del P
                for fam, level, t, rows_g, d in work:
                    j = int(t["target_col"])
                    q = Pt[j].astype(np.float64) / col[j]
                    rec = {"case_id": case["case_id"], "strategy": strategy, "method": method, "family": fam, "level": int(level),
                           "subclone": t[f"L{level}"], "cell": t["cell"], "site": t["site"], "status": "evaluable", "n_sources_subclone": len(rows_g)}
                    rec.update(c.cell_metrics(q, m, rows_g, d))
                    rows.append(rec)
                del Pt
        table = pd.DataFrame(rows)
        table.to_csv(args.out / "cells" / f"{case['case_id']}.tsv.gz", sep="\t", index=False, compression="gzip")
        print(f"{case['case_id']}: {len(work)} cell x level units scored, {skipped} skipped (< 5 subclone primaries or equal distances)", flush=True)
        allrows.append(table)
    cells_all = pd.concat(allrows, ignore_index=True)
    cells_all["dataset"] = cells_all["family"].str.split("_").str[0]
    cells_all["mouse"] = cells_all["family"].str.split("_").str[1]
    fam = summarise(cells_all, ["strategy", "method", "level", "dataset", "mouse", "family"])
    fam.to_csv(args.out / "family_summary.tsv", sep="\t", index=False)
    mouse = fam.groupby(["strategy", "method", "level", "dataset", "mouse"], sort=False)[["delta_D_norm", "delta_D", "c_ot", "c_ind"]].mean().reset_index()
    over = []
    for (s, m_, lev), g in fam.groupby(["strategy", "method", "level"], sort=False):
        gm = mouse[mouse["strategy"].eq(s) & mouse["method"].eq(m_) & mouse["level"].eq(lev)]
        over.append({"strategy": s, "method": m_, "level": lev, "families": len(g), "families_dDn_pos": int((g["delta_D_norm"] > 0).sum()),
                     "families_ci_above_0": int((g["delta_D_norm_lo"] > 0).sum()), "families_ci_below_0": int((g["delta_D_norm_hi"] < 0).sum()),
                     "median_family_dDn": float(g["delta_D_norm"].median()), "mice": len(gm), "mice_dDn_pos": int((gm["delta_D_norm"] > 0).sum()),
                     "median_mouse_dDn": float(gm["delta_D_norm"].median()),
                     "median_family_c_ot": float(g["c_ot"].median()), "median_family_c_ind": float(g["c_ind"].median())})
    over = pd.DataFrame(over)
    over.to_csv(args.out / "overall.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(over.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

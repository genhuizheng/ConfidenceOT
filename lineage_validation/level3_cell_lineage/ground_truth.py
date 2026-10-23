"""Level 3 lineage ground truth (PLAN.md): every metastatic cell's tree mapping and status, method-independent.

    python lineage_validation/level3_cell_lineage/ground_truth.py --runs LEVEL2_ROOT --out OUT

LEVEL2_ROOT is the Level 2 output folder holding full/, full_cnv/, full_rna_cnv/ (only cells.tsv.gz is read;
the three strategies must list the same cells in the same order). Reads the authors' trees and the frozen leaf
mapping (../within_family_subclones/output). No coupling is read.
Writes OUT/:
  targets.tsv.gz    every metastatic cell of every case: family, leaf, reliable, missing share, status
                    (evaluable / ambiguous / not_informative / not_evaluable) and reason, comparable primaries,
                    d_min, d_max, tree-nearest count and share
  coverage.tsv      per case and family: metastatic cells by status, sources and comparable sources, tree
                    resolution, flags (high missingness, low tree resolution)
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


def load_cells(runs: Path, case_id: str) -> pd.DataFrame:
    tables = {s: pd.read_csv(runs / f / case_id / "cells.tsv.gz", sep="\t", dtype=str) for s, f in c.STRATEGY_FOLDERS.items()
              if (runs / f / case_id / "cells.tsv.gz").exists()}
    if not tables:
        raise FileNotFoundError(f"{case_id}: no cells.tsv.gz under {runs}")
    first = next(iter(tables.values()))
    for s, t in tables.items():
        if not t[["side", "cell"]].equals(first[["side", "cell"]]):
            raise ValueError(f"{case_id}: strategy {s} lists other cells or another order")
    return first


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=c.L2_INPUTS / "cases.tsv")
    args = parser.parse_args()
    cases = pd.read_csv(args.cases, sep="\t", dtype=str)
    trees = c.tree_files()
    targets, cover = [], []
    for _, case in cases.iterrows():
        cells = load_cells(args.runs, case["case_id"])
        fams, allt, n_src = c.ground_truth(case, cells, trees)
        targets.append(allt)
        for fam, (ft, s_rows, ftab) in fams.items():
            table = c.family_cells(fam)
            st = ftab["status"].value_counts()
            ev = ftab[ftab["status"].eq("evaluable")]
            row = {"case_id": case["case_id"], "family": fam, "tree": ft.tree.path.name, "sources": n_src,
                   "sources_comparable": len(s_rows), "metastatic_cells_in_family": len(ftab),
                   **{f"n_{k}": int(st.get(k, 0)) for k in ("evaluable", "ambiguous", "not_informative", "not_evaluable")},
                   "tree_leaves_unreliable_share": float((~table["reliable"]).mean()),
                   "evaluable_mean_missing_share": float(ev["missing_share"].mean()) if len(ev) else np.nan,
                   **{k: v for k, v in ft.tree.resolution.items() if k != "tree_has_branch_lengths"}}
            row["flag_high_missingness"] = bool(row["tree_leaves_unreliable_share"] > 0.2 or (row["evaluable_mean_missing_share"] > 0.3))
            row["flag_low_tree_resolution"] = bool((row["n_ambiguous"] + row["n_not_informative"]) > 0.5 * len(ftab))
            cover.append(row)
        n_none = int((allt["family"] == "").sum())
        print(f"{case['case_id']}: {len(allt)} metastatic cells, {n_none} in no family tree; "
              + ", ".join(f"{k} {v}" for k, v in allt["status"].value_counts().items()), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.concat(targets, ignore_index=True).to_csv(args.out / "targets.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(cover).to_csv(args.out / "coverage.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

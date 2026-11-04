"""Reproduction check (PLAN.md section 3): each finished reproduction row (label ranknm256_noscale_ds_cos, PCA 30, the
earlier KP run's manifest and index) against that earlier run's output, cell by cell. Read only; runs on TACC.

    python lineage_validation/preprocessing_factorial_20261103/03_check_reproduction.py --root ROOT \
        [--earlier /scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/ot/kptracer/ranknm256_noscale_ds_cos]

Per row and method (M4-E, M4-R): cells in both / only one; cells whose retained call differs; maximum absolute
difference of signed_rejection_margin; the two calibrated rejection costs. Writes ROOT/REPRODUCTION_CHECK.tsv
(overwritten on each call: it is a check, recomputed from the row outputs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def one(path: Path, pattern: str) -> Path | None:
    hits = sorted(path.rglob(pattern))
    return hits[0] if hits else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--earlier", type=Path,
                   default=Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/ot/kptracer/ranknm256_noscale_ds_cos"))
    a = p.parse_args()
    w = pd.read_csv(a.root / "WORKLIST.tsv", sep="\t")
    units = pd.read_csv(a.root / "UNITS.tsv", sep="\t").set_index("unit_id")
    rows = []
    for r in w[w.config_id.str.startswith("label:")].itertuples():
        out = Path(r.out)
        if not (out / "SUCCESS").exists():
            continue
        pair = units.loc[r.unit_id, "pair_or_case"]
        new_c, old_c = one(out / "runs", "cell_confidence.csv"), one(a.earlier / str(pair), "cell_confidence.csv")
        new_m, old_m = one(out / "runs", "pair_metrics.csv"), one(a.earlier / str(pair), "pair_metrics.csv")
        if new_c is None or old_c is None:
            rows.append({"unit_id": r.unit_id, "pair_id": pair, "status": f"missing {'new' if new_c is None else 'earlier'} cell table"})
            continue
        n, o = pd.read_csv(new_c), pd.read_csv(old_c)
        pm_n = pd.read_csv(new_m).set_index("method") if new_m else None
        pm_o = pd.read_csv(old_m).set_index("method") if old_m else None
        for method in ("M4-E", "M4-R"):
            k = ["side", "observation_id"]
            m = n[n.method == method].merge(o[o.method == method], on=k, how="outer", suffixes=("_new", "_old"), indicator=True)
            both = m[m._merge == "both"]
            rows.append({"unit_id": r.unit_id, "pair_id": pair, "method": method, "status": "compared",
                         "cells_both": len(both), "cells_only_new": int((m._merge == "left_only").sum()),
                         "cells_only_earlier": int((m._merge == "right_only").sum()),
                         "retained_differs": int((both.retained_new.astype(bool) != both.retained_old.astype(bool)).sum()),
                         "max_abs_margin_diff": float(np.nanmax(np.abs(both.signed_rejection_margin_new - both.signed_rejection_margin_old))) if len(both) else np.nan,
                         "rejection_cost_new": float(pm_n.loc[method, "rejection_cost"]) if pm_n is not None and method in pm_n.index else np.nan,
                         "rejection_cost_earlier": float(pm_o.loc[method, "rejection_cost"]) if pm_o is not None and method in pm_o.index else np.nan})
    t = pd.DataFrame(rows)
    t.to_csv(a.root / "REPRODUCTION_CHECK.tsv", sep="\t", index=False)
    print(t.to_string(index=False))
    if len(t) and "retained_differs" in t:
        c = t[t.status == "compared"]
        print(f"\n{len(c)} comparisons: retained calls identical in {int((c.retained_differs == 0).sum())}; "
              f"max |margin diff| {c.max_abs_margin_diff.max():.3g}; cells missing on either side {int((c.cells_only_new + c.cells_only_earlier).sum())}")


if __name__ == "__main__":
    main()

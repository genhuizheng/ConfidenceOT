"""Collect the N = 5,000 benchmark rows into its deliverable tables (runs on TACC; reads small tables only).

    python realdata_benchmark/splatter_n5000/32_collect_n5000.py --root N5000_ROOT

SPLATTER_N5000_GATE_METRICS.tsv        40_score.py rows (M4-E, M4-R) of every row's fit (the frozen fit for reused
                                       L0-L2 rows) and the comparators' solver_comparison.csv (IC-POT, Vanilla UOT),
                                       with level / replicate / case / configuration
SPLATTER_N5000_COUPLING_METRICS.tsv    23_row_extras.py coupling.tsv (M4-E, M4-R, balanced full, balanced on the block)
SPLATTER_N5000_CALIBRATION_VALIDITY.tsv every fit's pair_metrics.csv: convergence, calibration flags, rejection cost
SPLATTER_N5000_DEG_VALIDATION.tsv      DEG-B (23_) per row and side, and DEG-A / C / D (24_) per condition
SPLATTER_N5000_REPRESENTATION_DIAGNOSTICS / DOWNSAMPLING_DIAGNOSTICS / RANK_STABILITY.tsv  26_ per condition pair
COLLECT_STATUS_N5000.tsv               every row's flag and which tables were found
The robustness views (batch: L2 -> L3, perturbation: L3 -> L4 and L2 -> A4, imbalance: L4 -> L5 and L2 -> A5,
depth / dropout: L0 -> L1 -> L2) are cuts of these tables by level, made in the report, not separate runs.
Nothing is filtered; rows without output are listed. Refuses if a table exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KEYS = ["level", "replicate", "case", "pair_id", "config_id"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    names = ["GATE_METRICS", "COUPLING_METRICS", "CALIBRATION_VALIDITY", "DEG_VALIDATION", "REPRESENTATION_DIAGNOSTICS",
             "DOWNSAMPLING_DIAGNOSTICS", "RANK_STABILITY"]
    outs = {n: a.root / f"SPLATTER_N5000_{n}.tsv" for n in names}
    for f in outs.values():
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    w = pd.read_csv(a.root / "TACC_N5000_WORKLIST.tsv", sep="\t", dtype=str)
    tabs = {n: [] for n in names}; status = []
    for _, r in w.iterrows():
        out = Path(r.out); meta = {k: r[k] for k in KEYS}
        st = {**meta, "dataset": r.dataset, "flag": "SUCCESS" if (out / "SUCCESS").exists() else "FAILED" if (out / "FAILED").exists() else "missing"}
        if r.dataset == "splatter_n5000_condition":
            f = out / "deg_condition.tsv"; st["deg_condition"] = f.exists()
            if f.exists():
                tabs["DEG_VALIDATION"].append(pd.read_csv(f, sep="\t").assign(**meta, table="DEG-A/C/D"))
            for fname, n in (("representation.tsv", "REPRESENTATION_DIAGNOSTICS"), ("downsampling.tsv", "DOWNSAMPLING_DIAGNOSTICS"),
                             ("rank_stability.tsv", "RANK_STABILITY")):
                f = out / "diagnostics" / fname; st[fname] = f.exists()
                if f.exists():
                    cond_meta = {k: v for k, v in meta.items() if k != "config_id"}
                    tabs[n].append(pd.read_csv(f, sep="\t").assign(**cond_meta))
            status.append(st); continue
        fit = Path(r.fit_out) if isinstance(r.fit_out, str) and r.fit_out else out
        st["fit_reused"] = fit != out
        sc = fit / "score.csv"; st["score"] = sc.exists()
        if sc.exists():
            tabs["GATE_METRICS"].append(pd.read_csv(sc).assign(**meta, solver_family="ConfidenceOT"))
        for name in ("icpot", "uot"):
            f = out / "comparators" / name / "solver_comparison.csv"; st[name] = f.exists()
            if f.exists():
                tabs["GATE_METRICS"].append(pd.read_csv(f).assign(**meta, solver_family=name))
        pm = list((fit / "runs").glob("*/scope_all/budget_*/pair_metrics.csv")); st["pair_metrics"] = bool(pm)
        if pm:
            tabs["CALIBRATION_VALIDITY"].append(pd.read_csv(pm[0]).assign(**meta))
        for f, n in ((out / "extras" / "coupling.tsv", "COUPLING_METRICS"), (out / "extras" / "deg_b.tsv", "DEG_VALIDATION")):
            st[f.stem] = f.exists()
            if f.exists():
                tabs[n].append(pd.read_csv(f, sep="\t").assign(**meta, **({"table": "DEG-B"} if n == "DEG_VALIDATION" else {})))
        status.append(st)
    for n, parts in tabs.items():
        (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()).to_csv(outs[n], sep="\t", index=False)
    s = pd.DataFrame(status); s.to_csv(a.root / "COLLECT_STATUS_N5000.tsv", sep="\t", index=False)
    print(s.groupby(["dataset", "flag"]).size().to_string())
    for n, f in outs.items():
        print(n, sum(1 for _ in open(f)) - 1, "rows")


if __name__ == "__main__":
    main()

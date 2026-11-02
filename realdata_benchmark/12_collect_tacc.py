"""Collect the TACC rows of the joint benchmark into the deliverable tables (runs on TACC; reads small tables only).

    python realdata_benchmark/12_collect_tacc.py --root ROOT

PDX_GATE_COUPLING_METRICS.tsv       every ltl331 worklist row's LTL331_OT_BENCHMARK_pair<i>.tsv (05_: convergence,
                                    calibration validity, rejected fractions, asymmetry, coupling entropy, type-level
                                    transport, coverage, NE vs non-NE, depth AUC, balanced-OT correspondence; formal_result
                                    False for any non-converged or calibration-invalid fit)
SPLATTER_GATE_COUPLING_METRICS.tsv  every splatter row's score.csv (benchmark/40_score.py: false rejection, recall,
                                    precision, depth AUC / rho, same-population mass, false asserted mass, ...) with the
                                    row's configuration, acceptance and the fitted run's calibration flags (pair_metrics.csv)
COLLECT_STATUS.tsv                  per row: SUCCESS / FAILED / missing and whether its table was found
Nothing is filtered: a row without output is listed, never dropped silently. Refuses if a table exists.
--ltl331-worklist NAME (e.g. worklist_ltl331_author_harmony.tsv): collect only that LTL331 worklist, into
PDX_GATE_COUPLING_METRICS_<NAME stem>.tsv and COLLECT_STATUS_<NAME stem>.tsv (the default tables are not touched).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--ltl331-worklist", default=None)
    a = p.parse_args()
    tag = "" if a.ltl331_worklist is None else "_" + Path(a.ltl331_worklist).stem
    outs = {d: a.root / f"{d}_GATE_COUPLING_METRICS{tag}.tsv" for d in (("PDX",) if tag else ("PDX", "SPLATTER"))}
    for f in outs.values():
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    status, pdx, spl = [], [], []
    for w in (("ltl331",) if tag else ("ltl331", "splatter")):
        work = pd.read_csv(a.root / (a.ltl331_worklist if tag else f"worklist_{w}.tsv"), sep="\t", dtype=str)
        for _, r in work.iterrows():
            out = Path(r.out)
            st = {"dataset": w, "config_id": r.config_id, "pair_id": r.pair_id, "acceptance": r.acceptance, "out": r.out,
                  "flag": "SUCCESS" if (out / "SUCCESS").exists() else "FAILED" if (out / "FAILED").exists() else "missing"}
            if w == "ltl331":
                tabs = list(out.glob("LTL331_OT_BENCHMARK_pair*.tsv"))
                st["table_found"] = bool(tabs)
                if tabs:
                    t = pd.read_csv(tabs[0], sep="\t"); t.insert(0, "pair_index", r.pair_index); pdx.append(t)
            else:
                sc = out / "score.csv"
                st["table_found"] = sc.exists()
                if sc.exists():
                    t = pd.read_csv(sc)
                    t.insert(0, "config_id", r.config_id); t.insert(1, "acceptance", float(r.acceptance))
                    t.insert(2, "fit_label", (out / "fit_label.txt").read_text().strip() if (out / "fit_label.txt").exists() else "")
                    pm = list((out / "runs").glob("*/scope_all/budget_*/pair_metrics.csv"))
                    if pm:
                        m = pd.read_csv(pm[0])[["method", "calibration_m4e_inference_valid", "calibration_valid_for_m4r",
                                                "inner_converged", "outer_converged", "cycle_detected", "rejection_cost"]]
                        t = t.merge(m, on="method", how="left", suffixes=("", "_fit"))
                    spl.append(t)
            status.append(st)
    (pd.concat(pdx, ignore_index=True) if pdx else pd.DataFrame()).to_csv(outs["PDX"], sep="\t", index=False)
    if "SPLATTER" in outs:
        (pd.concat(spl, ignore_index=True) if spl else pd.DataFrame()).to_csv(outs["SPLATTER"], sep="\t", index=False)
    s = pd.DataFrame(status); s.to_csv(a.root / f"COLLECT_STATUS{tag}.tsv", sep="\t", index=False)
    print(s.groupby(["dataset", "flag", "table_found"]).size().to_string())
    for k, f in outs.items():
        print(k, f, sum(1 for _ in open(f)) - 1, "rows")


if __name__ == "__main__":
    main()

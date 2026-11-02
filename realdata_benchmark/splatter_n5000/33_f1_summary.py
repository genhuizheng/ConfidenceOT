"""Rejection F1 of the N = 5,000 benchmark, the primary metric (user 2026-11-01): per side, per row, and summarised
by level x case x configuration x method. Reads the tables of 32_collect_n5000.py; no new scoring.

    python realdata_benchmark/splatter_n5000/33_f1_summary.py --root N5000_ROOT [--pilot]

F1 is the convention of benchmark/58_summarise_arms.directional_f1 (imported): 2PR / (P + R) from 40_score.py's
precision and recall of rejection, 0 when the side holds cells to reject and none was rejected. One addition: a side
with nothing to reject has no F1 (NaN; it is never averaged in), which is Case 1 on both sides and the shared side of
Case 2. The comparators (IC-POT, Vanilla UOT; solver_comparison.csv, one row per method and side) carry their deployable
fixed-cutoff precision / recall, put through the same rule.
Per case the reading is: Case 1 (all shared) false rejection rate; Case 2 (one side holds an unmatched population) F1
of that side, the primary scenario, with precision / recall to explain it and the false rejection rate of the other
side; Case 3 (nothing shared) recall.
Writes into N5000_ROOT (refuses if any exists):
  SPLATTER_N5000_F1_ROWS<suffix>.tsv     level, replicate, case, pair, configuration, solver family, method, side:
                                         positives, f1, precision, recall, false_rejection_rate; plus the row's
                                         coupling accuracy (same_population_mass of the plan, 23_) and convergence /
                                         calibration flags as diagnostics only
  SPLATTER_N5000_F1_SUMMARY<suffix>.tsv  level x case x configuration x method x side: rows, rows with defined F1, mean /
                                         SD / median F1 over the defined ones, mean precision, recall, false rejection
                                         rate and coupling accuracy
No PASS / FAIL, no weighting, no composite score.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("summarise_arms", REPO / "benchmark" / "58_summarise_arms.py")
summarise_arms = importlib.util.module_from_spec(_spec); sys.modules["summarise_arms"] = summarise_arms
_spec.loader.exec_module(summarise_arms)
F1_CASE = {"source": "population_lost", "target": "population_emerged"}  # makes directional_f1 read that side
KEYS = ["level", "replicate", "case", "pair_id", "config_id"]


def side_f1(frame: pd.DataFrame, side: str) -> pd.Series:
    """directional_f1 on the given side; NaN where the side has nothing to reject (recall undefined)."""
    f = summarise_arms.directional_f1(frame.assign(biological_case=F1_CASE[side]))
    return f.where(np.isfinite(frame[f"{side}_recall"].astype(float)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    sfx = "_PILOT" if a.pilot else ""
    outs = [a.root / f"SPLATTER_N5000_F1_ROWS{sfx}.tsv", a.root / f"SPLATTER_N5000_F1_SUMMARY{sfx}.tsv"]
    for f in outs:
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    g = pd.read_csv(a.root / f"SPLATTER_N5000_GATE_METRICS{sfx}.tsv", sep="\t", low_memory=False)
    rows = []
    cot = g[g.solver_family == "ConfidenceOT"]
    for side in ("source", "target"):
        x = cot[KEYS + ["method"]].copy()
        x["solver_family"] = "ConfidenceOT"; x["side"] = side
        x["f1"] = side_f1(cot, side).to_numpy()
        for c in ("precision", "recall", "false_rejection_rate"):
            x[c] = cot[f"{side}_{c}"].to_numpy()
        rows.append(x)
    comp = g[g.solver_family != "ConfidenceOT"]
    if len(comp):
        x = comp[KEYS + ["method", "side", "solver_family"]].copy()
        wide = pd.DataFrame({"biological_case": "population_lost", "source_precision": comp.fixed_precision.to_numpy(),
                             "source_recall": comp.fixed_recall.to_numpy()}, index=comp.index)
        x["f1"] = side_f1(wide, "source").to_numpy()
        x["precision"], x["recall"] = comp.fixed_precision.to_numpy(), comp.fixed_recall.to_numpy()
        x["false_rejection_rate"] = comp.fixed_false_rejection_rate.to_numpy()
        rows.append(x)
    r = pd.concat(rows, ignore_index=True)
    r["has_positives"] = np.isfinite(r.recall.astype(float))
    cm = a.root / f"SPLATTER_N5000_COUPLING_METRICS{sfx}.tsv"
    if cm.exists():
        c = pd.read_csv(cm, sep="\t")
        c = c[c.plan.isin(["M4-E", "M4-R"])].rename(columns={"plan": "method", "same_population_mass": "coupling_accuracy"})
        r = r.merge(c[KEYS + ["method", "coupling_accuracy"]], on=KEYS + ["method"], how="left")
    cv = a.root / f"SPLATTER_N5000_CALIBRATION_VALIDITY{sfx}.tsv"
    if cv.exists():
        v = pd.read_csv(cv, sep="\t")[KEYS + ["method", "calibration_m4e_inference_valid", "calibration_valid_for_m4r", "outer_converged", "cycle_detected"]]
        r = r.merge(v, on=KEYS + ["method"], how="left")
    r.to_csv(outs[0], sep="\t", index=False)
    k = ["level", "case", "config_id", "solver_family", "method", "side"]
    agg = {"rows": ("f1", "size"), "f1_defined": ("f1", "count"), "f1_mean": ("f1", "mean"), "f1_sd": ("f1", "std"),
           "f1_median": ("f1", "median"), "precision_mean": ("precision", "mean"), "recall_mean": ("recall", "mean"),
           "false_rejection_rate_mean": ("false_rejection_rate", "mean")}
    if "coupling_accuracy" in r:
        agg["coupling_accuracy_mean"] = ("coupling_accuracy", "mean")
    s = r.groupby(k, dropna=False).agg(**agg).reset_index()
    s.to_csv(outs[1], sep="\t", index=False)
    main_view = s[s.case.astype(str).str.startswith("Case2") & (s.f1_defined > 0)]
    print(main_view.sort_values(["case", "level", "f1_mean"], ascending=[True, True, False]).head(60).round(3).to_string(index=False))
    print(f"-> {outs[0]}\n-> {outs[1]}")


if __name__ == "__main__":
    main()

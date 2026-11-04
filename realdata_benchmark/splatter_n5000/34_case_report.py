"""Case-by-case report of an N = 5,000 root: Rejection F1 for Cases 1-3 separately, source / target rejected fractions
and the one-side reject-all diagnostic (read only; from 32_collect_n5000.py and 33_f1_summary.py tables).

    python realdata_benchmark/splatter_n5000/34_case_report.py --root ROOT

Per row (level, replicate, case, configuration, method), from SPLATTER_N5000_GATE_METRICS.tsv: source and target
rejected fractions; the false asserted mass; one_side_all = one side rejected >= 0.99 while the other <= 0.01 (which
side recorded), both_all = both sides >= 0.99; the calibrated rejection cost. ConfidenceOT M4-E / M4-R, and the
comparators (IC-POT, Vanilla UOT; their fixed-cutoff rejection rate per side). F1 from SPLATTER_N5000_F1_ROWS.tsv (the
33_ rule: F1 only on a side holding cells to reject).
Per case the reading is: Case 1 (all shared) false rejection per side; Case 2 F1 on the side holding the unmatched
population, its precision / recall, and the other side's false rejection; Case 3 (nothing shared) rejected fraction
per side and the false asserted mass. No composite, no PASS / FAIL.
Writes ROOT/CASE_REPORT_ROWS.tsv, ROOT/CASE_REPORT.tsv (level x case x configuration x method) and ROOT/CASE_REPORT.md
(M4-E pivots); refuses if any exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "A4_perturb_only", "L4_batch_perturb",
          "A5_imbalance_only", "L5_batch_perturb_imbalance"]
KEYS = ["level", "replicate", "case", "pair_id", "config_id", "method"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    outs = [a.root / n for n in ("CASE_REPORT_ROWS.tsv", "CASE_REPORT.tsv", "CASE_REPORT.md")]
    for f in outs:
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    g = pd.read_csv(a.root / "SPLATTER_N5000_GATE_METRICS.tsv", sep="\t", low_memory=False)
    f1 = pd.read_csv(a.root / "SPLATTER_N5000_F1_ROWS.tsv", sep="\t")
    cot = g[g.solver_family == "ConfidenceOT"].copy()
    cot["source_rejected_fraction"] = cot.source_rejected_n / cot.source_cells_n
    cot["target_rejected_fraction"] = cot.target_rejected_n / cot.target_cells_n
    comp = g[(g.solver_family != "ConfidenceOT") & ~g.method.astype(str).str.startswith("ConfidenceOT")]
    comp = comp.pivot_table(index=KEYS, columns="side", values="fixed_rejection_rate", aggfunc="first").add_suffix("_rejected_fraction").reset_index()
    rows = pd.concat([cot[KEYS + ["source_rejected_fraction", "target_rejected_fraction", "false_asserted_mass", "asserted_mass",
                                  "rejection_cost", "source_false_rejection_rate", "target_false_rejection_rate"]], comp], ignore_index=True)
    s, t = rows.source_rejected_fraction, rows.target_rejected_fraction
    rows["one_side_all"] = ((s >= 0.99) & (t <= 0.01)) | ((t >= 0.99) & (s <= 0.01))
    rows["one_side_all_side"] = np.where((s >= 0.99) & (t <= 0.01), "source", np.where((t >= 0.99) & (s <= 0.01), "target", ""))
    rows["both_all"] = (s >= 0.99) & (t >= 0.99)
    fw = f1.pivot_table(index=KEYS, columns="side", values=["f1", "precision", "recall"], aggfunc="first")
    fw.columns = [f"{side}_{m}" for m, side in fw.columns]
    rows = rows.merge(fw.reset_index(), on=KEYS, how="left")
    rows["calibration_null"] = np.where(rows.config_id.astype(str).str.endswith("|cross_side_rotation"), "cross_side_rotation", "within_side_split")
    rows.to_csv(outs[0], sep="\t", index=False)
    agg = {"rows": ("replicate", "size"), "source_rejected_fraction": ("source_rejected_fraction", "mean"),
           "target_rejected_fraction": ("target_rejected_fraction", "mean"), "one_side_all_share": ("one_side_all", "mean"),
           "one_side_all_source": ("one_side_all_side", lambda x: int((x == "source").sum())),
           "one_side_all_target": ("one_side_all_side", lambda x: int((x == "target").sum())), "both_all_share": ("both_all", "mean"),
           "false_asserted_mass": ("false_asserted_mass", "mean"), "rejection_cost_median": ("rejection_cost", "median")}
    for side in ("source", "target"):
        for m in ("f1", "precision", "recall"):
            if f"{side}_{m}" in rows:
                agg[f"{side}_{m}_mean"] = (f"{side}_{m}", "mean")
        agg[f"{side}_f1_defined"] = (f"{side}_f1", "count")
    rep = rows.groupby(["level", "case", "config_id", "method"]).agg(**agg).reset_index()
    rep["level"] = pd.Categorical(rep.level, LEVELS); rep = rep.sort_values(["case", "level", "method", "config_id"])
    rep.to_csv(outs[1], sep="\t", index=False)
    e = rep[rep.method == "M4-E"]
    md = [f"# Case report, {a.root.name} (M4-E; means over replicates)", ""]
    views = [("Case1_all_shared", ["source_rejected_fraction", "target_rejected_fraction", "one_side_all_share"]),
             ("Case2_source_unmatched", ["source_f1_mean", "source_precision_mean", "source_recall_mean", "target_rejected_fraction", "one_side_all_share"]),
             ("Case2_target_unmatched", ["target_f1_mean", "target_precision_mean", "target_recall_mean", "source_rejected_fraction", "one_side_all_share"]),
             ("Case3_none_shared", ["source_rejected_fraction", "target_rejected_fraction", "false_asserted_mass", "one_side_all_share"])]
    for case, cols in views:
        md += [f"## {case}", ""]
        for col in cols:
            v = e[e.case == case].pivot_table(index="config_id", columns="level", values=col, aggfunc="first", observed=False)
            v = v.reindex(columns=[l for l in LEVELS if l in v.columns])
            v.index = v.index.str.replace("|", r"\|", regex=False)  # config ids carry '|', a markdown cell separator
            md += [f"**{col}**", "", v.round(3).to_markdown(), ""]
    (outs[2]).write_text("\n".join(md) + "\n")
    print(f"-> {outs[0]}\n-> {outs[1]}\n-> {outs[2]}")


if __name__ == "__main__":
    main()

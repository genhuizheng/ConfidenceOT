"""Audit main_decisions_per_pair.csv case by case before trusting any figure.

Three checks, read straight from the table 56_comparator_metrics.py wrote:

1. Directional F1 per strategy and arm, case by case -- population_lost on
   the source, population_emerged on the target, populations_disjoint on the
   source -- with the number of pairs behind each mean, their macro mean, and
   the plain mean over pairs that 55_plot_method_comparison.py plots. The two
   agree when every case has the same number of pairs.
2. Whether the two ConfidenceOT gates are distinct rows: at each within-side
   acceptance minimum, how many pair-sides differ between M4-E and M4-R. Zero
   would mean one was copied from the other.
3. Partial OT (truth-derived m) on populations_disjoint, where m = 0 must mean
   every source cell is rejected: for a few pairs per arm, m, the source
   scores, the rejected count and the counts behind F1, read from the saved
   score file and provenance.json, and a count of disjoint pairs where not
   every source cell was rejected.

    python benchmark/57_audit_main_decisions.py \\
        BENCH_ROOT/comparator_metrics_final/main_decisions_per_pair.csv BENCH_ROOT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DIRECTIONAL = {"population_lost": "source", "population_emerged": "target",
               "populations_disjoint": "source"}
TRUTH_M = "Partial OT (truth-derived m)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decisions_csv", type=Path)
    parser.add_argument("bench_root", type=Path, nargs="?", default=None,
                        help="For check 3: holds solvers_oracle_budget/")
    parser.add_argument("--oracle-dir", default="solvers_oracle_budget")
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = pd.read_csv(args.decisions_csv)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(f"{args.decisions_csv}: {len(d)} rows")
    print("\nrows per strategy and arm (180 pairs x 2 sides = 360 when complete):")
    print(d.groupby(["strategy", "arm"]).size().unstack("arm").to_string())

    # 1. Directional F1, case by case.
    keep = np.array([DIRECTIONAL.get(c) == s for c, s in zip(d["biological_case"], d["side"])])
    directional = d[keep]
    by_case = (directional.groupby(["strategy", "arm", "biological_case"])["f1"]
               .agg(["mean", "count"]).unstack("biological_case"))
    table = pd.DataFrame(index=by_case.index)
    for case in DIRECTIONAL:
        present = ("mean", case) in by_case.columns
        table[f"{case} F1"] = by_case[("mean", case)] if present else np.nan
        table[f"{case} n"] = by_case[("count", case)] if present else 0
    table["macro mean of the three"] = table[[f"{c} F1" for c in DIRECTIONAL]].mean(axis=1)
    table["mean over pairs (plotted)"] = directional.groupby(["strategy", "arm"])["f1"].mean()
    print("\n1. directional F1 by case:")
    print(table.round(3).to_string())

    # 2. Are the two ConfidenceOT gates distinct, at each acceptance minimum?
    gates = sorted(s for s in d["strategy"].unique() if s.startswith("ConfidenceOT"))
    print(f"\n2. ConfidenceOT strategies present: {gates}")
    key = ["arm", "pair_id", "side"]
    for e_name in (s for s in gates if s.startswith("ConfidenceOT M4-E")):
        r_name = e_name.replace("M4-E", "M4-R", 1)
        if r_name not in gates:
            continue
        e = d[d["strategy"] == e_name].set_index(key)[["tp", "fp", "fn", "tn"]]
        r = d[d["strategy"] == r_name].set_index(key)[["tp", "fp", "fn", "tn"]]
        joined = e.join(r, lsuffix="_e", rsuffix="_r", how="inner")
        differ = ((joined["tp_e"] != joined["tp_r"]) | (joined["fp_e"] != joined["fp_r"]))
        print(f"   pair-sides compared: {len(joined)}; pair-sides where {e_name} and "
              f"{r_name} reject differently: {int(differ.sum())}")
        print(differ.groupby(level="arm").sum().rename("differing pair-sides").to_string())

    # 3. Partial OT with m = 0 on populations_disjoint.
    rows = d[(d["strategy"].str.startswith(TRUTH_M)) & (d["biological_case"] == "populations_disjoint")
             & (d["side"] == "source")]
    incomplete = rows[rows["rejected_n"] != rows["n_cells"]]
    print(f"\n3. {TRUTH_M} on populations_disjoint, source side: {len(rows)} rows; "
          f"rows where not every source cell was rejected: {len(incomplete)}")
    if len(incomplete):
        print(incomplete[["strategy", "arm", "pair_id", "rejected_n", "n_cells", "tp", "fp", "fn"]]
              .to_string(index=False))
    if args.bench_root is None:
        return
    shown = []
    for arm, part in rows[rows["strategy"].str.contains("fixed cutoff")].groupby("arm"):
        for pair_id in sorted(part["pair_id"])[: args.examples]:
            where = args.bench_root / args.oracle_dir / arm / pair_id
            m = json.loads((where / "provenance.json").read_text(encoding="utf-8"))["oracle_budget"]["m"]
            u = np.load(where / "score_Oracle-budget_Partial_OT_source.npy")
            row = part[part["pair_id"] == pair_id].iloc[0]
            shown.append({"arm": arm, "pair_id": pair_id, "partial_m": m,
                          "u_min": float(u.min()), "u_median": float(np.median(u)),
                          "u_max": float(u.max()),
                          "rejected": f"{int(row['rejected_n'])} / {int(row['n_cells'])}",
                          "tp": int(row["tp"]), "fp": int(row["fp"]), "fn": int(row["fn"]),
                          "f1": row["f1"]})
    print(pd.DataFrame(shown).to_string(index=False))


if __name__ == "__main__":
    main()

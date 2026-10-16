"""Does a coupling rerun reproduce the original gate? Read only.

    python kp_lineage/validation_v3/03_compare_reproduction.py --root ROOT --rerun RERUN_ROOT \\
        --labels ranknm256_noscale_ds_cos raw --out REPORT_DIR

For every pair with a rerun, per method (M4-E, M4-R), against the original output:

* cell sets and order: the (side, observation_id) sequence of cell_confidence.csv;
* gate: retained, raw_retained (count of disagreements);
* continuous: max |difference| of signed_rejection_margin and decision_cost;
* rejection cost: the rerun's rejection_cost against the original's;
* pair_metrics: transported mass, rejection rates, iteration counts, convergence;
* coupling: shape against the cell counts, total mass against transported_mass,
  and the mass carried by retained and by rejected source rows (descriptive).

A pair passes when cell order and every gate decision agree, the rejection costs
are equal, max |margin difference| <= 1e-8, |mass difference| <= 1e-9, and the
coupling has the right shape and mass. Exits 1 if any compared pair fails.
Writes reproduction.tsv, reproduction.json, SUCCESS (only when all pass).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN_TOL = 1e-8
MASS_TOL = 1e-9


def one(root: Path, pair_id: str) -> Path | None:
    done = sorted((root / pair_id).glob("scope_malignant/*/SUCCESS"))
    return done[0].parent if len(done) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", default=["ranknm256_noscale_ds_cos", "raw"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for label in args.labels:
        for new_dir in sorted((args.rerun / label).glob("*/scope_malignant/*/SUCCESS")):
            pair_id = new_dir.parents[2].name
            new = new_dir.parent
            old = one(args.root / "ot" / "kptracer" / label, pair_id)
            if old is None:
                rows.append({"label": label, "pair_id": pair_id, "pass": False, "problem": "no original output"})
                continue
            old_cells, new_cells = pd.read_csv(old / "cell_confidence.csv"), pd.read_csv(new / "cell_confidence.csv")
            old_metrics = pd.read_csv(old / "pair_metrics.csv").set_index("method")
            new_metrics = pd.read_csv(new / "pair_metrics.csv").set_index("method")
            old_run = json.loads((old / "run.json").read_text())
            new_run = json.loads((new / "run.json").read_text())
            for method in ("M4-E", "M4-R"):
                a = old_cells[old_cells["method"].eq(method)].reset_index(drop=True)
                b = new_cells[new_cells["method"].eq(method)].reset_index(drop=True)
                order_same = (len(a) == len(b) and a["side"].tolist() == b["side"].tolist()
                              and a["observation_id"].astype(str).tolist() == b["observation_id"].astype(str).tolist())
                record = {"label": label, "pair_id": pair_id, "method": method, "order_same": order_same,
                          "source_cells": int(a["side"].eq("source").sum()), "target_cells": int(a["side"].eq("target").sum()),
                          "rejection_cost_old": float(old_run["rejection_cost"]),
                          "rejection_cost_new": float(new_run["rejection_cost"])}
                if order_same:
                    record["retained_disagree"] = int((a["retained"].astype(bool) != b["retained"].astype(bool)).sum())
                    record["raw_retained_disagree"] = int((a["raw_retained"].astype(bool) != b["raw_retained"].astype(bool)).sum())
                    record["margin_max_abs_diff"] = float(np.max(np.abs(a["signed_rejection_margin"].to_numpy(float)
                                                                         - b["signed_rejection_margin"].to_numpy(float))))
                    record["decision_cost_max_abs_diff"] = float(np.max(np.abs(a["decision_cost"].to_numpy(float)
                                                                               - b["decision_cost"].to_numpy(float))))
                for key in ("transported_mass", "source_final_rejection_rate", "target_final_rejection_rate",
                            "outer_iterations", "total_inner_iterations", "inner_converged", "outer_converged",
                            "cycle_detected"):
                    record[f"{key}_old"] = old_metrics.loc[method, key]
                    record[f"{key}_new"] = new_metrics.loc[method, key]
                path = new / f"coupling_{method.lower().replace('-', '')}.npz"
                if path.exists():
                    coupling = np.load(path)["coupling"]
                    record["coupling_shape"] = "x".join(map(str, coupling.shape))
                    record["coupling_shape_ok"] = coupling.shape == (record["source_cells"], record["target_cells"])
                    record["coupling_mass"] = float(coupling.sum())
                    record["coupling_mass_vs_metrics"] = float(coupling.sum() - float(new_metrics.loc[method, "transported_mass"]))
                    if order_same and record["coupling_shape_ok"]:
                        retained = b.loc[b["side"].eq("source"), "retained"].astype(bool).to_numpy()
                        rows_mass = coupling.sum(axis=1)
                        record["mass_on_retained_rows"] = float(rows_mass[retained].sum())
                        record["mass_on_rejected_rows"] = float(rows_mass[~retained].sum())
                        record["coupling_negative_entries"] = int((coupling < 0).sum())
                else:
                    record["coupling_shape_ok"] = False
                mass_diff = abs(float(record["transported_mass_old"]) - float(record["transported_mass_new"]))
                record["mass_abs_diff"] = mass_diff
                record["pass"] = bool(order_same and record.get("retained_disagree") == 0
                                      and record.get("raw_retained_disagree") == 0
                                      and record["rejection_cost_old"] == record["rejection_cost_new"]
                                      and record.get("margin_max_abs_diff", np.inf) <= MARGIN_TOL
                                      and mass_diff <= MASS_TOL and record.get("coupling_shape_ok")
                                      and abs(record.get("coupling_mass_vs_metrics", np.inf)) <= MASS_TOL)
                rows.append(record)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "reproduction.tsv", sep="\t", index=False)
    summary = {"compared": int(len(table)), "passed": int(table["pass"].sum()) if len(table) else 0,
               "failed": table.loc[~table["pass"], ["label", "pair_id"]].astype(str).agg(" ".join, axis=1).tolist() if len(table) else [],
               "tolerances": {"margin": MARGIN_TOL, "mass": MASS_TOL}}
    (args.out / "reproduction.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        cols = ["label", "pair_id", "method", "order_same", "retained_disagree", "margin_max_abs_diff",
                "mass_abs_diff", "outer_iterations_old", "outer_iterations_new", "coupling_shape",
                "mass_on_retained_rows", "mass_on_rejected_rows", "pass"]
        print(table[[c for c in cols if c in table]].to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))
    if len(table) and table["pass"].all():
        (args.out / "SUCCESS").write_text("all reruns reproduce the original gate\n", encoding="utf-8")
    sys.exit(0 if len(table) and table["pass"].all() else 1)


if __name__ == "__main__":
    main()

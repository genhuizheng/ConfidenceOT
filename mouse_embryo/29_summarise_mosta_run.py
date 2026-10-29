"""Read-only: print what a finished MOSTA run found, from its audit tables.

Reads <root>/audit/ as 25_ wrote it (summary.json, pair_metrics.csv,
annotation_fractions.csv.gz, calibration.csv, depth_diagnostics.csv) and
prints, for every completed pair (nothing is filtered):

1. completeness;
2. per transition (adjacent) and per stage (same-stage): the M4-E rejected
   fraction of the source and the target side, median and range over pairs,
   and the same for M4-R and balanced OT;
3. per transition and side: the annotations with the highest mean M4-E
   rejected fraction over the pairs in which the annotation has bins on that
   side, with the number of those pairs and of partner-absent ones;
4. per transition: the nominal-mass split of the source side into retained
   with a same-annotation partner, retained with another annotation, retained
   without supported mass, and rejected (median over pairs; the four parts of
   one pair sum to 1);
5. per transition: where the bins M4-E rejects go under balanced OT (same
   annotation share of their mass), beside the same-annotation share of the
   M4-E retained mass;
6. diagnostics: calibration flags, M4-E / M4-R convergence and cycles, and the
   depth association of the decision (rank AUC of original depth, rejected
   against retained; Spearman of decision cost with original depth).

Fractions are of bins (uniform marginals, so equal to nominal mass).  Prints
only; writes nothing.

    python mouse_embryo/29_summarise_mosta_run.py <root> [--top 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 500)


def stage_key(group: str) -> float:
    return float(group.split("->")[0].lstrip("E"))


def spread(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return "n/a"
    return f"{values.median():.3f} [{values.min():.3f}, {values.max():.3f}]"


def ordered(groups) -> list[str]:
    return sorted(set(groups), key=stage_key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    audit = args.root / "audit"

    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    print("=== 1. completeness")
    print(json.dumps(summary.get("status_counts"), indent=1), "not completed:", summary.get("not_completed"))

    metrics = pd.read_csv(audit / "pair_metrics.csv")
    print("\n=== 2. rejected fraction per side, median [min, max] over pairs")
    for kind in ("adjacent", "same_stage"):
        table = metrics[metrics["kind"] == kind]
        rows = []
        for group in ordered(table["group"]):
            part = table[table["group"] == group]
            row = {"group": group, "pairs": part["pair_id"].nunique()}
            for method in ("M4-E", "M4-R", "balanced"):
                m = part[part["method"] == method]
                row[f"{method} source"] = spread(m["source_final_rejection_rate"])
                row[f"{method} target"] = spread(m["target_final_rejection_rate"])
            rows.append(row)
        print(f"--- {kind}")
        print(pd.DataFrame(rows).to_string(index=False))

    fractions = pd.read_csv(audit / "annotation_fractions.csv.gz")
    adjacent = fractions[fractions["kind"] == "adjacent"]
    print(f"\n=== 3. annotations with the highest mean M4-E rejected fraction (top {args.top}; adjacent pairs)")
    for group in ordered(adjacent["group"]):
        for side in ("source", "target"):
            part = adjacent[(adjacent["group"] == group) & (adjacent["side"] == side) & (adjacent["n_bins"] > 0)]
            ranked = part.groupby("annotation").agg(
                mean_rejected=("m4e_rejected_fraction", "mean"),
                pairs=("pair_id", "nunique"),
                partner_absent_pairs=("partner_present", lambda s: int((~s.astype(bool)).sum())),
                median_bins=("n_bins", "median")).sort_values("mean_rejected", ascending=False)
            print(f"--- {group} {side}")
            print(ranked.head(args.top).round(3).to_string())

    print("\n=== 4. source-side nominal mass split, median over pairs (adjacent)")
    source = adjacent[adjacent["side"] == "source"].copy()
    weight = source["n_bins"]
    parts = {"retained_same": source["co_same_share"], "retained_other": source["co_other_share"],
             "retained_without_support": source["co_retained_without_support_share"],
             "rejected": source["m4e_rejected_fraction"]}
    per_pair = pd.DataFrame({name: values * weight for name, values in parts.items()})
    per_pair["n_bins"] = weight
    per_pair[["pair_id", "group"]] = source[["pair_id", "group"]]
    per_pair = per_pair.groupby(["group", "pair_id"]).sum(numeric_only=True)
    for name in parts:
        per_pair[name] = per_pair[name] / per_pair["n_bins"]
    rows = []
    for group in ordered(per_pair.index.get_level_values("group")):
        part = per_pair.loc[group]
        rows.append({"group": group, "pairs": len(part), **{name: f"{part[name].median():.3f}" for name in parts},
                     "sum_of_medians_check": f"{sum(part[name].median() for name in parts):.3f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 5. same-annotation share: M4-E retained mass vs the M4-E rejected bins under balanced OT (adjacent, source)")
    rejected = source["m4e_rejected"]
    retained_mass = source["co_same_share"] + source["co_other_share"]
    frame = pd.DataFrame({
        "group": source["group"], "pair_id": source["pair_id"],
        "co_same": source["co_same_share"] * weight, "co_supported": retained_mass * weight,
        "bot_rej_same": source["bot_co_rejected_same_share"].fillna(0) * rejected, "rejected": rejected,
        "bot_same": source["bot_same_share"].fillna(0) * weight, "n_bins": weight})
    frame = frame.groupby(["group", "pair_id"]).sum(numeric_only=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["co_retained_same_share"] = frame["co_same"] / frame["co_supported"]
        frame["bot_on_co_rejected_same_share"] = frame["bot_rej_same"] / frame["rejected"]
        frame["bot_all_same_share"] = frame["bot_same"] / frame["n_bins"]
    rows = []
    for group in ordered(frame.index.get_level_values("group")):
        part = frame.loc[group]
        rows.append({"group": group, "pairs": len(part),
                     "M4-E retained: same annotation": spread(part["co_retained_same_share"]),
                     "balanced, all bins: same": spread(part["bot_all_same_share"]),
                     "balanced, M4-E-rejected bins: same": spread(part["bot_on_co_rejected_same_share"]),
                     "pairs with no rejected bin": int((part["rejected"] == 0).sum())})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== 6. diagnostics")
    calibration = pd.read_csv(audit / "calibration.csv")
    flags = ["calibration_valid", "feasible_cost_found", "m4e_calibration_clean", "m4r_validation_clean",
             "m4e_inference_valid", "validation_aggregate_valid"]
    print("calibration flags true / pairs:", {flag: f"{int(calibration[flag].astype(bool).sum())}/{len(calibration)}"
                                               for flag in flags if flag in calibration})
    print("rejection cost:", spread(calibration["rejection_cost"]))
    print("held-out raw acceptance, source:", spread(calibration["validation_source_raw_acceptance"]),
          " target:", spread(calibration["validation_target_raw_acceptance"]))
    for method in ("M4-E", "M4-R", "balanced"):
        m = metrics[metrics["method"] == method]
        print(f"{method}: outer converged {int(m['outer_converged'].astype(bool).sum())}/{len(m)}, "
              f"inner converged {int(m['inner_converged'].astype(bool).sum())}/{len(m)}, "
              f"cycles {int(m['cycle_detected'].astype(bool).sum())}, "
              f"outer iterations {spread(m['outer_iterations'].astype(float))}")
    depth = pd.read_csv(audit / "depth_diagnostics.csv")
    rows = []
    for (kind, method, side), part in depth.groupby(["kind", "method", "side"]):
        rows.append({"kind": kind, "method": method, "side": side, "pairs": len(part),
                     "AUC original depth (rejected vs retained)": spread(part["auc_original_depth_rejected_vs_retained"]),
                     "Spearman cost vs original depth": spread(part["spearman_decision_cost_original_depth"]),
                     "pairs without rejected bins": int((part["rejected"] == 0).sum())})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

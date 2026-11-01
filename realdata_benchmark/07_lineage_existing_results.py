"""KP_QUINN_LINEAGE_BENCHMARK.tsv, part 1: the existing KP-Tracer / Quinn Level 2 and Level 3 results in one table,
read from the committed result tables (lineage_validation/level2_pairwise_ot/results, level3_cell_lineage/results/
summary); nothing recomputed. New configurations (part 2) are TACC jobs (PROPOSED_TACC_JOBS.md) and append rows with
the same columns.

    python realdata_benchmark/07_lineage_existing_results.py --lineage-root lineage_validation --out OUT_DIR

Each row: dataset, mouse, family, strategy (rna / cnv / rna_cnv input), method, and the preprocessing / solver /
rejection that method used (from level2 diagnostics.tsv: A_standard_OT = logcpm_noscale on original counts, balanced
OT; B_our_OT = ranknm256_noscale_ds_cos on equalised counts, ConfidenceOT UOT with the rejection budget fixed at 0, so
no rejection), so preprocessing and solver are never read as one factor. Level 2: balanced accuracy of subclone
identification (enrichment rule) minus chance, per family and level (L1, L2), evaluable families only, with the
evaluation status kept. Level 3: delta_D_norm, top-1-in-nearest against its independent-coupling rate, per family.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lineage-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    target = a.out / "KP_QUINN_LINEAGE_BENCHMARK.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    L2 = a.lineage_root / "level2_pairwise_ot" / "results"; L3 = a.lineage_root / "level3_cell_lineage" / "results" / "summary"
    diag = pd.read_csv(L2 / "diagnostics.tsv", sep="\t")
    method_cfg = diag.groupby("method")[["preprocessing", "counts", "solver", "rejection"]].agg(lambda s: "; ".join(sorted(set(map(str, s)))))
    s2 = pd.read_csv(L2 / "summary.tsv", sep="\t")
    s2["dataset"] = s2.case_id.str.split("_").str[0]; s2["mouse"] = s2.case_id.str.split("__").str[0].str.split("_").str[-1]
    l2 = s2[["dataset", "mouse", "family", "strategy", "method", "level", "evaluation_status", "not_evaluable_reason", "scored",
             "chance", "balanced_accuracy_enrichment"]].copy()
    # chance as level2 records it; a family with one candidate (chance 1.0) is not evaluable and its balanced accuracy is 0.5
    l2["level2_balanced_accuracy_minus_chance"] = l2.balanced_accuracy_enrichment - l2.chance.where(l2.chance < 1, 0.5)
    l2 = l2.rename(columns={"level": "subclone_level"})
    l3 = pd.read_csv(L3 / "family_summary.tsv", sep="\t", dtype={"mouse": str})
    l3 = l3[["dataset", "mouse", "family", "strategy", "method", "cells", "delta_D_norm", "delta_D_norm_lo", "delta_D_norm_hi",
             "top1_in_nearest", "top1_in_nearest_ind_rate", "flag_high_missingness", "flag_low_tree_resolution"]]
    l3 = l3.rename(columns={c: f"level3_{c}" for c in l3.columns if c not in ("dataset", "mouse", "family", "strategy", "method")})
    out = l2.merge(l3, on=["dataset", "mouse", "family", "strategy", "method"], how="outer")
    out = out.join(method_cfg, on="method")
    out.insert(0, "source", "existing Level 2 (2026-10-22) / Level 3 results, frozen protocol 508609f")
    out.to_csv(target, sep="\t", index=False)
    print(method_cfg.to_string()); print(f"{len(out)} rows -> {target}")


if __name__ == "__main__":
    main()

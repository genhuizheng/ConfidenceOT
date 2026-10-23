"""Level 3 summaries (PLAN.md, Aggregation): cell -> family / clone -> mouse -> across mice, A vs B paired.

    python lineage_validation/level3_cell_lineage/summarize_level3.py --scores SCORES --ground-truth GT --out OUT

SCORES is score_level3.py's output (cells/<case_id>.tsv.gz, run_status.tsv); GT is ground_truth.py's (coverage.tsv).
Primary rows use evaluable cells; ambiguous cells get their own rows; nothing else is scored. Writes OUT/:
  family_summary.tsv   strategy x method x family: cells, coverage (c_ot, c_ind), D_ot, D_ind, d_min, delta_D,
                       delta_D_norm with a 95% bootstrap interval over cells, top-k mean distance, top-1 in the
                       tree-nearest set (OT and the independent expectation), log2 close-relative enrichment, flags
  site_summary.tsv     the same per metastatic site
  mouse_summary.tsv    strategy x method x mouse: mean over the mouse's families
  overall.tsv          strategy x method: mice with delta_D > 0, median of mouse means, family counts
  ab_paired.tsv        strategy x family: B - A on the same cells (delta_D_norm, top-1 rate, enrichment)
  ab_overall.tsv       strategy: families and mice where B > A, medians of B - A
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["c_ot", "c_ind", "D_ot", "D_ind", "d_min", "delta_D", "delta_D_norm", "top1_mean_d", "top5_mean_d", "top10_mean_d",
           "top1_in_nearest", "top1_in_nearest_ind_rate", "log2_close_enrichment", "M_ot", "M_ind", "M_ot_total", "M_ind_total",
           "top1_share_comparable", "top5_share_comparable", "top10_share_comparable"]
BOOT, SEED = 1000, 20261022


def boot_ci(x: np.ndarray, key: str) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng([SEED, zlib.crc32(key.encode())])  # deterministic across processes
    means = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["cells"] = len(g)
        for m in METRICS:
            if m in g:
                r[m] = float(pd.to_numeric(g[m], errors="coerce").mean())
        lo, hi = boot_ci(pd.to_numeric(g["delta_D_norm"], errors="coerce").to_numpy(float), "|".join(map(str, r.values())))
        r["delta_D_norm_lo"], r["delta_D_norm_hi"] = lo, hi
        r["share_cells_delta_D_pos"] = float((g["delta_D"] > 0).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    frames = [pd.read_csv(p, sep="\t", low_memory=False) for p in sorted((args.scores / "cells").glob("*.tsv.gz"))
              if not p.name.endswith("__targets.tsv.gz")]
    cells = pd.concat(frames, ignore_index=True)
    cells["dataset"] = cells["family"].str.split("_").str[0]
    cells["mouse"] = cells["family"].str.split("_").str[1]
    cover = pd.read_csv(args.ground_truth / "coverage.tsv", sep="\t")
    flags = cover.set_index("family")[["flag_high_missingness", "flag_low_tree_resolution", "n_evaluable", "metastatic_cells_in_family",
                                       "sources", "sources_comparable"]]
    args.out.mkdir(parents=True, exist_ok=True)
    ev = cells[cells["status"].eq("evaluable")]
    fam = summarise(ev, ["strategy", "method", "dataset", "mouse", "family"]).join(flags, on="family")
    fam.to_csv(args.out / "family_summary.tsv", sep="\t", index=False)
    summarise(ev, ["strategy", "method", "dataset", "mouse", "family", "site"]).to_csv(args.out / "site_summary.tsv", sep="\t", index=False)
    amb = cells[cells["status"].eq("ambiguous")]
    if len(amb):
        summarise(amb, ["strategy", "method", "family"]).to_csv(args.out / "family_summary_ambiguous.tsv", sep="\t", index=False)
    cols = ["delta_D", "delta_D_norm", "top1_in_nearest", "top1_in_nearest_ind_rate", "log2_close_enrichment", "c_ot", "c_ind"]
    mouse = fam.groupby(["strategy", "method", "dataset", "mouse"], sort=False).agg(
        families=("family", "size"), cells=("cells", "sum"), **{c: (c, "mean") for c in cols}).reset_index()
    mouse.to_csv(args.out / "mouse_summary.tsv", sep="\t", index=False)
    over = mouse.groupby(["strategy", "method"], sort=False).agg(
        mice=("mouse", "size"), mice_delta_D_pos=("delta_D", lambda x: int((x > 0).sum())),
        median_mouse_delta_D=("delta_D", "median"), median_mouse_delta_D_norm=("delta_D_norm", "median"),
        median_mouse_top1_in_nearest=("top1_in_nearest", "median"), median_mouse_top1_ind_rate=("top1_in_nearest_ind_rate", "median"),
        median_mouse_log2_enrichment=("log2_close_enrichment", "median")).reset_index()
    fam_counts = fam.groupby(["strategy", "method"], sort=False).agg(
        families=("family", "size"), families_delta_D_pos=("delta_D", lambda x: int((x > 0).sum())),
        families_ci_above_0=("delta_D_norm_lo", lambda x: int((x > 0).sum())),
        families_ci_below_0=("delta_D_norm_hi", lambda x: int((x < 0).sum()))).reset_index()
    over.merge(fam_counts, on=["strategy", "method"]).to_csv(args.out / "overall.tsv", sep="\t", index=False)
    # A vs B on the same cells: pair per family, then per mouse
    w = fam.pivot_table(index=["strategy", "dataset", "mouse", "family"], columns="method",
                        values=["delta_D_norm", "delta_D", "top1_in_nearest", "log2_close_enrichment", "cells"], aggfunc="first")
    w.columns = [f"{v}__{m}" for v, m in w.columns]
    w = w.reset_index()
    for v in ("delta_D_norm", "delta_D", "top1_in_nearest", "log2_close_enrichment"):
        w[f"{v}__B_minus_A"] = w[f"{v}__B_our_OT"] - w[f"{v}__A_standard_OT"]
    w.to_csv(args.out / "ab_paired.tsv", sep="\t", index=False)
    wm = w.groupby(["strategy", "dataset", "mouse"], sort=False)[["delta_D_norm__B_minus_A", "delta_D__B_minus_A"]].mean().reset_index()
    abo = []
    for s, g in w.groupby("strategy", sort=False):
        gm = wm[wm["strategy"].eq(s)]
        abo.append({"strategy": s, "families": len(g), "families_B_higher": int((g["delta_D_norm__B_minus_A"] > 0).sum()),
                    "families_A_higher": int((g["delta_D_norm__B_minus_A"] < 0).sum()),
                    "median_family_B_minus_A": float(g["delta_D_norm__B_minus_A"].median()),
                    "mice": len(gm), "mice_B_higher": int((gm["delta_D_norm__B_minus_A"] > 0).sum()),
                    "median_mouse_B_minus_A": float(gm["delta_D_norm__B_minus_A"].median())})
    pd.DataFrame(abo).to_csv(args.out / "ab_overall.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(over.merge(fam_counts, on=["strategy", "method"]).round(3).to_string(index=False))
    print(pd.DataFrame(abo).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

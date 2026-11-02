"""Worklist of the Splatter N = 5,000 benchmark, levels L0-L5 and the two ablations (runs on TACC; reads manifests).

    python realdata_benchmark/splatter_n5000/31_build_n5000_worklist.py --root N5000_ROOT --registry REGISTRY \
        --frozen-fits /scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/splatter/acc0.99

Configurations (SPLATTER_N5000_CONFIG_REGISTRY.tsv, the registry rows kept here, fixed before any L3-L5 fit): the four
R4 / R5 representatives of the joint benchmark's protocol (logcpm, ranknm256 production ties, ranknm1024 expected ties,
rankhvg2000 midrank; no gene scaling, cosine) at DS none and P10 seed 1 = 8 configurations, every level; chosen by the
protocol's rule, not by any result. The full 40-configuration grid at L0-L2 already exists (joint benchmark, Arm A)
and is reused, not refitted.
Rows:
  splatter_n5000            level x replicate (1-3) x case pair (4: Case 1, Case 2 source- and target-unmatched,
                            Case 3) x configuration; L0-L2 rows reuse the frozen fit (fit_out) when the regenerated
                            level is identical to the native one (NATIVE_IDENTITY.json), otherwise they are refitted
                            here; every row then gets IC-POT, Vanilla UOT and 23_row_extras.py
  splatter_n5000_condition  level x replicate x case pair: 24_condition_deg.py
Writes ROOT/TACC_N5000_WORKLIST.tsv, ROOT/SPLATTER_N5000_CONFIG_REGISTRY.tsv and ROOT/SPLATTER_N5000_SIMULATION_MANIFEST.tsv
(one row per level x replicate x pair with the generation record); refuses if the worklist exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
          "A4_perturb_only", "A5_imbalance_only"]
CONFIGS = [f"{enc}|{ds}|noscale|cos" for enc in ("logcpm", "ranknm256-production", "ranknm1024-expected", "rankhvg2000-midrank")
           for ds in ("none", "P10s1")]
COLUMNS = ["dataset", "config_id", "pair_index", "pair_id", "manifest", "manifest_ds", "run_root", "converted", "acceptance",
           "size", "level", "replicate", "case", "truth_dir", "fit_out", "out"]


def safe(c: str) -> str:
    return c.replace("|", "__").replace(":", "_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--frozen-fits", type=Path, required=True)
    p.add_argument("--replicates", type=int, nargs="*", default=[1, 2, 3])
    p.add_argument("--pilot", action="store_true",
                   help="TACC_N5000_PILOT_WORKLIST.tsv only: replicate 1, Case 2 source-unmatched pair, L2 (frozen fits reused, "
                        "native identity checked) and L4 (new fits with batch + perturbation), 8 configurations + the condition rows")
    a = p.parse_args()
    if a.pilot:
        a.replicates = [1]
    target = a.root / ("TACC_N5000_PILOT_WORKLIST.tsv" if a.pilot else "TACC_N5000_WORKLIST.tsv")
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    reg = pd.read_csv(a.registry, sep="\t")
    missing = [c for c in CONFIGS if c not in set(reg.config_id)]
    if missing:
        sys.exit(f"configurations not in the registry: {missing}")
    reg[reg.config_id.isin(CONFIGS)].assign(n5000_role="all levels").to_csv(a.root / "SPLATTER_N5000_CONFIG_REGISTRY.tsv", sep="\t", index=False)
    rows, sims, problems = [], [], []
    for level in LEVELS:
        for rep in a.replicates:
            cond = a.root / "conditions" / f"N5000" / f"rep{rep}" / level
            gen = a.root / "generated" / "N5000" / f"rep{rep}" / level
            if not (cond / "manifest.csv").exists():
                problems.append(f"no conditions for {level} rep{rep}"); continue
            m = pd.read_csv(cond / "manifest.csv")
            identical = None
            if level in ("L0_matched", "L1_depth", "L2_depth_dropout"):
                f = gen / "NATIVE_IDENTITY.json"
                identical = json.loads(f.read_text()).get("identical") if f.exists() else None
                if not identical:
                    problems.append(f"{level} rep{rep}: native identity {identical}; rows are refitted")
            for i, r in m.iterrows():
                sims.append({"level": level, "replicate": rep, "pair_id": r.pair_id, "benchmark_pair_id": r.benchmark_pair_id,
                             "case": r.case, "n_cells_per_side": r.n_cells, "native_identity": identical,
                             **{c: r[c] for c in m.columns if c.startswith("gen_")}})
                truth = cond / str(r.pair_id) / "truth"
                common = {"pair_index": i, "pair_id": r.pair_id, "manifest": str(cond / "manifest.csv"), "manifest_ds": "", "run_root": "",
                          "converted": "", "acceptance": 0.99, "size": 5000, "level": level, "replicate": rep, "case": r.case, "truth_dir": str(truth)}
                rows.append({**common, "dataset": "splatter_n5000_condition", "config_id": "", "fit_out": "",
                             "out": str(a.root / "rows" / "condition" / level / f"rep{rep}" / str(r.pair_id))})
                for cid in CONFIGS:
                    fit_out = ""
                    if identical:
                        fo = a.frozen_fits / safe(cid) / "N5000" / f"rep{rep}" / level / str(r.pair_id)
                        if (fo / "SUCCESS").exists():
                            fit_out = str(fo)
                        else:
                            problems.append(f"no frozen fit {fo}; refitted")
                    rows.append({**common, "dataset": "splatter_n5000", "config_id": cid, "fit_out": fit_out,
                                 "out": str(a.root / "rows" / safe(cid) / level / f"rep{rep}" / str(r.pair_id))})
    w = pd.DataFrame(rows, columns=COLUMNS)
    if a.pilot:
        w = w[w.level.isin(["L2_depth_dropout", "L4_batch_perturb"]) & (w.case == "Case2_source_unmatched")]
        problems = [x for x in problems if "rep1" in x and ("L2_depth" in x or "L4_" in x)]
    w.to_csv(target, sep="\t", index=False)
    if not a.pilot:
        pd.DataFrame(sims).to_csv(a.root / "SPLATTER_N5000_SIMULATION_MANIFEST.tsv", sep="\t", index=False)
    summary = w.assign(refit=(w.dataset == "splatter_n5000") & (w.fit_out.fillna("") == "")).groupby(["dataset", "level"]).agg(rows=("out", "size"), refit=("refit", "sum"))
    print(summary.to_string()); print(f"{len(w)} rows -> {target}")
    for x in problems:
        print("NOTE", x)


if __name__ == "__main__":
    main()

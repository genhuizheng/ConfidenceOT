"""Worklists of the joint benchmark's TACC fits (PROPOSED_TACC_JOBS.md). Runs on TACC; reads manifests only.

    python realdata_benchmark/11_build_worklists.py --registry REGISTRY --out-root ROOT \
        --splatter-root /scratch/10119/ghzheng/OT_project/benchmark_native \
        --ltl331-run-root /scratch/10119/ghzheng/OT_project/ltl331_confidenceot \
        --ltl331-converted /scratch/10119/ghzheng/OT_project/data/ltl331_GSE297328/converted

Design fixed by computational cost before any fit (2026-10-31; JOINT_BENCHMARK_PROTOCOL.md section 3):
  ltl331    all 7 adjacent pairs x (the 24 non-raw historical labels + every registered main configuration),
            acceptance 0.99 (the LTL331 factorial arguments, via 05_).
  splatter  Arm A conditions with n_cells 1,000 and 5,000, replicates 1-3, every technical level and every pair of the
            condition manifest (biological cases and directions), x every registered main configuration at acceptance
            0.99, and the eight R5 cosine representative configurations (logcpm, ranknm256 production, ranknm1024
            expected, rankhvg2000 midrank; DS none and frozen P10) at 0.90 for comparability with the original grid.
            n_cells 10,000 and replicates 4-5 are left out for cost; the historical labels are not refitted on Splatter
            (their 0.90 runs exist; ranknm256_noscale_ds_cos is refitted at 0.99 as a registered configuration, which
            also reproduces the archived 0.99 run).
The Splatter root is benchmark_native (the original benchmark: all_shared, population_lost, population_emerged,
populations_disjoint); benchmark_technical is the superseded thinning benchmark without the disjoint case (a first
worklist built from it on 2026-10-31 was set aside before any Splatter fit). The build stops if a condition manifest
lacks any of the four cases.
Rows are ordered largest-first so the long fits start early. Writes whichever of ROOT/worklist_ltl331.tsv and
ROOT/worklist_splatter.tsv is missing (an existing one is kept, never rebuilt) and ROOT/WORKLIST_SUMMARY_<dataset>.tsv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

COLUMNS = ["dataset", "config_id", "pair_index", "pair_id", "manifest", "manifest_ds", "run_root", "converted", "acceptance", "size", "out"]


def safe(config_id: str) -> str:
    return config_id.replace("|", "__").replace(":", "_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--splatter-root", type=Path, required=True)
    p.add_argument("--ltl331-run-root", type=Path, required=True)
    p.add_argument("--ltl331-converted", type=Path, required=True)
    a = p.parse_args()
    build = {d: not (a.out_root / f"worklist_{d}.tsv").exists() for d in ("ltl331", "splatter")}
    if not any(build.values()):
        sys.exit("both worklists exist; nothing rebuilt")
    print("building:", [d for d, b in build.items() if b])
    a.out_root.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(a.registry, sep="\t")
    main_cfg = reg[(reg.kind == "registered") & (reg.role == "main")].config_id.tolist()
    hist = reg[(reg.kind == "historical") & reg.in_new_grid.astype(bool)].config_id.tolist()
    r5 = reg[(reg.kind == "registered") & (reg.role == "main") & reg.contrasts.fillna("").str.contains("R5") & (reg.cost == "cos")].config_id.tolist()

    lt = []
    man = pd.read_csv(a.ltl331_run_root / "inputs" / "rawqc" / "pair_manifest_rawqc.csv")
    man = man[man.eligible.astype(bool)].reset_index(drop=True)
    for i, r in man.iterrows():
        for cid in hist + main_cfg:
            lt.append({"dataset": "ltl331", "config_id": cid, "pair_index": i, "pair_id": r.pair_id,
                       "manifest": str(a.ltl331_run_root / "inputs" / "rawqc" / "pair_manifest_rawqc.csv"),
                       "manifest_ds": str(a.ltl331_run_root / "inputs" / "ds" / "pair_manifest_downsampled.csv"),
                       "run_root": str(a.ltl331_run_root), "converted": str(a.ltl331_converted), "acceptance": 0.99,
                       "size": 0, "out": str(a.out_root / "ltl331" / safe(cid) / f"pair{i}")})
    lt = pd.DataFrame(lt, columns=COLUMNS)
    # the long pairs first: 8 -> 12, 12 -> 16 and 20 -> 22 weeks hold 12,694 / 19,386 cells on one side
    order = {3: 0, 4: 1, 6: 2, 1: 3, 2: 4, 0: 5, 5: 6}
    lt = lt.assign(_o=lt.pair_index.map(order)).sort_values(["_o", "config_id"]).drop(columns="_o")

    sp = []
    for mf in sorted(a.splatter_root.glob("conditions/*/*/*/manifest.csv")):
        m = pd.read_csv(mf)
        need = {"all_shared", "population_lost", "population_emerged", "populations_disjoint"}
        if not need <= set(m.biological_case.astype(str)):
            sys.exit(f"{mf} lacks {sorted(need - set(m.biological_case.astype(str)))}: wrong Splatter root?")
        n, rep = int(m.n_cells.iloc[0]), int(m.replicate.iloc[0])
        if n not in (1000, 5000) or rep not in (1, 2, 3):
            continue
        rel = mf.parent.relative_to(a.splatter_root / "conditions")
        mds = a.splatter_root / "conditions_ds" / rel / "pair_manifest_downsampled.csv"
        for i, r in m.iterrows():
            for acc, cfgs in ((0.99, main_cfg), (0.90, r5)):
                for cid in cfgs:
                    sp.append({"dataset": "splatter", "config_id": cid, "pair_index": i, "pair_id": r.pair_id, "manifest": str(mf),
                               "manifest_ds": str(mds) if mds.exists() else "", "run_root": "", "converted": "", "acceptance": acc,
                               "size": n, "out": str(a.out_root / "splatter" / f"acc{acc}" / safe(cid) / str(rel).replace("\\", "/") / str(r.pair_id))})
    sp = pd.DataFrame(sp, columns=COLUMNS).sort_values(["size", "config_id"], ascending=[False, True])
    for name, table, keys in (("ltl331", lt, ["dataset", "acceptance"]), ("splatter", sp, ["dataset", "acceptance", "size"])):
        if not build[name]:
            continue
        table.to_csv(a.out_root / f"worklist_{name}.tsv", sep="\t", index=False)
        summary = table.assign(acceptance=table.acceptance.astype(str)).groupby(keys).size().rename("rows").reset_index()
        summary.to_csv(a.out_root / f"WORKLIST_SUMMARY_{name}.tsv", sep="\t", index=False)
        print(summary.to_string(index=False))
    if build["splatter"]:
        pairs = sp.drop_duplicates(["manifest", "pair_index"])
        case = pairs.pair_id.str.extract(r"(all_shared|population_lost|population_emerged|populations_disjoint)")[0]
        print(f"splatter rows: {len(sp)}; condition-pairs per case: {case.value_counts().to_dict()}")
        print(f"splatter rows without a conditions_ds manifest: {int((sp.manifest_ds == '').sum())}")


if __name__ == "__main__":
    main()

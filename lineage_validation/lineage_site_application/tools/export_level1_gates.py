"""Read-only export of the Level 1 runs for Phase 2 (cell-level gate validation). Runs on TACC; re-runs nothing.

    python lineage_validation/lineage_site_application/tools/export_level1_gates.py --root LV_ROOT --out OUT

LV_ROOT is the lineage-validation root (inputs/, labels/, runs/acc0.99/). For every pair of
inputs/manifests/{kp_D_A,kp_D_B,kp_D_C3,quinn_D_B}_raw.csv x the 5 strategies x M4-E / M4-R, the run is loaded with
lv_common.load_run (the frozen protocol's S0 x T0, two-step extension, rho / tau, margins; the audited semantics).
Writes OUT/:
  cells_<block>.tsv.gz   one row per run x method x cell: side, cell, site, state (retained / rejected / excluded),
                         signed margin (inf if excluded at step 1), raw mass (row / column sum of P over S0 x T0)
                         and effective mass (the same sum of rho P tau: a rejected cell's raw mass is never counted)
  blocks_<block>.tsv.gz  per run x method: lineage x lineage and site x site blocks of P (raw) and rho P tau
                         (effective), with the labels of labels/cells.tsv.gz (applied after OT, for aggregation only)
  status.tsv             every run x method: ok / missing (with the reason)
  sides.tsv, cell_sites.tsv.gz   copies of inputs/ (the actual source / target sets of every comparison)
  SUCCESS
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LV = HERE.parents[1]
sys.path.insert(0, str(LV))
import lv_common as lv  # noqa: E402

BLOCKS = ("kp_D_A", "kp_D_B", "kp_D_C3", "quinn_D_B")


def block_rows(key: dict, P: np.ndarray, eff: np.ndarray, src_g: np.ndarray, tgt_g: np.ndarray, kind: str) -> list[dict]:
    gs, si = np.unique(src_g, return_inverse=True)
    gt, ti = np.unique(tgt_g, return_inverse=True)
    S = np.zeros((len(gs), len(P))); S[si, np.arange(len(P))] = 1
    T = np.zeros((P.shape[1], len(gt))); T[np.arange(P.shape[1]), ti] = 1
    raw, ef = S @ P @ T, S @ eff @ T
    return [{**key, "kind": kind, "source_group": gs[a], "target_group": gt[b], "raw_mass": float(raw[a, b]), "effective_mass": float(ef[a, b])}
            for a in range(len(gs)) for b in range(len(gt))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--blocks", default=",".join(BLOCKS))
    args = parser.parse_args()
    if (args.out / "SUCCESS").exists():
        sys.exit(f"{args.out}/SUCCESS exists")
    args.out.mkdir(parents=True, exist_ok=True)
    runs = args.root / "runs" / "acc0.99"
    labels = pd.read_csv(args.root / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    lab = {(d, m): g.set_index("cell")["lineage"] for (d, m), g in labels.groupby(["dataset", "mouse"])}
    status = []
    for block in args.blocks.split(","):
        manifest = pd.read_csv(args.root / "inputs" / "manifests" / f"{block}_raw.csv", dtype=str)
        cells_out, blocks_out = [], []
        for _, row in manifest.iterrows():
            pair, dataset, mouse = row["pair_id"], row["dataset_id"], str(row["patient_id"])
            L = lab.get((dataset, mouse), pd.Series(dtype=str))
            for strategy in lv.STRATEGIES:
                for method in lv.METHODS:
                    key = {"block": block, "dataset": dataset, "mouse": mouse, "pair": pair, "strategy": strategy, "method": method}
                    try:
                        r = lv.load_run(runs, strategy, block, pair, method)
                    except (lv.RunMissing, FileNotFoundError, ValueError) as error:
                        status.append({**key, "status": "missing", "reason": str(error)[:300]})
                        continue
                    P, rho, tau = r["P"], r["rho"], r["tau"]
                    eff = P * rho[:, None] * tau[None, :]
                    for side, ids, site, kept, margin, raw_m, eff_m in (
                            ("source", r["source"], r["source_site"], rho, r["source_margin"], P.sum(1), eff.sum(1)),
                            ("target", r["target"], r["target_site"], tau, r["target_margin"], P.sum(0), eff.sum(0))):
                        state = np.where(kept, "retained", np.where(np.isinf(margin), "excluded", "rejected"))
                        cells_out.append(pd.DataFrame({**key, "side": side, "cell": ids, "site": site, "state": state,
                                                       "signed_margin": margin, "raw_mass": raw_m, "effective_mass": eff_m}))
                    src_l = L.reindex(r["source"]).fillna("unknown").to_numpy(dtype=object)
                    tgt_l = L.reindex(r["target"]).fillna("unknown").to_numpy(dtype=object)
                    blocks_out += block_rows(key, P, eff, src_l, tgt_l, "lineage")
                    blocks_out += block_rows(key, P, eff, np.asarray(r["source_site"], dtype=object), np.asarray(r["target_site"], dtype=object), "site")
                    status.append({**key, "status": "ok", "reason": "", "n_source": len(r["source"]), "n_target": len(r["target"]),
                                   "total_mass": float(P.sum()), "effective_total": float(eff.sum())})
            print(f"{block} {pair} done", flush=True)
        if cells_out:
            pd.concat(cells_out, ignore_index=True).to_csv(args.out / f"cells_{block}.tsv.gz", sep="\t", index=False, compression="gzip")
            pd.DataFrame(blocks_out).to_csv(args.out / f"blocks_{block}.tsv.gz", sep="\t", index=False, compression="gzip")
    pd.DataFrame(status).to_csv(args.out / "status.tsv", sep="\t", index=False)
    for name in ("sides.tsv", "cell_sites.tsv.gz"):
        if (args.root / "inputs" / name).exists():
            shutil.copy2(args.root / "inputs" / name, args.out / name)
    st = pd.DataFrame(status)
    print(st.groupby(["block", "status"]).size().to_string())
    (args.out / "SUCCESS").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()

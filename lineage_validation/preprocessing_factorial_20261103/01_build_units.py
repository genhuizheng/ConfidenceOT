"""Units of the lineage preprocessing factorial (PLAN.md section 2) and the pooled depth targets (section 3). Runs on
TACC (reads the KP-Tracer and lineage-validation inputs there); writes only tables.

    python lineage_validation/preprocessing_factorial_20261103/01_build_units.py --root ROOT

Units:
  L1   one per row of the KP-Tracer equalised pair manifest (KP/downsampled/pair_manifest_downsampled.csv, the 104
       eligible pairs the earlier KP run fitted, in its row order, so --index and every seed equal that run's); truth
       columns (same_met_family, families) from the prepared manifest by pair_id.
  L23  one per row of lineage_validation/level2_pairwise_ot/inputs/cases.tsv (9 cases), loaded at run time exactly as
       run_pairwise.py loads them (files_for / load_side: the frozen equalised files' malignant cells, raw counts of
       the same cells, sites pooled).
Cells are the equalised files' malignant cells in every configuration; only their counts change with the
configuration (original or equalised here).
Depth targets (DS_TARGETS.json): P10 / P25 / P50 of the original total counts of the analysed cells, pooled over every
KP-Tracer tumour of the equalised manifest, and for Quinn per experiment over the case's LL and target cells.
Writes ROOT/UNITS.tsv and ROOT/DS_TARGETS.json (refuses if either exists).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
L2 = REPO / "lineage_validation" / "level2_pairwise_ot"
QUANTILES = {"P10": 0.10, "P25": 0.25, "P50": 0.50}


def run_pairwise():
    """level2_pairwise_ot/run_pairwise.py as a module: its files_for / load_side / load_part and TACC paths."""
    sys.path.insert(0, str(L2))
    spec = importlib.util.spec_from_file_location("l2_run_pairwise", L2 / "run_pairwise.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def totals(adata) -> np.ndarray:
    return np.asarray(adata.X.sum(axis=1)).ravel()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--cases", type=Path, default=L2 / "inputs" / "cases.tsv")
    a = p.parse_args()
    for f in (a.root / "UNITS.tsv", a.root / "DS_TARGETS.json"):
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    a.root.mkdir(parents=True, exist_ok=True)
    rp = run_pairwise()
    KP = rp.KP
    ds_manifest = KP / "downsampled" / "pair_manifest_downsampled.csv"
    ds = pd.read_csv(ds_manifest)
    prep = pd.read_csv(KP / "prepared" / "pair_manifest_eligible.csv")
    truth = prep.set_index("pair_id")[["same_met_family", "source_met_family", "target_met_family"]]
    units, depth = [], {}
    for i, r in ds.iterrows():
        if "eligible" in r and not bool(r["eligible"]):
            continue
        part = lambda t: {"ds": (rp.side_paths(r, "source" if t == r["source_sample"] else "target"), t),
                          "raw": ([str(KP / "prepared" / "h5ad" / f"{t}.h5ad")], t), "site": t, "site_from_obs": False}
        n = {}
        for side in ("source", "target"):
            t = str(r[f"{side}_sample"])
            raw, dsx, _ = rp.load_part(part(t), side)
            n[side] = raw.n_obs
            depth.setdefault(("KPTracer", t), totals(raw))
        tr = truth.loc[r["pair_id"]] if r["pair_id"] in truth.index else pd.Series(dtype=object)
        units.append({"unit_id": f"L1__{r['pair_id']}", "level": "L1", "dataset": "KPTracer", "mouse": str(r["patient_id"]),
                      "manifest": str(ds_manifest), "index": int(i), "pair_or_case": r["pair_id"],
                      "source": r["source_sample"], "target": r["target_sample"],
                      "same_met_family": tr.get("same_met_family", np.nan), "source_met_family": tr.get("source_met_family", ""),
                      "target_met_family": tr.get("target_met_family", ""), "source_cells": n["source"], "target_cells": n["target"]})
        print("L1", r["pair_id"], n, flush=True)
    cases = pd.read_csv(a.cases, sep="\t", dtype=str)
    for _, c in cases.iterrows():
        spec = rp.files_for(c); left = dict(spec["left_out"])
        (s_raw, s_ds, s_site), (t_raw, t_ds, t_site) = rp.load_side(spec["source"], "source", left), rp.load_side(spec["target"], "target", left)
        if c["dataset"] == "Quinn2021":
            depth[("Quinn2021:" + c["mouse"], c["case_id"])] = np.concatenate([totals(s_raw), totals(t_raw)])
        units.append({"unit_id": f"L23__{c['case_id']}", "level": "L23", "dataset": c["dataset"], "mouse": c["mouse"],
                      "manifest": str(a.cases), "index": int(c["index"]), "pair_or_case": c["case_id"],
                      "source": c["source_site"], "target": c["target_sites"], "same_met_family": np.nan,
                      "source_met_family": c["families"], "target_met_family": json.dumps(left),
                      "source_cells": int(s_raw.n_obs), "target_cells": int(t_raw.n_obs)})
        print("L23", c["case_id"], s_raw.n_obs, t_raw.n_obs, flush=True)
    u = pd.DataFrame(units)
    u["size"] = u.source_cells.astype(float) * u.target_cells.astype(float)
    u.to_csv(a.root / "UNITS.tsv", sep="\t", index=False)
    kp = np.concatenate([v for (d, _), v in depth.items() if d == "KPTracer"])
    targets = {"KPTracer": {k: int(np.quantile(kp, q)) for k, q in QUANTILES.items()}, "KPTracer_cells": int(kp.size)}
    for (d, _), v in depth.items():
        if d.startswith("Quinn2021:"):
            targets[d] = {k: int(np.quantile(v, q)) for k, q in QUANTILES.items()}; targets[d + "_cells"] = int(v.size)
    (a.root / "DS_TARGETS.json").write_text(json.dumps(targets, indent=1))
    print(u.groupby("level").agg(units=("unit_id", "size"), median_size=("size", "median"), max_size=("size", "max")).to_string())
    print(json.dumps(targets))


if __name__ == "__main__":
    main()

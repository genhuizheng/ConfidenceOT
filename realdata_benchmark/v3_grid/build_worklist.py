"""Worklist of the frozen-calibration-v3 grid for one dataset (v3_common.py): one row per unit x encoding key (37 keys,
8 configurations each, 296 configurations). Reads manifests only.

    python realdata_benchmark/v3_grid/build_worklist.py --dataset splatter --root ROOT --splatter-root SPLATTER_ROOT \
        [--replicates 1 2 3] [--levels L1_depth ...]
    python realdata_benchmark/v3_grid/build_worklist.py --dataset ltl331 --root ROOT --ltl331-run-root RUN_ROOT
    python realdata_benchmark/v3_grid/build_worklist.py --dataset kp --root ROOT --kp-units UNITS.tsv
    add --pilot N to also write PILOT_WORKLIST.tsv: N rows spread over the unit-size range and the encodings

  splatter  SPLATTER_ROOT/conditions/N5000/rep<r>/<level>/manifest.csv (20_ --param-set splatter_default --pairing
            paired, 21_), levels L1-L5 (the local grid's), every pair of the manifest (Case 1, Case 2 both directions,
            Case 3); truth <condition>/<pair>/truth/population_truth.tsv
  ltl331    the 7 adjacent pairs of RUN_ROOT/inputs/rawqc/pair_manifest_rawqc.csv (eligible), original counts
  kp        every unit of the lineage factorial's UNITS.tsv (01_build_units.py: KP L1 pairs, L2 / L3 cases)
Rows are ordered largest unit first (memory and the long tail). Writes ROOT/WORKLIST.tsv, ROOT/FROZEN_CALIBRATION_V3.json
(refuses to overwrite a different worklist; an identical one is left as it is).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_common as v3  # noqa: E402

LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
COLUMNS = ["dataset", "unit_id", "key", "spec", "size", "index", "pair_id", "manifest", "truth", "level", "replicate", "case",
           "run_root", "units", "mouse", "out"]


def safe(x: str) -> str:
    return str(x).replace("|", "__").replace(":", "_").replace("/", "_")


def units(a) -> list[dict]:
    out = []
    if a.dataset == "splatter":
        for rep in a.replicates:
            for lv in a.levels:
                cond = a.splatter_root / "conditions" / a.conditions_subdir / f"rep{rep}" / lv
                if not (cond / "manifest.csv").exists():
                    sys.exit(f"no conditions for {lv} rep{rep}: {cond}")
                m = pd.read_csv(cond / "manifest.csv")
                for i, r in m.iterrows():
                    truth = cond / str(r.pair_id) / "truth" / "population_truth.tsv"
                    out.append({"unit_id": f"{lv}__rep{rep}__{r.pair_id}", "size": 5000 * 5000, "index": i, "pair_id": r.pair_id,
                                "manifest": str(cond / "manifest.csv"), "truth": str(truth), "level": lv, "replicate": rep, "case": r.case})
    elif a.dataset == "ltl331":
        import anndata as ad
        import rb_common as rb
        m = rb.ltl_pairs(a.ltl331_run_root, False)
        for i, r in m.iterrows():
            n = [ad.read_h5ad(rb.local(r[f"{s}_h5ad"], a.ltl331_run_root), backed="r").n_obs for s in ("source", "target")]
            out.append({"unit_id": str(r.pair_id), "size": n[0] * n[1], "index": i, "pair_id": r.pair_id, "run_root": str(a.ltl331_run_root)})
    else:
        u = pd.read_csv(a.kp_units, sep="\t", dtype={"mouse": str})
        for _, r in u.iterrows():
            out.append({"unit_id": r.unit_id, "size": int(r["size"]), "index": int(r["index"]), "pair_id": r.pair_or_case,
                        "units": str(a.kp_units), "mouse": r.mouse, "level": r.level})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=("splatter", "ltl331", "kp"), required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--splatter-root", type=Path)
    p.add_argument("--conditions-subdir", default="N5000", help="splatter: conditions/<this>/rep<r>/<level> (empty for the local N = 200 root)")
    p.add_argument("--replicates", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--levels", nargs="+", default=LEVELS)
    p.add_argument("--ltl331-run-root", type=Path)
    p.add_argument("--kp-units", type=Path)
    p.add_argument("--pilot", type=int, default=0)
    a = p.parse_args()
    need = {"splatter": "splatter_root", "ltl331": "ltl331_run_root", "kp": "kp_units"}[a.dataset]
    if getattr(a, need) is None:
        sys.exit(f"--{need.replace('_', '-')} is required for --dataset {a.dataset}")
    keys = v3.grid_keys()
    rows = []
    for u in units(a):
        for s in keys:
            k = v3.key_name(s)
            rows.append({"dataset": a.dataset, **u, "key": k, "spec": json.dumps({x: (None if isinstance(y, float) and np.isnan(y) else y)
                                                                                    for x, y in s.items()}),
                         "out": str(a.root / "rows" / safe(u["unit_id"]) / safe(k))})
    t = pd.DataFrame(rows).reindex(columns=COLUMNS).sort_values("size", ascending=False, kind="stable").reset_index(drop=True)
    a.root.mkdir(parents=True, exist_ok=True)
    dest = a.root / "WORKLIST.tsv"
    if dest.exists():
        old = pd.read_csv(dest, sep="\t", dtype=str).fillna("")
        if not old.equals(t.astype(str).replace({"nan": "", "None": ""}).fillna("")):
            sys.exit(f"{dest} exists and differs; not overwritten (use a new root)")
    else:
        t.to_csv(dest, sep="\t", index=False)
    (a.root / "FROZEN_CALIBRATION_V3.json").write_text(json.dumps(v3.frozen_record(), indent=1))
    if a.pilot:
        # spread over sizes and encodings: every k-th unit, cycling through the keys
        us = t.drop_duplicates("unit_id").unit_id.tolist(); ks = [v3.key_name(s) for s in keys]
        pick = [(us[int(round(j * (len(us) - 1) / max(1, a.pilot - 1)))], ks[j % len(ks)]) for j in range(a.pilot)]
        pl = t.set_index(["unit_id", "key"]).loc[list(dict.fromkeys(pick))].reset_index()[COLUMNS]
        pl.to_csv(a.root / "PILOT_WORKLIST.tsv", sep="\t", index=False)
        print(f"pilot: {len(pl)} rows -> {a.root / 'PILOT_WORKLIST.tsv'}")
    n_units = t.unit_id.nunique()
    print(f"{a.dataset}: {n_units} units x {len(keys)} keys = {len(t)} rows, {len(t) * 8} fits ({len(keys) * 8} configurations) -> {dest}")


if __name__ == "__main__":
    main()

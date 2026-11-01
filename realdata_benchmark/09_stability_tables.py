"""RANK_K_HVG_RESULTS.tsv and DOWNSAMPLING_ROBUSTNESS.tsv from the representation diagnostics (no OT).

    python realdata_benchmark/09_stability_tables.py --diag-dir OUT_DIR

Reads REPRESENTATION_DIAGNOSTICS.tsv and source/diag_pair<i>.npz (04_) and, per pair:
  RANK_K_HVG_RESULTS   contrasts R1 (Top-K sweep), R2 (tie rule) and R3 (Rank-HVG against Rank-K and logcpm): every
                       configuration's diagnostics, plus its stability against a reference configuration of the same
                       contrast on the same cells: kNN-gate agreement (kappa, both sides), PCA subspace similarity of the
                       joint coordinates, Spearman correlation of the sampled costs. References: R1 / R3 -> ranknm256
                       expected (the production K with gene-order-free ties); R2 -> the production tie rule at the same K
                       and DS.
  DOWNSAMPLING_ROBUSTNESS  contrast R4: per representative and DS level, the diagnostics, the agreement between DS seeds
                       (seed 1 against seeds 2 and 3, and for P10 against the frozen draw: kappa, subspace, cost Spearman),
                       and the agreement of each DS level with DS none on the same cells.
Every number is a representation property before OT; gate here means the OT-free kNN gate of 04_.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import rb_common as rb


def compare(store, a, b):
    if f"{a}::coords" not in store or f"{b}::coords" not in store:
        return {}
    return {"source_gate_kappa": rb.kappa(store[f"{a}::source_gate"], store[f"{b}::source_gate"]),
            "target_gate_kappa": rb.kappa(store[f"{a}::target_gate"], store[f"{b}::target_gate"]),
            "subspace_similarity": rb.subspace_similarity(store[f"{a}::coords"].astype(np.float32), store[f"{b}::coords"].astype(np.float32)),
            "cost_spearman": float(spearmanr(store[f"{a}::costs"], store[f"{b}::costs"])[0])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diag-dir", type=Path, required=True)
    a = p.parse_args()
    out_rk, out_ds = a.diag_dir / "RANK_K_HVG_RESULTS.tsv", a.diag_dir / "DOWNSAMPLING_ROBUSTNESS.tsv"
    for f in (out_rk, out_ds):
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    diag_file = a.diag_dir / "REPRESENTATION_DIAGNOSTICS.tsv"
    if not diag_file.exists():
        diag_file = a.diag_dir / "REPRESENTATION_DIAGNOSTICS.partial.tsv"
    diag = pd.read_csv(diag_file, sep="\t")
    rk, dsr = [], []
    for index, g in diag.groupby("pair_index"):
        npz = a.diag_dir / "source" / f"diag_pair{index}.npz"
        if not npz.exists():
            continue
        store = dict(np.load(npz))
        for _, r in g.iterrows():
            cid, contrasts = r.config_id, str(r.contrasts)
            enc, ds_part = cid.split("|")[0], "|".join(cid.split("|")[1:])
            if any(c in contrasts for c in ("R1", "R2", "R3")):
                if "R2" in contrasts and enc.startswith("ranknm"):
                    ref = f"{enc.split('-')[0]}-production|{ds_part}"
                else:
                    ref = "ranknm256-expected|none|noscale|cos"
                rk.append({**r.to_dict(), "reference": ref, **{f"vs_reference_{k}": v for k, v in compare(store, cid, ref).items()}})
            if "R4" in contrasts:
                rep, ds_cost = enc, cid.split("|")
                level, rest = ds_cost[1], "|".join(ds_cost[2:])
                row = {**r.to_dict(), "representative": rep, "ds_level": level}
                row.update({f"vs_ds_none_{k}": v for k, v in compare(store, cid, f"{rep}|none|{rest}").items()})
                if level.endswith("s1"):
                    stem = level[:-2]
                    for other in ("s2", "s3") + (("sfrozen",) if stem == "P10" else ()):
                        row.update({f"vs_{stem}{other}_{k}": v for k, v in compare(store, cid, f"{rep}|{stem}{other}|{rest}").items()})
                dsr.append(row)
    pd.DataFrame(rk).to_csv(out_rk, sep="\t", index=False)
    pd.DataFrame(dsr).to_csv(out_ds, sep="\t", index=False)
    print(f"{len(rk)} rows -> {out_rk}; {len(dsr)} rows -> {out_ds}")


if __name__ == "__main__":
    main()

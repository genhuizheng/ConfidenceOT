"""RNA vs RNA + CNV on the lineage units, read-only on finished v3-grid rows (user 2026-11-23: no chance baseline; RNA and
RNA + CNV side by side). Runs where the rows are (TACC); writes small tables.

    python lineage_validation/rna_cnv_clade_20261123/10_score_arms.py --arm old=V3_2026-11-04_KP_ROOT \
        --arm rna=.../kp_rna --arm rna_cnv=.../kp_rna_cnv [--arm rna_cnv_b2=...] \
        --bundles CRISPR_BUNDLES --clades CELLS_TSV_GZ --out OUT

Units: the 49 units that have a bundle (40 KP same-family pairs, 5 KP pooled cases, 4 Quinn). Configurations: l5best
(ranknm256-midrank|none|scale|cos|pca50) and logcpm_pca (logcpm|none|noscale|sqeuc|pca30). Per unit x configuration x
arm, M4-E, from <root>/rows/<unit>/<key>/<config>/{cell_confidence.csv.gz, coupling_m4e.npz}:
  gates       source / target retained fraction
  same_clade  the clade analysis' own-clade mass (01_clade_overlap.py): on the retained block (renormalised), the share
              of retained-target mass with a clade that lands on source cells of the target's clade. Clades: KP A2 clades
              (CELLS_TSV_GZ: cell, family, clade; minor / unassigned = no clade), Quinn LineageGroup (bundle).
  ev_share    retained-target mass with a clade
No chance baseline (user). Environment check, every arm against --reference (default old): per side the share of cells
with the same M4-E retained flag, and the largest |decision_cost| difference.
Writes OUT/units.tsv, OUT/gate_agreement.tsv. Refuses to overwrite.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = {"l5best": ("ranknm256-midrank__none", "ranknm256-midrank__none__scale__cos__pca50"),
           "logcpm_pca": ("logcpm__none", "logcpm__none__noscale__sqeuc__pca30")}
NO_CLADE = {"minor", "unassigned", "nan", "", "None"}


def clade_of(b: dict, kp_map: dict, ids: np.ndarray, side: str) -> np.ndarray:
    if b["dataset"] == "KPTracer":
        return np.array([kp_map.get(i, "") for i in ids], dtype=object)
    lab = pd.Series(np.asarray(b[f"{side}_lineage"]).astype(str), index=np.asarray(b[f"{side}_ids"]).astype(str))
    lab = lab[~lab.index.duplicated()]
    out = lab.reindex(ids).fillna("").astype(str).to_numpy(dtype=object)
    out[np.isin(out, list(NO_CLADE))] = ""
    return out


def load(d: Path):
    g = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "retained", "decision_cost"],
                    dtype={"observation_id": str})
    g = g[g.method == "M4-E"]
    z = np.load(d / "coupling_m4e.npz", allow_pickle=True)
    return g, z["coupling"].astype(np.float64), z["source_ids"].astype(str), z["target_ids"].astype(str)


def same_clade(g, P, sid, tid, cs_of, ct_of):
    gs = g[g.side == "source"].drop_duplicates("observation_id").set_index("observation_id")
    gt = g[g.side == "target"].drop_duplicates("observation_id").set_index("observation_id")
    s = gs.retained.reindex(sid).fillna(False).to_numpy(bool); t = gt.retained.reindex(tid).fillna(False).to_numpy(bool)
    rec = {"source_retained": float(s.mean()), "target_retained": float(t.mean()), "n_source": len(sid), "n_target": len(tid)}
    if not s.any() or not t.any():
        return rec | {"status": "empty retained block"}
    F = P[np.ix_(s, t)]
    if F.sum() <= 0:
        return rec | {"status": "zero mass"}
    F = F / F.sum()
    cs, ct = cs_of[s], ct_of[t]
    w = F.sum(0); ev = np.array([bool(c) for c in ct])
    rec["ev_share"] = float(w[ev].sum())
    if not ev.any() or w[ev].sum() <= 0:
        return rec | {"status": "no evaluable target"}
    O = sum(F[np.ix_(cs == c, ct == c)].sum() for c in set(ct[ev]))
    return rec | {"status": "ok", "same_clade": float(O / w[ev].sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=ROOT (a v3-grid root with rows/)")
    ap.add_argument("--reference", default="old")
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--clades", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.out / "units.tsv").exists():
        raise SystemExit(f"{a.out}/units.tsv exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    arms = dict(x.split("=", 1) for x in a.arm)
    c = pd.read_csv(a.clades, sep="\t", usecols=["cell", "family", "clade"], dtype=str)
    ok = ~c.clade.fillna("").isin(NO_CLADE)
    kp_map = dict(zip(c.cell[ok], c.family[ok] + "/" + c.clade[ok]))
    rows, agree = [], []
    for f in sorted(a.bundles.glob("*.pkl")):
        b = pickle.load(open(f, "rb")); u = f.stem
        for cname, (key, cid) in CONFIGS.items():
            gates = {}
            for arm, root in arms.items():
                d = Path(root) / "rows" / u / key / cid
                base = {"unit": u, "dataset": b["dataset"], "level": u.split("__")[0], "mouse": str(b["mouse"]),
                        "config": cname, "arm": arm}
                if not (d / "coupling_m4e.npz").exists():
                    rows.append(base | {"status": "missing"}); continue
                g, P, sid, tid = load(d)
                gates[arm] = g
                rows.append(base | same_clade(g, P, sid, tid, clade_of(b, kp_map, sid, "source"), clade_of(b, kp_map, tid, "target")))
            ref = gates.get(a.reference)
            for arm, g in gates.items():
                if arm == a.reference or ref is None:
                    continue
                for side in ("source", "target"):
                    x = ref[ref.side == side].set_index("observation_id"); y = g[g.side == side].set_index("observation_id")
                    common = x.index.intersection(y.index)
                    agree.append({"unit": u, "config": cname, "arm": arm, "side": side, "cells_ref": len(x), "cells_arm": len(y),
                                  "common": len(common),
                                  "same_flag": float((x.retained.reindex(common) == y.retained.reindex(common)).mean()) if len(common) else np.nan,
                                  "max_abs_decision_cost_diff": float((x.decision_cost.reindex(common) - y.decision_cost.reindex(common)).abs().max())
                                  if len(common) else np.nan})
        print(u, flush=True)
    pd.DataFrame(rows).to_csv(a.out / "units.tsv", sep="\t", index=False)
    pd.DataFrame(agree).to_csv(a.out / "gate_agreement.tsv", sep="\t", index=False)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()

"""Does RNA-OT pair primary and metastatic cells within the same CRISPR clade? (2026-11-19, read-only, no refit)

    python lineage_validation/clade_overlap_20261119/01_clade_overlap.py \
        --root results/rna_lineage_discordance_20261110 \
        --bundles results/crispr_matrix_benchmark_20261108/bundles \
        --out results/clade_overlap_20261119

Truth (CRISPR, never an OT input for RNA-OT):
  KP     the all-eligible-cell A2 clades (state_lineage_ext/all_cells__original/cells.tsv.gz: UPGMA on 1 - s within the
         family, cut 0.5), keyed (family, clade). "minor" (pooled clusters < 20 cells), "unassigned" and cells without
         a row have no clade and are never counted as a same-clade match.
  Quinn  the authors' LineageGroup (the bundle's source_lineage / target_lineage); missing = no clade.

Couplings (existing): RNA-OT = inputs_A/v3_grid_20261104/kp/rows/<unit>/<config>/coupling_m4e.npz with that row's
cell_confidence.csv.gz (M4-E gates), configs l5best (ranknm256-midrank|none|scale|cos|pca50) and logcpm_pca
(logcpm|none|noscale|sqeuc|pca30). CRISPR-OT = crispr_ot/<unit>/coupling_m4e.npz + gates.tsv.gz: its cost is built
from the same edits as the clades, so it is a circular diagnostic, never validation.

Per unit and coupling, on the M4-E retained block (retained sources x retained targets, renormalised to mass 1):
  evaluable targets   retained targets with a clade; ev_share = their share of retained-target mass
  O        overlap: share of evaluable-target mass that lands on a source cell of the target's own clade
  I_ret    independent coupling with the block's marginals: sum_j w_j r_c(j), r_c = row-mass share of clade c
  I_all    no gate: sum_j w_j pi_c(j), pi_c = share of ALL analysed source cells in clade c (uniform)
  U        upper bound: evaluable-target mass whose clade has >= 1 retained source cell
  gate_gain = I_ret - I_all, coupling_gain = O - I_ret, O_norm = (O - I_ret) / (U - I_ret) when U - I_ret > 0.01
  absent   share of analysed targets whose clade has no analysed source cell; their M4-E rejected fraction beside the
           rejected fraction of targets whose clade is present (the gate should reject what has no counterpart)
  n_target_clades     clades among evaluable targets (1 = O is decided by the gate alone)
Aggregation: unit -> mouse mean (KP mouse, Quinn experiment); no cell is a replicate. Rules fixed here, before any
value was read. Writes OUT/units.tsv and OUT/mice.tsv; refuses to overwrite.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

CONFIGS = {"l5best": "ranknm256-midrank__none/ranknm256-midrank__none__scale__cos__pca50",
           "logcpm_pca": "logcpm__none/logcpm__none__noscale__sqeuc__pca30"}
NO_CLADE = {"minor", "unassigned", "nan", "", "None"}


def clade_maps(root: Path) -> dict:
    c = pd.read_csv(root / "state_lineage_ext" / "all_cells__original" / "cells.tsv.gz", sep="\t",
                    usecols=["cell", "family", "clade"], dtype=str)
    ok = ~c["clade"].fillna("").isin(NO_CLADE)
    return dict(zip(c.loc[ok, "cell"], c.loc[ok, "family"] + "/" + c.loc[ok, "clade"]))


def unit_clades(b: dict, kp_map: dict, ids: np.ndarray, side: str) -> np.ndarray:
    if b["dataset"] == "KPTracer":
        return np.array([kp_map.get(i, "") for i in ids.astype(str)], dtype=object)
    lab = pd.Series(np.asarray(b[f"{side}_lineage"]).astype(str), index=np.asarray(b[f"{side}_ids"]).astype(str))
    lab = lab[~lab.index.duplicated()]
    out = lab.reindex(ids.astype(str)).fillna("").astype(str).to_numpy(dtype=object)
    out[np.isin(out, list(NO_CLADE))] = ""
    return out


def score(P: np.ndarray, src_ids, tgt_ids, s_ret: np.ndarray, t_ret: np.ndarray, cs: np.ndarray, ct: np.ndarray):
    """cs / ct: clades of all analysed sources / targets (in coupling order); s_ret / t_ret: M4-E retained masks."""
    row = {"n_source": len(src_ids), "n_target": len(tgt_ids),
           "source_retained": float(s_ret.mean()), "target_retained": float(t_ret.mean())}
    # absent clades and the gate, on all analysed cells
    src_set = {c for c in cs if c}
    has = np.array([bool(c) for c in ct]); present = np.array([c in src_set for c in ct])
    row["target_clade_share"] = float(has.mean())
    row["absent_share"] = float((has & ~present).sum() / max(has.sum(), 1))
    row["rej_absent"] = float((~t_ret[has & ~present]).mean()) if (has & ~present).any() else np.nan
    row["rej_present"] = float((~t_ret[has & present]).mean()) if (has & present).any() else np.nan
    if s_ret.sum() == 0 or t_ret.sum() == 0:
        return row | {"status": "empty retained block"}
    F = P[np.ix_(s_ret, t_ret)].astype(np.float64)
    if F.sum() <= 0:
        return row | {"status": "zero mass"}
    F /= F.sum()
    csr, ctr = cs[s_ret], ct[t_ret]
    w = F.sum(0)
    ev = np.array([bool(c) for c in ctr])
    row["ev_share"] = float(w[ev].sum())
    if not ev.any() or w[ev].sum() <= 0:
        return row | {"status": "no evaluable target"}
    wn = w[ev] / w[ev].sum()
    rmass = pd.Series(F.sum(1)).groupby(csr).sum()
    rmass = rmass / rmass.sum()
    pi = pd.Series(1.0, index=range(len(cs))).groupby(cs).sum() / len(cs)
    ret_set = {c for c in csr if c}
    O = 0.0
    Fe = F[:, ev]
    for c in set(ctr[ev]):
        cols = ctr[ev] == c
        O += Fe[np.ix_(csr == c, cols)].sum()
    O /= w[ev].sum()
    I_ret = float(sum(wn[k] * rmass.get(c, 0.0) for k, c in enumerate(ctr[ev])))
    I_all = float(sum(wn[k] * pi.get(c, 0.0) for k, c in enumerate(ctr[ev])))
    U = float(sum(wn[k] for k, c in enumerate(ctr[ev]) if c in ret_set))
    row |= {"status": "ok", "O": float(O), "I_ret": I_ret, "I_all": I_all, "U": U,
            "gate_gain": I_ret - I_all, "coupling_gain": float(O) - I_ret,
            "O_norm": (float(O) - I_ret) / (U - I_ret) if U - I_ret > 0.01 else np.nan,
            "n_target_clades": int(len(set(ctr[ev]))), "n_source_clades": int(len(ret_set))}
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if (a.out / "units.tsv").exists():
        raise SystemExit(f"{a.out}/units.tsv exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    kp_map = clade_maps(a.root)
    rows_dir = a.root / "inputs_A" / "v3_grid_20261104" / "kp" / "rows"
    rows = []
    for udir in sorted(rows_dir.iterdir()):
        u = udir.name
        b = pickle.load(open(a.bundles / f"{u}.pkl", "rb"))
        same_family = np.nan
        if b["dataset"] == "KPTracer":
            sl = pd.Series(np.asarray(b["source_lineage"]).astype(str)).mode()
            same_family = float((np.asarray(b["target_lineage"]).astype(str) == (sl.iloc[0] if len(sl) else "")).mean())
        base = {"unit": u, "level": u.split("__")[0], "dataset": b["dataset"], "mouse": str(b["mouse"]),
                "target_same_family_share": same_family}
        couplings = [(k, udir / v, "rna") for k, v in CONFIGS.items()] + [("crispr_ot", a.root / "crispr_ot" / u, "crispr")]
        for name, d, kind in couplings:
            if not (d / "coupling_m4e.npz").exists():
                rows.append(base | {"coupling": name, "status": "missing"}); continue
            z = np.load(d / "coupling_m4e.npz", allow_pickle=True)
            P, sid, tid = z["coupling"], z["source_ids"].astype(str), z["target_ids"].astype(str)
            if kind == "rna":
                g = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "retained"],
                                dtype={"observation_id": str})
            else:
                g = pd.read_csv(d / "gates.tsv.gz", sep="\t", dtype={"observation_id": str})
            g = g[g["method"] == "M4-E"]
            gs = g[g.side == "source"].drop_duplicates("observation_id").set_index("observation_id")["retained"]
            gt = g[g.side == "target"].drop_duplicates("observation_id").set_index("observation_id")["retained"]
            s_ret = gs.reindex(sid).fillna(False).astype(bool).to_numpy()
            t_ret = gt.reindex(tid).fillna(False).astype(bool).to_numpy()
            cs, ct = unit_clades(b, kp_map, sid, "source"), unit_clades(b, kp_map, tid, "target")
            rows.append(base | {"coupling": name} | score(P, sid, tid, s_ret, t_ret, cs, ct))
        print(u, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(a.out / "units.tsv", sep="\t", index=False)
    ok = t[t.status == "ok"]
    cols = ["O", "I_ret", "I_all", "U", "gate_gain", "coupling_gain", "O_norm", "ev_share", "absent_share",
            "rej_absent", "rej_present", "source_retained", "target_retained"]
    m = ok.groupby(["coupling", "dataset", "level", "mouse"])[cols].mean()
    m["units"] = ok.groupby(["coupling", "dataset", "level", "mouse"]).size()
    m.reset_index().to_csv(a.out / "mice.tsv", sep="\t", index=False)
    print(f"{len(t)} rows, {len(ok)} ok -> {a.out}")


if __name__ == "__main__":
    main()

"""PLAN_v2.md amendment A4: RNA-OT vs CRISPR-OT conditional coupling disagreement on a common support (M4-E).

    python lineage_validation/rna_lineage_discordance_20261110/75_common_support.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles

Per unit and RNA configuration:
  common block  sources retained by both gates x targets retained by both gates; pairs restricted to evaluable cell
                pairs (10_crispr_ot.UnitDistance D finite; Quinn cells of different LineageGroups count as evaluable,
                D = 1, an observed difference)
  each coupling restricted to that block and every column renormalised to 1
  per common target: total-variation distance, top-k source overlap (k = min(15, common sources)) and its chance
                level k / common sources
  per unit: common sources / targets, each coupling's mass kept on the block before renormalisation (share of its
                retained-block mass), share of evaluable pairs
Gate disagreement stays in compare/ (kappa, cross-tables). Writes ROOT/common_support/{<unit>__<tag>.tsv.gz,
units.tsv}. Skips a unit without a CRISPR-OT or compare result.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONFIGS = {"l5best": "ranknm256-midrank|none|scale|cos|pca50", "logcpm_pca": "logcpm|none|noscale|sqeuc|pca30"}
K = 15


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "common_support"; out.mkdir(exist_ok=True)
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module(); tabs = bb.load_tables(a.inputs)
    A = a.root / "inputs_A"
    rows = []
    for f in sorted(glob.glob(str(a.bundles / "*.pkl")), key=lambda f: Path(f).stat().st_size):
        b = pickle.load(open(f, "rb")); u = b["unit_id"]
        crd = a.root / "crispr_ot" / u
        if not (crd / "audit.json").exists():
            print(f"{u}: no CRISPR-OT result, skipped", flush=True); continue
        key = "KPTracer" if b["dataset"] == "KPTracer" else f"Quinn2021:{b['mouse']}"
        t, prior = tabs[key]
        U = cot.UnitDistance(b, t, prior, bb.map_ids)
        ns, nt = len(b["source_ids"]), len(b["target_ids"])
        zc = np.load(crd / "coupling_m4e.npz"); Pc = zc["coupling"].astype(np.float64)
        gates = pd.read_csv(crd / "gates.tsv.gz", sep="\t", dtype={"observation_id": str})
        gm = gates[gates.method == "M4-E"]
        cs = gm[gm.side == "source"].set_index("observation_id").retained.reindex(b["source_ids"]).to_numpy(bool)
        ct = gm[gm.side == "target"].set_index("observation_id").retained.reindex(b["target_ids"]).to_numpy(bool)
        E = None
        for tag, cid in CONFIGS.items():
            row = A / "v3_grid_20261104" / "kp" / "rows" / u / (cid.split("|")[0] + "__none") / cid.replace("|", "__")
            if not (row / "coupling_m4e.npz").exists():
                continue
            z = np.load(row / "coupling_m4e.npz")
            P = z["coupling"][np.ix_(pd.Index(z["source_ids"].astype(str)).get_indexer(b["source_ids"]),
                                     pd.Index(z["target_ids"].astype(str)).get_indexer(b["target_ids"]))].astype(np.float64)
            cc = pd.read_csv(row / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected"])
            cc = cc[cc.method == "M4-E"]; cc["observation_id"] = cc.observation_id.astype(str)
            rs = ~cc[cc.side == "source"].set_index("observation_id").rejected.reindex(b["source_ids"]).to_numpy(bool)
            rt = ~cc[cc.side == "target"].set_index("observation_id").rejected.reindex(b["target_ids"]).to_numpy(bool)
            S, T = np.flatnonzero(rs & cs), np.flatnonzero(rt & ct)
            rec = {"unit_id": u, "dataset": b["dataset"], "mouse": b["mouse"], "config": tag, "sources": ns, "targets": nt,
                   "common_sources": len(S), "common_targets": len(T),
                   "rna_retained_sources": int(rs.sum()), "crispr_retained_sources": int(cs.sum()),
                   "rna_retained_targets": int(rt.sum()), "crispr_retained_targets": int(ct.sum())}
            if len(S) == 0 or len(T) == 0:
                rows.append(rec); print(f"{u} {tag}: empty common block", flush=True); continue
            if E is None:   # evaluable cross pairs, all sources x targets
                E = np.zeros((ns, nt), bool)
                for k0 in range(0, nt, 2000):
                    cols = np.arange(k0, min(nt, k0 + 2000))
                    E[:, cols] = np.isfinite(U.block("source", np.arange(ns), "target", cols))
            Eb = E[np.ix_(S, T)]
            res = {}
            for name, Q, keep_s, keep_t in (("rna", P, rs, rt), ("crispr", Pc, cs, ct)):
                full = Q[np.ix_(np.flatnonzero(keep_s), np.flatnonzero(keep_t))].sum()
                B = Q[np.ix_(S, T)] * Eb
                rec[f"{name}_mass_on_block"] = float(B.sum() / full) if full > 0 else np.nan
                col = B.sum(0); res[name] = (B / np.where(col > 0, col, np.nan)[None, :], col > 0)
            qR, okR = res["rna"]; qC, okC = res["crispr"]; ok = okR & okC
            qR, qC = np.nan_to_num(qR), np.nan_to_num(qC)
            tv = np.where(ok, 0.5 * np.abs(qR - qC).sum(0), np.nan)
            k = min(K, len(S))
            topR = np.argpartition(-qR, k - 1, axis=0)[:k] if k < len(S) else np.tile(np.arange(len(S))[:, None], (1, len(T)))
            topC = np.argpartition(-qC, k - 1, axis=0)[:k] if k < len(S) else topR
            ov = np.array([len(np.intersect1d(topR[:, j], topC[:, j])) / k if ok[j] else np.nan for j in range(len(T))])
            pd.DataFrame({"target": np.asarray(b["target_ids"])[T], "tv": tv, "topk_overlap": ov, "k": k}).to_csv(
                out / f"{u}__{tag}.tsv.gz", sep="\t", index=False, compression="gzip")
            rec.update(evaluable_share=float(Eb.mean()), targets_scored=int(ok.sum()), k=k, chance_overlap=k / len(S),
                       tv_median=float(np.nanmedian(tv)), tv_q25=float(np.nanpercentile(tv, 25)), tv_q75=float(np.nanpercentile(tv, 75)),
                       overlap_mean=float(np.nanmean(ov)))
            rows.append(rec)
            print(f"{u} {tag}: block {len(S)} x {len(T)}, TV {rec['tv_median']:.3f}, overlap {rec['overlap_mean']:.3f} (chance {rec['chance_overlap']:.3f})", flush=True)
    pd.DataFrame(rows).to_csv(out / "units.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()

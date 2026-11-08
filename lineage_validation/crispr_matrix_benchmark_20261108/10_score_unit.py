"""Score every configuration of one unit of the v3 grid against the CRISPR bundle (MATRIX_METRICS.md): matrix L1
(Quinn clone mass), L2 (intBC-allele and UPGMA partitions), L3 (edit agreement), the tree-secondary L2 / L3, and the
sequencing-depth correlations. M4-E and M4-R. Reads couplings; writes small tables.

    python lineage_validation/crispr_matrix_benchmark_20261108/10_score_unit.py --bundle BUNDLES/<unit>.pkl \
        --rows ROWS/<unit> --out OUT [--depth predownsample_depth.csv.gz] [--keep-cells CONFIG ...]

ROWS/<unit>/<key>/<config>/ holds coupling_m4e.npz, coupling_m4r.npz (source_ids, target_ids) and cell_confidence.csv.gz
(v3 grid worker). Every coupling is aligned to the bundle's cell order by id; a cell set that differs from the bundle's
is an error. Writes OUT/<unit>/{l3_lineage,l2_partition,l1_clone,tree_l2,tree_l3,depth,status}.tsv.gz and, for the
--keep-cells configurations, cells_l3.tsv.gz. Refuses if OUT/<unit>/SUCCESS exists.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "lineage_validation" / "level3_cell_lineage"))
sys.path.insert(0, str(REPO / "lineage_validation" / "level2_pairwise_ot"))
import matrix_core as mc  # noqa: E402

UNLABELLED, OTHER = "family unlabelled", "other"


def rho(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok])[0]) if ok.sum() >= 10 and np.unique(x[ok]).size > 1 else np.nan


def load_fit(d: Path, method: str, b: dict):
    tag = method.lower().replace("-", "")
    with np.load(d / f"coupling_{tag}.npz") as z:
        P, sid, tid = z["coupling"], z["source_ids"].astype(str), z["target_ids"].astype(str)
    i_s = pd.Index(sid).get_indexer(b["source_ids"]); i_t = pd.Index(tid).get_indexer(b["target_ids"])
    if (i_s < 0).any() or (i_t < 0).any() or len(sid) != len(b["source_ids"]) or len(tid) != len(b["target_ids"]):
        raise SystemExit(f"{d}: coupling cells differ from the bundle")
    P = P[np.ix_(i_s, i_t)]
    cc = pd.read_csv(d / "cell_confidence.csv.gz", usecols=["method", "side", "observation_id", "rejected", "signed_rejection_margin"])
    cc = cc[cc.method == method]
    g = {side: cc[cc.side == side].set_index(cc[cc.side == side].observation_id.astype(str)) for side in ("source", "target")}
    rs = ~g["source"].rejected.reindex(b["source_ids"]).to_numpy(bool)
    rt = ~g["target"].rejected.reindex(b["target_ids"]).to_numpy(bool)
    margin = (g["source"].signed_rejection_margin.reindex(b["source_ids"]).to_numpy(float),
              g["target"].signed_rejection_margin.reindex(b["target_ids"]).to_numpy(float))
    return P, rs, rt, margin


def score(b: dict, P, rs, rt, margin, key: dict, keep: bool, out: dict):
    from score_pairwise import score_one
    import l3_core as c3
    Q, m, col = mc.retained_block(P, rs, rt)
    n_t = len(b["target_ids"])
    # ---- L3 and L2 per lineage
    cell_frames = []
    for lin, v in b["lineages"].items():
        rows_s, rows_t = v["source_rows"], v["target_rows"]
        Qb = Q[np.ix_(rows_s, rows_t)]
        cells = mc.l3_cells(Qb, m[rows_s], col[rows_t], v["s"].astype(np.float64), v["ok"], rt[rows_t])
        sc = cells[cells.scored]
        out["l3"].append({**key, "lineage": lin, "targets": len(rows_t), "retained": int(rt[rows_t].sum()), "scored": len(sc),
                          "mean_c": float(cells.c.mean()) if cells.c.notna().any() else np.nan,
                          "median_dS_norm": float(sc.dS_norm.median()) if sc.dS_norm.notna().any() else np.nan,
                          "mean_dS": float(sc.dS.mean()) if len(sc) else np.nan,
                          "mean_S_ot": float(sc.S_ot.mean()) if len(sc) else np.nan,
                          "mean_S_ind": float(sc.S_ind.mean()) if len(sc) else np.nan,
                          "rho_dS_depth": rho(cells.dS.to_numpy(), b["target_depth"][rows_t])})
        if keep:
            cell_frames.append(cells.assign(cell=b["target_ids"][rows_t], lineage=lin, depth=b["target_depth"][rows_t]))
        ns = len(rows_s)
        for name, lab in v["partitions"].items():
            r = mc.l2_partition(Qb, m[rows_s], col[rows_t], lab[:ns], lab[ns:], rt[rows_t])
            if r is not None:
                out["l2"].append({**key, "lineage": lin, "partition": name, "kind": "upgma" if name == "upgma" else "intbc", **r})
    if keep and cell_frames:
        out["cells"].append(pd.concat(cell_frames).assign(**key))
    # ---- Quinn L1: own-clone mass gain
    if "quinn_clone" in b:
        cs, ct = b["quinn_clone"]
        pi = m / m.sum() if m.sum() > 0 else m
        q = Q / np.where(col > 0, col, np.nan)[None, :]
        src_clones = set(cs[(m > 0) & pd.notna(cs)])
        gs = []
        for j in range(n_t):
            if not rt[j] or not isinstance(ct[j], str) or ct[j] not in src_clones or col[j] <= 0:
                continue
            same = cs == ct[j]
            gs.append(q[same, j].sum() - pi[same].sum())
        out["l1"].append({**key, "targets": n_t, "scored": len(gs), "coverage": len(gs) / n_t,
                          "mean_gain": float(np.mean(gs)) if gs else np.nan})
    # ---- tree-secondary
    if "tree" in b:
        tr = b["tree"]
        for (fam, level), v in tr["l2"].items():
            names = v["allowed"] + [UNLABELLED, OTHER]
            onehot = np.stack([(v["source_group"] == x) for x in names], 1).astype(np.float64)
            tm = pd.DataFrame(Q.T @ onehot, columns=[f"mass__{x}" for x in names])
            tm.insert(0, "column_mass", col); tm.insert(0, "site", b["target_site"]); tm.insert(0, "cell", b["target_ids"])
            src = pd.DataFrame({"group": names, "cells": onehot.sum(0).astype(int), "row_mass": m @ onehot})
            labels = pd.Series(v["target_label"], index=b["target_ids"]).dropna()
            s, *_ = score_one(tm, src, labels, v["allowed"])
            out["tree_l2"].append({**key, "family": fam, "level": level, **{k: s[k] for k in (
                "evaluation_status", "target_family_cells", "scored", "coverage_scored", "true_classes", "candidates",
                "balanced_accuracy_enrichment", "balanced_accuracy_size_rule", "chance")}})
        for fam, v in tr["l3"].items():
            vals = []
            tg = v["targets"]
            for k, (j, status) in enumerate(zip(tg["target_col"], tg["status"])):
                j = int(j)
                if status != "evaluable" or not rt[j] or col[j] <= 0:
                    continue
                met = c3.cell_metrics(Q[:, j] / col[j], m, v["s_rows"], v["D"][k].astype(np.float64))
                vals.append(met.get("delta_D_norm", np.nan))
            ev = int((tg["status"] == "evaluable").sum())
            out["tree_l3"].append({**key, "family": fam, "evaluable": ev, "scored": len(vals), "coverage": len(vals) / ev if ev else np.nan,
                                   "median_dD_norm": float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan,
                                   "mean_dD_norm": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan})
    # ---- gate score vs depth
    out["depth"].append({**key, "rho_margin_depth_source": rho(margin[0], b["source_depth"]),
                         "rho_margin_depth_target": rho(margin[1], b["target_depth"]),
                         "retained_source": float(rs.mean()), "retained_target": float(rt.mean())})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth", type=Path, help="observation_id, predownsample_total_counts (fills depth missing in the bundle)")
    p.add_argument("--keep-cells", nargs="*", default=["ranknm256-midrank|none|scale|cos|pca50", "logcpm|none|noscale|sqeuc|pca30",
                                                        "ranknm256-production|none|noscale|cos|pca30"])
    a = p.parse_args()
    with open(a.bundle, "rb") as fh:
        b = pickle.load(fh)
    o = a.out / b["unit_id"]
    if (o / "SUCCESS").exists():
        sys.exit(f"{o} done; not overwritten")
    o.mkdir(parents=True, exist_ok=True)
    if a.depth is not None:
        d = pd.read_csv(a.depth).set_index("observation_id")["predownsample_total_counts"]
        for side in ("source", "target"):
            x = b[f"{side}_depth"]; miss = ~np.isfinite(x)
            x[miss] = pd.Series(b[f"{side}_ids"][miss]).map(d).to_numpy(float)
    out = {k: [] for k in ("l3", "l2", "l1", "tree_l2", "tree_l3", "depth", "status", "cells")}
    t0 = time.perf_counter()
    for cdir in sorted(a.rows.glob("*/*")):
        if not cdir.is_dir() or cdir.name.startswith(".") or cdir.parent.name.startswith("."):
            continue
        cid = cdir.name.replace("__", "|")
        for method in ("M4-E", "M4-R"):
            key = {"unit_id": b["unit_id"], "config_id": cid, "method": method}
            if not (cdir / "DONE").exists():
                out["status"].append({**key, "status": "missing"}); continue
            P, rs, rt, margin = load_fit(cdir, method, b)
            score(b, P, rs, rt, margin, key, keep=(cid in a.keep_cells), out=out)
            out["status"].append({**key, "status": "scored"})
        print(f"{cid}: {time.perf_counter() - t0:.0f} s", flush=True)
    names = {"l3": "l3_lineage", "l2": "l2_partition", "l1": "l1_clone", "tree_l2": "tree_l2", "tree_l3": "tree_l3",
             "depth": "depth", "status": "status"}
    for k, f in names.items():
        if out[k]:
            pd.DataFrame(out[k]).to_csv(o / f"{f}.tsv.gz", sep="\t", index=False, compression="gzip")
    if out["cells"]:
        pd.concat(out["cells"]).to_csv(o / "cells_l3.tsv.gz", sep="\t", index=False, compression="gzip")
    (o / "SUCCESS").write_text(f"{len(out['status'])} fits, {time.perf_counter() - t0:.0f} s\n")


if __name__ == "__main__":
    main()

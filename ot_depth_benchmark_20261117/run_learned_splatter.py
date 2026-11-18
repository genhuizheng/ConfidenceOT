"""S3 (PLAN.md post-hoc extension 2026-11-18): CINEMA-OT and CellOT on the local Splatter N = 200 data. Runs in the otbench env.

    python -I ot_depth_benchmark_20261117/run_learned_splatter.py [--methods cinemaot cellot] [--jobs 6]
        [--cellot-iters 100000] [--ae-iters 250000] [--max-pairs 0] [--check-cinemaot]

Outputs: results/ot_depth_benchmark_20261117/splatter_learned/parts/ (one file per pair and method, resumable), merged
by --collect into METRICS.tsv / ERRORS.txt.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import otdb_common as oc  # noqa: E402

DATA = oc.REPO / "results" / "splatter_local_20261103" / "conditions"
OUT = oc.REPO / "results" / "ot_depth_benchmark_20261117" / "splatter_learned"
THIN = ("Group1", "Group2")


def pairs() -> list[dict]:
    out = []
    for rep in (1, 2, 3):
        for lv, case, f in ([("L0_matched", "all_shared", f) for f in (1.0, 0.25, 0.1)] +
                            [("L1_depth", "all_shared", 1.0), ("L2_depth_dropout", "all_shared", 1.0),
                             ("L5_batch_perturb_imbalance", "all_shared", 1.0), ("L1_depth", "population_lost", 1.0)]):
            cond = DATA / f"rep{rep}" / lv
            man = pd.read_csv(cond / "manifest.csv")
            i = int(np.flatnonzero(man.biological_case.to_numpy() == case)[0]); r = man.iloc[i]
            arm = "conflict" if lv == "L0_matched" else "native"
            out.append({"replicate": rep, "level": lv, "case": case, "arm": arm, "f": f, "pair_id": r.pair_id,
                        "source": str(oc.REPO / r.source_h5ad), "target": str(oc.REPO / r.target_h5ad),
                        "truth": str(cond / r.pair_id / "truth" / "population_truth.tsv"),
                        "name": f"{lv}__{r.pair_id}__{case}__{arm}__f{f}"})
    return out


def load_pair(job):
    import anndata as ad
    import run_splatter as rs
    S, T = ad.read_h5ad(job["source"]), ad.read_h5ad(job["target"])
    if job["arm"] == "conflict" and job["f"] < 1:
        seed = 20261117 + 100 * job["replicate"] + rs.THIN_FACTORS.index(job["f"])
        T = rs.thin(T, np.isin(T.obs.group.astype(str), THIN), job["f"], seed)
    X = sp.vstack([sp.csr_matrix(S.X, dtype=np.float32), sp.csr_matrix(T.X, dtype=np.float32)], format="csr")
    g = np.concatenate([S.obs.group.astype(str).to_numpy(), T.obs.group.astype(str).to_numpy()])
    nc = np.asarray(X.sum(1)).ravel().astype(float); nf = np.diff(X.indptr).astype(float)
    return X, g, nc, nf, S.n_obs


def s2_categories(gs, gt_partner):
    same = gs == gt_partner
    thin_src = np.isin(gs, THIN)
    out = {}
    if thin_src.any():
        c = gt_partner[thin_src]; s_ = gs[thin_src]
        out = {"g12_correct": float((c == s_).mean()), "g12_wrong_dissimilar_depth": float(((c != s_) & np.isin(c, THIN)).mean()),
               "g12_wrong_similar_depth": float((~np.isin(c, THIN)).mean())}
    return {"correct_share": float(same.mean())} | out


def fit(job) -> dict:
    import run_learned as rl
    rows, err = [], []
    base = {k: job[k] for k in ("replicate", "level", "case", "arm", "f", "pair_id")}
    try:
        X, g, nc, nf, n = load_pair(job)
        src, tgt = np.arange(n), np.arange(n, X.shape[0])
        gs, gt = g[src], g[tgt]
        if "cinemaot" in job["methods"]:
            try:
                row, P = rl.run_cinemaot(X, src, tgt, nc, nf, return_coupling=True)
                r = base | {"method": "cinemaot", "solver": "cinemaot", "status": "ok"} | row | oc.group_metrics(P, gs, gt, nc[src], nc[tgt])
                M = pd.DataFrame(P, index=gs, columns=gt).T.groupby(level=0).sum().T.groupby(level=0).sum()
                if any(x in M.index for x in THIN):
                    sub = M.loc[[x for x in THIN if x in M.index]]
                    tot = sub.values.sum()
                    r |= {"g12_correct": float(sum(sub.loc[x, x] for x in sub.index if x in sub.columns) / tot),
                          "g12_wrong_similar_depth": float(sub[[c for c in sub.columns if c not in THIN]].values.sum() / tot)}
                rows.append(r)
            except Exception:
                rows.append(base | {"method": "cinemaot", "status": "failed"}); err.append(traceback.format_exc()[-1500:])
        if "cellot" in job["methods"]:
            try:
                m = rl.cellot_train(X, src, tgt, Path(job["cellot_repo"]), job["cellot_iters"],
                                    (OUT / "work" / job["name"]).resolve(), job["ae_iters"])
                F, cond, split, idx = m["F"], m["cond"], m["split"], m["idx"]
                isc, tr = cond == "control", split == "train"
                te_src = isc & ~tr
                Ft = F[~isc]; gt_all = g[idx[~isc]]; lct = np.log10(nc[idx[~isc]])
                xs = F[te_src]; gsrc = g[idx[te_src]]; lcs = np.log10(nc[idx[te_src]])
                from sklearn.neighbors import NearestNeighbors
                nn = NearestNeighbors(n_neighbors=1).fit(Ft)
                preds = {"cellot": m["push"](xs), "baseline_no_transport": xs,
                         "baseline_mean_shift": xs + (F[~isc & tr].mean(0) - F[isc & tr].mean(0))}
                for k, v in preds.items():
                    j = nn.kneighbors(v, return_distance=False)[:, 0]
                    r = base | {"method": "cellot", "solver": k, "status": "ok", "n_eval_source": int(te_src.sum()),
                                "cellot_iters": job["cellot_iters"], "ae_iters": job["ae_iters"]}
                    r |= s2_categories(gsrc, gt_all[j])
                    r["nn_depth_corr_count"] = float(np.corrcoef(lcs, lct[j])[0, 1]) if np.std(lct[j]) > 0 else np.nan
                    if k == "cellot":
                        r |= m["diag"]
                        r["map_displacement_rel"] = float(np.linalg.norm(v - xs, axis=1).mean() / np.linalg.norm(xs - xs.mean(0), axis=1).mean())
                    rows.append(r)
                import shutil
                shutil.rmtree(m["work"], ignore_errors=True)
            except Exception:
                rows.append(base | {"method": "cellot", "status": "failed"}); err.append(traceback.format_exc()[-2500:])
    except Exception:
        err.append(traceback.format_exc()[-1500:])
    return {"rows": rows, "err": [f"{job['name']}: {e}" for e in err]}


def check_cinemaot(smoothness: float = 1.0):
    """Package matching (linear-domain Sinkhorn-Knopp) vs the log-domain matching of run_learned at a smoothness where the
    package does not underflow; same confounders, same objective."""
    import anndata as ad
    import ot as pot
    import scanpy as sc
    import sklearn.metrics
    from cinemaot import cinemaot as cm
    from sklearn.decomposition import FastICA
    import run_learned as rl
    job = pairs()[0]
    X, g, nc, nf, n = load_pair(job)
    L = rl.lognorm(X); hv = rl.hvg(L, 2000)
    a = ad.AnnData(X=L[:, hv].toarray(), obs=pd.DataFrame({"condition": ["control"] * n + ["perturbed"] * (X.shape[0] - n)},
                                                         index=[f"c{i}" for i in range(X.shape[0])]))
    a.raw = a.copy(); sc.pp.scale(a, max_value=10); sc.tl.pca(a, n_comps=50, svd_solver="arpack", random_state=rl.SEED); a.X = a.raw.X.copy()
    cf, Ppkg, _ = cm.cinemaot_unweighted(a, obs_label="condition", ref_label="control", expr_label="perturbed",
                                         dim=20, thres=0.15, smoothness=smoothness, eps=1e-3)
    Xt = FastICA(n_components=20, random_state=0, whiten="arbitrary-variance").fit_transform(a.obsm["X_pca"][:, :20])
    xi = np.array([cm.Xi(c, (a.obs.condition == "control").to_numpy().astype(int)).correlation for c in Xt.T])
    c2 = Xt[:, xi < 0.15]
    d2 = sklearn.metrics.pairwise_distances(c2[:n], c2[n:]) ** 2
    Pours = pot.sinkhorn(np.full(n, 1 / n), np.full(X.shape[0] - n, 1 / (X.shape[0] - n)), d2, smoothness * c2.shape[1],
                         method="sinkhorn_log", numItermax=100000, stopThr=1e-9)
    Ppkg = np.asarray(Ppkg); Ppkg = Ppkg / Ppkg.sum()
    res = {"smoothness": smoothness, "package_nan": bool(np.isnan(Ppkg).any()), "same_confounders": bool(np.allclose(np.abs(cf), np.abs(c2))),
           "max_abs_diff": float(np.nanmax(np.abs(Ppkg - Pours))), "max_entry": float(Pours.max()),
           "rowwise_argmax_agreement": float((Ppkg.argmax(1) == Pours.argmax(1)).mean())}
    print(res)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", default=["cinemaot", "cellot"])
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--cellot-iters", type=int, default=100000)
    p.add_argument("--ae-iters", type=int, default=250000)
    p.add_argument("--cellot-repo", default=os.environ.get("CELLOT_REPO", "D:/Xia_lab/CellOT_tools/cellot"))
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--collect", action="store_true")
    p.add_argument("--check-cinemaot", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.check_cinemaot:
        rows = [check_cinemaot(s) for s in (1.0, 0.1, 1e-4)]
        pd.DataFrame(rows).to_csv(OUT / "CHECK_CINEMAOT_vs_package.tsv", sep="\t", index=False); return
    parts = OUT / f"parts{('_' + a.tag) if a.tag else ''}"
    if a.collect:
        oc.collect_parts(parts, OUT, "", a.tag); return
    jobs = [j | {"methods": a.methods, "cellot_iters": a.cellot_iters, "ae_iters": a.ae_iters, "cellot_repo": a.cellot_repo} for j in pairs()]
    if a.max_pairs:
        jobs = jobs[: a.max_pairs]
    oc.run_resumable(jobs, fit, parts, lambda j: f"{j['name']}__{'_'.join(a.methods)}", a.jobs)


if __name__ == "__main__":
    main()

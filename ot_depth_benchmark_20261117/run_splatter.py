"""Dataset A (Splatter, existing local N = 200 simulations): S1 native scenarios and S2 depth-identity conflict (PLAN.md).

    python ot_depth_benchmark_20261117/run_splatter.py --stage s1 [--replicates 1 2 ...] [--jobs 12]
    python ot_depth_benchmark_20261117/run_splatter.py --stage s2

Per fit (pair x cost x solver) one row of METRICS_<stage>.tsv; the 5 x 5 source-group x target-group mass of every fit in
TRANSITIONS_<stage>.tsv; per-cell gates of ConfidenceOT in GATES_<stage>.tsv.gz; couplings (float32) in
couplings_<stage>/<fit>.npz. Outputs: results/ot_depth_benchmark_20261117/splatter/.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import otdb_common as oc  # noqa: E402

DATA = oc.REPO / "results" / "splatter_local_20261103"
OUT = oc.REPO / "results" / "ot_depth_benchmark_20261117" / "splatter"
LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
          "A4_perturb_only", "A5_imbalance_only"]
THIN_FACTORS = [1.0, 0.5, 0.25, 0.1]
THIN_GROUPS = ("Group1", "Group2")
S2_ARMS = ("conflict", "aligned", "global")


def thin(adata, mask, f, seed):
    """Binomial thinning X' ~ Binomial(X, f) of the rows in mask (integer counts)."""
    import scipy.sparse as sp
    if f >= 1.0 or not mask.any():
        return adata
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(adata.X, dtype=np.float64, copy=True)
    for i in np.flatnonzero(mask):
        lo, hi = X.indptr[i], X.indptr[i + 1]
        X.data[lo:hi] = rng.binomial(X.data[lo:hi].astype(np.int64), f)
    X.eliminate_zeros()
    out = adata.copy(); out.X = X
    return out


def nc_nf(adata):
    import scipy.sparse as sp
    X = sp.csr_matrix(adata.X)
    return np.asarray(X.sum(1)).ravel(), np.diff(X.indptr)


def fit_one(job: dict) -> dict:
    import anndata as ad
    rows, trans, gates, err = [], [], [], []
    try:
        S, T = ad.read_h5ad(job["source"]), ad.read_h5ad(job["target"])
        truth = pd.read_csv(job["truth"], sep="\t").set_index(["side", "observation_id"])
        if job["stage"] == "s2":
            f, arm, rep = job["f"], job["arm"], job["replicate"]
            seed = 20261117 + 100 * rep + THIN_FACTORS.index(f)
            tmask = np.isin(T.obs.group.astype(str), THIN_GROUPS) if arm != "global" else np.ones(T.n_obs, bool)
            T = thin(T, tmask, f, seed)
            if arm == "aligned":
                S = thin(S, np.isin(S.obs.group.astype(str), THIN_GROUPS), f, seed + 50)
        gs = S.obs.group.astype(str).to_numpy(); gt = T.obs.group.astype(str).to_numpy()
        rs = truth.loc[[("source", x) for x in S.obs_names]].should_reject.to_numpy(bool)
        rt = truth.loc[[("target", x) for x in T.obs_names]].should_reject.to_numpy(bool)
        sc_, sf_ = nc_nf(S); tc_, tf_ = nc_nf(T)
        groups = sorted(set(gs) | set(gt))
        # depth-biology association in the data: eta^2 of log10 nCount by group, per side
        def eta2(lc, g):
            s = pd.Series(lc); m = s.groupby(g).transform("mean")
            return float(((m - s.mean()) ** 2).sum() / max(((s - s.mean()) ** 2).sum(), 1e-300))
        data_info = {"eta2_depth_group_source": eta2(np.log10(sc_), gs), "eta2_depth_group_target": eta2(np.log10(tc_), gt),
                     "median_ncount_source": float(np.median(sc_)), "median_ncount_target": float(np.median(tc_)),
                     "median_nfeature_source": float(np.median(sf_)), "median_nfeature_target": float(np.median(tf_))}
        for cost in ("C1", "C2"):
            idx = job["index"]
            X, Y, diag = (oc.c1_coordinates if cost == "C1" else oc.c2_coordinates)(S, T, 20260925 + idx)
            C, Xc, Yc, scale, rng = oc.cost_matrix(X, Y, cost == "C1", idx)
            res = oc.solve_all(C, Xc, Yc, scale, rng, idx)
            for solver, r in res.items():
                key = {k: job[k] for k in job["keys"]} | {"cost": cost, "solver": solver}
                if "error" in r:
                    rows.append(key | {"status": r["error"]}); continue
                P = r["coupling"]
                n, m = P.shape
                row = key | {"status": "ok", **data_info, "n_source": n, "n_target": m,
                             "mnncov_weight": diag.get("mnncred_weight", np.nan), "marginal_error": r.get("marginal_error", np.nan)}
                row |= oc.group_metrics(P, gs, gt, sc_, tc_)
                row |= oc.depth_metrics(P, sc_, tc_, sf_, tf_)
                # transported share of each population's marginal mass, unmatched (should_reject) vs shared
                rsum, csum = P.sum(1) * n, P.sum(0) * m
                for side, s_, msk in (("source", rsum, rs), ("target", csum, rt)):
                    row[f"transported_share_shared_{side}"] = float(s_[~msk].mean()) if (~msk).any() else np.nan
                    row[f"transported_share_unmatched_{side}"] = float(s_[msk].mean()) if msk.any() else np.nan
                if r["source_gate"] is not None:
                    sg, tg = r["source_gate"], r["target_gate"]
                    row["rejection_cost"] = r["rejection_cost"]
                    from scipy.stats import spearmanr
                    for side, g, msk, cnt in (("source", sg, rs, sc_), ("target", tg, rt, tc_)):
                        rej = ~g
                        row[f"rejected_fraction_shared_{side}"] = float(rej[~msk].mean()) if (~msk).any() else np.nan
                        row[f"rejected_fraction_unmatched_{side}"] = float(rej[msk].mean()) if msk.any() else np.nan
                        sh = ~msk
                        row[f"rejection_depth_corr_{side}"] = (float(spearmanr(np.log10(cnt[sh]), rej[sh]).statistic)
                                                               if sh.sum() > 2 and 0 < rej[sh].sum() < sh.sum() else np.nan)
                    keep = np.outer(sg, tg)
                    A = P * keep
                    row["asserted_mass"] = float(A.sum())
                    if A.sum() > 0:
                        row["false_asserted_share"] = float((A * (gs[:, None] != gt[None, :])).sum() / A.sum())
                        row |= {f"asserted_{k}": v for k, v in oc.depth_metrics(A, sc_, tc_, sf_, tf_).items()}
                    for side, ids, g, cnt, fea, grp, msk in (("source", S.obs_names, sg, sc_, sf_, gs, rs), ("target", T.obs_names, tg, tc_, tf_, gt, rt)):
                        gates.append(pd.DataFrame({**{k: job[k] for k in job["keys"]}, "cost": cost, "side": side, "cell": ids,
                                                   "group": grp, "should_reject": msk, "rejected": ~g, "ncount": cnt, "nfeature": fea}))
                rows.append(row)
                M = pd.DataFrame(P, index=gs, columns=gt).groupby(level=0).sum().T.groupby(level=0).sum().T
                M = M.reindex(index=groups, columns=groups, fill_value=0.0)
                for a_ in groups:
                    for b_ in groups:
                        trans.append(key | {"source_group": a_, "target_group": b_, "mass": float(M.loc[a_, b_])})
                if job.get("save_dir"):
                    np.savez_compressed(Path(job["save_dir"]) / f"{job['fit_name']}__{cost}__{solver}.npz", coupling=P.astype(np.float32))
    except Exception:
        err.append(traceback.format_exc()[-1500:])
    return {"rows": rows, "trans": trans, "gates": gates, "err": err, "name": job["fit_name"]}


def jobs_for(stage: str, reps: list[int], save: bool, data: Path = DATA, out: Path = OUT, levels_s1=None, cond_sub: str = "") -> list[dict]:
    jobs = []
    sd = out / f"couplings_{stage}"
    if save:
        sd.mkdir(parents=True, exist_ok=True)
    for rep in reps:
        levels = (levels_s1 or LEVELS) if stage == "s1" else ["L0_matched"]
        for lv in levels:
            cond = data / "conditions" / cond_sub / f"rep{rep}" / lv if cond_sub else data / "conditions" / f"rep{rep}" / lv
            man = pd.read_csv(cond / "manifest.csv")
            for i, r in man.iterrows():
                base = {"stage": stage, "replicate": rep, "level": lv, "case": r.biological_case, "pair_id": r.pair_id, "index": i,
                        "source": str(oc.REPO / r.source_h5ad), "target": str(oc.REPO / r.target_h5ad),
                        "truth": str(cond / r.pair_id / "truth" / "population_truth.tsv"), "save_dir": str(sd) if save else ""}
                if stage == "s1":
                    jobs.append(base | {"keys": ["stage", "replicate", "level", "case", "pair_id"], "fit_name": r.pair_id})
                elif r.biological_case == "all_shared":
                    for arm in S2_ARMS:
                        for f in THIN_FACTORS:
                            jobs.append(base | {"arm": arm, "f": f, "keys": ["stage", "replicate", "level", "case", "pair_id", "arm", "f"],
                                                "fit_name": f"{r.pair_id}__{arm}__f{f}"})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("s1", "s2"), required=True)
    p.add_argument("--replicates", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--jobs", type=int, default=12)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--no-save-couplings", action="store_true")
    p.add_argument("--data-root", type=Path, default=DATA, help="Splatter root holding conditions/ (TACC: the N = 5,000 paired root)")
    p.add_argument("--conditions-subdir", default="", help="e.g. N5000 when the root keeps conditions/N5000/rep*/")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--levels", nargs="+", default=None)
    a = p.parse_args()

    globals()["OUT"] = a.out
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = jobs_for(a.stage, a.replicates, not a.no_save_couplings, a.data_root, a.out, a.levels, a.conditions_subdir)
    if a.max_jobs:
        jobs = jobs[: a.max_jobs]
    print(f"{len(jobs)} jobs", flush=True)
    rows, trans, gates, errs = [], [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = [ex.submit(fit_one, j) for j in jobs]
        for k, fu in enumerate(as_completed(futs), 1):
            r = fu.result(); rows += r["rows"]; trans += r["trans"]; gates += r["gates"]
            errs += [f"{r['name']}: {e}" for e in r["err"]]
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(OUT / f"METRICS_{a.stage}.tsv", sep="\t", index=False)
    pd.DataFrame(trans).to_csv(OUT / f"TRANSITIONS_{a.stage}.tsv", sep="\t", index=False)
    if gates:
        pd.concat(gates).to_csv(OUT / f"GATES_{a.stage}.tsv.gz", sep="\t", index=False)
    (OUT / f"ERRORS_{a.stage}.txt").write_text("\n\n".join(errs))
    print(f"{len(rows)} rows, {len(errs)} errors -> {OUT}")


if __name__ == "__main__":
    main()

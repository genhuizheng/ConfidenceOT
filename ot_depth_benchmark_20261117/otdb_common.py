"""Shared code of the OT depth benchmark (PLAN.md): the two cost families, the fixed-cost solvers, ConfidenceOT M4-E
with the frozen v3 calibration, and the coupling metrics.

C1 main      ranknm256-midrank|none|scale|cos|pca50 (factorial_common.encode_sides -> rb_common.pca_coordinates, 2,000
             top-variance encoded genes, gene scaling, PCA 50, seed) -> unit rows -> run_local_benchmark.correct
             'mnncovlocal' (the v3 correction, imported) -> unit rows again (02_run_pair's cosine label normalises the
             precomputed coordinates) -> squared Euclidean / median cross-side pair cost.
C2 standard  normalize_total 1e4, log1p, 2,000 HVG (seurat, joint), scale (max 10), PCA 50 -> squared Euclidean /
             median cross-side pair cost; no correction.

ConfidenceOT is called the way cancer_metastasis/02_run_pair.py calls it (lines 723-818: median scale, within-side
split nulls, calibrate_confidence_cost, ConfidenceOT M4-E), with the Splatter fit arguments of
run_local_benchmark.FIT_ARGS. check_against_02.py verifies gate and coupling equality with the production path.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path[:0] = [str(REPO / "realdata_benchmark"), str(REPO / "realdata_benchmark" / "splatter_local"), str(REPO / "src"),
                str(REPO / "cancer_metastasis"), str(REPO)]

FIT_SEED = 20260925          # run_local_benchmark.FIT_ARGS --seed
EPSILON, LAMBDA, ACCEPT, TOL = 0.1, 1.0, 0.99, 1e-4
PARTIAL_MASS = 0.8
SOLVERS = ("independent", "emd", "sinkhorn", "uot", "partial", "confidenceot", "confidenceot_bot")


# ----------------------------------------------------------------------------------------------------------- costs
# cost families on the v3 grid path (encoding -> 2,000 top-variance genes -> [scaling] -> PCA -> unit rows -> mnncovlocal)
GRID_COSTS = {
    "C1": ({"encoding": "ranknm", "K": 256, "tie": "midrank"}, True, 50),    # ranknm256-midrank|none|scale|cos|pca50 (main, L5 best)
    "C3": ({"encoding": "ranknm", "K": 512, "tie": "expected"}, False, 30),  # ranknm512-expected|none|noscale|cos|pca30 (local v3 grid best)
    "C4": ({"encoding": "logcpm"}, False, 30),                                # logcpm|none|noscale|cos|pca30 (local v3 grid second)
}


def grid_coordinates(S, T, pca_seed: int, cost: str):
    """A v3-grid cost family: corrected coordinates (before the final unit-row step) and the correction diagnostics."""
    import factorial_common as fc
    import rb_common as rb
    import run_local_benchmark as lb
    from confidenceot.preprocessing import unit_rows
    want, scale, n_pcs = GRID_COSTS[cost]
    spec = next(s for s in fc.grid() if s["ds"] == "none" and all(s.get(k) == v for k, v in want.items()))
    se, te, _, _ = fc.encode_sides(spec, S, T)
    joint = sp.vstack([sp.csr_matrix(se.X), sp.csr_matrix(te.X)], format="csr"); n = se.n_obs
    P, _ = rb.pca_coordinates(joint, fc.N_HVG, n_pcs, scale, False, pca_seed)
    C = unit_rows(P)
    lb._LAST.clear()
    X, Y = lb.correct(C[:n], C[n:], "mnncovlocal")
    diag = {k: v for k, v in lb._LAST.items() if not k.startswith("_")}
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), diag


def coordinates(S, T, pca_seed: int, cost: str):
    """(X, Y, diagnostics, cosine?) for a cost family name."""
    if cost == "C2":
        return (*c2_coordinates(S, T, pca_seed), False)
    return (*grid_coordinates(S, T, pca_seed, cost), True)


def c1_coordinates(S, T, pca_seed: int):
    """C1 main configuration: corrected coordinates (before the final unit-row step) and the correction diagnostics."""
    import factorial_common as fc
    import rb_common as rb
    import run_local_benchmark as lb
    from confidenceot.preprocessing import unit_rows
    spec = next(s for s in fc.grid() if s["ds"] == "none" and s["encoding"] == "ranknm" and s["K"] == 256 and s["tie"] == "midrank")
    se, te, _, _ = fc.encode_sides(spec, S, T)
    joint = sp.vstack([sp.csr_matrix(se.X), sp.csr_matrix(te.X)], format="csr"); n = se.n_obs
    P, _ = rb.pca_coordinates(joint, fc.N_HVG, 50, True, False, pca_seed)
    C = unit_rows(P)
    lb._LAST.clear()
    X, Y = lb.correct(C[:n], C[n:], "mnncovlocal")
    diag = {k: v for k, v in lb._LAST.items() if not k.startswith("_")}
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), diag


def c2_coordinates(S, T, pca_seed: int):
    """C2 standard scanpy path on the joint counts."""
    import anndata as ad
    import scanpy as sc
    genes = S.var_names.intersection(T.var_names)
    joint = ad.AnnData(X=sp.vstack([sp.csr_matrix(S[:, genes].X, dtype=np.float32), sp.csr_matrix(T[:, genes].X, dtype=np.float32)], format="csr"),
                       var=pd.DataFrame(index=genes.astype(str)))
    sc.pp.normalize_total(joint, target_sum=1e4)
    sc.pp.log1p(joint)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.pp.highly_variable_genes(joint, n_top_genes=2000, flavor="seurat")
    joint = joint[:, joint.var.highly_variable].copy()
    sc.pp.scale(joint, max_value=10)
    sc.tl.pca(joint, n_comps=50, svd_solver="arpack", random_state=pca_seed)
    Z = joint.obsm["X_pca"].astype(np.float32)
    return Z[: S.n_obs], Z[S.n_obs:], {}


def cost_matrix(X, Y, cosine: bool, index: int):
    """The cost 02_run_pair.py builds from precomputed coordinates, and the generator it continues with."""
    from confidenceot import median_pair_scale, squared_euclidean
    from confidenceot.preprocessing import unit_rows
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    if cosine:
        X, Y = unit_rows(X), unit_rows(Y)
    rng = np.random.default_rng(FIT_SEED + index * 104729)
    scale = median_pair_scale(X, Y, rng=rng)
    return squared_euclidean(X, Y) / scale, X, Y, scale, rng


# --------------------------------------------------------------------------------------------------------- solvers
def confidenceot_fit(Xc, Yc, C, scale, rng, index: int, device: str = "cpu", backbone: str = "uot"):
    """02_run_pair.py lines 723-818 for M4-E with the within-side split null; Xc / Yc are the coordinates the cost was
    built from, rng the generator after the median scale."""
    from confidenceot import ConfidenceOT, calibrate_confidence_cost, within_side_null_costs
    limit = min(2000, len(Xc), len(Yc))
    si = np.sort(rng.choice(len(Xc), limit, replace=False)); ti = np.sort(rng.choice(len(Yc), limit, replace=False))
    sn = within_side_null_costs(Xc[si], observed_scale=scale, seed=FIT_SEED + index, n_replicates=10)
    tn = within_side_null_costs(Yc[ti], observed_scale=scale, seed=FIT_SEED + index + 7, n_replicates=10)
    cal = calibrate_confidence_cost(sn[:5] + tn[:5], sn[5:] + tn[5:], backbone=backbone, epsilon=EPSILON, lambda_a=LAMBDA,
                                    lambda_b=LAMBDA, null_semantics="within_side_split", within_side_acceptance_minimum=ACCEPT,
                                    source_rejection_budget=0.95, target_rejection_budget=0.95, tolerance=TOL, grid_size=5,
                                    device=device, workers=1, fallback_to_cpu=False)
    model = ConfidenceOT(backbone=backbone, variant="exact", rejection_cost=float(cal.rejection_cost), epsilon=EPSILON,
                         lambda_a=LAMBDA, lambda_b=LAMBDA, source_rejection_budget=0.95, target_rejection_budget=0.95,
                         enforce_rejection_budget=False, tolerance=TOL, device=device)
    res = model.fit(C)
    return np.asarray(res.coupling, np.float64), np.asarray(res.source_gate, bool), np.asarray(res.target_gate, bool), float(cal.rejection_cost)


def solve_all(C, Xc=None, Yc=None, scale=None, rng=None, index: int = 0, solvers=SOLVERS, device: str = "cpu") -> dict:
    """name -> dict(coupling, source_gate, target_gate). Uniform marginals; the gate is None for gate-free solvers."""
    import ot
    n, m = C.shape
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    out = {}
    for s in solvers:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if s == "independent":
                    P = np.outer(a, b)
                elif s == "emd":
                    P = ot.emd(a, b, C, numItermax=10_000_000)
                elif s == "sinkhorn":
                    P = ot.sinkhorn(a, b, C, EPSILON, method="sinkhorn_log", numItermax=20000, stopThr=1e-9)
                elif s == "uot":
                    # log-domain KL-UOT (traditional_ot.unbalanced via confidenceot._cpu_uot), every cell open: the ConfidenceOT
                    # backbone without the gate. POT's non-log 'sinkhorn' underflows on large costs (PLAN post-hoc note).
                    from confidenceot._cpu_uot import solve_fixed_bidirectional_uot
                    P = solve_fixed_bidirectional_uot(C, np.ones(n, bool), np.ones(m, bool), rejection_cost=1.0, epsilon=EPSILON,
                                                      lambda_a=LAMBDA, lambda_b=LAMBDA, threshold=TOL, max_iterations=100_000).coupling
                elif s == "partial":
                    P = ot.partial.partial_wasserstein(a, b, C, m=PARTIAL_MASS)
                elif s in ("confidenceot", "confidenceot_bot"):
                    # the generator state is copied so both backbones draw the same calibration subsample and nulls
                    import copy
                    P, sg, tg, rc = confidenceot_fit(Xc, Yc, C, scale, copy.deepcopy(rng), index, device,
                                                     backbone="balanced" if s == "confidenceot_bot" else "uot")
                    out[s] = {"coupling": P, "source_gate": sg, "target_gate": tg, "rejection_cost": rc}
                    continue
            P = np.asarray(P, np.float64)
            # balanced solvers: largest relative marginal error (convergence diagnostic)
            merr = max(np.abs(P.sum(1) - a).max() / a[0], np.abs(P.sum(0) - b).max() / b[0]) if s in ("emd", "sinkhorn") else np.nan
            out[s] = {"coupling": P, "source_gate": None, "target_gate": None, "marginal_error": float(merr)}
        except Exception as e:  # recorded, never silently dropped
            out[s] = {"error": f"{type(e).__name__}: {e}"[:300]}
    return out


# --------------------------------------------------------------------------------------------------------- metrics
def wcorr(x, y, W):
    """Coupling-weighted Pearson correlation between x (rows) and y (columns) under weights W (any non-negative mass)."""
    tot = W.sum()
    if tot <= 0:
        return np.nan
    W = W / tot
    r, c = W.sum(1), W.sum(0)
    mx, my = r @ x, c @ y
    cov = x @ W @ y - mx * my
    vx, vy = r @ (x - mx) ** 2, c @ (y - my) ** 2
    return float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def depth_metrics(P, s_count, t_count, s_feat, t_feat, prefix=""):
    """Coupling-depth association: weighted correlations and the depth-gap ratio against the independent coupling."""
    lc_s, lc_t = np.log10(np.maximum(s_count, 1)), np.log10(np.maximum(t_count, 1))
    lf_s, lf_t = np.log10(np.maximum(s_feat, 1)), np.log10(np.maximum(t_feat, 1))
    gap = np.abs(lc_s[:, None] - lc_t[None, :])
    tot = P.sum()
    ind = np.outer(P.sum(1), P.sum(0)) / tot if tot > 0 else None   # independent coupling with the same marginals
    out = {f"{prefix}depth_corr_count": wcorr(lc_s, lc_t, P), f"{prefix}depth_corr_feature": wcorr(lf_s, lf_t, P)}
    if tot > 0:
        out[f"{prefix}depth_gap_ratio"] = float((P * gap).sum() / tot / max((ind * gap).sum() / tot, 1e-12))
    return out


def group_metrics(P, gs, gt, s_count, t_count):
    """Identity: correct-group share, its independent-coupling expectation, and the depth structure of wrong-group mass."""
    same = gs[:, None] == gt[None, :]
    tot = P.sum()
    out = {"transported_mass": float(tot)}
    if tot <= 0:
        return out
    r, c = P.sum(1) / tot, P.sum(0) / tot
    out["correct_mass"] = float((P * same).sum() / tot)
    out["correct_mass_indep"] = float((np.outer(r, c) * same).sum())
    lc_s, lc_t = np.log10(np.maximum(s_count, 1)), np.log10(np.maximum(t_count, 1))
    gap = np.abs(lc_s[:, None] - lc_t[None, :])
    wrong = ~same
    if wrong.any() and (P * wrong).sum() > 0:
        thr = np.median(gap[wrong])
        out["wrong_simdepth_mass"] = float((P * wrong * (gap < thr)).sum() / (P * wrong).sum())
        ind = np.outer(r, c)
        out["wrong_simdepth_mass_indep"] = float((ind * wrong * (gap < thr)).sum() / max((ind * wrong).sum(), 1e-300))
    if (P * same).sum() > 0:
        # depth preference among biologically equivalent partners: log depth centred on each group's mean per side, so
        # a depth difference between groups (S2 aligned arm) does not count as within-group preference
        cs = lc_s - pd.Series(lc_s).groupby(gs).transform("mean").to_numpy()
        ct = lc_t - pd.Series(lc_t).groupby(gt).transform("mean").to_numpy()
        out["depth_corr_count_within"] = wcorr(cs, ct, P * same)
    return out


# ------------------------------------------------------------------------------------------------- resumable runs
def run_resumable(jobs: list[dict], fn, parts: Path, name_of, workers: int, shard: int = 0, nshards: int = 1) -> None:
    """Run fn(job) -> {"rows", "gates"?, "err"} over the jobs of this shard (index mod nshards), each finished job written
    at once to parts/<name>.metrics.tsv (+ .gates.tsv.gz, .err.txt; written to a temporary name, then renamed).
    Jobs whose metrics file exists are skipped, so a resubmission continues where a job stopped (walltime, node loss)."""
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parts.mkdir(parents=True, exist_ok=True)
    todo = [j for i, j in enumerate(jobs) if i % nshards == shard and not (parts / f"{name_of(j)}.metrics.tsv").exists()]
    print(f"shard {shard}/{nshards}: {len(todo)} jobs to run ({len(jobs)} in total)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, j): j for j in todo}
        for k, fu in enumerate(as_completed(futs), 1):
            j = futs[fu]; nm = name_of(j)
            try:
                r = fu.result()
            except Exception as e:  # a crashed worker is recorded, not dropped
                r = {"rows": [], "gates": [], "err": [f"{nm}: worker crashed: {type(e).__name__}: {e}"]}
            if r.get("gates"):
                pd.concat(r["gates"]).to_csv(parts / f"{nm}.gates.tmp.gz", sep="\t", index=False, compression="gzip")
                os.replace(parts / f"{nm}.gates.tmp.gz", parts / f"{nm}.gates.tsv.gz")
            for tname, trows in (r.get("tables") or {}).items():
                if trows:
                    pd.DataFrame(trows).to_csv(parts / f"{nm}.{tname}.tsv", sep="\t", index=False)
            if r.get("err"):
                (parts / f"{nm}.err.txt").write_text("\n\n".join(r["err"]))
            if r["rows"]:   # no metrics file -> the job runs again next time
                pd.DataFrame(r["rows"]).to_csv(parts / f"{nm}.metrics.tmp", sep="\t", index=False)
                os.replace(parts / f"{nm}.metrics.tmp", parts / f"{nm}.metrics.tsv")
            if k % 10 == 0 or k == len(todo):
                print(f"{k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)


def collect_parts(parts: Path, out: Path, prefix: str, tag: str = "") -> None:
    """Concatenate parts into <out>/<prefix>METRICS<tag>.tsv, GATES<tag>.tsv.gz and ERRORS<tag>.txt."""
    t = f"_{tag}" if tag else ""
    m = [pd.read_csv(f, sep="\t") for f in sorted(parts.glob("*.metrics.tsv"))]
    g = [pd.read_csv(f, sep="\t") for f in sorted(parts.glob("*.gates.tsv.gz"))]
    e = [f.read_text() for f in sorted(parts.glob("*.err.txt"))]
    if m:
        pd.concat(m, ignore_index=True).to_csv(out / f"{prefix}METRICS{t}.tsv", sep="\t", index=False)
    if g:
        pd.concat(g, ignore_index=True).to_csv(out / f"{prefix}GATES{t}.tsv.gz", sep="\t", index=False)
    (out / f"{prefix}ERRORS{t}.txt").write_text("\n\n".join(e))
    extra = sorted({f.name.split(".")[-2] for f in parts.glob("*.*.tsv") if not f.name.endswith(".metrics.tsv")})
    for tname in extra:
        x = [pd.read_csv(f, sep="\t") for f in sorted(parts.glob(f"*.{tname}.tsv"))]
        pd.concat(x, ignore_index=True).to_csv(out / f"{prefix}{tname}{t}.tsv", sep="\t", index=False)
    print(f"collected {len(m)} jobs with metrics, {len(e)} with errors -> {out}")

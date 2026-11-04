"""Local small-N Splatter benchmark (N = 200 per side): L−1 calibration selection, then the L1-L5 preprocessing grid
with the frozen calibration (SELECTION_RULES.md).

    python realdata_benchmark/splatter_local/run_local_benchmark.py --root ROOT --stage calibration [--replicates 1 2]
    python realdata_benchmark/splatter_local/run_local_benchmark.py --root ROOT --stage calibration --replicates 3 4 --tag validation
    python realdata_benchmark/splatter_local/run_local_benchmark.py --root ROOT --stage grid [--replicates 1 2]

Data: 20_generate_n5000.R --param-set splatter_default --pairing paired (Splatter 1.30.0 defaults, batch 0.1 / 0.1),
seeds as generate_n5000.slurm (20260925 + 1000 rep + 137 level index; L3-L5, A4, A5 take L2's), conditions and truth by
21_build_conditions_n5000.py. Generated once and kept (small at N = 200) for every stage.

Arms (calibration null + correction of the joint PCA): within_side_split, cross_side_rotation, each optionally +harmony
(harmonypy 0.0.10 defaults, batch = side) and, for exploration only, +center / +quantile. No rejection cap.
A task is one pair and one representation: depth handling (equalise_depth to the pooled quantile of the condition's
files, seeds 1001 / 1002, as the TACC worker), the encoding, PCA, the correction -- computed once and shared by every
fit that uses them -- then each fit through 02_run_pair.py in this process (no interpreter start per fit) as a
precomputed coordinate matrix (label coord_allgenes_nopca_noscale[_cos]; the N = 5,000 Splatter arguments, acceptance
0.99, couplings saved). Every fit is scored against the 21_ truth and its run directory deleted.
  calibration  representation fixed: ranknm256-production|P10s1|noscale|cos (production path, PCA 30); the arms given
  grid         the frozen arm (ROOT/FROZEN_CALIBRATION.json); the 592 configurations of factorial_common.grid() x PCA
               {30, 50} (rb_common.pca_coordinates: 2,000 top-variance encoded genes, all H for rankhvg), plus the
               baseline through the production path; levels L1-L5
Writes ROOT/RESULTS_<stage>[_<tag>].tsv (appended; finished keys skipped), one row per fit x method x side.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "realdata_benchmark"), str(REPO / "src"), str(REPO / "cancer_metastasis"), str(REPO)]

ALL_LEVELS = ["L0_matched", "L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
              "A4_perturb_only", "A5_imbalance_only"]
GRID_LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"]
BASELINE = "ranknm256-production|P10s1|noscale|cos"
ARMS = ["within_side_split", "cross_side_rotation", "within_side_split+harmony", "cross_side_rotation+harmony"]
FIT_ARGS = ["--analysis-scope", "all", "--epsilon", "0.1", "--lambda-a", "1.0", "--lambda-b", "1.0", "--tolerance", "1e-4",
            "--save-coupling", "--device", "cpu", "--workers", "1", "--seed", "20260925", "--within-side-acceptance-minimum", "0.99",
            "--allow-precomputed"]
KEYS = ["stage", "level", "replicate", "case", "pair_id", "config_id", "arm"]
# one fixed column order for every appended chunk (arms write different diagnostics)
COLUMNS = KEYS + ["seconds", "status", "method", "side", "cells", "to_reject", "rejected", "rejected_fraction", "precision", "recall", "f1",
                  "false_rejection_rate", "rejection_cost", "m4e_inference_valid", "asserted_mass", "false_asserted_mass",
                  "false_asserted_share", "error", "diag_genebatch_scale", "diag_genebatch_clipped_share", "diag_mnnc_consistency",
                  "diag_mnncred_share", "diag_mnncred_reference", "diag_mnncred_weight", "diag_mnncred_disjoint_reference"]
_RUN_PAIR = None
_DEPTH = {}


def env() -> dict:
    sep = ";" if os.name == "nt" else ":"
    return {**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO / "realdata_benchmark", REPO]))}


def seed_of(rep: int, level: str) -> int:
    return 20260925 + 1000 * rep + 137 * {"L0_matched": 0, "L1_depth": 1}.get(level, 2)


def generate(root: Path, rscript: str, n: int, rep: int, level: str) -> str:
    gen = root / "generated" / f"rep{rep}" / level; cond = root / "conditions" / f"rep{rep}" / level
    if not (gen / "GENERATION_DONE").exists():
        r = subprocess.run([rscript, str(REPO / "realdata_benchmark" / "splatter_n5000" / "20_generate_n5000.R"), "--out", str(gen), "--n", str(n),
                            "--replicate", str(rep), "--level", level, "--seed", str(seed_of(rep, level)), "--param-set", "splatter_default",
                            "--pairing", "paired"], capture_output=True, text=True, cwd=REPO)
        if r.returncode:
            return f"generation failed {level} rep{rep}: {r.stderr[-500:]}"
    if not (cond / "manifest.csv").exists():
        r = subprocess.run([sys.executable, str(REPO / "realdata_benchmark" / "splatter_n5000" / "21_build_conditions_n5000.py"), "--generated", str(gen),
                            "--out", str(cond), "--n-cells", str(n), "--replicate", str(rep), "--level", level], capture_output=True, text=True,
                           cwd=REPO, env=env())
        if r.returncode:
            return f"conditions failed {level} rep{rep}: {r.stdout[-300:]} {r.stderr[-500:]}"
    return f"{level} rep{rep} ready"


def run_pair(argv: list[str]) -> tuple[int, str]:
    """02_run_pair.py main() in this process (loaded once per worker)."""
    global _RUN_PAIR
    if _RUN_PAIR is None:
        spec = importlib.util.spec_from_file_location("run_pair_02", REPO / "cancer_metastasis" / "02_run_pair.py")
        _RUN_PAIR = importlib.util.module_from_spec(spec); spec.loader.exec_module(_RUN_PAIR)
    old, buf, code = sys.argv, io.StringIO(), 0
    sys.argv = ["02_run_pair.py", *argv]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _RUN_PAIR.main()
    except SystemExit as e:
        code = int(e.code or 0) if not isinstance(e.code, str) else 1
    except Exception:
        code = 1; buf.write(traceback.format_exc())
    finally:
        sys.argv = old
    return code, buf.getvalue()


def depth_target(manifest: str, ds: str) -> int:
    import anndata as ad
    import rb_common as rb
    if manifest not in _DEPTH:
        man = pd.read_csv(manifest)
        files = sorted(set(man.source_h5ad) | set(man.target_h5ad))
        _DEPTH[manifest] = np.concatenate([np.asarray(ad.read_h5ad(f).X.sum(1)).ravel() for f in files])
    return int(np.quantile(_DEPTH[manifest], rb.DS_QUANTILES[ds]))


_LAST: dict = {}


def gene_batch(S, T):
    """Splatter-informed gene-wise batch estimate on the counts, needing no matched cells: the batch is a dense, small,
    population-independent log shift of every gene, biology a sparse, large one ('most genes are not differential',
    as DESeq2 / TMM). Per gene with CPM >= 1 on both sides: d = log2(CPM_target + 1) - log2(CPM_source + 1); robust
    centre m = median(d), scale s = 1.4826 MAD(d); batch b = m + clip(d - m, -1.345 s, 1.345 s) (Huber's constant);
    genes below CPM 1 on either side are left alone. The target's counts are divided by 2^b gene by gene (they become
    non-integer); the source is untouched."""
    import scipy.sparse as sp
    xs = np.asarray(S.X.sum(0)).ravel(); xt = np.asarray(T.X.sum(0)).ravel()
    cs, ct = xs / max(xs.sum(), 1) * 1e6, xt / max(xt.sum(), 1) * 1e6
    ok = (cs >= 1) & (ct >= 1)
    d = np.log2(ct[ok] + 1) - np.log2(cs[ok] + 1)
    m = np.median(d); s = 1.4826 * np.median(np.abs(d - m))
    b = np.zeros(len(cs)); b[ok] = m + np.clip(d - m, -1.345 * s, 1.345 * s)
    _LAST["genebatch_scale"] = float(s); _LAST["genebatch_clipped_share"] = float((np.abs(d - m) > 1.345 * s).mean())
    T = T.copy(); T.X = sp.csr_matrix(sp.csr_matrix(T.X, dtype=np.float64) @ sp.diags(2.0 ** (-b)))
    return S, T


def correct(X: np.ndarray, Y: np.ndarray, correction: str) -> tuple[np.ndarray, np.ndarray]:
    if not correction:
        return X, Y
    if correction == "harmony":
        import harmonypy
        Z = np.vstack([X, Y]).astype(np.float64)
        meta = pd.DataFrame({"side": ["source"] * len(X) + ["target"] * len(Y)})
        import logging; logging.getLogger("harmonypy").setLevel(logging.WARNING)
        Zc = np.asarray(harmonypy.run_harmony(Z, meta, ["side"], random_state=0, verbose=False).Z_corr).T
        return Zc[: len(X)], Zc[len(X):]
    if correction == "center":
        return X - X.mean(0), Y - Y.mean(0)
    if correction in ("mnnc", "genebatch-mnnc"):
        # MNN shift weighted by its consistency across source regions (8 k-means regions of the source, regions with >= 5
        # anchors): c = |mean of the regional median displacements|^2 / mean |regional displacement|^2, in [0, 1]. A shift
        # common to every region (a batch acting on all populations alike) gives c near 1; anchors pulled towards
        # different populations in different regions (nothing shared) give a low c; no shift at all gives noise and a
        # low c. The target moves by c x the global median displacement; no threshold.
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
        k = min(20, len(X) - 1, len(Y) - 1)
        nx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Y, return_distance=False)
        ny = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(X, return_distance=False)
        pairs = [(i, j) for j in range(len(Y)) for i in nx[j] if j in set(ny[i])]
        if not pairs:
            return X, Y
        i, j = map(np.asarray, zip(*pairs))
        D = Y[j] - X[i]
        region = KMeans(n_clusters=min(8, len(X)), n_init=10, random_state=0).fit(X).labels_[i]
        V = np.array([np.median(D[region == r], axis=0) for r in np.unique(region) if (region == r).sum() >= 5])
        if len(V) == 0:
            return X, Y
        c = float(np.sum(V.mean(0) ** 2) / max(np.mean(np.sum(V ** 2, axis=1)), 1e-12))
        _LAST["mnnc_consistency"] = c
        return X, Y - c * np.median(D, axis=0)
    if correction in ("mnncred", "mnncred2", "mnncred3"):
        # credible-anchor MNN translation, soft-weighted by data-internal evidence (no labels, no threshold tuned on F1):
        # 1. anchors = cross-side mutual nearest neighbours (k = 20); shift = their median displacement;
        # 2. credible anchors = pairs whose distance after the shift is below the 20th same-side neighbour distance of
        #    both cells; the shift is re-estimated from the credible anchors (3 rounds);
        # 3. reference = the same credible share between two random halves of the source (a split known to share
        #    everything, no batch); weight w = min(1, observed credible share / reference);
        # 4. the target moves by w x shift. Nothing shared -> few credible anchors -> w small -> no forced alignment.
        from sklearn.neighbors import NearestNeighbors
        def credible(A, B):
            k = min(20, len(A) - 1, len(B) - 1)
            ra = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[0][:, k]
            rb = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B)[0][:, k]
            na = NearestNeighbors(n_neighbors=k).fit(A).kneighbors(B, return_distance=False)
            nb = NearestNeighbors(n_neighbors=k).fit(B).kneighbors(A, return_distance=False)
            pairs = [(i, j) for j in range(len(B)) for i in na[j] if j in set(nb[i])]
            if not pairs:
                return np.zeros(A.shape[1]), 0.0
            i, j = map(np.asarray, zip(*pairs))
            D = B[j] - A[i]; shift = np.median(D, axis=0); share = 0.0
            for _ in range(3):
                ok = np.linalg.norm(D - shift, axis=1) < np.minimum(ra[i], rb[j])
                share = float(ok.mean())
                if ok.sum() < 5:
                    break
                shift = np.median(D[ok], axis=0)
            if correction == "mnncred3":
                # the evidence is the most credible region: 8 k-means regions of the B side, regions with >= 10 anchors;
                # one sizeable shared population is enough to trust the translation, while with nothing shared every
                # region stays low. The shift is re-estimated from that region's credible anchors.
                from sklearn.cluster import KMeans
                reg = KMeans(n_clusters=min(8, len(B)), n_init=10, random_state=0).fit(B).labels_[j]
                best, best_r = -1.0, None
                for r_ in np.unique(reg):
                    m_ = reg == r_
                    if m_.sum() >= 10 and ok[m_].mean() > best:
                        best, best_r = float(ok[m_].mean()), r_
                if best_r is None:
                    return shift, share
                sel = (reg == best_r) & ok
                if sel.sum() >= 5:
                    shift = np.median(D[sel], axis=0)
                return shift, best
            return shift, share
        shift, share = credible(X, Y)
        rng = np.random.default_rng(0); order = rng.permutation(len(X)); h = len(X) // 2
        _, ref = credible(X[order[:h]], X[order[h:2 * h]])
        if correction == "mnncred":
            w = float(min(1.0, share / ref)) if ref > 0 else 0.0
        else:  # mnncred2: also a 'nothing shared' reference from the data -- the source split into its two k-means halves
            from sklearn.cluster import KMeans
            lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X).labels_
            _, dis = credible(X[lab == 0], X[lab == 1]) if min((lab == 0).sum(), (lab == 1).sum()) > 21 else (None, 0.0)
            w = float(np.clip((share - dis) / (ref - dis), 0.0, 1.0)) if ref > dis else 0.0
            _LAST.update(mnncred_disjoint_reference=dis)
        _LAST.update(mnncred_share=share, mnncred_reference=ref, mnncred_weight=w)
        return X, Y - w * shift
    if correction in ("mnncov", "mnncovgate", "mnncovlocal"):
        # credible-coverage MNN (shared_evidence_v2.py, statistic cover_min): after the iterative credible-anchor
        # translation (k = 20 anchors, credible = closer than both cells' 20th same-side neighbour), the share of each
        # side's cells covered by a credible anchor, the smaller side's; normalised between two references from the
        # data itself (random halves of the source: shares everything; its two k-means halves: shares nothing):
        # score = clip((observed - disjoint) / (shared - disjoint), 0, 1).
        # mnncov: target moves by score x shift; mnncovgate: by the full shift when score > 0 (coverage above what two
        # disjoint population sets give), else not at all.
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
        def evidence(A, B):
            k = min(20, len(A) - 1, len(B) - 1)
            ra = NearestNeighbors(n_neighbors=k + 1).fit(A).kneighbors(A)[0][:, k]
            rb = NearestNeighbors(n_neighbors=k + 1).fit(B).kneighbors(B)[0][:, k]
            na = NearestNeighbors(n_neighbors=k).fit(A).kneighbors(B, return_distance=False)
            nb = NearestNeighbors(n_neighbors=k).fit(B).kneighbors(A, return_distance=False)
            pairs = [(i, j) for j in range(len(B)) for i in na[j] if j in set(nb[i])]
            if not pairs:
                return np.zeros(A.shape[1]), 0.0
            i, j = map(np.asarray, zip(*pairs))
            D = B[j] - A[i]; shift = np.median(D, axis=0)
            for _ in range(3):
                ok = np.linalg.norm(D - shift, axis=1) < np.minimum(ra[i], rb[j])
                if ok.sum() < 5:
                    break
                shift = np.median(D[ok], axis=0)
            _LAST["_anchors"] = (j[ok], D[ok], float(np.median(rb)))
            return shift, float(min(len(set(i[ok])) / len(A), len(set(j[ok])) / len(B)))
        shift, obs = evidence(X, Y)
        anchors = _LAST.pop("_anchors", None)
        rng = np.random.default_rng(0); o = rng.permutation(len(X)); h = len(X) // 2
        _, ref = evidence(X[o[:h]], X[o[h:2 * h]])
        lab = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X).labels_
        _, dis = evidence(X[lab == 0], X[lab == 1]) if min((lab == 0).sum(), (lab == 1).sum()) > 21 else (None, 0.0)
        score = float(np.clip((obs - dis) / (ref - dis), 0.0, 1.0)) if ref > dis else 0.0
        w = score if correction == "mnncov" else float(score > 0)
        _LAST.pop("_anchors", None)
        _LAST.update(mnncred_share=obs, mnncred_reference=ref, mnncred_disjoint_reference=dis, mnncred_weight=w)
        if correction == "mnncovlocal" and w > 0 and anchors is not None and len(anchors[0]) >= 5:
            # same gate; when open, each target cell moves by the Gaussian-kernel average of the credible anchor
            # displacements around it (bandwidth = median 20th same-side neighbour distance of the target), shrunk
            # towards the global credible shift by one pseudo-anchor of weight 1
            ja, Da, h = anchors
            d2 = ((Y[:, None, :] - Y[ja][None, :, :]) ** 2).sum(-1)
            K = np.exp(-d2 / (2 * h * h))
            disp = (K @ Da + shift[None, :]) / (K.sum(1, keepdims=True) + 1.0)
            return X, Y - disp
        return X, Y - w * shift
    if correction == "genebatch":  # applied on the counts (gene_batch), nothing left to do on the PCA
        return X, Y
    if correction == "mnn":
        # rigid translation of the target by the median displacement of cross-side mutual nearest neighbours (k = 20,
        # the fastMNN default): estimated from matched pairs, so a population present on one side only barely moves it;
        # every within-side distance is unchanged
        from sklearn.neighbors import NearestNeighbors
        k = min(20, len(X) - 1, len(Y) - 1)
        nx = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(Y, return_distance=False)  # for each target, its source NN
        ny = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(X, return_distance=False)  # for each source, its target NN
        pairs = [(i, j) for j in range(len(Y)) for i in nx[j] if j in set(ny[i])]
        if not pairs:
            return X, Y
        i, j = map(np.asarray, zip(*pairs))
        shift = np.median(Y[j] - X[i], axis=0)
        return X, Y - shift
    if correction == "quantile":
        from scipy.stats import rankdata
        pooled = np.sort(np.vstack([X, Y]), axis=0); m = pooled.shape[0]
        def qmap(V):
            idx = np.clip((((rankdata(V, axis=0) - 0.5) / V.shape[0]) * m).astype(int), 0, m - 1)
            return np.take_along_axis(pooled, idx, axis=0)
        return qmap(X), qmap(Y)
    raise ValueError(correction)


def score(run: Path, truth: pd.DataFrame, base: dict) -> list[dict]:
    cells = pd.read_csv(run / "cell_confidence.csv"); pm = pd.read_csv(run / "pair_metrics.csv").set_index("method")
    tr = truth.set_index(["side", "observation_id"])
    rows = []
    for method in ("M4-E", "M4-R"):
        c = cells[cells.method == method].copy()
        t = tr.reindex(pd.MultiIndex.from_frame(c[["side", "observation_id"]]))
        c["should_reject"] = t.should_reject.to_numpy(bool); c["group"] = t.group.to_numpy()
        coupling = np.load(run / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
        sc, tc = c[c.side == "source"], c[c.side == "target"]
        keep = np.outer(sc.retained.to_numpy(bool), tc.retained.to_numpy(bool))
        diff = sc.group.to_numpy()[:, None] != tc.group.to_numpy()[None, :]
        asserted = float((coupling * keep).sum()); false_asserted = float((coupling * keep * diff).sum())
        for side, d in (("source", sc), ("target", tc)):
            rej = d.rejected.to_numpy(bool); pos = d.should_reject.to_numpy(bool)
            tp, fp, fn = int((rej & pos).sum()), int((rej & ~pos).sum()), int((~rej & pos).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / (tp + fn) if pos.any() else np.nan
            f1 = np.nan if not pos.any() else (0.0 if tp == 0 else 2 * prec * rec / (prec + rec))
            rows.append(base | {"status": "ok", "method": method, "side": side, "cells": len(d), "to_reject": int(pos.sum()),
                                "rejected": int(rej.sum()), "rejected_fraction": float(rej.mean()), "precision": prec, "recall": rec, "f1": f1,
                                "false_rejection_rate": fp / int((~pos).sum()) if (~pos).any() else np.nan,
                                "rejection_cost": float(pm.loc[method, "rejection_cost"]),
                                "m4e_inference_valid": pm.loc[method].get("calibration_m4e_inference_valid", np.nan),
                                "asserted_mass": asserted, "false_asserted_mass": false_asserted,
                                "false_asserted_share": false_asserted / asserted if asserted > 0 else np.nan})
    return rows


def fit_coords(task: dict, X: np.ndarray, Y: np.ndarray, cos: bool, base: dict, work: Path) -> list[dict]:
    import anndata as ad
    man = pd.read_csv(task["manifest"]); i = int(task["pair_index"])
    shutil.rmtree(work, ignore_errors=True); (work / "input").mkdir(parents=True)
    cols = pd.Index([f"PC{k + 1}" for k in range(X.shape[1])], dtype=str)
    obs = task["_obs"]
    for coords, side in ((X, "source"), (Y, "target")):
        ad.AnnData(X=np.asarray(coords, np.float32), obs=obs[side], var=pd.DataFrame(index=cols)).write_h5ad(work / "input" / f"{side}.h5ad")
    m = man.copy()
    for side in ("source", "target"):
        m.loc[i, f"{side}_h5ad"] = str(work / "input" / f"{side}.h5ad")
        if f"{side}_h5ads_json" in m:
            m.loc[i, f"{side}_h5ads_json"] = json.dumps([str(work / "input" / f"{side}.h5ad")])
    m.to_csv(work / "manifest.csv", index=False)
    null = base["arm"].split("+")[0]
    started = time.time()
    code, log = run_pair([str(work / "manifest.csv"), str(work / "runs"), "--index", str(i), "--preprocessing",
                          "coord_allgenes_nopca_noscale" + ("_cos" if cos else ""), "--calibration-null", null, *FIT_ARGS])
    base = base | {"seconds": round(time.time() - started, 2)}
    if code:
        (Path(task["root"]) / "failed").mkdir(exist_ok=True)
        (Path(task["root"]) / "failed" / f"{work.name}.log").write_text(log[-4000:])
        shutil.rmtree(work, ignore_errors=True)
        return [base | {"status": f"failed {code}"}]
    rows = score(next((work / "runs").rglob("cell_confidence.csv")).parent, task["_truth"], base)
    shutil.rmtree(work, ignore_errors=True)
    return rows


def run_task(task: dict) -> list[dict]:
    """One pair, one representation group: build once, fit every variant."""
    import anndata as ad
    import rb_common as rb
    import factorial_common as fc
    man = pd.read_csv(task["manifest"]); i = int(task["pair_index"]); r = man.iloc[i]
    S, T = ad.read_h5ad(r.source_h5ad), ad.read_h5ad(r.target_h5ad)
    task["_obs"] = {"source": S.obs.copy(), "target": T.obs.copy()}
    task["_truth"] = pd.read_csv(task["truth"], sep="\t")
    work = Path(task["work"]); rows = []
    head = {k: task[k] for k in ("stage", "level", "replicate", "case", "pair_id")}
    try:
        if task["kind"] == "production":  # calibration stage and the grid baseline: registry spec, production path
            spec = task["spec"]
            if spec["ds"] != "none":
                tg = depth_target(task["manifest"], spec["ds"]); s = int(float(spec["ds_seed"]))
                S, T = rb.equalised(S, tg, 1000 * s + 1), rb.equalised(T, tg, 1000 * s + 2)
            coords = {}
            for arm in task["arms"]:
                corr = (arm.split("+") + [""])[1]
                key = "genebatch" if corr.startswith("genebatch") else "raw"
                if key not in coords:
                    S2, T2 = gene_batch(S, T) if key == "genebatch" else (S, T)
                    coords[key] = rb.build_config(dict(spec), S2, T2, 20260925 + i)[:2] + (dict(_LAST),)
                X0, Y0, diag = coords[key]
                _LAST.clear(); _LAST.update(diag)
                X, Y = correct(X0, Y0, corr)
                extra = {f"diag_{k}": v for k, v in _LAST.items()}
                rows += fit_coords(task, X, Y, spec["cost"] == "cos", head | {"config_id": task["config_id"], "arm": arm} | extra,
                                   work / arm.replace("+", "_"))
        else:  # grid: one encoding key; PCA {30, 50} x scaling; both costs share the PCA
            spec = task["spec"]
            s_, t_ = (S, T) if spec["ds"] == "none" else fc.equalise_sides(S, T, spec["ds"], {spec["ds"]: depth_target(task["manifest"], spec["ds"])})
            se, te, _, _ = fc.encode_sides(spec, s_, t_)
            import scipy.sparse as sp
            joint = sp.vstack([sp.csr_matrix(se.X), sp.csr_matrix(te.X)], format="csr"); n = se.n_obs
            n_hvg = int(spec["H"]) if spec["encoding"] == "rankhvg" else fc.N_HVG
            for n_pcs in task["pcs"]:
                for scale in (False, True):
                    P, _ = rb.pca_coordinates(joint, n_hvg, n_pcs, scale, False, 20260925 + i)
                    for cost in ("cos", "sqeuc"):
                        from confidenceot.preprocessing import unit_rows
                        C = unit_rows(P) if cost == "cos" else P
                        X, Y = correct(C[:n], C[n:], (task["arm"].split("+") + [""])[1])
                        cid = fc.config_id({**spec, "scale": scale, "cost": cost}).replace(f"|pca{fc.N_PCS}", f"|pca{n_pcs}")
                        rows += fit_coords(task, X, Y, cost == "cos", head | {"config_id": cid, "arm": task["arm"]},
                                           work / f"{n_pcs}_{int(scale)}_{cost}")
    except Exception:
        rows.append(head | {"config_id": task.get("config_id", task.get("key", "")), "arm": task.get("arm", ""), "status": "failed task",
                            "error": traceback.format_exc()[-500:]})
    shutil.rmtree(work, ignore_errors=True)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("calibration", "grid"), required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--replicates", type=int, nargs="+", default=[1, 2])
    p.add_argument("--levels", nargs="+", default=None)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--pcs", type=int, nargs="+", default=[30, 50])
    p.add_argument("--jobs", type=int, default=24)
    p.add_argument("--max-tasks", type=int, default=0, help="run only the first N tasks (smoke test; finished keys are skipped later)")
    p.add_argument("--rscript", default="D:/R-4.4.1/bin/Rscript.exe")
    p.add_argument("--registry", type=Path, default=REPO / "results" / "realdata_benchmark_20261031" / "PREPROCESSING_CONFIG_REGISTRY.tsv")
    p.add_argument("--config", default=BASELINE, help="calibration stage: the fixed representation (registry config id)")
    p.add_argument("--frozen", default="FROZEN_CALIBRATION.json", help="grid stage: the frozen calibration file (arm, config)")
    p.add_argument("--no-ds", action="store_true", help="grid stage: depth-equalised configurations left out (DS none only)")
    p.add_argument("--drop-duplicates", action="store_true", help="grid stage: rankhvg expected left out (identical to midrank, no cut)")
    a = p.parse_args()
    levels = a.levels or (ALL_LEVELS if a.stage == "calibration" else GRID_LEVELS)
    a.root.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(generate, a.root, a.rscript, a.n, rep, lv) for rep in a.replicates for lv in levels]):
            print(f.result(), flush=True)
    reg = pd.read_csv(a.registry, sep="\t").set_index("config_id")
    res = a.root / f"RESULTS_{a.stage}{'_' + a.tag if a.tag else ''}.tsv"
    done = set()
    if res.exists():
        old = pd.read_csv(res, sep="\t")
        if list(old.columns) != COLUMNS:  # realign a file written with an older column set before appending to it
            old.reindex(columns=COLUMNS).to_csv(res, sep="\t", index=False)
        old = old[old.status == "ok"]
        done = set(zip(old.level, old.replicate.astype(int), old.pair_id, old.config_id, old.arm))
    cfg = a.config
    if a.stage == "grid":
        fz = json.loads((a.root / a.frozen).read_text())
        frozen, cfg = fz["arm"], fz.get("config", BASELINE)
        import factorial_common as fc
        keys = {}
        for s in fc.grid():
            if (a.no_ds and s["ds"] != "none") or (a.drop_duplicates and s["encoding"] == "rankhvg" and s["tie"] == "expected"):
                continue
            k = (s["encoding"], s["K"], s["H"], s["tie"], s["ds"])
            keys.setdefault(k, s)
    tasks = []
    for rep in a.replicates:
        for lv in levels:
            cond = a.root / "conditions" / f"rep{rep}" / lv
            man = pd.read_csv(cond / "manifest.csv")
            for i, r in man.iterrows():
                base = {"root": str(a.root), "stage": a.stage, "level": lv, "replicate": rep, "case": r.case, "pair_id": r.pair_id,
                        "manifest": str(cond / "manifest.csv"), "pair_index": i, "truth": str(cond / str(r.pair_id) / "truth" / "population_truth.tsv")}
                tag = f"{lv}_rep{rep}_{i}"
                if a.stage == "calibration":
                    arms = [x for x in a.arms if (lv, rep, r.pair_id, cfg, x) not in done]
                    if arms:
                        tasks.append(base | {"kind": "production", "config_id": cfg, "spec": reg.loc[cfg].to_dict(), "arms": arms,
                                             "work": str(a.root / "work" / tag)})
                else:
                    if (lv, rep, r.pair_id, cfg, frozen) not in done:
                        tasks.append(base | {"kind": "production", "config_id": cfg, "spec": reg.loc[cfg].to_dict(), "arms": [frozen],
                                             "work": str(a.root / "work" / f"{tag}_baseline")})
                    for k, s in keys.items():
                        probe = fc.config_id({**s, "scale": True, "cost": "sqeuc"}).replace(f"|pca{fc.N_PCS}", f"|pca{a.pcs[-1]}")
                        if (lv, rep, r.pair_id, probe, frozen) in done:
                            continue
                        tasks.append(base | {"kind": "grid", "spec": s, "arm": frozen, "pcs": a.pcs, "key": str(k),
                                             "work": str(a.root / "work" / f"{tag}_{'_'.join(map(str, k))}")})
    if a.max_tasks:
        tasks = tasks[: a.max_tasks]
    print(f"{len(tasks)} tasks ({a.stage}), {a.jobs} jobs -> {res}", flush=True)
    started = time.time(); n = 0; fits = 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futures = [ex.submit(run_task, t) for t in tasks]
        for f in as_completed(futures):
            rows = f.result(); n += 1; fits += len({(x.get("config_id"), x.get("arm")) for x in rows})
            pd.DataFrame(rows).reindex(columns=COLUMNS).to_csv(res, sep="\t", index=False, mode="a", header=not res.exists())
            if n % 20 == 0 or n == len(tasks):
                el = time.time() - started
                print(f"{n} / {len(tasks)} tasks, {fits} fits, {el:.0f} s, eta {el / n * (len(tasks) - n):.0f} s", flush=True)
    shutil.rmtree(a.root / "work", ignore_errors=True)
    print(f"-> {res}")


if __name__ == "__main__":
    main()

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
            X0, Y0, _ = rb.build_config(dict(spec), S, T, 20260925 + i)
            for arm in task["arms"]:
                X, Y = correct(X0, Y0, (arm.split("+") + [""])[1])
                rows += fit_coords(task, X, Y, spec["cost"] == "cos", head | {"config_id": task["config_id"], "arm": arm},
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
        old = old[old.status == "ok"]
        done = set(zip(old.level, old.replicate.astype(int), old.pair_id, old.config_id, old.arm))
    if a.stage == "grid":
        frozen = json.loads((a.root / "FROZEN_CALIBRATION.json").read_text())["arm"]
        import factorial_common as fc
        keys = {}
        for s in fc.grid():
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
                    arms = [x for x in a.arms if (lv, rep, r.pair_id, BASELINE, x) not in done]
                    if arms:
                        tasks.append(base | {"kind": "production", "config_id": BASELINE, "spec": reg.loc[BASELINE].to_dict(), "arms": arms,
                                             "work": str(a.root / "work" / tag)})
                else:
                    if (lv, rep, r.pair_id, BASELINE, frozen) not in done:
                        tasks.append(base | {"kind": "production", "config_id": BASELINE, "spec": reg.loc[BASELINE].to_dict(), "arms": [frozen],
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
            pd.DataFrame(rows).to_csv(res, sep="\t", index=False, mode="a", header=not res.exists())
            if n % 20 == 0 or n == len(tasks):
                el = time.time() - started
                print(f"{n} / {len(tasks)} tasks, {fits} fits, {el:.0f} s, eta {el / n * (len(tasks) - n):.0f} s", flush=True)
    shutil.rmtree(a.root / "work", ignore_errors=True)
    print(f"-> {res}")


if __name__ == "__main__":
    main()

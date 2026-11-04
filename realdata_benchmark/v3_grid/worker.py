"""One worker of the frozen-calibration-v3 grid on TACC (v3_common.py): claims worklist rows one at a time and fits them.
Any number of workers on any number of nodes share one worklist; a row is claimed by creating OUT/.claim atomically.

    python realdata_benchmark/v3_grid/worker.py --worklist WORKLIST.tsv [--device cpu|cuda] [--fit-workers W]
        [--retry-failed] [--max-rows N]

A row is one unit (a Splatter pair, an LTL331 adjacent pair, a KP-Tracer lineage unit) and one encoding key; its 8
configurations (PCA 30 / 50 x scaling x cos / sqeuc) share the encoding and each PCA. Per configuration OUT/<config>/:
  cell_confidence.csv.gz, pair_metrics.csv, run.json, diag.json (the correction's gate statistics), and
  coupling_m4e.npz / coupling_m4r.npz: the transition matrices of every fit, every dataset (user, 2026-11-04):
            float32, compressed; source_ids / target_ids are their rows / columns, the gates' cells
  splatter  score.tsv: run_local_benchmark.score() against the pair's 21_ truth (raw rows; summarize_local.py derives
            the effective readout)
  ltl331    coupling_clusters.tsv.gz: M4-E / M4-R coupling mass between the authors' seurat_clusters, over all cells and
            over the retained block
A claim whose Slurm job has ended without SUCCESS / FAILED is taken over (claim()). A configuration with DONE is
skipped, so a row interrupted part-way resumes; OUT/SUCCESS when all 8 are done, OUT/FAILED
(with the configurations that failed) otherwise. Inputs are deleted after each fit (reproducible from the row).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import shutil
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_common as v3  # noqa: E402

REPO = v3.REPO


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def set_arg(args: list[str], flag: str, value: str) -> list[str]:
    args = list(args)
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]
    return args


def fit_args(dataset: str, device: str, workers: int) -> list[str]:
    if dataset == "splatter":  # the local benchmark's arguments (run_local_benchmark.FIT_ARGS), within-side split null
        args = [x for x in v3.lb.FIT_ARGS if x != "--allow-precomputed"] + ["--calibration-null", "within_side_split"]
    elif dataset == "ltl331":  # 05_ltl331_ot_pilot.FACTORIAL_ARGS; --cell-qc dropped (0 / 0 / 100 passes every cell, and it refuses negative values)
        pilot = load_module("ltl331_pilot_05", REPO / "realdata_benchmark" / "05_ltl331_ot_pilot.py")
        args = [x for x in pilot.FACTORIAL_ARGS if x != "--cell-qc"]
    elif dataset == "kp":  # factorial_common.fit_command (the KP-Tracer runs) without the row's own arguments and --cell-qc
        cmd = v3.fc.fit_command("python", "02_run_pair.py", "M", "O", 0, "L", 50, workers, precomputed=False)[4:]
        args, skip = [], 0
        for x in cmd:
            if skip:
                skip -= 1; continue
            if x in ("--index", "--preprocessing", "--n-pcs"):
                skip = 1; continue
            if x != "--cell-qc":
                args.append(x)
    else:
        raise ValueError(dataset)
    return set_arg(set_arg(args, "--device", device), "--workers", str(workers))


class Inputs:
    """The unit's count sides, manifest, index and PCA seed; cached for the worker's next row of the same unit."""

    def __init__(self):
        self.key, self.value, self._kp = None, None, None

    def get(self, row: pd.Series) -> dict:
        if self.key == row.unit_id:
            return self.value
        import anndata as ad
        if row.dataset == "splatter":
            man = pd.read_csv(row.manifest); i = int(row["index"]); r = man.iloc[i]
            if str(r.pair_id) != str(row.pair_id):
                raise SystemExit(f"manifest row {i} is {r.pair_id}, worklist says {row.pair_id}")
            S, T = ad.read_h5ad(r.source_h5ad), ad.read_h5ad(r.target_h5ad)
            v = {"S": S, "T": T, "manifest": man, "index": i, "seed": 20260925 + i, "truth": pd.read_csv(row.truth, sep="\t")}
        elif row.dataset == "ltl331":
            import rb_common as rb
            run_root = Path(row.run_root)
            man = rb.ltl_pairs(run_root, False).copy(); i = int(row["index"])
            for side in ("source", "target"):
                man[f"{side}_h5ad"] = [str(rb.local(x, run_root).resolve()) for x in man[f"{side}_h5ad"]]
                man[f"{side}_h5ads_json"] = [json.dumps([x]) for x in man[f"{side}_h5ad"]]
            if str(man.pair_id.iloc[i]) != str(row.pair_id):
                raise SystemExit(f"pair {i} is {man.pair_id.iloc[i]}, worklist says {row.pair_id}")
            S, T = ad.read_h5ad(man.source_h5ad.iloc[i]), ad.read_h5ad(man.target_h5ad.iloc[i])
            v = {"S": S, "T": T, "manifest": man, "index": i, "seed": rb.SEED + i}
        elif row.dataset == "kp":
            if self._kp is None:
                lw = load_module("lf_worker_10", REPO / "lineage_validation" / "preprocessing_factorial_20261103" / "10_worker.py")
                self._kp = (lw, lw.Sides(lw.run_pairwise()))
            lw, sides = self._kp
            units = pd.read_csv(row.units, sep="\t", dtype={"mouse": str}).set_index("unit_id", drop=False)
            u = units.loc[row.unit_id]; s = sides.get(u)
            if u.level == "L1":
                man = pd.read_csv(u.manifest); i = int(u["index"])
            else:
                man = pd.DataFrame([{"pair_id": u.pair_or_case, "dataset_id": u.dataset, "patient_id": u.mouse,
                                     "source_sample": f"{u.pair_or_case}__source", "target_sample": f"{u.pair_or_case}__target",
                                     "source_h5ad": "", "target_h5ad": "", "eligible": True}]); i = 0
            v = {"S": s["source"], "T": s["target"], "manifest": man, "index": i, "seed": 20260831 + i}
        else:
            raise ValueError(row.dataset)
        self.key, self.value = row.unit_id, v
        return v


def cluster_coupling(run: Path, S, T) -> pd.DataFrame:
    """LTL331: coupling mass between seurat_clusters, all cells and the retained block, M4-E and M4-R."""
    cc = pd.read_csv(run / "cell_confidence.csv"); rows = []
    sc = pd.Series(S.obs["seurat_clusters"].astype(str).to_numpy(), index=S.obs_names.astype(str))
    tc = pd.Series(T.obs["seurat_clusters"].astype(str).to_numpy(), index=T.obs_names.astype(str))
    for method in ("M4-E", "M4-R"):
        g = cc[cc.method == method]
        src, tgt = g[g.side == "source"], g[g.side == "target"]
        P = np.load(run / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
        if P.shape != (len(src), len(tgt)):
            raise SystemExit(f"coupling {P.shape} vs cells {len(src)} x {len(tgt)}")
        a, b = sc.loc[src.observation_id.astype(str)].to_numpy(), tc.loc[tgt.observation_id.astype(str)].to_numpy()
        for block, rs, rt in (("all", np.ones(len(src), bool), np.ones(len(tgt), bool)),
                              ("retained", ~src.rejected.to_numpy(bool), ~tgt.rejected.to_numpy(bool))):
            Q = P[np.ix_(rs, rt)]
            ua, ub = np.unique(a[rs]), np.unique(b[rt])
            for x in ua:
                rx = Q[a[rs] == x]
                for y in ub:
                    m = float(rx[:, b[rt] == y].sum())
                    if m > 0:
                        rows.append({"method": method, "block": block, "source_cluster": x, "target_cluster": y, "mass": m})
    return pd.DataFrame(rows)


def save_couplings(run: Path, d: Path) -> None:
    """Every dataset: the M4-E and M4-R couplings (transition matrices), float32, compressed, with the source / target
    observation ids of their rows and columns in the order of cell_confidence (the gates)."""
    cc = pd.read_csv(run / "cell_confidence.csv")
    for method in ("M4-E", "M4-R"):
        tag = method.lower().replace("-", "")
        P = np.load(run / f"coupling_{tag}.npz")["coupling"]
        g = cc[cc.method == method]
        src = np.asarray(g[g.side == "source"].observation_id.astype(str).tolist(), dtype=str)  # fixed-width: loads without pickle
        tgt = np.asarray(g[g.side == "target"].observation_id.astype(str).tolist(), dtype=str)
        if P.shape != (len(src), len(tgt)):
            raise SystemExit(f"coupling {P.shape} vs cells {len(src)} x {len(tgt)}")
        np.savez_compressed(d / f"coupling_{tag}.npz", coupling=P.astype(np.float32), source_ids=src, target_ids=tgt,
                            total_mass_float64=np.float64(P.sum()))


def run_row(row: pd.Series, inputs: Inputs, args: list[str], out: Path, work: Path) -> tuple[int, dict]:
    v = inputs.get(row)
    spec = json.loads(row.spec)
    obs = {"source": v["S"].obs.copy(), "target": v["T"].obs.copy()}
    rec = {"configs": {}, "failed": []}
    t0 = time.perf_counter()
    for cid, X, Y, cos, diag in v3.representations(spec, v["S"], v["T"], v["seed"]):
        d = out / cid.replace("|", "__")
        if (d / "DONE").exists():
            continue
        started = time.perf_counter()
        code, log, run = v3.fit(X, Y, cos, obs, v["manifest"], v["index"], args, work)
        if code != 0 or run is None:
            d.mkdir(parents=True, exist_ok=True); (d / "fit.log").write_text(log[-20000:])
            rec["failed"].append(cid); shutil.rmtree(work, ignore_errors=True); continue
        d.mkdir(parents=True, exist_ok=True)
        pd.read_csv(run / "cell_confidence.csv").to_csv(d / "cell_confidence.csv.gz", index=False, compression="gzip")
        for f in ("pair_metrics.csv", "run.json"):
            shutil.copy2(run / f, d / f)
        (d / "diag.json").write_text(json.dumps(diag, default=float))
        if row.dataset == "splatter":
            base = {"stage": "grid", "level": row.level, "replicate": int(row.replicate), "case": row.case, "pair_id": row.pair_id,
                    "config_id": cid, "arm": v3.ARM, "seconds": round(time.perf_counter() - started, 2)} | diag
            pd.DataFrame(v3.lb.score(run, v["truth"], base)).reindex(columns=v3.lb.COLUMNS).to_csv(d / "score.tsv", sep="\t", index=False)
        elif row.dataset == "ltl331":
            cluster_coupling(run, v["S"], v["T"]).to_csv(d / "coupling_clusters.tsv.gz", sep="\t", index=False, compression="gzip")
        save_couplings(run, d)
        shutil.rmtree(work, ignore_errors=True)
        rec["configs"][cid] = round(time.perf_counter() - started, 1)
        (d / "DONE").write_text("")
    rec["row_seconds"] = round(time.perf_counter() - t0, 1)
    return (0 if not rec["failed"] else 1), rec


def take_slot(slots: Path, n: int) -> Path | None:
    slots.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        try:
            (slots / f"slot{k}").mkdir(); return slots / f"slot{k}"
        except FileExistsError:
            continue
    return None


def max_large(a) -> int:
    """--max-large, or the number in MAX_LARGE_PER_NODE next to the worklist when that file exists: read before every
    large row, so running and queued jobs take a new limit without a resubmission (scheduling only, results unchanged)."""
    try:
        return max(1, int((a.worklist.parent / "MAX_LARGE_PER_NODE").read_text().strip()))
    except (OSError, ValueError):
        return a.max_large


_FINISHED: set[str] = set()


def claim(out: Path, owner: str) -> bool:
    """Claim a row atomically. A claim left by a Slurm job that is no longer queued or running (a pilot or node that hit
    its time limit) is taken over: renamed away (atomic, so one worker wins), then claimed again."""
    try:
        (out / ".claim").mkdir(); return True
    except FileExistsError:
        pass
    try:
        job = (out / ".claim" / "owner").read_text().rsplit("job ", 1)[1].strip()
    except (OSError, IndexError):
        return False
    me = os.environ.get("SLURM_JOB_ID", "-")
    if job in ("-", me) or not job.isdigit():
        return False
    if job not in _FINISHED:
        import subprocess
        try:
            q = subprocess.run(["squeue", "-h", "-j", job], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        if q.returncode == 0 and q.stdout.strip():
            return False  # its job is still queued or running
        if q.returncode != 0 and "Invalid job id" not in q.stderr:
            return False
        _FINISHED.add(job)
    try:
        os.rename(out / ".claim", out / f".claim_stale_{job}_{os.getpid()}")
    except OSError:
        return False
    try:
        (out / ".claim").mkdir(); return True
    except FileExistsError:
        return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--fit-workers", type=int, default=1, help="02_run_pair --workers (parallel null fits within one fit)")
    p.add_argument("--scratch", type=Path, default=None, help="node-local work directory (default OUT/.work)")
    p.add_argument("--large-slots", type=Path, default=None, help="node-local directory of the large-unit slots")
    p.add_argument("--max-large", type=int, default=4, help="rows of units larger than --large-size fitted at once per node")
    p.add_argument("--large-size", type=float, default=2e7, help="source x target cells above which a row takes a large slot")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--max-rows", type=int, default=0, help="stop after this many rows (0: until none is left)")
    a = p.parse_args()
    work = pd.read_csv(a.worklist, sep="\t", dtype={"mouse": str, "pair_id": str})
    inputs = Inputs(); args = {}
    owner = f"{socket.gethostname()} pid {os.getpid()} job {os.environ.get('SLURM_JOB_ID', '-')}"
    done, stop, tried = 0, False, set()
    while not stop:  # passes over the worklist until no row is left that a large slot held back
        waiting = progress = 0
        for _, row in work.iterrows():
            out = Path(row.out)
            if out in tried or (out / "SUCCESS").exists() or ((out / "FAILED").exists() and not a.retry_failed):
                continue
            out.mkdir(parents=True, exist_ok=True)
            slot = None
            if a.large_slots is not None and float(row["size"]) > a.large_size:
                slot = take_slot(a.large_slots, max_large(a))
                if slot is None:
                    waiting += 1; continue  # every large slot of this node is busy: come back to the row
            if not claim(out, owner) and not (a.retry_failed and (out / "FAILED").exists()):
                if slot is not None:
                    slot.rmdir()
                continue
            (out / ".claim" / "owner").write_text(owner)
            (out / "FAILED").unlink(missing_ok=True); tried.add(out)
            scratch = (a.scratch / f"{os.getpid()}") if a.scratch else out / ".work"
            started = time.time()
            try:
                if row.dataset not in args:
                    args[row.dataset] = fit_args(row.dataset, a.device, a.fit_workers)
                code, rec = run_row(row, inputs, args[row.dataset], out, scratch)
            except Exception:
                code, rec = -1, {}
                (out / "error.txt").write_text(traceback.format_exc())
            shutil.rmtree(scratch, ignore_errors=True)
            if slot is not None:
                slot.rmdir()
            rec.update(unit_id=row.unit_id, key=row.key, exit=code, seconds=round(time.time() - started, 1), owner=owner, device=a.device)
            (out / "row.json").write_text(json.dumps(rec, indent=1, default=str))
            (out / ("SUCCESS" if code == 0 else "FAILED")).write_text(json.dumps({"exit": code, "failed": rec.get("failed", [])}))
            print(row.dataset, row.unit_id, row.key, "ok" if code == 0 else f"FAILED {code}", f"{rec['seconds']:.0f}s", flush=True)
            done += 1; progress += 1
            if a.max_rows and done >= a.max_rows:
                stop = True; break
        if not stop:
            if not waiting:
                break
            if not progress:
                time.sleep(60)
    print(f"worker finished: {done} rows", flush=True)


if __name__ == "__main__":
    main()

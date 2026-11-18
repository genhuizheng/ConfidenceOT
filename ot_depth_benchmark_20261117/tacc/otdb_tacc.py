"""OT depth benchmark on TACC, organised like the v3 grid (realdata_benchmark/v3_grid): a worklist of rows, claim-based
workers on any number of gg nodes, resumable, then collect. The claim / large-slot logic is the v3 worker's (imported).

    python ot_depth_benchmark_20261117/tacc/otdb_tacc.py worklist --dataset splatter --root ROOT --splatter-root SROOT [--pilot 12]
    python ot_depth_benchmark_20261117/tacc/otdb_tacc.py worklist --dataset norman|vcc --root ROOT --perturb-root PROOT [--pilot 12]
    python ot_depth_benchmark_20261117/tacc/otdb_tacc.py work     --worklist ROOT/WORKLIST.tsv [--large-slots DIR --max-large 4]
    python ot_depth_benchmark_20261117/tacc/otdb_tacc.py collect  --root ROOT
    python ot_depth_benchmark_20261117/tacc/otdb_tacc.py status   --root ROOT

Row = one unit x one cost family (C1-C4); every row fits all solvers of otdb_common.SOLVERS (independent, exact,
entropic balanced, entropic UOT, partial, ConfidenceOT with the UOT backbone, ConfidenceOT with the balanced backbone).
  splatter  S1 (every level x case), S2 (L0_matched all_shared x conflict / aligned / global x f) and S2b (the same on
            L1-L5, A4, A5); unit = level x replicate x pair (x arm x f); run_splatter.fit_one
  norman / vcc  unit = perturbation x control reference (A, B) + control vs control; run_perturb.fit_unit on the prepared
            root (run_perturb.py --stage prepare)
Per row ROOT/rows/<row>/: metrics.tsv, gates.tsv.gz, transitions.tsv (splatter), error.txt, then SUCCESS or FAILED.
collect writes ROOT/METRICS.tsv, GATES.tsv.gz, TRANSITIONS.tsv, ERRORS.txt, STATUS.tsv.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib.util
import json
import socket
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
REPO = BENCH.parent
sys.path[:0] = [str(BENCH)]
COSTS = ["C1", "C2", "C3", "C4"]
S2B_LEVELS = ["L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance",
              "A4_perturb_only", "A5_imbalance_only"]


def v3_worker():
    """The v3 worker module (claim, take_slot, max_large), imported, not copied."""
    spec = importlib.util.spec_from_file_location("v3_worker", REPO / "realdata_benchmark" / "v3_grid" / "worker.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def safe(x: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(x))


# --------------------------------------------------------------------------------------------------------- worklist
def worklist(a) -> None:
    rows = []
    if a.dataset == "splatter_ext":
        # the 60 v3 grid units (L1-L5 x every pair of the manifest x replicates), one row per unit x external method
        envs = json.loads(Path(a.envs).read_text())
        for rep in a.replicates:
            for lv in ("L1_depth", "L2_depth_dropout", "L3_batch", "L4_batch_perturb", "L5_batch_perturb_imbalance"):
                cond = a.splatter_root / "conditions" / a.conditions_subdir / f"rep{rep}" / lv
                if not (cond / "manifest.csv").exists():
                    sys.exit(f"no conditions for {lv} rep{rep}: {cond}")
                for r in pd.read_csv(cond / "manifest.csv").itertuples():
                    unit = f"{lv}__rep{rep}__{r.pair_id}"
                    for method, env in envs["methods"].items():
                        job = {"method": method, "python": envs["envs"][env], "source": r.source_h5ad, "target": r.target_h5ad,
                               "truth": str(cond / str(r.pair_id) / "truth" / "population_truth.tsv"),
                               "meta": {"level": lv, "replicate": rep, "pair_id": r.pair_id, "case": r.case, "unit": unit},
                               "extra": envs.get("extra", {}).get(method, [])}
                        rows.append({"dataset": "splatter_ext", "row_id": safe(f"{unit}__{method}"), "stage": method, "cost": "own",
                                     "size": a.cells_per_side ** 2, "job": json.dumps(job)})
    elif a.dataset == "splatter":
        import run_splatter as rs
        n = a.cells_per_side
        for stage, levels in (("s1", None), ("s2", ["L0_matched"]), ("s2", S2B_LEVELS)):
            for j in rs.jobs_for(stage, a.replicates, False, a.splatter_root, a.root, levels, a.conditions_subdir):
                j.pop("save_dir", None)
                for cost in COSTS:
                    rows.append({"dataset": "splatter", "row_id": safe(f"{stage}__{j['fit_name']}__{cost}"), "stage": stage,
                                 "cost": cost, "size": n * n, "job": json.dumps(j)})
    else:
        units = pd.read_csv(a.perturb_root / "UNITS.tsv", sep="\t")
        cells = pd.read_csv(a.perturb_root / "cells.tsv.gz", sep="\t", usecols=["perturbation", "split"], low_memory=False)
        ntr = cells[cells.split == "train"].perturbation.value_counts()
        import run_perturb as rp
        sep = rp.DATASETS[a.dataset]["target_sep"]
        for r in units.itertuples():
            m = int(0.8 * rp.N_CONTROL) if r.kind == "negative_control" else int(ntr.get(r.unit, 0))
            for cost in COSTS:
                job = {"out": str(a.perturb_root), "unit": r.unit, "control_ref": r.control_ref, "kind": r.kind, "target_sep": sep}
                rows.append({"dataset": a.dataset, "row_id": safe(f"{r.unit}__{r.control_ref}__{cost}"), "stage": "perturb",
                             "cost": cost, "size": int(0.8 * rp.N_CONTROL) * m, "job": json.dumps(job)})
    t = pd.DataFrame(rows)
    if t.row_id.duplicated().any():
        sys.exit(f"duplicate row ids: {t.row_id[t.row_id.duplicated()].head().tolist()}")
    t["out"] = [str(a.root / "rows" / x) for x in t.row_id]
    t = t.sort_values("size", ascending=False, kind="stable").reset_index(drop=True)
    a.root.mkdir(parents=True, exist_ok=True)
    dest = a.root / "WORKLIST.tsv"
    if dest.exists() and not pd.read_csv(dest, sep="\t").row_id.equals(t.row_id):
        sys.exit(f"{dest} exists and differs; not overwritten (use a new root)")
    t.to_csv(dest, sep="\t", index=False)
    print(f"{len(t)} rows -> {dest}")
    if a.pilot:
        step = max(1, len(t) // a.pilot)
        t.iloc[::step].head(a.pilot).to_csv(a.root / "PILOT_WORKLIST.tsv", sep="\t", index=False)
        print(f"pilot: {min(a.pilot, len(t))} rows spread over the size range -> {a.root / 'PILOT_WORKLIST.tsv'}")


# ------------------------------------------------------------------------------------------------------------- rows
def run_external_row(row, out: Path) -> int:
    import subprocess
    job = json.loads(row.job)
    cmd = [job["python"], "-I", str(BENCH / "run_external.py"), "--method", job["method"], "--source", job["source"],
           "--target", job["target"], "--truth", job["truth"], "--out", str(out), "--meta", json.dumps(job["meta"]), *job["extra"]]
    env = {**os.environ, "CELLOT_REPO": os.environ.get("CELLOT_REPO", ""), "PYTHONNOUSERSITE": "1"}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    (out / "run.log").write_text((r.stdout + r.stderr)[-20000:])
    return r.returncode


def run_row(row) -> dict:
    job = json.loads(row.job) | {"costs": [row.cost], "solvers": None}
    job["solvers"] = None
    if row.dataset == "splatter":
        import run_splatter as rs
        job.pop("solvers")
        return rs.fit_one(job)
    import run_perturb as rp
    job.pop("solvers")
    return rp.fit_unit(job)


def write_row(out: Path, r: dict) -> int:
    if r.get("rows"):
        pd.DataFrame(r["rows"]).to_csv(out / "metrics.tsv", sep="\t", index=False)
    if r.get("gates"):
        pd.concat(r["gates"]).to_csv(out / "gates.tsv.gz", sep="\t", index=False)
    for name, trows in (r.get("tables") or {}).items():
        if trows:
            pd.DataFrame(trows).to_csv(out / f"{name.lower()}.tsv", sep="\t", index=False)
    if r.get("err"):
        (out / "error.txt").write_text("\n\n".join(r["err"]))
    ok = bool(r.get("rows")) and not r.get("err") and all(str(x.get("status", "ok")) == "ok" for x in r["rows"])
    return 0 if ok else 1


def work(a) -> None:
    w = v3_worker()
    wl = pd.read_csv(a.worklist, sep="\t")
    owner = f"{socket.gethostname()} pid {os.getpid()} job {os.environ.get('SLURM_JOB_ID', '-')}"
    done, stop, tried = 0, False, set()
    while not stop:
        waiting = progress = 0
        for row in wl.itertuples():
            out = Path(row.out)
            if out in tried or (out / "SUCCESS").exists() or ((out / "FAILED").exists() and not a.retry_failed):
                continue
            out.mkdir(parents=True, exist_ok=True)
            slot = None
            if a.large_slots is not None and float(row.size) > a.large_size:
                slot = w.take_slot(a.large_slots, w.max_large(a))
                if slot is None:
                    waiting += 1; continue
            if not w.claim(out, owner) and not (a.retry_failed and (out / "FAILED").exists()):
                if slot is not None:
                    slot.rmdir()
                continue
            (out / ".claim" / "owner").write_text(owner)
            (out / "FAILED").unlink(missing_ok=True); tried.add(out)
            started = time.time()
            try:
                code = run_external_row(row, out) if row.dataset == "splatter_ext" else write_row(out, run_row(row))
            except Exception:
                code = -1; (out / "error.txt").write_text(traceback.format_exc())
            if slot is not None:
                slot.rmdir()
            rec = {"row_id": row.row_id, "exit": code, "seconds": round(time.time() - started, 1), "owner": owner}
            (out / "row.json").write_text(json.dumps(rec))
            (out / ("SUCCESS" if code == 0 else "FAILED")).write_text(json.dumps(rec))
            print(row.row_id, "ok" if code == 0 else f"FAILED {code}", f"{rec['seconds']:.0f}s", flush=True)
            done += 1; progress += 1
            if a.max_rows and done >= a.max_rows:
                stop = True; break
        if not stop:
            if not waiting:
                break
            if not progress:
                time.sleep(60)
    print(f"worker finished: {done} rows", flush=True)


# ------------------------------------------------------------------------------------------------- collect / status
def status_table(root: Path) -> pd.DataFrame:
    wl = pd.read_csv(root / "WORKLIST.tsv", sep="\t")
    def st(o):
        o = Path(o)
        return ("SUCCESS" if (o / "SUCCESS").exists() else "FAILED" if (o / "FAILED").exists()
                else "claimed" if (o / ".claim").exists() else "open")
    wl["status"] = [st(o) for o in wl.out]
    return wl[["row_id", "dataset", "stage", "cost", "size", "status", "out"]]


def collect(a) -> None:
    s = status_table(a.root)
    s.to_csv(a.root / "STATUS.tsv", sep="\t", index=False)
    for name, fn in (("METRICS.tsv", "metrics.tsv"), ("GATES.tsv.gz", "gates.tsv.gz"), ("TRANSITIONS.tsv", "transitions.tsv")):
        parts = [pd.read_csv(Path(o) / fn, sep="\t") for o in s.out if (Path(o) / fn).exists()]
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(a.root / name, sep="\t", index=False)
    errs = [f"{Path(o).name}: {(Path(o) / 'error.txt').read_text()}" for o in s.out if (Path(o) / "error.txt").exists()]
    (a.root / "ERRORS.txt").write_text("\n\n".join(errs))
    print(s.status.value_counts().to_string()); print(f"{len(errs)} rows with errors -> {a.root}")


def status(a) -> None:
    s = status_table(a.root)
    print(s.groupby(["stage", "status"]).size().unstack(fill_value=0).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worklist")
    w.add_argument("--dataset", choices=("splatter", "splatter_ext", "norman", "vcc"), required=True)
    w.add_argument("--envs", default=str(HERE / "external_envs.json"), help="splatter_ext: method -> environment python map")
    w.add_argument("--root", type=Path, required=True)
    w.add_argument("--splatter-root", type=Path)
    w.add_argument("--conditions-subdir", default="N5000")
    w.add_argument("--replicates", type=int, nargs="+", default=[1, 2, 3])
    w.add_argument("--cells-per-side", type=int, default=5000)
    w.add_argument("--perturb-root", type=Path)
    w.add_argument("--pilot", type=int, default=0)
    k = sub.add_parser("work")
    k.add_argument("--worklist", type=Path, required=True)
    k.add_argument("--large-slots", type=Path, default=None)
    k.add_argument("--max-large", type=int, default=4)
    k.add_argument("--large-size", type=float, default=4e6)
    k.add_argument("--retry-failed", action="store_true")
    k.add_argument("--max-rows", type=int, default=0)
    for name in ("collect", "status"):
        c = sub.add_parser(name); c.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    {"worklist": worklist, "work": work, "collect": collect, "status": status}[a.cmd](a)


if __name__ == "__main__":
    main()

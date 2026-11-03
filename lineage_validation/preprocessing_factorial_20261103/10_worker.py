"""One worker of the lineage preprocessing factorial: claims worklist rows one at a time and fits them (PLAN.md section
4). Many workers per node and many nodes share one worklist; a row is claimed by creating OUT/.claim atomically, so no
row is fitted twice and any number of workers can join or leave.

    python lineage_validation/preprocessing_factorial_20261103/10_worker.py --root ROOT --worklist WORKLIST.tsv
        [--large-slots DIR --max-large 4] [--retry-failed]

Per row (OUT = the row's out):
  grid          the unit's sides (the frozen equalised files' malignant cells; original counts of the same cells,
                level2_pairwise_ot/run_pairwise.py load_part / load_side) -> depth (factorial_common.equalise_sides,
                DS_TARGETS.json) -> encoding (factorial_common.encode_sides) -> OUT/input/{source,target}.h5ad ->
                02_run_pair.py (factorial_common.fit_command, --n-pcs 50, --save-coupling) into OUT/runs.
                L1: the KP equalised manifest copied with this row's two files replaced, --index = the unit's row, so
                seeds equal the earlier KP run's. L23: a one-row manifest (sample ids <case>__source / __target).
  reproduction  02_run_pair.py on the KP equalised manifest itself, the row's label and --n-pcs 30, as the earlier run.
Then the input h5ads are deleted (reproducible from the configuration), OUT/row.json records the configuration, the
encoding and the timings, and OUT/SUCCESS or OUT/FAILED (with the exit code) is written. A row with SUCCESS or FAILED
is skipped (--retry-failed re-runs FAILED rows). Rows whose unit holds more than LARGE cells x cells take one of
--max-large node-local slots, so a node never fits more than that many of them at once (memory).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "realdata_benchmark"), str(REPO / "src"), str(REPO / "cancer_metastasis")]
import factorial_common as fc  # noqa: E402

L2 = REPO / "lineage_validation" / "level2_pairwise_ot"
RUN_PAIR = str(REPO / "cancer_metastasis" / "02_run_pair.py")
LARGE = 2e7
SOLVER_WORKERS = 8


def run_pairwise():
    sys.path.insert(0, str(L2))
    spec = importlib.util.spec_from_file_location("l2_run_pairwise", L2 / "run_pairwise.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def env() -> dict:
    sep = ";" if os.name == "nt" else ":"
    return {**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO / "realdata_benchmark", REPO]))}


class Sides:
    """The original counts of a unit's two sides, cached for the worker's next row of the same unit."""

    def __init__(self, rp):
        self.rp, self.key, self.value = rp, None, None

    def get(self, unit: pd.Series):
        if self.key == unit.unit_id:
            return self.value
        rp = self.rp
        if unit.level == "L1":
            r = pd.read_csv(unit.manifest).iloc[int(unit["index"])]
            out = {}
            for side in ("source", "target"):
                t = str(r[f"{side}_sample"])
                part = {"ds": (rp.side_paths(r, side), t), "raw": ([str(rp.KP / "prepared" / "h5ad" / f"{t}.h5ad")], t),
                        "site": t, "site_from_obs": False}
                raw, _, _ = rp.load_part(part, side)
                out[side] = raw
        else:
            c = pd.read_csv(unit.manifest, sep="\t", dtype=str)
            c = c[c["index"].astype(int) == int(unit["index"])].iloc[0]
            spec = rp.files_for(c); left = dict(spec["left_out"])
            (s_raw, _, s_site), (t_raw, _, t_site) = rp.load_side(spec["source"], "source", left), rp.load_side(spec["target"], "target", left)
            out = {}
            for side, raw, site in (("source", s_raw, s_site), ("target", t_raw, t_site)):
                raw = raw.copy(); raw.obs["site"] = pd.Categorical(site); raw.obs["sample_id"] = f"{unit.pair_or_case}__{side}"
                raw.obs["malignant"] = "malignant"
                out[side] = raw
        self.key, self.value = unit.unit_id, out
        return out


def fit_grid(row, spec: dict, unit: pd.Series, sides: dict, targets: dict, out: Path) -> tuple[int, dict]:
    key = "KPTracer" if unit.dataset == "KPTracer" else f"Quinn2021:{unit.mouse}"
    t0 = time.perf_counter()
    s, t = fc.equalise_sides(sides["source"], sides["target"], spec["ds"], targets[key])
    se, te, label, rec = fc.encode_sides(spec, s, t)
    rec["encode_seconds"] = time.perf_counter() - t0
    rec["ds_target"] = None if spec["ds"] == "none" else int(targets[key][spec["ds"]])
    (out / "input").mkdir(parents=True, exist_ok=True)
    files = {"source": out / "input" / "source.h5ad", "target": out / "input" / "target.h5ad"}
    se.write_h5ad(files["source"]); te.write_h5ad(files["target"])
    if unit.level == "L1":
        man = pd.read_csv(unit.manifest); i = int(unit["index"])
        for side in ("source", "target"):
            man.loc[i, f"{side}_h5ad"] = str(files[side])
            if f"{side}_h5ads_json" in man:
                man.loc[i, f"{side}_h5ads_json"] = json.dumps([str(files[side])])
        index = i
    else:
        man = pd.DataFrame([{"pair_id": unit.pair_or_case, "dataset_id": unit.dataset, "patient_id": unit.mouse,
                             "source_sample": f"{unit.pair_or_case}__source", "target_sample": f"{unit.pair_or_case}__target",
                             "source_h5ad": str(files["source"]), "target_h5ad": str(files["target"]), "eligible": True}])
        index = 0
    man.to_csv(out / "manifest.csv", index=False)
    cmd = fc.fit_command(sys.executable, RUN_PAIR, str(out / "manifest.csv"), str(out / "runs"), index, label, fc.N_PCS,
                         SOLVER_WORKERS, precomputed=True)
    t1 = time.perf_counter()
    with open(out / "run.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=REPO).returncode
    rec["fit_seconds"] = time.perf_counter() - t1
    rec["label"] = label; rec["index"] = index
    for f in files.values():
        f.unlink(missing_ok=True)
    return code, rec


def fit_reproduction(row, spec: dict, unit: pd.Series, out: Path) -> tuple[int, dict]:
    cmd = fc.fit_command(sys.executable, RUN_PAIR, unit.manifest, str(out / "runs"), int(unit["index"]), spec["label"],
                         int(spec["n_pcs"]), SOLVER_WORKERS, precomputed=False)
    t1 = time.perf_counter()
    with open(out / "run.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=REPO).returncode
    return code, {"label": spec["label"], "index": int(unit["index"]), "fit_seconds": time.perf_counter() - t1}


def take_slot(slots: Path | None, n: int) -> Path | None:
    if slots is None:
        return Path(os.devnull)
    slots.mkdir(parents=True, exist_ok=True)
    for k in range(n):
        d = slots / f"slot{k}"
        try:
            d.mkdir(); return d
        except FileExistsError:
            continue
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--large-slots", type=Path, default=None, help="node-local directory for the large-unit slots")
    p.add_argument("--max-large", type=int, default=4)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--max-rows", type=int, default=0, help="stop after this many rows (0: until none is left)")
    a = p.parse_args()
    work = pd.read_csv(a.worklist, sep="\t", dtype={"mouse": str})
    reg = pd.read_csv(a.root / "FACTORIAL_CONFIG_REGISTRY.tsv", sep="\t").set_index("config_id")
    units = pd.read_csv(a.root / "UNITS.tsv", sep="\t", dtype={"mouse": str}).set_index("unit_id", drop=False)
    targets = json.loads((a.root / "DS_TARGETS.json").read_text())
    sides = Sides(run_pairwise())
    owner = f"{socket.gethostname()} pid {os.getpid()} job {os.environ.get('SLURM_JOB_ID', '-')}"
    done = 0
    for _, row in work.iterrows():
        out = Path(row.out)
        if (out / "SUCCESS").exists() or ((out / "FAILED").exists() and not a.retry_failed):
            continue
        out.mkdir(parents=True, exist_ok=True)
        large = float(row["size"]) > LARGE
        slot = None
        if large:
            slot = take_slot(a.large_slots, a.max_large)
            if slot is None:
                continue  # every large slot of this node is busy: leave the row for later
        try:
            (out / ".claim").mkdir()
        except FileExistsError:
            if large and slot is not None and a.large_slots is not None:
                slot.rmdir()
            continue
        (out / ".claim" / "owner").write_text(owner)
        (out / "FAILED").unlink(missing_ok=True)
        started = time.time()
        try:
            spec = reg.loc[row.config_id].to_dict(); spec["config_id"] = row.config_id
            unit = units.loc[row.unit_id]
            if spec["role"] == "reproduction":
                code, rec = fit_reproduction(row, spec, unit, out)
            else:
                code, rec = fit_grid(row, spec, unit, sides.get(unit), targets, out)
        except Exception:
            code, rec = -1, {}
            (out / "error.txt").write_text(traceback.format_exc())
        finally:
            if large and slot is not None and a.large_slots is not None:
                slot.rmdir()
        rec.update(config_id=row.config_id, unit_id=row.unit_id, exit=code, seconds=time.time() - started, owner=owner)
        (out / "row.json").write_text(json.dumps(rec, indent=1, default=str))
        (out / ("SUCCESS" if code == 0 else "FAILED")).write_text(json.dumps({"exit": code}))
        print(row.unit_id, row.config_id, "ok" if code == 0 else f"FAILED {code}", f"{rec['seconds']:.0f}s", flush=True)
        done += 1
        if a.max_rows and done >= a.max_rows:
            break
    print(f"worker finished: {done} rows", flush=True)


if __name__ == "__main__":
    main()

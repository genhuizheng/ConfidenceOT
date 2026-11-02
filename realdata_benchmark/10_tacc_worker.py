"""One TACC worker of the joint benchmark: fits its share of a worklist (rows k, k + n, k + 2n, ...).

    python realdata_benchmark/10_tacc_worker.py --worklist WORKLIST.tsv --registry REGISTRY.tsv --worker K --workers N

Worklist columns (11_build_worklists.py, splatter_n5000/31_build_n5000_worklist.py): dataset (ltl331 | splatter |
splatter_n5000 | splatter_n5000_condition), config_id, pair_index, pair_id, manifest (the original-count manifest),
manifest_ds (the frozen equalised manifest, Splatter conditions_ds or LTL331 inputs/ds), run_root (LTL331 run root),
converted (LTL331 converted dir), acceptance, out (this row's output directory); for the N = 5,000 rows also
truth_dir (the pair's 21_ truth) and fit_out (an existing fit to reuse, empty to fit here).
A row whose out/SUCCESS exists is skipped (resubmission continues); a failed row writes out/FAILED with the exit code
and the worker goes on.

ltl331 rows call 05_ltl331_ot_pilot.py --pair-index --only CONFIG --no-check --out OUT (the factorial arguments, frozen
manifests, metrics and balanced OT exactly as the local pilot). splatter rows: the configuration's input is prepared
as in 05_ (DS none: the manifest's files; P10 frozen: the conditions_ds files of that pair; P10 / P25 / P50 seed s:
the original counts equalised with confidenceot.equalise_depth to the pooled quantile over every file of the
condition manifest, seeds 1000 s + 1 / + 2; production-equivalent configurations by label, the others as precomputed
encodings with labels bmenc[_allgenes]...), then 02_run_pair.py with the original Splatter runner arguments
(benchmark/tacc/ot_array.slurm: scope all, within-side split null, epsilon 0.1, lambda 1 / 1, tolerance 1e-4,
--save-coupling --save-pairing-edges, seed 20260925, workers 8, cpu) and the row's acceptance. Scoring is a separate
stage (12_score_splatter.py) on the original-count manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import rb_common as rb

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SPLATTER_ARGS = ["--analysis-scope", "all", "--calibration-null", "within_side_split", "--epsilon", "0.1", "--lambda-a", "1.0",
                 "--lambda-b", "1.0", "--tolerance", "1e-4", "--save-coupling", "--save-pairing-edges", "--device", "cpu",
                 "--workers", "8", "--seed", "20260925"]


def env():
    sep = ";" if os.name == "nt" else ":"
    return {**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", HERE, REPO]))}


def pooled_targets(manifest: pd.DataFrame) -> dict:
    files = sorted(set(manifest.source_h5ad) | set(manifest.target_h5ad))
    depth = np.concatenate([np.asarray(ad.read_h5ad(f).X.sum(1)).ravel() for f in files])
    return {k: int(np.quantile(depth, q)) for k, q in rb.DS_QUANTILES.items()}


def splatter_row(row, spec, out: Path) -> int:
    man = pd.read_csv(row.manifest)
    r = man.iloc[int(row.pair_index)]
    if str(r.pair_id) != str(row.pair_id):
        raise SystemExit(f"manifest row {row.pair_index} is {r.pair_id}, worklist says {row.pair_id}")
    (out / "input").mkdir(parents=True, exist_ok=True)
    ds = spec["ds"]; seed = spec["ds_seed"]
    if ds == "none":
        s, t = ad.read_h5ad(r.source_h5ad), ad.read_h5ad(r.target_h5ad)
    elif str(seed).startswith("frozen"):
        mds = pd.read_csv(row.manifest_ds); q = mds[mds.pair_id.astype(str) == str(row.pair_id)].iloc[0]
        s, t = ad.read_h5ad(q.source_h5ad), ad.read_h5ad(q.target_h5ad)
    else:
        tg = pooled_targets(man)[ds]; seed = int(float(seed))
        s, t = rb.equalised(ad.read_h5ad(r.source_h5ad), tg, 1000 * seed + 1), rb.equalised(ad.read_h5ad(r.target_h5ad), tg, 1000 * seed + 2)
    label = None
    if spec["encoding"] == "logcpm" or spec["tie"] == "production":
        label = rb.production_label(spec["encoding"], int(spec["K"]) if spec["K"] == spec["K"] else 0, bool(spec["scale"]), ds != "none", spec["cost"])
    if spec["kind"] == "historical":
        label = spec["label"]
    precomputed = label is None
    if precomputed:
        joint = rb.joint_counts(s, t); genes = pd.Index(rb.common_genes(s, t)[0])
        if spec["encoding"] == "ranknm":
            rule, tseed = ("random", int(spec["tie"][6:])) if str(spec["tie"]).startswith("random") else (spec["tie"], 0)
            enc, _ = rb.rank_encode(joint, int(spec["K"]), False, rule, tseed); stem = "bmenc"
        else:
            enc, hvg = rb.rankhvg_encode(joint, int(spec["H"]), spec["tie"]); genes = genes[hvg]; stem = "bmenc_allgenes"
        label = stem + ("" if spec["scale"] else "_noscale") + ("_cos" if spec["cost"] == "cos" else "")
        enc = sp.csr_matrix(enc, dtype=np.float32); n = s.n_obs
        s = ad.AnnData(X=enc[:n], obs=s.obs.copy(), var=pd.DataFrame(index=genes))
        t = ad.AnnData(X=enc[n:], obs=t.obs.copy(), var=pd.DataFrame(index=genes))
    files = (out / "input" / "source.h5ad", out / "input" / "target.h5ad")
    s.write_h5ad(files[0]); t.write_h5ad(files[1])
    m = man.copy()
    m.loc[int(row.pair_index), "source_h5ad"], m.loc[int(row.pair_index), "target_h5ad"] = str(files[0]), str(files[1])
    for side in ("source", "target"):
        if f"{side}_h5ads_json" in m:
            m.loc[int(row.pair_index), f"{side}_h5ads_json"] = json.dumps([str(files[0 if side == "source" else 1])])
    m.to_csv(out / "manifest.csv", index=False)
    cmd = [sys.executable, str(REPO / "cancer_metastasis" / "02_run_pair.py"), str(out / "manifest.csv"), str(out / "runs"),
           "--index", str(int(row.pair_index)), "--preprocessing", label, *SPLATTER_ARGS,
           "--within-side-acceptance-minimum", str(row.acceptance)] + (["--allow-precomputed"] if precomputed or label.startswith("raw") else [])
    (out / "fit_label.txt").write_text(label)
    with open(out / "run.log", "w") as log:
        code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=REPO).returncode
    if code != 0:
        return code
    # scored against the original-count row (the scorer reads nCount / nFeature from the h5ads it is given)
    man.iloc[[int(row.pair_index)]].to_csv(out / "score_manifest.csv", index=False)
    score = [sys.executable, str(REPO / "benchmark" / "40_score.py"), "--run-root", str(out / "runs"), "--manifest",
             str(out / "score_manifest.csv"), "--preprocessing", row.config_id, "--scope", "all", "--out", str(out / "score.csv")]
    with open(out / "score.log", "w") as log:
        return subprocess.run(score, stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=REPO).returncode


def run_logged(cmd: list, log: Path) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env(), cwd=REPO).returncode


def n5000_row(row, spec, out: Path) -> int:
    """Splatter N = 5,000 (L0-L5): fit unless the row names an existing fit (frozen L0-L2 rows), then IC-POT and Vanilla
    UOT on the same cost (benchmark/50_solver_comparison.py, which reads the run's saved joint coordinates and checks
    them against run.json before solving), then 23_row_extras.py (balanced OT, coupling metrics, DEG-B)."""
    fit = Path(row.fit_out) if isinstance(row.fit_out, str) and row.fit_out else out
    if fit == out and not (out / "score.csv").exists():
        code = splatter_row(row, spec, out)
        if code != 0:
            return code
    label = (fit / "fit_label.txt").read_text().strip()
    for name, extra in (("icpot", ["--icpot-cost", "0.5"]), ("uot", ["--uot-only"])):
        d = out / "comparators" / name
        if (d / "solver_comparison.csv").exists():
            continue
        code = run_logged([sys.executable, str(REPO / "benchmark" / "50_solver_comparison.py"), str(fit / "manifest.csv"), str(d),
                           "--index", str(int(row.pair_index)), "--preprocessing", label, "--confidenceot-root", str(fit / "runs"),
                           "--scope", "all", "--seed", "20260925", *extra], out / f"comparator_{name}.log")
        if code != 0:
            return code
    if not (out / "extras" / "coupling.tsv").exists():
        return run_logged([sys.executable, str(HERE / "splatter_n5000" / "23_row_extras.py"), "--fit-out", str(fit), "--manifest", row.manifest,
                           "--pair-index", str(int(row.pair_index)), "--truth", row.truth_dir, "--out", str(out / "extras")], out / "extras.log")
    return 0


def n5000_condition_row(row, out: Path) -> int:
    code = 0
    if not (out / "deg_condition.tsv").exists():
        code = run_logged([sys.executable, str(HERE / "splatter_n5000" / "24_condition_deg.py"), "--manifest", row.manifest,
                           "--pair-index", str(int(row.pair_index)), "--truth", row.truth_dir, "--out", str(out)], out / "deg_condition.log")
    if code == 0 and not (out / "diagnostics" / "representation.tsv").exists():
        code = run_logged([sys.executable, str(HERE / "splatter_n5000" / "26_representation_diagnostics.py"), "--manifest", row.manifest,
                           "--pair-index", str(int(row.pair_index)), "--truth", row.truth_dir, "--out", str(out / "diagnostics")], out / "diagnostics.log")
    return code


def ltl331_row(row, out: Path, registry: Path) -> int:
    cmd = [sys.executable, str(HERE / "05_ltl331_ot_pilot.py"), "--run-root", row.run_root, "--converted", row.converted,
           "--registry", str(registry), "--out", str(out), "--pair-index", str(int(row.pair_index)), "--only", row.config_id, "--no-check"]
    with open(out / "worker.log", "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env(), cwd=REPO).returncode


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worklist", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--worker", type=int, required=True)
    p.add_argument("--workers", type=int, required=True)
    a = p.parse_args()
    work = pd.read_csv(a.worklist, sep="\t", dtype=str)
    reg = pd.read_csv(a.registry, sep="\t").set_index("config_id")
    mine = work.iloc[a.worker::a.workers]
    print(f"worker {a.worker} of {a.workers}: {len(mine)} rows", flush=True)
    for _, row in mine.iterrows():
        out = Path(row.out); out.mkdir(parents=True, exist_ok=True)
        if (out / "SUCCESS").exists():
            continue
        (out / "FAILED").unlink(missing_ok=True)
        try:
            if row.dataset == "splatter_n5000_condition":
                code = n5000_condition_row(row, out)
            else:
                spec = reg.loc[row.config_id].to_dict(); spec["config_id"] = row.config_id
                code = (ltl331_row(row, out, a.registry) if row.dataset == "ltl331" else
                        n5000_row(row, spec, out) if row.dataset == "splatter_n5000" else splatter_row(row, spec, out))
        except Exception:
            code = -1; (out / "error.txt").write_text(traceback.format_exc())
        (out / ("SUCCESS" if code == 0 else "FAILED")).write_text(json.dumps({"config_id": row.config_id, "pair_id": row.pair_id, "exit": code}))
        print(row.dataset, row.pair_id, row.config_id, "ok" if code == 0 else f"FAILED {code}", flush=True)


if __name__ == "__main__":
    main()

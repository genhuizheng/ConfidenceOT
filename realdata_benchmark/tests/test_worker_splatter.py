"""End-to-end test of 10_tacc_worker.py's Splatter path on a small condition shaped like the real ones (local).

    python realdata_benchmark/tests/test_worker_splatter.py --run-root RUN_ROOT --converted CONVERTED

The condition is built from LTL331 cells (300 16-week source, 300 20-week target; truth: all_shared, nothing to
reject), with a frozen-equalised copy at the pooled P10 (conditions_ds) and a manifest carrying the Splatter columns.
Three worklist rows exercise the three input paths: a production label at DS none, a precomputed expected-tie
encoding at frozen P10, and Rank-HVG on a seeded P25 draw. Checks: every row SUCCESS, score.csv has M4-E and M4-R
rows with the scorer's columns, the precomputed fit used a bmenc label, a rerun skips finished rows. PASS / FAIL.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RB = HERE.parent
REPO = RB.parent
sys.path[:0] = [str(RB), str(REPO / "src"), str(REPO / "cancer_metastasis")]
import rb_common as rb  # noqa: E402

RESULTS = []


class _keep:
    def __init__(self, path): self.path = Path(path)
    def __enter__(self):
        if self.path.exists():
            raise SystemExit(f"{self.path} exists")
        self.path.mkdir(parents=True); return str(self.path)
    def __exit__(self, *exc): return False


def check(name, ok, detail=""):
    RESULTS.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--workdir", type=Path, default=None, help="keep everything here instead of a temporary directory")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    with (tempfile.TemporaryDirectory() if a.workdir is None else _keep(a.workdir)) as tmp:
        tmp = Path(tmp); cond = tmp / "conditions" / "N300" / "rep1" / "L0_matched"; cond_ds = tmp / "conditions_ds" / "N300" / "rep1" / "L0_matched"
        cond.mkdir(parents=True); cond_ds.mkdir(parents=True)
        row = rb.ltl_pairs(a.run_root, False).iloc[5]
        s = ad.read_h5ad(rb.local(row.source_h5ad, a.run_root)); t = ad.read_h5ad(rb.local(row.target_h5ad, a.run_root))
        s = s[np.sort(rng.choice(s.n_obs, 300, replace=False))].copy(); t = t[np.sort(rng.choice(t.n_obs, 300, replace=False))].copy()
        s.write_h5ad(cond / "source.h5ad"); t.write_h5ad(cond / "target.h5ad")
        depth = np.concatenate([np.asarray(s.X.sum(1)).ravel(), np.asarray(t.X.sum(1)).ravel()])
        p10 = int(np.quantile(depth, 0.1))
        rb.equalised(s, p10, 11).write_h5ad(cond_ds / "source.h5ad"); rb.equalised(t, p10, 12).write_h5ad(cond_ds / "target.h5ad")
        ctype = rb.cell_types(a.converted)
        truth = pd.concat([pd.DataFrame({"side": "source", "group": ctype.loc[s.obs_names].to_numpy(), "should_reject": False}),
                           pd.DataFrame({"side": "target", "group": ctype.loc[t.obs_names].to_numpy(), "should_reject": False})])
        truth.to_csv(cond / "truth.csv.gz", index=False)
        base = {"pair_id": "test_all_shared", "dataset_id": "splatter_technical", "patient_id": "N300_rep1", "source_sample": str(s.obs["sample_id"].iloc[0]),
                "target_sample": str(t.obs["sample_id"].iloc[0]), "eligible": True, "n_cells": 300, "replicate": 1, "technical_level": "L0_matched",
                "biological_case": "all_shared", "R_nCount": 1.0, "R_nFeature": 1.0, "rank_cut_valid": True, "truth_csv": str(cond / "truth.csv.gz")}
        pd.DataFrame([{**base, "source_h5ad": str(cond / "source.h5ad"), "target_h5ad": str(cond / "target.h5ad")}]).to_csv(cond / "manifest.csv", index=False)
        pd.DataFrame([{**base, "source_h5ad": str(cond_ds / "source.h5ad"), "target_h5ad": str(cond_ds / "target.h5ad")}]).to_csv(cond_ds / "pair_manifest_downsampled.csv", index=False)
        reg_dir = tmp / "reg"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", RB, REPO]))}
        subprocess.run([sys.executable, str(RB / "01_config_registry.py"), "--out", str(reg_dir)], env=env, check=True, capture_output=True)
        configs = ["ranknm256-production|none|noscale|cos", "ranknm1024-expected|P10sfrozen|noscale|cos", "rankhvg2000-midrank|P25s1|noscale|cos"]
        work = pd.DataFrame([{"dataset": "splatter", "config_id": c, "pair_index": 0, "pair_id": "test_all_shared",
                              "manifest": str(cond / "manifest.csv"), "manifest_ds": str(cond_ds / "pair_manifest_downsampled.csv"),
                              "run_root": "", "converted": "", "acceptance": 0.99, "size": 300, "out": str(tmp / "out" / c.replace("|", "__"))} for c in configs])
        work.to_csv(tmp / "worklist.tsv", sep="\t", index=False)
        cmd = [sys.executable, str(RB / "10_tacc_worker.py"), "--worklist", str(tmp / "worklist.tsv"), "--registry", str(reg_dir / "PREPROCESSING_CONFIG_REGISTRY.tsv"), "--worker", "0", "--workers", "1"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)
        print(r.stdout[-1500:], r.stderr[-1500:])
        for c in configs:
            out = tmp / "out" / c.replace("|", "__")
            check(f"{c}: SUCCESS", (out / "SUCCESS").exists(), (out / "error.txt").read_text()[-600:] if (out / "error.txt").exists() else "")
            sc = pd.read_csv(out / "score.csv") if (out / "score.csv").exists() else pd.DataFrame()
            check(f"{c}: score.csv has M4-E and M4-R with the scorer's metrics",
                  {"M4-E", "M4-R"} <= set(sc.get("method", [])) and "same_population_mass" in sc, f"{len(sc)} rows")
        label = (tmp / "out" / configs[1].replace("|", "__") / "fit_label.txt").read_text()
        check("expected-tie configuration fitted through the precomputed path", label == "bmenc_noscale_cos", label)
        check("production configuration fitted by its label", (tmp / "out" / configs[0].replace("|", "__") / "fit_label.txt").read_text() == "ranknm256_noscale_cos")
        r2 = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)
        check("a rerun skips finished rows", " ok" not in r2.stdout and "FAILED" not in r2.stdout, r2.stdout[-300:])
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

"""Export the LTL331 results small enough to check locally: every pair's gate tables and every cell's labels.

    python ltl331_nepc/04_export_results.py --root ROOT --converted CONVERTED --out OUT

Copies, keeping their path under ROOT/runs: pair_metrics.csv, cell_confidence.csv, cell_qc.csv.gz,
population_rejection.csv, population_transitions.csv, calibration.json, run.json of every pair directory (a directory
holding pair_metrics.csv). Couplings (coupling_*.npz) stay on TACC; COUPLINGS_NOT_EXPORTED.tsv lists them with sizes.
Writes cells_obs.tsv.gz: the obs of every converted time point (observation_id = the cell name the gate tables use;
annotated obs, coarse_celltype, phase, S_score, G2M_score, latent_time, sample_id, time_weeks, relapse), and copies the
conversion and equalisation reports. MANIFEST.tsv holds bytes and md5 of every file; OUT.tar.gz packs OUT.
OUT must not exist. Reads only obs from the h5ad files (backed), so it is light.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

import anndata as ad
import pandas as pd

PAIR_FILES = ["pair_metrics.csv", "cell_confidence.csv", "cell_qc.csv.gz", "population_rejection.csv",
              "population_transitions.csv", "calibration.json", "run.json"]
INPUT_FILES = ["inputs/ds/downsample_per_sample.csv", "inputs/ds/downsample_report.json",
               "inputs/ds/predownsample_depth.csv.gz"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    runs = a.root / "runs"
    pair_dirs = sorted({f.parent for f in runs.rglob("pair_metrics.csv")})
    if not pair_dirs:
        sys.exit(f"no pair_metrics.csv under {runs}")
    a.out.mkdir(parents=True)

    couplings = []
    for d in pair_dirs:
        rel = d.relative_to(a.root)
        (a.out / rel).mkdir(parents=True, exist_ok=True)
        for name in PAIR_FILES:
            if (d / name).is_file():
                shutil.copy2(d / name, a.out / rel / name)
        couplings += [{"path": f.relative_to(a.root).as_posix(), "bytes": f.stat().st_size} for f in sorted(d.glob("coupling_*.npz"))]
    pd.DataFrame(couplings, columns=["path", "bytes"]).to_csv(a.out / "COUPLINGS_NOT_EXPORTED.tsv", sep="\t", index=False)

    samples = pd.read_csv(a.converted / "samples.tsv", sep="\t")
    obs = []
    for h5ad in samples["h5ad"]:
        data = ad.read_h5ad(h5ad, backed="r")
        o = data.obs.copy()
        data.file.close()
        o.insert(0, "observation_id", o.index.astype(str))
        obs.append(o)
    obs = pd.concat(obs, ignore_index=True)
    if obs["observation_id"].duplicated().any():
        sys.exit("cell names repeat across time points")
    obs.to_csv(a.out / "cells_obs.tsv.gz", sep="\t", index=False, compression="gzip")
    shutil.copy2(a.converted / "samples.tsv", a.out / "samples.tsv")
    shutil.copy2(a.converted / "conversion_report.json", a.out / "conversion_report.json")
    for rel in INPUT_FILES:
        if (a.root / rel).is_file():
            (a.out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(a.root / rel, a.out / rel)

    files = sorted(f for f in a.out.rglob("*") if f.is_file())
    pd.DataFrame([{"path": f.relative_to(a.out).as_posix(), "bytes": f.stat().st_size, "md5": md5(f)}
                  for f in files]).to_csv(a.out / "MANIFEST.tsv", sep="\t", index=False)
    tar = a.out.with_name(a.out.name + ".tar.gz")
    with tarfile.open(tar, "w:gz") as t:
        t.add(a.out, arcname=a.out.name)
    print(f"{len(pair_dirs)} pair directories, {len(obs)} cells, {len(couplings)} couplings left on TACC")
    print(f"{tar} ({tar.stat().st_size / 1e6:.1f} MB), md5 {md5(tar)}")


if __name__ == "__main__":
    main()

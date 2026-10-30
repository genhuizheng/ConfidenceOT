"""End-to-end test of the LTL331 preparation on fake data (local): 01_convert_geo -> 02_build_manifest ->
cancer_metastasis/27_downsample_counts.py -> 03_raw_on_ds_cells.

    python ltl331_nepc/tests/test_prepare.py

A fake annotated.h5ad (X log-normalised, .raw integer UMIs, obs sample_id preCX_1 / 4wkCX_1 / 20wkCX_1_R with aggr
suffixes) and one CellRanger tarball per time point. Checks: the counts written are the integer .raw; time points
ordered with the relapse last; the RAW cross-check finds every cell and equal counts; adjacent pairs only; 27_
drops low-quality cells; 03 writes the original counts of exactly the equalised cells, in their order.
Prints PASS / FAIL; exit 1 on any failure.
"""

from __future__ import annotations

import gzip
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def run(*cmd):
    r = subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True, cwd=REPO,
                       env={**__import__("os").environ, "PYTHONPATH": f"{REPO / 'src'};{REPO / 'cancer_metastasis'};{REPO}"})
    return r


def main() -> None:
    rng = np.random.default_rng(1)
    genes = [f"G{i}" for i in range(380)] + [f"MT-{i}" for i in range(20)]
    samples = {"preCX_1": 1, "4wkCX_1": 2, "20wkCX_1_R": 3}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw_dir = tmp / "raw"; raw_dir.mkdir()
        blocks, obs, names = [], [], []
        for s, k in samples.items():
            n = 60
            lam = rng.gamma(0.6, 4.0, len(genes))
            X = rng.poisson(lam * rng.uniform(4, 12, (n, 1))).astype(np.int64)
            X[:5, -20:] += 400  # five mitochondria-heavy cells per sample, which 27_'s QC removes
            bc = ["".join(rng.choice(list("ACGT"), 16)) + "-1" for _ in range(n)]
            blocks.append(X); names += [b.replace("-1", f"-{k}") for b in bc]
            obs.append(pd.DataFrame({"sample_id": s, "model": "LTL331R" if s.endswith("_R") else "LTL331",
                                     "cell_type": rng.choice(["PRAD", "NE"], n)}))
            # the CellRanger archive of this time point (plus 4 barcodes the authors' QC removed)
            extra = rng.poisson(2, (4, len(genes)))
            M = sp.csr_matrix(np.vstack([X, extra]).T)
            buf = {"matrix.mtx.gz": io.BytesIO(), "barcodes.tsv.gz": io.BytesIO(), "features.tsv.gz": io.BytesIO()}
            mm = io.BytesIO(); scipy.io.mmwrite(mm, M); buf["matrix.mtx.gz"] = gzip.compress(mm.getvalue())
            buf["barcodes.tsv.gz"] = gzip.compress("\n".join(bc + ["".join(rng.choice(list("ACGT"), 16)) + "-1" for _ in range(4)]).encode())
            buf["features.tsv.gz"] = gzip.compress("\n".join(f"ENSG{i}\t{g}\tGene Expression" for i, g in enumerate(genes)).encode())
            with tarfile.open(raw_dir / f"GSM88{k}_{s}.tar.gz", "w:gz") as t:
                for fname, data in buf.items():
                    info = tarfile.TarInfo(f"{s}/filtered_feature_bc_matrix/{fname}"); info.size = len(data)
                    t.addfile(info, io.BytesIO(data))
        Xr = sp.csr_matrix(np.vstack(blocks))
        o = pd.concat(obs, ignore_index=True); o.index = names
        a = ad.AnnData(X=sp.csr_matrix(np.log1p(Xr.toarray() / Xr.sum(1).A * 1e4)), obs=o, var=pd.DataFrame(index=genes))
        a.raw = ad.AnnData(X=Xr, obs=o, var=pd.DataFrame(index=genes))
        a.write_h5ad(tmp / "annotated.h5ad")
        conv = tmp / "converted"
        r = run(PKG / "01_convert_geo.py", "--annotated", tmp / "annotated.h5ad", "--raw-dir", raw_dir, "--out", conv)
        check("01_convert_geo exits 0", r.returncode == 0, r.stderr[-500:] if r.returncode else "")
        rep = json.loads((conv / "conversion_report.json").read_text())
        check("counts are the integer .raw", rep["counts_from"] == "raw" and rep["counts_check"]["integer"])
        s = pd.read_csv(conv / "samples.tsv", sep="\t")
        check("time points ordered, relapse last: w00_preCX_1, w04_4wkCX_1, w20_20wkCX_1_R",
              s["sample_id"].tolist() == ["w00_preCX_1", "w04_4wkCX_1", "w20_20wkCX_1_R"] and s["relapse"].tolist() == [False, False, True])
        cc = rep["raw_cross_check"]
        check("RAW cross-check: every author cell found in its CellRanger matrix, counts equal",
              all(v["author_cells_found"] == v["author_cells"] == 60 and v["counts_equal"] for v in cc.values()), str(cc)[:300])
        w0 = ad.read_h5ad(conv / "h5ad" / "w00_preCX_1.h5ad")
        check("per time point: malignant column, sample_id, time_weeks, raw counts equal to .raw",
              set(w0.obs["malignant"]) == {"malignant"} and set(w0.obs["sample_id"]) == {"w00_preCX_1"} and set(w0.obs["time_weeks"]) == {0}
              and np.array_equal(w0.X.toarray(), blocks[0]))
        man = tmp / "root" / "inputs" / "manifest_raw.csv"
        r = run(PKG / "02_build_manifest.py", "--converted", conv, "--out", man)
        m = pd.read_csv(man)
        check("02: adjacent pairs only (preCX->4wk, 4wk->20wk R)", r.returncode == 0 and m["pair_id"].tolist() ==
              ["LTL331__w00_preCX_1__w04_4wkCX_1", "LTL331__w04_4wkCX_1__w20_20wkCX_1_R"], r.stderr[-300:])
        ds = tmp / "root" / "inputs" / "ds"
        r = run(REPO / "cancer_metastasis" / "27_downsample_counts.py", man, ds, "--malignant-column", "malignant",
                "--minimum-total-counts", "100", "--minimum-detected-genes", "50")
        check("27_downsample_counts.py runs on the manifest", r.returncode == 0, r.stderr[-500:] if r.returncode else "")
        dsm = pd.read_csv(ds / "pair_manifest_downsampled.csv")
        d0 = ad.read_h5ad(dsm.loc[0, "source_h5ad"])
        check("27_ QC removed the mitochondria-heavy cells", d0.n_obs <= 55, f"{d0.n_obs} of 60 kept")
        rq = tmp / "root" / "inputs" / "rawqc"
        r = run(PKG / "03_raw_on_ds_cells.py", "--raw-manifest", man, "--ds-manifest", ds / "pair_manifest_downsampled.csv", "--out", rq)
        check("03 exits 0", r.returncode == 0, r.stderr[-500:] if r.returncode else "")
        rqm = pd.read_csv(rq / "pair_manifest_rawqc.csv")
        q0 = ad.read_h5ad(rqm.loc[0, "source_h5ad"])
        orig = ad.read_h5ad(conv / "h5ad" / "w00_preCX_1.h5ad")
        check("03: the equalised file's cells in its order, with the original (not equalised) counts",
              list(q0.obs_names) == list(d0.obs_names) and np.array_equal(q0.X.toarray(), orig[d0.obs_names].X.toarray())
              and q0.X.sum() > d0.X.sum())
        check("03: same pairs as the equalised manifest", rqm["pair_id"].tolist() == dsm["pair_id"].tolist())
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

"""End-to-end test of the LTL331 preparation on fake data shaped like the real GEO files (local): 01_convert_geo ->
02_build_manifest -> cancer_metastasis/27_downsample_counts.py -> 03_raw_on_ds_cells.

    python ltl331_nepc/tests/test_prepare.py

Fake files as found on TACC 2026-10-30: an annotated.h5ad whose .raw is log1p(1e4 * counts / nCount_RNA) on a gene
subset (no counts) and whose obs is a Seurat export (orig.ident, pretty, week_num, is_regrowth, nCount_RNA, cellid,
seurat_clusters; cell names <barcode>-<orig.ident>); a harmony object with the LTL331 cells under other names plus
another cohort, linked by cellid, carrying coarse_celltype; a velocity object lacking some cells, carrying phase and
latent_time; one CellRanger archive per time point (plain files in a folder, extra barcodes, extra and repeated gene
symbols). Checks: the counts written are the archive's for exactly the authors' cells; .raw reproduced; labels carried
(missing where velocity lacks the cell); time points ordered with the relapse last; adjacent pairs only; 27_ drops
low-quality cells; 03 writes the original counts of exactly the equalised cells. Prints PASS / FAIL; exit 1 on failure.
"""

from __future__ import annotations

import io
import json
import os
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
    sep = ";" if os.name == "nt" else ":"
    return subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO]))})


def main() -> None:
    rng = np.random.default_rng(1)
    raw_genes = [f"G{i}" for i in range(360)] + ["DUP", "DUP"] + [f"MT-{i}" for i in range(20)]
    ann_genes = [f"G{i}" for i in range(300)] + [f"MT-{i}" for i in range(20)]  # the authors' filtered gene set
    gi = [raw_genes.index(g) for g in ann_genes]
    samples = [("GSM101", "preCX_1", 0, "B_g11_preCx1", 2, False), ("GSM102", "4wkCX_1", 4, "F_g11_4wk1", 6, False),
               ("GSM103", "20wkCX_1_R", 20, "S_g11_20wk_R", 16, True)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); data = tmp / "data"; (data / "raw").mkdir(parents=True)
        counts, obs = {}, []
        for gsm, name, w, pretty, oi, regrowth in samples:
            n = 60
            X = rng.poisson(rng.gamma(0.6, 4.0, len(raw_genes)) * rng.uniform(4, 12, (n + 6, 1))).astype(np.int64)
            X[:5, -20:] += 400  # five mitochondria-heavy cells, which 27_'s QC removes
            bc = ["".join(rng.choice(list("ACGT"), 16)) + "-1" for _ in range(n + 6)]  # the last 6: removed by the authors
            mm = io.BytesIO(); scipy.io.mmwrite(mm, sp.coo_matrix(X.T))
            files = {"matrix.mtx": mm.getvalue(), "barcodes.tsv": "\n".join(bc).encode(),
                     "features.tsv": "\n".join(f"ENSG{i}\t{g}\tGene Expression" for i, g in enumerate(raw_genes)).encode()}
            with tarfile.open(data / "raw" / f"{gsm}_{name}.tar.gz", "w:gz") as t:
                for f, b in files.items():
                    info = tarfile.TarInfo(f"{name}/{f}"); info.size = len(b); t.addfile(info, io.BytesIO(b))
            cells = [b.replace("-1", f"-{oi}") for b in bc[:n]]
            counts[name] = (cells, X[:n])
            sub = X[:n][:, gi]
            obs.append(pd.DataFrame({"orig.ident": oi, "pretty": pretty, "week_num": w, "is_regrowth": regrowth,
                                     "nCount_RNA": sub.sum(1).astype(float), "nFeature_RNA": (sub > 0).sum(1),
                                     "seurat_clusters": rng.integers(0, 4, n), "cellid": cells}, index=cells))
        o = pd.concat(obs)
        o["week_num"] = o["week_num"].astype("Int64"); o["is_regrowth"] = o["is_regrowth"].astype("boolean")
        o["pretty"] = o["pretty"].astype("category")
        S = np.vstack([counts[s[1]][1][:, gi] for s in samples]).astype(float)
        L = np.log1p(1e4 * S / o["nCount_RNA"].to_numpy()[:, None])
        a = ad.AnnData(X=np.clip((L - L.mean(0)) / (L.std(0) + 1e-9), -10, 10), obs=o, var=pd.DataFrame(index=ann_genes))
        a.raw = ad.AnnData(X=sp.csr_matrix(L), obs=o, var=pd.DataFrame(index=ann_genes))
        a.write_h5ad(data / "GSE297328_ltl331_annotated.h5ad")
        h = pd.DataFrame({"cellid": o.index.tolist() + [f"P{i}" for i in range(10)],
                          "dataset": ["LTL331"] * len(o) + ["Gao"] * 10,
                          "coarse_celltype": rng.choice(["PRAD", "NE", "Basal", "Other"], len(o) + 10)})
        h.index = [f"LTL331_{c}" for c in h["cellid"]]
        for c in h: h[c] = h[c].astype("category")
        ad.AnnData(X=np.zeros((len(h), 3)), obs=h).write_h5ad(data / "GSE297328_ltl331_harmony_integrated.h5ad")
        v = o.iloc[3:].copy()  # velocity lacks the first three cells
        v["phase"] = pd.Categorical(rng.choice(["G1", "S", "G2M"], len(v))); v["latent_time"] = rng.uniform(size=len(v))
        v["S_score"] = rng.normal(size=len(v)); v["G2M_score"] = rng.normal(size=len(v))
        ad.AnnData(X=np.zeros((len(v), 3)), obs=v).write_h5ad(data / "GSE297328_ltl331_velocity.h5ad")

        conv = tmp / "converted"
        r = run(PKG / "01_convert_geo.py", "--data", data, "--out", conv)
        check("01_convert_geo exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        rep = json.loads((conv / "conversion_report.json").read_text())
        check("authors' .raw reproduced from the archive counts", all(x["raw_vs_log1p_counts_max_abs_diff"] < 1e-9 for x in rep["samples"].values()))
        check("nCount_RNA equals the sum over the authors' genes", all(x["nCount_RNA_equal_to_annotated_gene_sum_fraction"] == 1 for x in rep["samples"].values()))
        s = pd.read_csv(conv / "samples.tsv", sep="\t")
        check("time points ordered, relapse last", s["sample_id"].tolist() == ["w00_preCX_1", "w04_4wkCX_1", "w20_20wkCX_1_R"]
              and s["relapse"].tolist() == [False, False, True])
        w0 = ad.read_h5ad(conv / "h5ad" / "w00_preCX_1.h5ad")
        cells, X0 = counts["preCX_1"]
        check("counts: the archive's, exactly the authors' cells in their order, every feature",
              list(w0.obs_names) == cells and np.array_equal(w0.X.toarray(), X0) and w0.n_vars == len(raw_genes))
        check("var: unique symbols with gene_ids", w0.var_names.is_unique and "gene_ids" in w0.var and "DUP" in set(w0.var["gene_symbol"]))
        check("labels: coarse_celltype from harmony, phase missing only where velocity lacks the cell",
              w0.obs["coarse_celltype"].notna().all() and w0.obs["phase"].isna().sum() == 3 and w0.obs["phase"].iloc[3:].notna().all())
        check("obs: sample_id, time_weeks, malignant, gsm", set(w0.obs["sample_id"]) == {"w00_preCX_1"} and set(w0.obs["time_weeks"]) == {0}
              and set(w0.obs["malignant"]) == {"malignant"} and set(w0.obs["gsm"]) == {"GSM101"})
        man = tmp / "root" / "inputs" / "manifest_raw.csv"
        r = run(PKG / "02_build_manifest.py", "--converted", conv, "--out", man)
        m = pd.read_csv(man)
        check("02: adjacent pairs only", r.returncode == 0 and m["pair_id"].tolist() ==
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
        check("03: the equalised file's cells in its order, with the original counts",
              list(q0.obs_names) == list(d0.obs_names) and np.array_equal(q0.X.toarray(), w0[d0.obs_names].X.toarray())
              and q0.X.sum() > d0.X.sum())
        check("03: same pairs as the equalised manifest", rqm["pair_id"].tolist() == dsm["pair_id"].tolist())
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()

"""Convert GSE297328 (LTL331 PDX, neuroendocrine transdifferentiation time course; Sar et al., Genome Med 2026) to one
raw-count H5AD per time point: the CellRanger counts of the authors' QC cells, with the authors' labels.

    python ltl331_nepc/01_convert_geo.py --data DATA_DIR --out OUT_DIR

DATA_DIR holds the GEO files: GSE297328_ltl331_annotated.h5ad, GSE297328_ltl331_harmony_integrated.h5ad,
GSE297328_ltl331_velocity.h5ad and raw/ (GSE297328_RAW.tar extracted: one GSM*_<sample>.tar.gz per time point).

What the files hold (checked on TACC 2026-10-30 by cancer_application/GSE297328_longitudinal/scripts/00_ and 00b_;
the annotated object does not match its README):
  annotated  the authors' 51,726 QC cells (Seurat export). .X scaled, .raw = log1p(1e4 * counts / nCount_RNA) on their
             26,175 genes (not counts), no layers. obs: orig.ident, pretty, timepoint, week_num, is_regrowth,
             phase_castration, nCount_RNA, nFeature_RNA, percent.mt, seurat_clusters (= RNA_snn_res.0.4), cellid.
             Cell names <16-nt barcode>-<orig.ident>, equal to cellid.
  RAW        per time point the CellRanger matrix (29,622 features, integer). Every annotated cell's barcode is in the
             archive of its week; its counts reproduce .raw to 1e-15.
  harmony    joint integration with two patient cohorts; its LTL331 cells are linked by cellid (all 51,726) and carry
             coarse_celltype (PRAD / NE / Basal / Other); ltl331_cluster_original equals seurat_clusters.
  velocity   51,499 of the cells (same cell names) with the cell-cycle phase, S_score, G2M_score and latent_time.

Nothing is normalised, selected or filtered here (the project's QC and depth equalisation come later,
27_downsample_counts.py). Writes OUT_DIR/:
  h5ad/<sample>.h5ad   X = the CellRanger integer counts of the authors' cells of one time point, every RAW feature;
                       var index = gene symbol made unique, gene_ids, feature_types, gene_symbol; obs = the authors'
                       annotated columns, coarse_celltype (harmony), phase / S_score / G2M_score / latent_time
                       (velocity, missing for the 227 cells it lacks), and dataset_id, patient_id (LTL331), gsm,
                       author_sample_id (GSM sample name), sample_id (w<weeks>_<GSM sample name>), time_weeks, relapse
                       (is_regrowth: LTL331R), malignant ("malignant": human-specific reads of a PDX)
  samples.tsv          per time point: GSM, author_sample_id, sample_id, weeks, relapse, cells, median UMIs, file
  conversion_report.json   per time point: cells found, integer check, counts against .raw, nCount_RNA against the
                       counts; harmony and velocity link counts; obs value counts
Stops if any author cell is missing from its archive. Refuses if conversion_report.json exists.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

SEED = 20261030
ANN = "GSE297328_ltl331_annotated.h5ad"
HARMONY = "GSE297328_ltl331_harmony_integrated.h5ad"
VELOCITY = "GSE297328_ltl331_velocity.h5ad"
HARMONY_COLUMNS = ["coarse_celltype"]
VELOCITY_COLUMNS = ["phase", "S_score", "G2M_score", "latent_time"]
CHECK_CELLS = 200


def strings(values) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.ravel(values)], dtype=object)


def read_obs(path: Path) -> pd.DataFrame:
    """obs of an h5ad with h5py only (the objects carry large layers / uns that need not be loaded)."""
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        key = obs.attrs.get("_index", "_index")
        key = key.decode() if isinstance(key, bytes) else str(key)
        out = {}
        for k in obs.keys():
            if k == key:
                continue
            o = obs[k]
            if isinstance(o, h5py.Group) and "categories" in o:
                cats = strings(o["categories"][()]); codes = o["codes"][()]
                out[k] = pd.Categorical.from_codes(codes, categories=pd.Index(cats))
            elif isinstance(o, h5py.Group) and "values" in o:  # nullable integer / boolean
                v = pd.array(o["values"][()]); out[k] = pd.Series(v).mask(o["mask"][()].astype(bool)).to_numpy()
            elif isinstance(o, h5py.Dataset):
                out[k] = strings(o[()]) if o.dtype.kind in "OSU" else o[()]
        return pd.DataFrame(out, index=pd.Index(strings(obs[key][()])))


def weeks_of(sample: str) -> int:
    """GSM sample name -> weeks after castration: 'preCX_1' -> 0, '12wkCX_2' -> 12, '20wkCX_1_R' -> 20."""
    if re.search(r"pre", sample, re.I):
        return 0
    m = re.search(r"(\d+)wk", sample)
    if not m:
        raise ValueError(f"cannot read a time point from {sample!r}")
    return int(m.group(1))


def bare(barcode: str) -> str:
    return str(barcode).split("-")[0]


def read_archive(tgz: Path):
    """barcodes, features (DataFrame), cells x genes CSR of one GSM archive (files plain or gzipped)."""
    with tarfile.open(tgz) as t:
        members = {Path(m.name).name: m for m in t.getmembers() if m.isfile()}
        def get(prefix):
            name = next(n for n in sorted(members) if n.startswith(prefix))
            data = t.extractfile(members[name]).read()
            return gzip.decompress(data) if name.endswith(".gz") else data
        barcodes = get("barcodes").decode().split()
        feat_prefix = "features" if any(n.startswith("features") for n in members) else "genes"
        features = pd.read_csv(io.BytesIO(get(feat_prefix)), sep="\t", header=None, dtype=str)
        M = scipy.io.mmread(io.BytesIO(get("matrix")))
    M = sp.csr_matrix(M.T) if M.shape == (len(features), len(barcodes)) else sp.csr_matrix(M)
    return barcodes, features, M


def annotated_raw_rows(path: Path, rows: np.ndarray) -> tuple[np.ndarray, pd.Index]:
    with h5py.File(path, "r") as f:
        var = f["raw"]["var"]
        genes = pd.Index(strings(var[var.attrs.get("_index", "_index")][()]))
        g = f["raw"]["X"]; ip = g["indptr"]
        R = np.zeros((len(rows), len(genes)))
        for i, r in enumerate(rows):
            s, e = ip[r], ip[r + 1]
            R[i, g["indices"][s:e]] = g["data"][s:e]
    return R, genes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "conversion_report.json").exists():
        raise SystemExit(f"{args.out}/conversion_report.json exists; not rewritten")
    (args.out / "h5ad").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ann = read_obs(args.data / ANN)
    ann_pos = pd.Series(np.arange(len(ann)), index=ann.index)
    report = {"annotated_cells": int(len(ann)), "annotated_obs_columns": list(ann.columns)}
    harm = read_obs(args.data / HARMONY)
    harm = harm[harm["dataset"].astype(str).eq("LTL331")].set_index("cellid")
    vel = read_obs(args.data / VELOCITY)
    report["harmony_cells_linked"] = int(ann.index.isin(harm.index).sum())
    report["velocity_cells_linked"] = int(ann.index.isin(vel.index).sum())
    labels = ann.copy()
    for c in HARMONY_COLUMNS:
        labels[c] = harm[c].reindex(ann.index).to_numpy()
    for c in VELOCITY_COLUMNS:
        labels[c] = vel[c].reindex(ann.index).to_numpy()
    week = pd.to_numeric(pd.Series(ann["week_num"], index=ann.index)).astype(int)

    report["samples"], rows = {}, []
    archives = sorted((args.data / "raw").glob("GSM*.tar.gz"))
    if len(archives) != week.nunique():
        raise SystemExit(f"{len(archives)} archives but {week.nunique()} time points in the annotated object")
    for tgz in archives:
        gsm, name = re.match(r"^(GSM\d+)_(.+)\.tar\.gz$", tgz.name).groups()
        w = weeks_of(name)
        cells = ann.index[week.eq(w).to_numpy()]
        barcodes, features, M = read_archive(tgz)
        raw_pos = pd.Series(np.arange(len(barcodes)), index=[bare(b) for b in barcodes])
        if raw_pos.index.has_duplicates:
            raise SystemExit(f"{tgz.name}: duplicated barcodes")
        pos = raw_pos.reindex([bare(c) for c in cells])
        if pos.isna().any():
            raise SystemExit(f"{tgz.name}: {int(pos.isna().sum())} of {len(cells)} author cells not in the archive")
        X = M[pos.to_numpy(dtype=int)]
        rec = {"gsm": gsm, "weeks": w, "archive_barcodes": len(barcodes), "author_cells": int(len(cells)),
               "integer": bool(np.allclose(X.data, np.round(X.data))), "nonnegative": bool((X.data >= 0).all())}
        if not (rec["integer"] and rec["nonnegative"]):
            raise SystemExit(f"{tgz.name}: counts are not nonnegative integers")
        symbols = pd.Index(features[1] if features.shape[1] > 1 else features[0])
        unique = pd.Index(list(ad.utils.make_index_unique(symbols.copy()).astype(str)))
        # the counts against the authors' .raw (log1p(1e4 * counts / nCount_RNA)) on genes named once on both sides
        take = np.sort(rng.choice(len(cells), min(CHECK_CELLS, len(cells)), replace=False))
        R, genes = annotated_raw_rows(args.data / ANN, ann_pos[cells[take]].to_numpy())
        s_ok = ~symbols.duplicated(keep=False); g_ok = ~genes.duplicated(keep=False)
        common = genes[g_ok].intersection(symbols[s_ok])
        ci = np.flatnonzero(s_ok)[symbols[s_ok].get_indexer(common)]; gi = np.flatnonzero(g_ok)[genes[g_ok].get_indexer(common)]
        nc = pd.to_numeric(pd.Series(ann.loc[cells, "nCount_RNA"])).to_numpy()
        C = X[take][:, ci].toarray()
        rec["raw_check_cells"] = int(len(take)); rec["raw_check_genes"] = int(len(common))
        rec["raw_vs_log1p_counts_max_abs_diff"] = float(np.abs(np.log1p(1e4 * C / nc[take][:, None]) - R[:, gi]).max())
        sums = np.asarray(X.sum(1)).ravel()
        rec["nCount_RNA_equal_to_all_feature_sum_fraction"] = float(np.mean(np.isclose(sums, nc)))
        rec["nCount_RNA_equal_to_annotated_gene_sum_fraction"] = float(np.mean(np.isclose(
            np.asarray(X[:, np.flatnonzero(symbols.isin(genes))].sum(1)).ravel(), nc)))
        if rec["raw_vs_log1p_counts_max_abs_diff"] > 1e-6:
            raise SystemExit(f"{tgz.name}: counts do not reproduce the authors' .raw: {rec}")

        sample = f"w{w:02d}_{name}"
        obs = labels.loc[cells].copy()
        obs["dataset_id"] = "GSE297328"
        obs["patient_id"] = "LTL331"
        obs["gsm"] = gsm
        obs["author_sample_id"] = name
        obs["sample_id"] = sample
        obs["time_weeks"] = w
        obs["relapse"] = pd.Series(ann.loc[cells, "is_regrowth"]).astype(bool).to_numpy()
        obs["malignant"] = "malignant"
        for c in obs.columns:
            if obs[c].dtype == object:
                obs[c] = obs[c].astype("category")
        var = pd.DataFrame({"gene_ids": features[0].to_numpy(), "gene_symbol": symbols.to_numpy(),
                            "feature_types": features[2].to_numpy() if features.shape[1] > 2 else "Gene Expression"}, index=unique)
        part = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
        path = args.out / "h5ad" / f"{sample}.h5ad"
        part.write_h5ad(path, compression="lzf")
        relapse = bool(obs["relapse"].all())
        if obs["relapse"].nunique() != 1:
            raise SystemExit(f"{sample}: mixed is_regrowth within one time point")
        rows.append({"gsm": gsm, "author_sample_id": name, "sample_id": sample, "time_weeks": w, "relapse": relapse,
                     "cells": part.n_obs, "median_umis": float(np.median(sums)), "h5ad": str(path)})
        report["samples"][sample] = rec
        print(f"{sample}: {part.n_obs} cells, median UMIs {np.median(sums):.0f}, .raw reproduced to "
              f"{rec['raw_vs_log1p_counts_max_abs_diff']:.1e}", flush=True)
    report["obs_value_counts"] = {c: labels[c].astype(str).value_counts().to_dict() for c in labels.columns
                                  if labels[c].astype(str).nunique() < 100}
    samples = pd.DataFrame(rows).sort_values(["time_weeks", "relapse"])
    samples.to_csv(args.out / "samples.tsv", sep="\t", index=False, lineterminator="\n")
    (args.out / "conversion_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

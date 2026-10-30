"""Convert GSE297328 (LTL331 PDX, neuroendocrine transdifferentiation time course; Sar et al., Genome Med 2026) to one
H5AD per time point, raw counts only.

    python ltl331_nepc/01_convert_geo.py --annotated GSE297328_ltl331_annotated.h5ad --raw-dir RAW_DIR --out OUT_DIR

--annotated  the authors' ltl331_annotated.h5ad: their QC-passed cells (51,726 in the paper) with `.raw` = raw UMI
             counts on the pre-filtering gene set (README), and obs sample_id, model, leiden, cell_type, phase, QC.
--raw-dir    GSE297328_RAW.tar extracted as is (one GSM*_<sample>.tar.gz per time point: CellRanger human-specific
             filtered matrices). Used only to cross-check that the counts are the deposited raw UMIs.

Conversion only; nothing is normalised, selected or filtered (the project's QC and depth equalisation come later,
27_downsample_counts.py). Writes OUT_DIR/:
  h5ad/<sample>.h5ad   X = raw integer UMI counts (the authors' .raw, else layers['counts']), cells of one time point;
                       var = gene names (made unique, originals kept); obs = every authors' column plus dataset_id,
                       patient_id (LTL331), sample_id (w<weeks>_<authors' id>), time_weeks, relapse (LTL331R),
                       malignant ("malignant": the PDX reads are human-specific, so every cell is a tumour cell)
  samples.tsv          per time point: authors' sample_id, our sample_id, weeks, relapse, cells, median UMIs, file
  conversion_report.json   shapes, integer check, the authors' obs columns and their value counts (< 100 values),
                       the RAW cross-check (barcodes found per sample, counts equal on a random set of cells)
Refuses if conversion_report.json exists.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

SEED = 20261030


def weeks_of(sample: str) -> tuple[int, bool]:
    """'preCX_1' -> (0, False); '12wkCX_2' -> (12, False); '20wkCX_1_R' -> (20, True)."""
    s = str(sample)
    if re.search(r"pre", s, re.I):
        w = 0
    else:
        m = re.search(r"(\d+)\s*w", s, re.I)
        if not m:
            raise ValueError(f"cannot read a time point from sample id {s!r}")
        w = int(m.group(1))
    return w, bool(re.search(r"(_R$|_R_|LTL331R)", s))


def barcode_of(name: str) -> str:
    """The 16-nt 10x barcode inside a cell name, whatever prefix or aggr suffix the name carries."""
    m = re.search(r"[ACGTN]{16}", str(name))
    return m.group(0) if m else str(name)


def integer_check(X) -> dict:
    data = X.data if sp.issparse(X) else np.asarray(X).ravel()
    sample = data[: min(len(data), 2_000_000)]
    return {"nonnegative": bool((sample >= 0).all()), "integer": bool(np.allclose(sample, np.round(sample))),
            "max": float(sample.max()) if len(sample) else 0.0}


def read_10x(folder: Path):
    bc = next(folder.rglob("barcodes.tsv*")); feat = next(iter(sorted(folder.rglob("features.tsv*")) + sorted(folder.rglob("genes.tsv*"))))
    mtx = next(folder.rglob("matrix.mtx*"))
    opener = lambda p: gzip.open(p, "rt") if p.suffix == ".gz" else open(p)
    with opener(bc) as f:
        barcodes = [x.strip() for x in f]
    genes = pd.read_csv(feat, sep="\t", header=None, dtype=str)
    with (gzip.open(mtx) if mtx.suffix == ".gz" else open(mtx, "rb")) as f:
        M = scipy.io.mmread(f).tocsr()
    if M.shape == (len(genes), len(barcodes)):
        M = M.T.tocsr()
    return barcodes, genes, M


def cross_check(raw_dir: Path, data: ad.AnnData, author_sample: pd.Series, gene_names: pd.Index) -> dict:
    out = {}
    rng = np.random.default_rng(SEED)
    for tgz in sorted(raw_dir.glob("GSM*.tar.gz")):
        name = re.sub(r"^GSM\d+_", "", tgz.name.replace(".tar.gz", ""))
        dest = raw_dir / "extracted" / tgz.name.replace(".tar.gz", "")
        if not dest.exists():
            dest.mkdir(parents=True)
            with tarfile.open(tgz) as t:
                t.extractall(dest)
        try:
            barcodes, genes, M = read_10x(dest)
        except StopIteration:
            out[name] = {"error": "no 10x matrix found in the archive"}
            continue
        cells = author_sample.index[author_sample.astype(str).eq(name)]
        if len(cells) == 0:  # the authors' sample ids may be written differently; match on the weeks and relapse
            w, r = weeks_of(name)
            cells = author_sample.index[[weeks_of(s) == (w, r) for s in author_sample.astype(str)]]
        bare = pd.Index([barcode_of(c) for c in cells])
        bare_raw = pd.Index([barcode_of(b) for b in barcodes])
        if bare_raw.has_duplicates:
            out[name] = {"error": "duplicated barcodes in the archive"}
            continue
        pos = bare_raw.get_indexer(bare)
        found = pos >= 0
        rec = {"archive_cells": len(barcodes), "author_cells": int(len(cells)), "author_cells_found": int(found.sum())}
        sym = pd.Index(genes[1] if genes.shape[1] > 1 else genes[0])
        common = gene_names.intersection(sym.drop_duplicates(keep=False))
        if found.any() and len(common):
            pick = rng.choice(np.flatnonzero(found), min(200, int(found.sum())), replace=False)
            a = data[cells[pick], :][:, common].X
            b = M[pos[pick]][:, sym.get_indexer(common)]
            a = a.toarray() if sp.issparse(a) else np.asarray(a)
            b = b.toarray() if sp.issparse(b) else np.asarray(b)
            rec.update(genes_compared=int(len(common)), cells_compared=int(len(pick)), counts_equal=bool(np.array_equal(a, b)),
                       max_abs_difference=float(np.abs(a - b).max()))
        out[name] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotated", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.out / "conversion_report.json").exists():
        raise SystemExit(f"{args.out}/conversion_report.json exists; not rewritten")
    (args.out / "h5ad").mkdir(parents=True, exist_ok=True)
    full = ad.read_h5ad(args.annotated)
    report = {"annotated": str(args.annotated), "shape": list(full.shape), "obs_columns": list(full.obs.columns),
              "layers": list(full.layers.keys()), "has_raw": full.raw is not None}
    if full.raw is not None:
        X, var, source = full.raw.X, full.raw.var.copy(), "raw"
    elif "counts" in full.layers:
        X, var, source = full.layers["counts"], full.var.copy(), "layers['counts']"
    else:
        raise SystemExit("no raw counts: neither .raw nor layers['counts']")
    X = sp.csr_matrix(X)
    report["counts_from"] = source
    report["counts_check"] = integer_check(X)
    if not (report["counts_check"]["integer"] and report["counts_check"]["nonnegative"]):
        raise SystemExit(f"{source} is not raw integer counts: {report['counts_check']}")
    names = pd.Index(var.index.astype(str))
    var["gene_name_original"] = names
    unique = ad.utils.make_index_unique(names)
    counts = ad.AnnData(X=X, obs=full.obs.copy(), var=var.set_index(unique))
    counts.var_names = counts.var_names.astype(str)
    if "sample_id" not in counts.obs:
        raise SystemExit(f"no sample_id in obs: {list(counts.obs.columns)}")
    report["obs_value_counts"] = {c: counts.obs[c].astype(str).value_counts().to_dict() for c in counts.obs.columns
                                  if counts.obs[c].astype(str).nunique() < 100}
    if args.raw_dir is not None:
        report["raw_cross_check"] = cross_check(args.raw_dir, counts, counts.obs["sample_id"], counts.var_names)
    rows = []
    for author_sample, idx in counts.obs.groupby("sample_id", observed=True).groups.items():
        w, relapse = weeks_of(author_sample)
        if "model" in counts.obs and counts.obs.loc[idx, "model"].astype(str).eq("LTL331R").all():
            relapse = True
        sample = f"w{w:02d}_{author_sample}"
        part = counts[idx].copy()
        part.obs["dataset_id"] = "GSE297328"
        part.obs["patient_id"] = "LTL331"
        part.obs["author_sample_id"] = part.obs["sample_id"].astype(str)
        part.obs["sample_id"] = sample
        part.obs["time_weeks"] = w
        part.obs["relapse"] = relapse
        part.obs["malignant"] = "malignant"
        for c in part.obs.columns:
            if part.obs[c].dtype == object:
                part.obs[c] = part.obs[c].astype(str)
        path = args.out / "h5ad" / f"{sample}.h5ad"
        part.write_h5ad(path, compression="lzf")
        umis = np.asarray(part.X.sum(1)).ravel()
        rows.append({"author_sample_id": author_sample, "sample_id": sample, "time_weeks": w, "relapse": relapse,
                     "cells": part.n_obs, "median_umis": float(np.median(umis)), "h5ad": str(path)})
        print(f"{sample}: {part.n_obs} cells, median UMIs {np.median(umis):.0f}", flush=True)
    samples = pd.DataFrame(rows).sort_values(["time_weeks", "relapse"])
    samples.to_csv(args.out / "samples.tsv", sep="\t", index=False)
    report["samples"] = samples.to_dict("records")
    (args.out / "conversion_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()

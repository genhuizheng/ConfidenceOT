"""Read-only inspection of the downloaded GSE297328 files: structure, counts, labels, sample matching.

    python cancer_application/GSE297328_longitudinal/scripts/00_inspect_data.py --data DATA_DIR --out OUT_DIR

DATA_DIR holds what was downloaded from GEO (GSE297328_ltl331_annotated.h5ad, GSE297328_RAW.tar extracted into raw/,
and any of README / filelist / velocity / harmony / RDS that is present). Nothing is written to DATA_DIR. Reports:
  annotated.h5ad   HDF5 layout (groups, shapes, dtypes); obs columns with dtype, missing values and value counts (< 100
                   values); per-sample cell counts and sample x cell_type / leiden / phase / model tables; .X (log-normalised?
                   expm1 row sums), .raw.X and layers['counts'] (integer? equal on the common genes?), median UMIs per sample,
                   gene naming (symbols / Ensembl, MT- genes), obs_names format per sample
  raw/             per GSM archive: members and sizes, matrix dimensions, barcode count, features head; how many of the
                   authors' cells (16-nt barcode) each archive holds
  other files      velocity / harmony h5ad: obs columns and obsm / layers keys (HDF5 only, nothing loaded)
Writes OUT_DIR/inspect_report.json and OUT_DIR/INSPECT_SUMMARY.md and prints the summary.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

SAMPLE_ROWS = 3000


def h5_layout(path: Path, max_depth: int = 3) -> list[dict]:
    rows = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if name.count("/") >= max_depth:
                return
            rec = {"path": name, "kind": "group" if isinstance(obj, h5py.Group) else "dataset",
                   "encoding": obj.attrs.get("encoding-type", b"").decode() if isinstance(obj.attrs.get("encoding-type", b""), bytes) else str(obj.attrs.get("encoding-type", ""))}
            if isinstance(obj, h5py.Dataset):
                rec.update(shape=list(obj.shape), dtype=str(obj.dtype))
            if "shape" in obj.attrs:
                rec["sparse_shape"] = [int(x) for x in obj.attrs["shape"]]
            rows.append(rec)
        f.visititems(visit)
    return rows


def barcode_of(name: str) -> str:
    m = re.search(r"[ACGTN]{16}", str(name))
    return m.group(0) if m else str(name)


def integer_like(values: np.ndarray) -> dict:
    v = np.asarray(values).ravel()
    return {"n_values": int(v.size), "nonnegative": bool((v >= 0).all()) if v.size else None,
            "integer": bool(np.allclose(v, np.round(v))) if v.size else None, "max": float(v.max()) if v.size else None}


def dense(X) -> np.ndarray:
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def inspect_annotated(path: Path) -> dict:
    import anndata as ad
    rep = {"file": str(path), "bytes": path.stat().st_size, "hdf5_layout": h5_layout(path)}
    a = ad.read_h5ad(path, backed="r")
    rep["shape"] = list(a.shape)
    rep["layers"] = list(a.layers.keys())
    rep["obsm"] = {k: list(np.asarray(a.obsm[k]).shape) for k in a.obsm.keys()}
    rep["uns_keys"] = list(a.uns.keys())
    rep["has_raw"] = a.raw is not None
    obs = a.obs.copy()
    rep["obs_columns"] = []
    for c in obs.columns:
        s = obs[c]
        rec = {"column": c, "dtype": str(s.dtype), "missing": int(s.isna().sum()), "n_unique": int(s.nunique())}
        if s.nunique() < 100:
            rec["values"] = {str(k): int(v) for k, v in s.astype(str).value_counts().items()}
        elif pd.api.types.is_numeric_dtype(s):
            rec["quantiles"] = {q: float(s.quantile(q)) for q in (0, 0.01, 0.5, 0.99, 1)}
        rep["obs_columns"].append(rec)
    rep["obs_names_examples"] = {}
    sample_col = "sample_id" if "sample_id" in obs else None
    if sample_col:
        for s, idx in obs.groupby(sample_col, observed=True).groups.items():
            rep["obs_names_examples"][str(s)] = [str(x) for x in list(idx[:3])]
        rep["cells_per_sample"] = {str(k): int(v) for k, v in obs[sample_col].astype(str).value_counts().sort_index().items()}
        for c in ("cell_type", "leiden", "phase", "model"):
            if c in obs:
                rep[f"sample_x_{c}"] = pd.crosstab(obs[sample_col].astype(str), obs[c].astype(str)).to_dict()
    rep["var_columns"] = list(a.var.columns)
    rep["var_names_examples"] = [str(x) for x in a.var_names[:10]]
    rep["var_mt_genes"] = int(pd.Index(a.var_names.astype(str)).str.upper().str.startswith("MT-").sum())
    # matrices: X, raw.X, layers['counts'] on a sample of rows spread over the file
    rng = np.random.default_rng(20261030)
    rows = np.sort(rng.choice(a.n_obs, min(SAMPLE_ROWS, a.n_obs), replace=False))
    X = dense(a.X[rows])
    rep["X"] = {"dtype": str(X.dtype), **integer_like(X[:200]), "expm1_row_sums_median": float(np.median(np.expm1(X).sum(1)))}
    if a.raw is not None:
        R = dense(a.raw.X[rows])
        rep["raw"] = {"shape": list(a.raw.shape), "dtype": str(R.dtype), **integer_like(R),
                      "row_sum_median": float(np.median(R.sum(1))), "var_names_examples": [str(x) for x in a.raw.var_names[:10]],
                      "mt_genes": int(pd.Index(a.raw.var_names.astype(str)).str.upper().str.startswith("MT-").sum())}
    if "counts" in a.layers:
        C = dense(a.layers["counts"][rows])
        rep["layers_counts"] = {"dtype": str(C.dtype), **integer_like(C), "row_sum_median": float(np.median(C.sum(1)))}
        if a.raw is not None:
            common = pd.Index(a.var_names).intersection(pd.Index(a.raw.var_names))
            ri = pd.Index(a.raw.var_names).get_indexer(common); ci = pd.Index(a.var_names).get_indexer(common)
            rep["layers_counts"]["equal_to_raw_on_common_genes"] = bool(np.array_equal(C[:, ci], R[:, ri]))
            rep["layers_counts"]["common_genes"] = int(len(common))
    if a.raw is not None and sample_col:  # median UMIs per sample over every cell, in chunks
        sums = np.zeros(a.n_obs)
        for start in range(0, a.n_obs, 5000):
            sums[start:start + 5000] = np.asarray(a.raw.X[start:start + 5000].sum(1)).ravel()
        rep["median_raw_umis_per_sample"] = {str(k): float(np.median(sums[np.asarray(obs[sample_col].astype(str) == k)]))
                                             for k in sorted(obs[sample_col].astype(str).unique())}
    rep["_obs"] = obs
    return rep


def inspect_raw_dir(raw: Path, author_obs: pd.DataFrame | None) -> dict:
    rep = {"dir": str(raw), "files": sorted(p.name for p in raw.iterdir()) if raw.exists() else []}
    archives = {}
    for tgz in sorted(raw.glob("GSM*.tar.gz")):
        rec = {"bytes": tgz.stat().st_size}
        with tarfile.open(tgz) as t:
            members = [m for m in t.getmembers() if m.isfile()]
            rec["members"] = [{"name": m.name, "bytes": m.size} for m in members]
            def read(pattern):
                m = next((m for m in members if re.search(pattern, m.name)), None)
                if m is None:
                    return None
                data = t.extractfile(m).read()
                return gzip.decompress(data) if m.name.endswith(".gz") else data
            bc = read(r"barcodes\.tsv")
            ft = read(r"(features|genes)\.tsv")
            mx = read(r"matrix\.mtx")
        if bc is not None:
            barcodes = bc.decode().split()
            rec["barcodes"] = len(barcodes)
            rec["barcode_examples"] = barcodes[:3]
        if ft is not None:
            lines = ft.decode().splitlines()
            rec["features"] = len(lines)
            rec["features_head"] = lines[:3]
        if mx is not None:
            header = [l for l in io.StringIO(mx.decode(errors="ignore")).read(4000).splitlines() if not l.startswith("%")][:1]
            rec["mtx_dims_rows_cols_nnz"] = header[0] if header else ""
        if author_obs is not None and bc is not None and "sample_id" in author_obs:
            name = re.sub(r"^GSM\d+_", "", tgz.name.replace(".tar.gz", ""))
            mine = author_obs[author_obs["sample_id"].astype(str).eq(name)]
            raw_bc = pd.Index([barcode_of(b) for b in barcodes])
            if len(mine):
                found = pd.Index([barcode_of(c) for c in mine.index]).isin(raw_bc)
                rec["author_sample_id_match"] = name
                rec["author_cells"] = int(len(mine)); rec["author_cells_in_archive"] = int(found.sum())
            else:
                rec["author_sample_id_match"] = f"no author sample_id equals {name!r}"
        archives[tgz.name] = rec
    rep["archives"] = archives
    return rep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    present = {p.name: p.stat().st_size for p in args.data.iterdir() if p.is_file()}
    report = {"data_dir": str(args.data), "files_present": present}
    ann = args.data / "GSE297328_ltl331_annotated.h5ad"
    obs = None
    if ann.exists():
        a = inspect_annotated(ann)
        obs = a.pop("_obs")
        report["annotated"] = a
    else:
        report["annotated"] = "missing"
    report["raw"] = inspect_raw_dir(args.data / "raw", obs)
    for name in ("GSE297328_ltl331_velocity.h5ad", "GSE297328_ltl331_harmony_integrated.h5ad"):
        p = args.data / name
        if p.exists():
            with h5py.File(p, "r") as f:
                report[name] = {"obs": list(f["obs"].keys()) if "obs" in f else [], "obsm": list(f["obsm"].keys()) if "obsm" in f else [],
                                "layers": list(f["layers"].keys()) if "layers" in f else [], "uns": list(f["uns"].keys()) if "uns" in f else []}
    (args.out / "inspect_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    lines = ["# GSE297328 data inspection", "", f"data dir `{args.data}`", "", "## Files present", ""]
    lines += [f"- {k}: {v / 1e6:.1f} MB" for k, v in sorted(present.items())]
    if isinstance(report["annotated"], dict):
        a = report["annotated"]
        lines += ["", "## annotated.h5ad", "", f"- shape {a['shape']}, layers {a['layers']}, obsm {a['obsm']}, has .raw {a['has_raw']}",
                  f"- X: {a['X']}", f"- raw: {a.get('raw')}", f"- layers['counts']: {a.get('layers_counts')}",
                  f"- cells per sample: {a.get('cells_per_sample')}", f"- median raw UMIs per sample: {a.get('median_raw_umis_per_sample')}",
                  f"- obs_names examples: {a.get('obs_names_examples')}", "", "### obs columns", ""]
        for c in a["obs_columns"]:
            lines.append(f"- `{c['column']}` {c['dtype']}, missing {c['missing']}, unique {c['n_unique']}: "
                         + (json.dumps(c.get('values'))[:400] if 'values' in c else json.dumps(c.get('quantiles', {}))))
        for c in ("cell_type", "leiden", "model", "phase"):
            if f"sample_x_{c}" in a:
                lines += ["", f"### sample x {c}", "", "```", pd.DataFrame(a[f"sample_x_{c}"]).fillna(0).astype(int).to_string(), "```"]
    lines += ["", "## RAW archives", ""]
    for k, v in report["raw"]["archives"].items():
        lines.append(f"- {k}: {v.get('barcodes')} barcodes, {v.get('features')} features, mtx `{v.get('mtx_dims_rows_cols_nnz')}`, "
                     f"author cells {v.get('author_cells')} / in archive {v.get('author_cells_in_archive')} ({v.get('author_sample_id_match')}); "
                     f"members {[m['name'] for m in v['members']]}")
    for name in ("GSE297328_ltl331_velocity.h5ad", "GSE297328_ltl331_harmony_integrated.h5ad"):
        if name in report:
            lines += ["", f"## {name}", "", f"- {report[name]}"]
    (args.out / "INSPECT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

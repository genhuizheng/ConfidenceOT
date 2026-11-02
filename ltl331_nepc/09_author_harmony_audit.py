"""Audit the authors' Harmony object of GSE297328 and export its LTL331 coordinates (runs on TACC; h5py only, so the
large layers are never loaded).

    python ltl331_nepc/09_author_harmony_audit.py --data DATA_DIR --converted CONVERTED_DIR --out OUT_DIR

DATA_DIR holds GSE297328_ltl331_harmony_integrated.h5ad and GSE297328_ltl331_annotated.h5ad (as for 01_); CONVERTED_DIR
is 01_'s output (h5ad/<sample>.h5ad, whose obs_names are the cell ids every OT run uses). Writes OUT_DIR/ (refuses if it
exists):
  HARMONY_AUDIT.json   the object: shape; var (gene count, first names, var columns: which genes it was built on);
                       every obs column (dtype, missing, value counts when < 200 levels: the integration variables);
                       every obsm / varm / layers key with shape and dtype; the uns tree (keys, small values); and the
                       alignment with our cells: LTL331 rows (dataset == LTL331), duplicated cellids, cellids in the
                       harmony object and not in ours and the reverse, per-cell equality of ltl331_cluster_original with
                       the annotated object's seurat_clusters, of coarse_celltype with our converted obs, of every shared
                       obs column (orig.ident, nCount_RNA, ...)
  HARMONY_AUDIT.md     the same, readable
  AUTHOR_EMBEDDINGS_LTL331.npz   cellid (the converted obs_names, in that order) and, for every obsm key, its rows for
                       those cells (float32; NaN where a cell is missing); nothing is recomputed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HARMONY = "GSE297328_ltl331_harmony_integrated.h5ad"
ANNOTATED = "GSE297328_ltl331_annotated.h5ad"


def strings(values) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.ravel(values)], dtype=object)


def read_frame(group: h5py.Group) -> pd.DataFrame:
    key = group.attrs.get("_index", "_index"); key = key.decode() if isinstance(key, bytes) else str(key)
    out = {}
    for k in group.keys():
        if k == key:
            continue
        o = group[k]
        if isinstance(o, h5py.Group) and "categories" in o:
            out[k] = pd.Categorical.from_codes(o["codes"][()], categories=pd.Index(strings(o["categories"][()])))
        elif isinstance(o, h5py.Group) and "values" in o:
            out[k] = pd.Series(pd.array(o["values"][()])).mask(o["mask"][()].astype(bool)).to_numpy()
        elif isinstance(o, h5py.Dataset):
            out[k] = strings(o[()]) if o.dtype.kind in "OSU" else o[()]
    return pd.DataFrame(out, index=pd.Index(strings(group[key][()])))


def shape_of(o) -> dict:
    if isinstance(o, h5py.Dataset):
        return {"shape": list(o.shape), "dtype": str(o.dtype)}
    enc = o.attrs.get("encoding-type", b""); enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    sh = o.attrs.get("shape")
    return {"encoding": enc, "shape": [int(x) for x in sh] if sh is not None else None}


def uns_tree(o, depth=0, limit=4) -> dict:
    if isinstance(o, h5py.Dataset):
        if o.size <= 20:
            v = o[()]
            return {"value": [str(x) for x in strings(v)] if o.dtype.kind in "OSU" else np.asarray(v).tolist()}
        return {"shape": list(o.shape), "dtype": str(o.dtype)}
    if depth >= limit:
        return {"keys": list(o.keys())[:50]}
    return {k: uns_tree(o[k], depth + 1, limit) for k in list(o.keys())[:100]}


def dense_rows(o, rows: np.ndarray) -> np.ndarray:
    """Rows of an obsm entry (dense dataset, or a csr / csc group) for the given positions."""
    if isinstance(o, h5py.Dataset):
        order = np.argsort(rows); out = np.empty((len(rows),) + o.shape[1:], dtype=np.float32)
        out[order] = o[np.sort(rows)]
        return out
    import scipy.sparse as sp
    enc = o.attrs.get("encoding-type", b""); enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    m = (sp.csr_matrix if enc.startswith("csr") else sp.csc_matrix)((o["data"][()], o["indices"][()], o["indptr"][()]),
                                                                     shape=tuple(o.attrs["shape"]))
    return np.asarray(m.tocsr()[rows].todense(), dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    ours = []
    for f in sorted((a.converted / "h5ad").glob("*.h5ad")):
        with h5py.File(f, "r") as h:
            o = read_frame(h["obs"]); o["file_sample"] = f.stem; ours.append(o)
    ours = pd.concat(ours)
    rep = {"file": HARMONY, "our_cells": int(len(ours))}
    with h5py.File(a.data / ANNOTATED, "r") as h:
        ann = read_frame(h["obs"])
    with h5py.File(a.data / HARMONY, "r") as h:
        obs = read_frame(h["obs"]); var = read_frame(h["var"])
        rep["shape"] = [int(len(obs)), int(len(var))]
        rep["var"] = {"n": int(len(var)), "first_names": list(var.index[:20]), "columns": list(var.columns),
                      **{f"{c}_true": int(var[c].astype(bool).sum()) for c in var.columns if var[c].dtype == bool}}
        cols = {}
        for c in obs.columns:
            s = obs[c]; r = {"dtype": str(s.dtype), "missing": int(pd.isna(s).sum()), "levels": int(s.astype(str).nunique())}
            if r["levels"] < 200:
                r["value_counts"] = {str(k): int(v) for k, v in s.astype(str).value_counts().items()}
            cols[c] = r
        rep["obs_columns"] = cols
        rep["obs_names_examples"] = list(obs.index[:5])
        for grp in ("obsm", "varm", "layers", "obsp"):
            rep[grp] = {k: shape_of(h[grp][k]) for k in h[grp].keys()} if grp in h else {}
        rep["X"] = shape_of(h["X"]) if "X" in h else None
        rep["raw"] = {k: shape_of(h["raw"][k]) for k in h["raw"].keys()} if "raw" in h else None
        rep["uns"] = uns_tree(h["uns"]) if "uns" in h else {}

        # alignment with our cells
        ds = obs["dataset"].astype(str) if "dataset" in obs else pd.Series("?", index=obs.index)
        rep["dataset_counts"] = {str(k): int(v) for k, v in ds.value_counts().items()}
        ltl = obs[ds.eq("LTL331").to_numpy()]
        cid = ltl["cellid"].astype(str) if "cellid" in ltl else pd.Series(ltl.index, index=ltl.index)
        al = {"ltl331_rows": int(len(ltl)), "duplicated_cellids": int(cid.duplicated().sum()),
              "cellids_not_in_ours": int((~cid.isin(ours.index)).sum()), "ours_not_in_harmony": int((~ours.index.isin(cid)).sum()),
              "harmony_obs_name_equals_cellid": int((pd.Index(ltl.index) == pd.Index(cid)).sum())}
        pos = pd.Series(np.flatnonzero(ds.eq("LTL331").to_numpy()), index=cid.to_numpy())
        pos = pos[~pos.index.duplicated()]
        j = ours.index[ours.index.isin(pos.index)]
        hl = ltl.set_index(cid.to_numpy()).loc[j]
        if "ltl331_cluster_original" in hl and "seurat_clusters" in ann:
            al["cluster_original_equals_seurat_clusters"] = int((hl["ltl331_cluster_original"].astype(str).to_numpy()
                                                                 == ann.loc[j, "seurat_clusters"].astype(str).to_numpy()).sum())
        if "coarse_celltype" in hl:
            al["coarse_celltype_equals_ours"] = int((hl["coarse_celltype"].astype(str).to_numpy() == ours.loc[j, "coarse_celltype"].astype(str).to_numpy()).sum())
        shared = {}
        for c in hl.columns:
            if c in ann.columns and c not in ("cellid",):
                x, y = hl[c].to_numpy(), ann.loc[j, c].to_numpy()
                try:
                    eq = np.isclose(x.astype(float), y.astype(float), equal_nan=True)
                except (TypeError, ValueError):
                    eq = x.astype(str) == y.astype(str)
                shared[c] = int(eq.sum())
        al["shared_obs_columns_equal_cells"] = shared
        al["compared_cells"] = int(len(j))
        rep["alignment"] = al
        # per sample: cells and clusters of the LTL331 rows (the integration variables are in obs_columns)
        emb = {"cellid": np.asarray(ours.index, dtype=str)}
        rows = pos.reindex(ours.index)
        have = rows.notna().to_numpy(); r = rows[have].astype(int).to_numpy()
        for k in (h["obsm"].keys() if "obsm" in h else []):
            o = h["obsm"][k]
            if isinstance(o, h5py.Group) and "categories" in o:
                continue
            try:
                m = dense_rows(o, r)
            except Exception as e:  # dataframe-valued obsm etc.
                rep.setdefault("obsm_not_exported", {})[k] = repr(e); continue
            full = np.full((len(ours),) + m.shape[1:], np.nan, dtype=np.float32); full[have] = m
            emb[k] = full
    np.savez_compressed(a.out / "AUTHOR_EMBEDDINGS_LTL331.npz", **emb)
    rep["exported_obsm"] = {k: list(v.shape) for k, v in emb.items() if k != "cellid"}
    (a.out / "HARMONY_AUDIT.json").write_text(json.dumps(rep, indent=1, default=str))
    md = [f"# Author Harmony object audit ({HARMONY})", "", f"- shape {rep['shape']}; datasets {rep['dataset_counts']}",
          f"- obs names e.g. {rep['obs_names_examples']}", f"- var: {rep['var']['n']} genes, columns {rep['var']['columns']}",
          f"- X {rep['X']}; raw {rep['raw']}", "", "## obsm / varm / layers", ""]
    md += [f"- {g}.{k}: {v}" for g in ("obsm", "varm", "layers", "obsp") for k, v in rep[g].items()]
    md += ["", "## Alignment with our cells", ""] + [f"- {k}: {v}" for k, v in al.items()]
    md += ["", "## obs columns", ""]
    for c, r_ in cols.items():
        vc = r_.get("value_counts"); md.append(f"- {c} ({r_['dtype']}, {r_['levels']} levels, {r_['missing']} missing)"
                                              + (f": {dict(list(vc.items())[:30])}" if vc else ""))
    md += ["", "## uns", "", "```", json.dumps(rep["uns"], indent=1, default=str)[:20000], "```"]
    (a.out / "HARMONY_AUDIT.md").write_text("\n".join(md))
    print("\n".join(md[:40]))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

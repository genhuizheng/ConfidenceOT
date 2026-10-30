"""Second read-only check of GSE297328: where the raw counts are and how the cells of the three h5ad files and the RAW
CellRanger matrices correspond.

    python cancer_application/GSE297328_longitudinal/scripts/00b_match_cells.py --data DATA_DIR --out OUT_DIR

Why: 00_inspect_data.py found that the annotated h5ad does not match its README. Its .raw is log-normalised, not
integer (max 7.58), it has no layers['counts'], and its obs is a Seurat export (orig.ident, pretty, timepoint, week_num,
nCount_RNA, nFeature_RNA, seurat_clusters, cellid) with no sample_id / cell_type / leiden / phase. Cell-type labels
are only in the harmony object (coarse_celltype), whose cell names share none with the annotated object; the cell cycle
phase is only in the velocity object. This script checks, without writing to DATA_DIR:
  1. identifiers: obs_names, cellid, Barcode, orig_obs_names examples per time point in each object;
  2. RAW matrices: integer?, barcode format, features columns; for every annotated cell, its 16-nt barcode in the
     archive of its time point (matched by week number); nCount_RNA / nFeature_RNA equal to the RAW column sum / number
     of detected genes?; annotated .raw equal to log1p(1e4 * counts / total) on the common genes for 300 cells;
  3. harmony: which identifier links its LTL331 cells to the annotated cells; coarse_celltype x seurat cluster and x
     time point; ltl331_cluster_original equal to the annotated seurat_clusters on the linked cells?;
  4. velocity: cells linked to annotated, phase and latent_time per time point.
Writes OUT_DIR/match_report.json and OUT_DIR/MATCH_SUMMARY.md and prints the summary.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

ANN = "GSE297328_ltl331_annotated.h5ad"
HARMONY = "GSE297328_ltl331_harmony_integrated.h5ad"
VELOCITY = "GSE297328_ltl331_velocity.h5ad"


def strings(values) -> np.ndarray:
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in np.ravel(values)], dtype=object)


def read_obs(path: Path) -> pd.DataFrame:
    """obs of an h5ad with h5py: categoricals decoded, nullable columns masked, the index as the frame index."""
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
                out[k] = np.where(codes >= 0, cats[np.clip(codes, 0, None)], None)
            elif isinstance(o, h5py.Group) and "values" in o:
                v = o["values"][()].astype(object); v[o["mask"][()].astype(bool)] = None
                out[k] = v
            elif isinstance(o, h5py.Dataset):
                out[k] = strings(o[()]) if o.dtype.kind in "OSU" else o[()]
        return pd.DataFrame(out, index=pd.Index(strings(obs[key][()]), name=None))


def barcode_of(name: str) -> str | None:
    m = re.search(r"[ACGTN]{16}", str(name))
    return m.group(0) if m else None


def weeks_of(name: str) -> int:
    """Week number of a GSM sample name: preCX -> 0, '<n>wk' -> n."""
    if re.search(r"pre", name, re.I):
        return 0
    return int(re.search(r"(\d+)wk", name).group(1))


def read_archive(tgz: Path) -> dict:
    with tarfile.open(tgz) as t:
        members = {Path(m.name).name: m for m in t.getmembers() if m.isfile()}
        def get(prefix):
            m = next((members[n] for n in members if n.startswith(prefix)), None)
            data = t.extractfile(m).read()
            return __import__("gzip").decompress(data) if m.name.endswith(".gz") else data
        barcodes = get("barcodes").decode().split()
        features = [l.split("\t") for l in get("features").decode().splitlines()] if any(n.startswith("features") for n in members) \
            else [l.split("\t") for l in get("genes").decode().splitlines()]
        M = sp.csc_matrix(scipy.io.mmread(io.BytesIO(get("matrix"))))  # genes x cells
    return {"barcodes": barcodes, "features": features, "M": M}


def link(a: pd.Series, b: pd.Series) -> int:
    return int(pd.Index(a.astype(str)).isin(pd.Index(b.astype(str))).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-cells", type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rep, lines = {}, ["# GSE297328 cell matching", ""]

    ann = read_obs(args.data / ANN)
    ann["obs_name"] = ann.index
    ann["week"] = pd.to_numeric(ann["week_num"]).astype(int)
    with h5py.File(args.data / ANN, "r") as f:
        var = strings(f["raw"]["var"][f["raw"]["var"].attrs.get("_index", "_index")][()])
    lines += ["## 1. identifiers", "", "### annotated", ""]
    for w, g in ann.groupby("week"):
        lines.append(f"- week {w} ({g['pretty'].iloc[0]}, {len(g)} cells): obs_name {list(g.index[:2])}, cellid {list(g['cellid'].iloc[:2])}")
    harm = read_obs(args.data / HARMONY)
    harm["obs_name"] = harm.index
    hl = harm[harm["dataset"].eq("LTL331")]
    lines += ["", "### harmony (LTL331 cells)", ""]
    for p, g in hl.groupby("pretty"):
        lines.append(f"- {p} ({len(g)} cells): obs_name {list(g.index[:2])}, cellid {list(g['cellid'].iloc[:2])}, Barcode {list(g['Barcode'].iloc[:2])}")
    vel = read_obs(args.data / VELOCITY)
    vel["obs_name"] = vel.index
    lines += ["", "### velocity", ""]
    for p, g in vel.groupby("pretty"):
        lines.append(f"- {p} ({len(g)} cells): obs_name {list(g.index[:2])}, cellid {list(g['cellid'].iloc[:2])}, orig_obs_names {list(g['orig_obs_names'].iloc[:2])}")

    # 2. RAW matrices
    lines += ["", "## 2. RAW matrices against annotated cells", ""]
    raw_rep = {}
    rng = np.random.default_rng(20261030)
    for tgz in sorted((args.data / "raw").glob("GSM*.tar.gz")):
        name = re.sub(r"^GSM\d+_", "", tgz.name.replace(".tar.gz", ""))
        w = weeks_of(name)
        r = read_archive(tgz)
        M, bcs, feats = r["M"], r["barcodes"], r["features"]
        rec = {"sample": name, "week": w, "genes_x_cells": list(M.shape), "nnz": int(M.nnz),
               "integer": bool(np.allclose(M.data, np.round(M.data))), "max": float(M.data.max()) if M.nnz else None,
               "barcode_examples": bcs[:2], "features_columns": len(feats[0]), "features_head": feats[:2],
               "barcodes_unique_16nt": int(pd.Index([barcode_of(b) for b in bcs]).nunique())}
        mine = ann[ann["week"].eq(w)]
        rec["author_cells"] = int(len(mine))
        raw_pos = pd.Series(np.arange(len(bcs)), index=[barcode_of(b) for b in bcs])
        raw_pos = raw_pos[~raw_pos.index.duplicated(keep=False)]
        for col in ("obs_name", "cellid"):
            bc = mine[col].map(barcode_of)
            rec[f"via_{col}_with_16nt"] = int(bc.notna().sum())
            rec[f"via_{col}_found"] = int(bc.isin(raw_pos.index).sum())
        col = max(("obs_name", "cellid"), key=lambda c: rec[f"via_{c}_found"])
        bc = mine[col].map(barcode_of)
        found = mine[bc.isin(raw_pos.index)]
        if len(found):
            pos = raw_pos[found[col].map(barcode_of)].to_numpy()
            colsum = np.asarray(M[:, pos].sum(0)).ravel(); ndet = np.diff(M[:, pos].indptr)
            nc = pd.to_numeric(found["nCount_RNA"]).to_numpy(); nf = pd.to_numeric(found["nFeature_RNA"]).to_numpy()
            rec.update(match_column=col, nCount_equal_fraction=float(np.mean(np.isclose(colsum, nc))),
                       nCount_ratio_median=float(np.median(colsum / nc)), nFeature_equal_fraction=float(np.mean(ndet == nf)),
                       nFeature_ratio_median=float(np.median(ndet / nf)))
            # annotated .raw against log1p(1e4 * counts / total) on the common genes
            sym = pd.Index([f[1] if len(f) > 1 else f[0] for f in feats])
            # gene symbols that occur once on each side (CellRanger repeats some; Seurat makes them unique)
            s_ok = ~sym.duplicated(keep=False); v_idx = pd.Index(var); v_ok = ~v_idx.duplicated(keep=False)
            common = v_idx[v_ok].intersection(sym[s_ok])
            rec["raw_symbols_duplicated"] = int((~s_ok).sum()); rec["annotated_genes_not_in_raw_symbols"] = int((~v_idx.isin(sym)).sum())
            take = np.sort(rng.choice(len(found), min(args.check_cells, len(found)), replace=False))
            rows = ann.index.get_indexer(found.index[take])
            with h5py.File(args.data / ANN, "r") as f:
                g = f["raw"]["X"]
                ip = g["indptr"]; data = g["data"]; ind = g["indices"]
                R = np.zeros((len(rows), len(var)))
                for i, rr in enumerate(rows):
                    s, e = ip[rr], ip[rr + 1]
                    R[i, ind[s:e]] = data[s:e]
            C = M[:, pos[take]].T.toarray()
            ci = np.flatnonzero(s_ok)[sym[s_ok].get_indexer(common)]
            vi = np.flatnonzero(v_ok)[v_idx[v_ok].get_indexer(common)]
            for label, tot in (("total_all_raw_genes", C.sum(1)), ("total_common_genes", C[:, ci].sum(1)), ("nCount_RNA", nc[take])):
                L = np.log1p(1e4 * C[:, ci] / tot[:, None])
                rec[f"raw_vs_log1p_{label}_max_abs_diff"] = float(np.abs(L - R[:, vi]).max())
            rec["common_genes"] = int(len(common))
            rec["annotated_raw_expm1_row_sum_median"] = float(np.median(np.expm1(R).sum(1)))
        raw_rep[tgz.name] = rec
        lines.append(f"- {tgz.name}: week {w}, {M.shape[1]} barcodes, integer {rec['integer']} (max {rec['max']}), barcode {bcs[:1]}, "
                     f"features {feats[0]}; author cells {len(mine)}, found via obs_name {rec['via_obs_name_found']} / via cellid {rec['via_cellid_found']}; "
                     f"nCount equal {rec.get('nCount_equal_fraction')} (ratio {rec.get('nCount_ratio_median')}), nFeature equal {rec.get('nFeature_equal_fraction')}; "
                     f".raw vs log1p: all genes {rec.get('raw_vs_log1p_total_all_raw_genes_max_abs_diff')}, common {rec.get('raw_vs_log1p_total_common_genes_max_abs_diff')}, "
                     f"nCount {rec.get('raw_vs_log1p_nCount_RNA_max_abs_diff')}; expm1 row sum {rec.get('annotated_raw_expm1_row_sum_median')}; "
                     f"common genes {rec.get('common_genes')}, RAW symbols duplicated {rec.get('raw_symbols_duplicated')}, annotated genes not in RAW {rec.get('annotated_genes_not_in_raw_symbols')}")
    rep["raw"] = raw_rep

    # 3. harmony and 4. velocity against annotated
    lines += ["", "## 3. harmony against annotated", ""]
    hlinks = {f"harmony.{h} -> annotated.{a}": link(hl[h], ann[a]) for h in ("obs_name", "cellid", "Barcode") for a in ("obs_name", "cellid")}
    rep["harmony_links"] = hlinks
    lines += [f"- {k}: {v} of {len(hl)}" for k, v in hlinks.items()]
    best = max(hlinks, key=hlinks.get)
    hcol, acol = best.split(" -> ")[0].split(".")[1], best.split(" -> ")[1].split(".")[1]
    lines += ["", "### coarse_celltype x ltl331_cluster_original (LTL331 cells)", "", "```",
              pd.crosstab(hl["ltl331_cluster_original"], hl["coarse_celltype"]).to_string(), "```",
              "", "### coarse_celltype x pretty (LTL331 cells)", "", "```", pd.crosstab(hl["pretty"], hl["coarse_celltype"]).to_string(), "```"]
    if hlinks[best]:
        m = ann.reset_index(drop=True).merge(hl.reset_index(drop=True), left_on=acol, right_on=hcol, suffixes=("", "_h"))
        same = (m["seurat_clusters"].astype(str) == m["ltl331_cluster_original"].astype(str)).mean()
        rep["harmony_best_link"] = {"link": best, "linked": int(len(m)), "unique_linked_annotated": int(m[acol].nunique()),
                                    "seurat_cluster_equal_fraction": float(same)}
        lines.append(f"- best link {best}: {len(m)} rows, {m[acol].nunique()} distinct annotated cells; seurat_clusters equal on {same:.4f}")
    lines += ["", "## 4. velocity against annotated", ""]
    vlinks = {f"velocity.{v} -> annotated.{a}": link(vel[v], ann[a]) for v in ("obs_name", "cellid", "orig_obs_names") for a in ("obs_name", "cellid")}
    rep["velocity_links"] = vlinks
    lines += [f"- {k}: {v} of {len(vel)}" for k, v in vlinks.items()]
    lt = pd.to_numeric(vel["latent_time"])
    lines += ["", "### phase x pretty", "", "```", pd.crosstab(vel["pretty"], vel["phase"]).to_string(), "```", "",
              "### latent_time per pretty (median)", "", "```", lt.groupby(vel["pretty"]).median().to_string(), "```"]
    (args.out / "match_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (args.out / "MATCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

"""Content checks on the collected data, beyond format (07_verify_inputs.py): does each label mean what it says?

Reads the prepared inputs (APP/<dataset>/inputs/*.h5ad, each cell once) and, for SPLINTR and zebrafish, the raw downloads.
Expression summaries are mean log1p(CP10k) per group; nothing here feeds the OT or the scoring. Writes APP/content/*.tsv
and prints the tables. Reads only.

  kang         ISG15 / IFI6 / IFIT1 / MX1 / ISG20 by condition x cell type (expected: stim >> ctrl in every type)
  sciplex2     CDKN1A / MDM2 (Nutlin), TSC22D3 / FKBP5 (Dex), MKI67 / TOP2A by drug x dose
  schiebinger  Nanog / Zfp42 / Dppa5a (iPSC), Col1a1 / Thy1 (fibroblast) by condition x stage
  pancreas     Neurog3 / Fev / Ins1 / Gcg / Sst / Ghrl / Cpa1 / Sox9 by cell type
  larry_invivo Ltf / Elane (Neu), Csf1r / Ly6c2 (Mono), Klf1 / Hba-a1 (Ery), Cd79a (B), Pf4 (Meg), Kit by cell type;
               total counts by time x population
  celegans     embryo_time range per bin; lineage-name depth per bin; founder prefix per bin
  splintr      argmax hashtag per barcoded sample (authors' HTO map) in KF16 POOL_1 / POOL_2
  zebrafish    metadata cells vs GEO filtered h5 barcodes, per library

    python -I 08_content_checks.py --app $SCRATCH/OT_project/app_search_20261117 --only kang sciplex2 ...
"""
from __future__ import annotations

import argparse
import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd


def load(app: Path, ds: str):
    """Every cell of a dataset once (a cell can sit in several units)."""
    import anndata as ad
    parts, seen = [], set()
    for f in sorted((app / ds / "inputs").glob("*.h5ad")):
        A = ad.read_h5ad(f)
        # obs names carry the unit; the original row index is the suffix after the last "_"
        A.obs["row"] = [n.rsplit("_", 1)[1] for n in A.obs_names]
        keep = ~A.obs["row"].isin(seen).to_numpy()
        seen.update(A.obs["row"][keep])
        parts.append(A[keep].copy())
    A = ad.concat(parts)
    return A


def lcp(A, genes):
    import scipy.sparse as sp
    X = A.X.tocsr() if sp.issparse(A.X) else sp.csr_matrix(A.X)
    tot = np.asarray(X.sum(1)).ravel()
    out = {}
    for g in genes:
        if g in A.var_names:
            v = np.asarray(X[:, A.var_names.get_loc(g)].todense()).ravel()
            out[g] = np.log1p(v / tot * 1e4)
        else:
            out[g] = np.full(A.n_obs, np.nan)
    return pd.DataFrame(out, index=A.obs_names)


def table(A, genes, by):
    E = lcp(A, genes)
    for c in by:
        E[c] = A.obs[c].astype(str).to_numpy()
    t = E.groupby(by).mean().round(3)
    t.insert(0, "cells", E.groupby(by).size())
    return t


def kang(app, out):
    A = load(app, "kang")
    return table(A, ["ISG15", "IFI6", "IFIT1", "MX1", "ISG20"], ["cell_type", "condition"])


def sciplex2(app, out):
    A = load(app, "sciplex2")
    t = table(A, ["CDKN1A", "MDM2", "TSC22D3", "FKBP5", "MKI67", "TOP2A"], ["drug", "dose"]).reset_index()
    t["dose_num"] = t.dose.astype(float)
    return t.sort_values(["drug", "dose_num"]).drop(columns="dose_num").set_index(["drug", "dose"])


def sciplex2_wells(app, out):
    """Per well: is the p53 signal at BMS 1 uM (and the Nutlin 100 uM drop) carried by a few wells or by all?"""
    A = load(app, "sciplex2")
    t = table(A, ["CDKN1A", "MDM2", "TSC22D3", "FKBP5", "MKI67"], ["drug", "dose", "well"]).reset_index()
    t["dose_num"] = t.dose.astype(float)
    return t.sort_values(["drug", "dose_num", "well"]).drop(columns="dose_num").set_index(["drug", "dose", "well"])


def schiebinger(app, out):
    A = load(app, "schiebinger")
    order = {s: i for i, s in enumerate(["D0", "D2", "D4", "D6", "D8", "D9", "D10", "D11", "D12", "D16", "iPSC"])}
    t = table(A, ["Nanog", "Zfp42", "Dppa5a", "Col1a1", "Thy1"], ["condition", "stage"]).reset_index()
    t["o"] = t.stage.map(order)
    return t.sort_values(["condition", "o"]).drop(columns="o").set_index(["condition", "stage"])


def pancreas(app, out):
    A = load(app, "pancreas")
    return table(A, ["Neurog3", "Fev", "Ins1", "Gcg", "Sst", "Ghrl", "Cpa1", "Sox9"], ["cell_type"])


def larry_invivo(app, out):
    import scipy.sparse as sp
    A = load(app, "larry_invivo")
    t = table(A, ["Ltf", "Elane", "Csf1r", "Ly6c2", "Klf1", "Hba-a1", "Cd79a", "Pf4", "Kit"], ["cell_type"])
    tot = np.asarray(A.X.sum(1)).ravel()
    d = pd.DataFrame({"time": A.obs.time.astype(str).to_numpy(), "population": A.obs.population.astype(str).to_numpy(), "total": tot})
    dep = d.groupby(["population", "time"]).total.describe()[["count", "25%", "50%", "75%"]]
    dep.to_csv(out / "larry_invivo_depth.tsv", sep="\t"); print(dep.to_string())
    return t


def celegans(app, out):
    A = load(app, "celegans")
    o = A.obs.copy()
    o["lin"] = o.lineage.astype(str)
    ann = ~o.lin.isin(["", "nan", "None"])
    o["depth"] = np.where(ann, o.lin.str.split("/").str[0].str.len(), np.nan)
    o["founder"] = np.where(ann, o.lin.str.extract(r"^(ABa|ABp|AB|MS|E|C|D|P4|Z2|Z3|P\d|EMS)", expand=False).fillna("other"), "unannotated")
    bins = ["< 100", "100-130", "130-170", "170-210", "210-270", "270-330", "330-390", "390-450", "450-510", "510-580", "580-650", "> 650"]
    g = o.groupby("time_bin").agg(cells=("lin", "size"), t_min=("embryo_time", "min"), t_median=("embryo_time", "median"),
                                  t_max=("embryo_time", "max"), annotated=("depth", "count"), name_len_median=("depth", "median"))
    g = g.reindex(bins)
    f = pd.crosstab(o.time_bin, o.founder).reindex(bins)
    f.to_csv(out / "celegans_founder.tsv", sep="\t"); print(f.to_string())
    return g


def splintr(app, out):
    import scipy.io
    data = app / "data"
    sheet = pd.read_csv(data / "splintr_singletcode_figshare/extracted/allBarcodeSheets/SPLINTR/stepFourStarcodeShavedReads50.csv")
    sheet = sheet[sheet["sample"].str.startswith("chemo")].drop_duplicates(["sample", "cellID"])
    rows = []
    with tarfile.open(data / "splintr_gse186305/GSE186305_RAW.tar") as tf:
        for pool, gsm in (("POOL_1", "GSM5643584_KF16_POOL_1_HTO"), ("POOL_2", "GSM5643585_KF16_POOL_2_HTO")):
            M = scipy.io.mmread(io.BytesIO(gzip.decompress(tf.extractfile(f"{gsm}_matrix.mtx.gz").read()))).tocsc()
            bc = pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(f"{gsm}_barcodes.tsv.gz").read())), header=None)[0].str.replace("-1$", "", regex=True)
            ft = pd.read_csv(io.BytesIO(gzip.decompress(tf.extractfile(f"{gsm}_features.tsv.gz").read())), sep="\t", header=None)
            names = ft[1].astype(str).to_numpy() if ft.shape[1] > 1 else ft[0].astype(str).to_numpy()
            if M.shape[0] != len(names):
                M = M.T.tocsc()
            D = pd.DataFrame(M.toarray().T, index=bc.to_numpy(), columns=names)
            top = D.idxmax(axis=1); frac = D.max(axis=1) / D.sum(axis=1).replace(0, np.nan)
            for s, g in sheet.groupby("sample"):
                hit = g.cellID[g.cellID.isin(D.index)]
                if len(hit) == 0:
                    continue
                vc = top[hit].value_counts()
                rows.append({"pool": pool, "sample": s, "cells_in_hto": len(hit), "top_hashtag": vc.index[0],
                             "top_share": round(vc.iloc[0] / len(hit), 3), "second": vc.index[1] if len(vc) > 1 else "",
                             "second_share": round(vc.iloc[1] / len(hit), 3) if len(vc) > 1 else 0.0,
                             "median_top_fraction": round(float(frac[hit].median()), 3)})
    return pd.DataFrame(rows).set_index(["pool", "sample"])


def zebrafish(app, out):
    import h5py
    data = app / "data"
    m = pd.read_csv(data / "zebrafish_heart_moslin_figshare/Zebrafish_metadata.csv", usecols=["Cell", "orig.ident"], low_memory=False)
    m["bc"] = m.Cell.str.split("_", n=1).str[1]
    libs = {}
    for tar in (data / "zebrafish_heart_gse159032/GSE159032_RAW.tar", data / "zebrafish_heart_gse158919/GSE158919_RAW.tar"):
        with tarfile.open(tar) as tf:
            for mem in tf.getmembers():
                if mem.name.endswith("_filtered_matrix.h5"):
                    with h5py.File(io.BytesIO(tf.extractfile(mem).read()), "r") as h:
                        g = h["matrix"] if "matrix" in h else h[list(h.keys())[0]]
                        b = pd.Index(pd.Series(g["barcodes"][:].astype(str)).str.replace("-1$", "", regex=True))
                    lib = mem.name.split("_", 1)[1].replace("_filtered_matrix.h5", "")
                    libs[f"{tar.parent.name.split('_')[-1]}:{lib}"] = b
    rows = []
    for ident, g in m.groupby("orig.ident"):
        bcs = pd.Index(g.bc)
        best = sorted(((int(bcs.isin(b).sum()), k) for k, b in libs.items()), reverse=True)[:2]
        rows.append({"orig.ident": ident, "metadata_cells": len(bcs), "best_library": best[0][1], "matched": best[0][0],
                     "share": round(best[0][0] / len(bcs), 3), "second_library": best[1][1], "second_matched": best[1][0]})
    return pd.DataFrame(rows).set_index("orig.ident")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--app", type=Path, required=True)
    p.add_argument("--only", nargs="+", default=["kang", "sciplex2", "schiebinger", "pancreas", "larry_invivo", "celegans", "splintr", "zebrafish"])
    a = p.parse_args()
    out = a.app / "content"; out.mkdir(exist_ok=True)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    for ds in a.only:
        print(f"===== {ds}", flush=True)
        t = globals()[ds](a.app, out)
        t.to_csv(out / f"{ds}.tsv", sep="\t")
        print(t.to_string(), flush=True)


if __name__ == "__main__":
    main()

"""PLAN_TACC.md step 1: unit inputs for the v3 worker's manifest route, one dataset per call, on TACC. Reads the downloaded
files in place (archives through tarfile, nothing extracted next to the data) and writes

    OUT/<dataset>/inputs/<unit>__{source,target}.h5ad   integer counts; obs: sample_id, malignant = "malignant" (every
                                                       cell in scope) and the labels used for scoring only
    OUT/<dataset>/manifest.csv                         02_run_pair columns (absolute paths)
    OUT/<dataset>/prep/units.tsv                       cells per side before / after the cap, priority cells kept
    OUT/<dataset>/prep/*.tsv                           dataset checks (integer recovery, barcode match, crosstabs)

    python -I lineage_validation/app_search_20261117/06_prepare_tacc.py --dataset kang \
        --data $SCRATCH/OT_project/app_search_20261117/data --out $SCRATCH/OT_project/app_search_20261117

Cap: at most 10,000 cells per side, seed 20261118; cells carrying the dataset's truth label first (LARRY: clonally
informative cells; C. elegans: lineage-annotated cells), then random others. Datasets (PLAN_TACC.md):
  larry_invivo  GSM4185643; integer counts recovered as in larry_20261112/01_audit.py (a cell is rescaled by its smallest
                non-zero value; refused unless the median share of integer entries is >= 0.99); units day 2 -> day 9 /
                day 16 per starting population (LSK = Lin-Kit+Sca1hi, LK = Lin-Kit), recipients pooled
  kang          GSE96583 batch 2 singlets with a cell type; ctrl -> stim pooled, and per donor (8)
  schiebinger   pertpy subset of GSE122662; conditions from the GEO sample titles (schiebinger_gsm_map.tsv): Dox phase
                D0 -> D2 -> D4 -> D6 -> D8, then 2i D8 -> D9 -> D10 -> D11 -> D12 -> D16 -> iPSCs(2i) and serum
                D8 -> D10 -> D12 -> D16 -> iPSCs(serum). The iPSC samples are an established line, a reference rather
                than descendants of the time course
  sciplex2      pertpy sciPlex2; per drug, dose 0 of that drug -> each non-zero dose (7)
  celegans      GSE126954, QC-passed cells; adjacent embryo.time.bin
  pancreas      GSE132188: the authors' cells (h5ad obs), raw counts from RAW.tar matched by barcode and stage;
                E12.5 -> E13.5 -> E14.5 -> E15.5
"""

from __future__ import annotations

import argparse
import gzip
import io
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CAP, SEED = 10_000, 20261118
HERE = Path(__file__).resolve().parent


def read_mtx(fh, dtype=np.float32, chunk=20_000_000, transpose=False):
    """MatrixMarket coordinate reader (text handle); returns CSR, rows x cols as stored unless transpose."""
    line = fh.readline()
    while line.startswith("%"):
        line = fh.readline()
    nr, nc, _ = map(int, line.split())
    rows, cols, vals = [], [], []
    for df in pd.read_csv(fh, sep=" ", header=None, names=["r", "c", "v"], chunksize=chunk,
                          dtype={"r": np.int32, "c": np.int32, "v": np.float64}):
        rows.append(df.r.to_numpy() - 1); cols.append(df.c.to_numpy() - 1); vals.append(df.v.to_numpy().astype(dtype))
    r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    if transpose:
        r, c, nr, nc = c, r, nc, nr
    return sp.csr_matrix((v, (r, c)), shape=(nr, nc))


def is_integer(X) -> bool:
    d = X.data if sp.issparse(X) else np.asarray(X).ravel()
    return bool(np.allclose(d, np.round(d)))


class Writer:
    def __init__(self, out: Path, dataset: str):
        self.root = (out / dataset).resolve()  # manifest paths must be absolute for the worker
        (self.root / "inputs").mkdir(parents=True, exist_ok=True); (self.root / "prep").mkdir(exist_ok=True)
        self.dataset, self.man, self.units = dataset, [], []

    def unit(self, u, patient, A, src, tgt, obs_cols, priority=None, meta=None):
        """A: AnnData with integer counts; src / tgt: integer row indices; priority: boolean per row of A."""
        import anndata as ad
        rng = np.random.default_rng(SEED)
        rec = {"pair_id": u, "dataset_id": self.dataset, "patient_id": patient, **(meta or {})}
        man = {"pair_id": u, "dataset_id": self.dataset, "patient_id": patient}
        empty = [s for s, i in (("source", src), ("target", tgt)) if len(i) == 0]
        if empty:  # the unit does not exist in the data (no threshold: only zero cells); listed, never written
            rec.update({"source_all": len(src), "target_all": len(tgt), "formed": False, "note": "no cells on " + " and ".join(empty)})
            self.units.append(rec); print(rec, flush=True)
            return
        rec["formed"] = True
        for side, idx in (("source", src), ("target", tgt)):
            idx = np.asarray(idx)
            pri = idx[priority[idx]] if priority is not None else idx[:0]
            rest = np.setdiff1d(idx, pri)
            if len(pri) >= CAP:
                keep = np.sort(rng.choice(pri, CAP, replace=False))
            else:
                keep = np.sort(np.r_[pri, rng.choice(rest, min(CAP - len(pri), len(rest)), replace=False)]).astype(int)
            B = A[keep].copy()
            B.obs = pd.DataFrame({"sample_id": f"{u}__{side}", "malignant": "malignant",
                                  **{c: (A.obs[c].to_numpy() if pd.api.types.is_numeric_dtype(A.obs[c]) else A.obs[c].astype(str).to_numpy())[keep]
                                     for c in obs_cols}},
                                 index=[f"{u}_{side}_{i}" for i in keep])
            B.var_names = B.var_names.astype(str)
            assert is_integer(B.X), f"{u} {side}: counts are not integers"
            f = self.root / "inputs" / f"{u}__{side}.h5ad"
            ad.AnnData(X=sp.csr_matrix(B.X, dtype=np.float32), obs=B.obs, var=pd.DataFrame(index=B.var_names)).write_h5ad(f, compression="gzip")
            rec.update({f"{side}_all": len(idx), f"{side}_cells": len(keep), f"{side}_priority_kept": int(min(len(pri), len(keep)))})
            man.update({f"{side}_sample": f"{u}__{side}", f"{side}_h5ad": str(f)})
        man["eligible"] = True
        self.man.append(man); self.units.append(rec); print(rec, flush=True)

    def close(self):
        cols = ["pair_id", "dataset_id", "patient_id", "source_sample", "target_sample", "source_h5ad", "target_h5ad", "eligible"]
        pd.DataFrame(self.man)[cols].to_csv(self.root / "manifest.csv", index=False)
        pd.DataFrame(self.units).to_csv(self.root / "prep" / "units.tsv", sep="\t", index=False)
        print("units", len(self.units), "->", self.root, flush=True)


def tar_text(tf: tarfile.TarFile, name: str):
    raw = tf.extractfile(name)
    return io.TextIOWrapper(gzip.GzipFile(fileobj=raw) if name.endswith(".gz") else raw)


# ----------------------------------------------------------------------------------------------------------------- data

def larry_invivo(d: Path, W: Writer):
    import anndata as ad
    pre = d / "larry_invivo_gse140802" / "GSM4185643_stateFate_inVivo_"
    meta = pd.read_csv(f"{pre}metadata.txt.gz", sep="\t")
    genes = pd.read_csv(f"{pre}gene_names.txt.gz", header=None)[0].astype(str).to_numpy()
    with gzip.open(f"{pre}clone_matrix.mtx.gz", "rt") as fh:
        clones = read_mtx(fh)
    with gzip.open(f"{pre}normed_counts.mtx.gz", "rt") as fh:
        X = read_mtx(fh)
    print("counts", X.shape, X.nnz, "clones", clones.shape, "metadata", meta.shape, flush=True)
    if clones.shape[0] != X.shape[0] and clones.shape[1] == X.shape[0]:
        clones = clones.T.tocsr()  # the in vivo clone matrix ships as clones x cells
    assert X.shape[0] == len(meta) == clones.shape[0]
    per_cell = np.diff((clones > 0).astype(np.int8).tocsr().indptr)
    print("clones per cell:", pd.Series(per_cell).value_counts().sort_index().to_dict(), flush=True)
    assert per_cell.max() <= 1, "a cell carries more than one clone"
    X.sort_indices()
    mins = np.array([X.data[X.indptr[i]:X.indptr[i + 1]].min() if X.indptr[i + 1] > X.indptr[i] else np.nan for i in range(X.shape[0])])
    R = X.data.astype(np.float64) * np.repeat(1 / mins, np.diff(X.indptr))
    share = np.add.reduceat(np.abs(R - np.round(R)) <= 0.02, X.indptr[:-1]) / np.maximum(np.diff(X.indptr), 1)
    summ = {"cells": X.shape[0], "integer_share_median": float(np.median(share)), "integer_share_p05": float(np.percentile(share, 5)),
            "cells_integer_share_ge_0.99": float(np.mean(share >= 0.99))}
    pd.Series(summ).to_csv(W.root / "prep" / "integer_recovery.tsv", sep="\t", header=False); print(summ, flush=True)
    if summ["integer_share_median"] < 0.99:
        raise SystemExit("integer recovery failed; not run (PLAN_TACC.md: integer counts only)")
    Rc = sp.csr_matrix((np.round(R).astype(np.float32), X.indices, X.indptr), shape=X.shape)
    cl = np.full(len(meta), -1); r_, c_ = clones.nonzero(); cl[r_] = c_
    o = pd.DataFrame({"time": meta["Time point"].astype(int).to_numpy(),
                      "population": np.where(meta["Starting population"].astype(str).str.contains("Sca1", regex=False), "LSK", "LK"),
                      "cell_type": meta["Cell type annotation"].astype(str).to_numpy(), "recipient": meta["Recipient mouse"].astype(str).to_numpy(),
                      "library": meta["Library"].astype(str).to_numpy(), "clone": cl}, index=[f"c{i}" for i in range(len(meta))])
    pd.crosstab([o.population, o.time], o.cell_type).to_csv(W.root / "prep" / "population_time_celltype.tsv", sep="\t")
    A = ad.AnnData(Rc, obs=o, var=pd.DataFrame(index=ad.utils.make_index_unique(pd.Index(genes))))
    for pop in ("LSK", "LK"):
        for day in (9, 16):
            src = np.flatnonzero(((o.population == pop) & (o.time == 2)).to_numpy())
            tgt = np.flatnonzero(((o.population == pop) & (o.time == day)).to_numpy())
            shared = (set(o.clone.to_numpy()[src]) & set(o.clone.to_numpy()[tgt])) - {-1}
            pri = np.isin(o.clone.to_numpy(), list(shared))
            W.unit(f"LARRYvivo_{pop}_d2_to_d{day}", pop, A, src, tgt, ["time", "population", "cell_type", "recipient", "library", "clone"],
                   priority=pri, meta={"shared_clones": len(shared)})


def kang(d: Path, W: Writer):
    import anndata as ad
    import scipy.io
    k = d / "kang_gse96583"
    genes = pd.read_csv(k / "GSE96583_batch2.genes.tsv.gz", sep="\t", header=None)
    sym = pd.Index(ad.utils.make_index_unique(pd.Index(genes[1].astype(str).to_numpy())), name=None)
    lab = pd.read_csv(k / "GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
    with tarfile.open(k / "GSE96583_RAW.tar") as tf:
        def lane(gsm, mtx):
            X = scipy.io.mmread(gzip.GzipFile(fileobj=tf.extractfile(mtx))).tocsr().T.tocsr()
            bc = pd.read_csv(tf.extractfile(f"{gsm}_barcodes.tsv.gz"), compression="gzip", header=None)[0].astype(str).to_numpy()
            return X, bc
        Xc, bc_c = lane("GSM2560248", "GSM2560248_2.1.mtx.gz")
        Xs, bc_s = lane("GSM2560249", "GSM2560249_2.2.mtx.gz")
    key_s = np.array([b.replace("-1", "-11") if b in set(bc_c) else b for b in bc_s])   # as in 02_prepare_kang.py
    parts = []
    for name, X, keys in (("ctrl", Xc, bc_c), ("stim", Xs, key_s)):
        L = lab.reindex(keys)
        ok = (L.stim == name).to_numpy() & (L.multiplets == "singlet").to_numpy() & L.cell.notna().to_numpy()
        parts.append(ad.AnnData(X=sp.csr_matrix(X[ok], dtype=np.float32),
                                obs=pd.DataFrame({"cell_type": L.cell[ok].astype(str).to_numpy(), "donor": L.ind[ok].astype(str).to_numpy(),
                                                  "condition": name}, index=[f"{name}_{x}" for x in keys[ok]]), var=pd.DataFrame(index=sym)))
    A = ad.concat(parts); A.var_names = sym
    pd.crosstab([A.obs.condition, A.obs.donor], A.obs.cell_type).to_csv(W.root / "prep" / "condition_donor_celltype.tsv", sep="\t")
    c, s = (A.obs.condition == "ctrl").to_numpy(), (A.obs.condition == "stim").to_numpy()
    cols = ["cell_type", "donor", "condition"]
    W.unit("Kang_ctrl_to_stim_pooled", "pooled_8_donors", A, np.flatnonzero(c), np.flatnonzero(s), cols)
    for dn in sorted(A.obs.donor.unique()):
        m = (A.obs.donor == dn).to_numpy()
        W.unit(f"Kang_ctrl_to_stim_donor{dn}", f"donor{dn}", A, np.flatnonzero(c & m), np.flatnonzero(s & m), cols)


def schiebinger(d: Path, W: Writer):
    import anndata as ad
    A = ad.read_h5ad(d / "schiebinger_pertpy" / "schiebinger_2019_16day.h5ad")
    gm = pd.read_csv(HERE / "schiebinger_gsm_map.tsv", sep="\t", dtype=str).set_index("GSM")
    A.obs["condition"] = gm.condition.reindex(A.obs.GSM.astype(str)).to_numpy()
    A.obs["stage"] = gm.stage.reindex(A.obs.GSM.astype(str)).to_numpy()
    assert A.obs.condition.notna().all(), "GSM without a condition"
    pd.crosstab(A.obs.stage, A.obs.condition).to_csv(W.root / "prep" / "stage_condition.tsv", sep="\t")
    pd.crosstab(A.obs.GSM, [A.obs.age, A.obs.perturbation.astype(str)]).to_csv(W.root / "prep" / "gsm_age_pertpy_label.tsv", sep="\t")
    o = A.obs
    def cells(stage, cond):
        return np.flatnonzero(((o.stage == stage) & (o.condition == cond)).to_numpy())
    chains = {"dox": [("D0", "dox"), ("D2", "dox"), ("D4", "dox"), ("D6", "dox"), ("D8", "dox")],
              "2i": [("D8", "dox"), ("D9", "2i"), ("D10", "2i"), ("D11", "2i"), ("D12", "2i"), ("D16", "2i"), ("iPSC", "2i")],
              "serum": [("D8", "dox"), ("D10", "serum"), ("D12", "serum"), ("D16", "serum"), ("iPSC", "serum")]}
    for ch, steps in chains.items():
        for (s0, c0), (s1, c1) in zip(steps[:-1], steps[1:]):
            W.unit(f"Schiebinger_{ch}_{s0}_to_{s1}", ch, A, cells(s0, c0), cells(s1, c1), ["GSM", "stage", "condition"])


def sciplex2(d: Path, W: Writer):
    import anndata as ad
    A = ad.read_h5ad(d / "sciplex2_pertpy" / "srivatsan_2020_sciplex2.h5ad")
    o = A.obs
    o["drug"] = o.perturbation.astype(str); o["dose"] = o.dose_value.astype(str)
    pd.crosstab(o.drug, o.dose).to_csv(W.root / "prep" / "drug_dose.tsv", sep="\t")
    for drug in ("Dex", "Nutlin", "SAHA", "BMS"):
        veh = np.flatnonzero(((o.drug == drug) & (o.dose == "0")).to_numpy())
        for dose in ("0.1", "0.5", "1", "5", "10", "50", "100"):
            tgt = np.flatnonzero(((o.drug == drug) & (o.dose == dose)).to_numpy())
            if len(veh) and len(tgt):
                W.unit(f"sciPlex2_{drug}_0_to_{dose}uM", drug, A, veh, tgt, ["drug", "dose", "well"])
            else:
                print("no cells", drug, dose, len(veh), len(tgt), flush=True)


BINS = ["< 100", "100-130", "130-170", "170-210", "210-270", "270-330", "330-390", "390-450", "450-510", "510-580", "580-650", "> 650"]


def celegans(d: Path, W: Writer):
    import anndata as ad
    c = d / "celegans_gse126954"
    ann = pd.read_csv(c / "GSE126954_cell_annotation.csv.gz")
    genes = pd.read_csv(c / "GSE126954_gene_annotation.csv.gz")
    with gzip.open(c / "GSE126954_gene_by_cell_count_matrix.txt.gz", "rt") as fh:
        X = read_mtx(fh, transpose=True)                      # cells x genes
    print("counts", X.shape, X.nnz, flush=True)
    assert X.shape == (len(ann), len(genes)) and is_integer(X)
    o = pd.DataFrame({"time_bin": ann["embryo.time.bin"].astype(str).to_numpy(), "embryo_time": ann["embryo.time"].to_numpy(),
                      "lineage": ann["lineage"].fillna("").astype(str).to_numpy(), "cell_type": ann["cell.type"].fillna("").astype(str).to_numpy(),
                      "cell_subtype": ann["cell.subtype"].fillna("").astype(str).to_numpy(), "batch": ann["batch"].astype(str).to_numpy(),
                      "qc": ann["passed_initial_QC_or_later_whitelisted"].to_numpy()}, index=ann["cell"].astype(str).to_numpy())
    A = ad.AnnData(X, obs=o, var=pd.DataFrame(index=ad.utils.make_index_unique(pd.Index(genes.gene_short_name.astype(str)))))
    qc = o.qc.to_numpy(bool); pri = (o.lineage != "").to_numpy()
    pd.crosstab(o.time_bin[qc], pri[qc]).reindex(BINS).to_csv(W.root / "prep" / "bin_lineage_annotated.tsv", sep="\t")
    for b0, b1 in zip(BINS[:-1], BINS[1:]):
        src = np.flatnonzero(qc & (o.time_bin == b0).to_numpy()); tgt = np.flatnonzero(qc & (o.time_bin == b1).to_numpy())
        u = "Celegans_" + b0.replace("< ", "lt").replace("> ", "gt") + "_to_" + b1.replace("< ", "lt").replace("> ", "gt")
        W.unit(u, "embryo", A, src, tgt, ["time_bin", "embryo_time", "lineage", "cell_type", "cell_subtype", "batch"], priority=pri)


def pancreas(d: Path, W: Writer):
    import anndata as ad
    p = d / "pancreas_gse132188"
    H = ad.read_h5ad(p / "GSE132188_adata.h5ad.h5", backed="r")
    ho = H.obs[["day", "clusters_fig6_broad_final", "clusters_fig6_fine_final", "proliferation"]].copy()
    suf = pd.Series(H.obs_names).str.rsplit("-", n=1).str[1].to_numpy()
    chk = pd.crosstab(suf, ho.day.astype(str).to_numpy()); chk.to_csv(W.root / "prep" / "obs_suffix_vs_day.tsv", sep="\t")
    assert (chk.gt(0).sum(1) == 1).all(), "an obs-name suffix spans more than one day"
    stage_files = {"12.5": "GSM3852752_E12_5_counts.tar.gz", "13.5": "GSM3852753_E13_5_counts.tar.gz",
                   "14.5": "GSM3852754_E14_5_counts.tar.gz", "15.5": "GSM3852755_E15_5_counts.tar.gz"}
    parts, genes0, rep = [], None, []
    with tarfile.open(p / "GSE132188_RAW.tar") as outer:
        for day, fn in stage_files.items():
            with tarfile.open(fileobj=io.BytesIO(outer.extractfile(fn).read()), mode="r:gz") as inner:
                names = {Path(m.name).name: m.name for m in inner.getmembers() if m.isfile()}
                X = read_mtx(io.TextIOWrapper(inner.extractfile(names["matrix.mtx"])), transpose=True)
                bc = pd.read_csv(inner.extractfile(names["barcodes.tsv"]), header=None)[0].astype(str).to_numpy()
                g = pd.read_csv(inner.extractfile(names["genes.tsv"]), sep="\t", header=None)
            genes0 = g if genes0 is None else genes0
            assert (g[0].to_numpy() == genes0[0].to_numpy()).all(), "gene order differs between stages"
            sfx = chk.index[chk[day] > 0][0]
            want = ho.index[(suf == sfx)]
            raw_key = pd.Index(bc)
            pos = raw_key.get_indexer(pd.Index(want).str.rsplit("-", n=1).str[0])
            rep.append({"day": day, "obs_cells": len(want), "raw_barcodes": len(bc), "matched": int((pos >= 0).sum())})
            assert (pos >= 0).all(), f"E{day}: {(pos < 0).sum()} h5ad cells not in the raw barcodes"
            parts.append(ad.AnnData(X[pos], obs=ho.loc[want], var=pd.DataFrame(index=ad.utils.make_index_unique(pd.Index(g[1].astype(str).to_numpy(), name=None)))))
    pd.DataFrame(rep).to_csv(W.root / "prep" / "barcode_match.tsv", sep="\t", index=False); print(rep, flush=True)
    A = ad.concat(parts); A.var_names = parts[0].var_names
    tot = np.asarray(A.X.sum(1)).ravel()
    pd.DataFrame({"raw_total": tot, "authors_n_counts": H.obs.n_counts.reindex(A.obs_names).to_numpy()}).describe().to_csv(
        W.root / "prep" / "raw_vs_authors_total.tsv", sep="\t")
    A.obs["cell_type"] = A.obs.clusters_fig6_broad_final.astype(str); A.obs["cell_type_fine"] = A.obs.clusters_fig6_fine_final.astype(str)
    A.obs["day"] = A.obs.day.astype(str)
    days = ["12.5", "13.5", "14.5", "15.5"]
    for d0, d1 in zip(days[:-1], days[1:]):
        W.unit(f"Pancreas_E{d0}_to_E{d1}", "pancreas", A, np.flatnonzero((A.obs.day == d0).to_numpy()), np.flatnonzero((A.obs.day == d1).to_numpy()),
               ["day", "cell_type", "cell_type_fine", "proliferation"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["larry_invivo", "kang", "schiebinger", "sciplex2", "celegans", "pancreas"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    W = Writer(a.out, a.dataset)
    globals()[a.dataset](a.data, W)
    W.close()


if __name__ == "__main__":
    main()

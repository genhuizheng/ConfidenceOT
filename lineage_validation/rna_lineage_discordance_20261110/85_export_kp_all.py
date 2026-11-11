"""PLAN_v2.md extension E1 / E2 on TACC: export, from the authors' processed KP-Tracer object, what the all-cell
state-lineage analysis and the harmonization checks need. Small outputs only, so the rest runs locally.

    python lineage_validation/rna_lineage_discordance_20261110/85_export_kp_all.py \
        --h5ad /scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data/expression/adata_processed.combined.h5ad \
        --out OUT

Readers are kp_lineage/01_prepare_kptracer_pairs.py's (old-anndata categoricals, integer raw/X).
Writes OUT/:
  obs_all.tsv.gz     every cell of the object: cell, tumor, met_family, mouse, genotype, is_normal, cluster_name,
                     sub_tumor, lane, tumour_kind (README rule), total_counts and detected_genes (raw/X)
  analysed cells     is_normal False, cluster_name present, tumour_kind primary / metastasis (the allele-table
                     criterion of E2 is applied locally)
  pca_logcpm30.npz   analysed cells: logCPM (1e4, log1p) -> 2,000 highest-variance genes -> PCA 30, seed 20261111
                     (rb_common.pca_coordinates, no gene scaling)
  markers.tsv.gz     analysed cells: scanpy Wilcoxon of each cluster_name vs all others on logCPM, top 200 per label
                     (names, scores, logfoldchanges, pvals_adj)
  label_means.tsv.gz mean logCPM per cluster_name over the union of every label's top-50 genes
  group_means.tsv.gz the same per mouse x tumor x cluster_name (groups with >= 20 cells), for per-mouse checks
Refuses an existing OUT.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse

REPO = Path(__file__).resolve().parents[2]
SEED, TOP_KEEP, TOP_UNION, MIN_GROUP = 20261111, 200, 50, 20


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists")
    a.out.mkdir(parents=True)
    prep = load("prep", REPO / "kp_lineage" / "01_prepare_kptracer_pairs.py")
    sys.path.insert(0, str(REPO / "realdata_benchmark")); sys.path.insert(0, str(REPO / "src"))
    import rb_common as rb
    with h5py.File(a.h5ad, "r") as f:
        cells = prep.read_index(f["obs"])
        obs = pd.DataFrame({"cell": cells})
        for col, name in prep.OBS_COLUMNS.items():
            obs[name] = prep.read_obs_column(f["obs"], col) if col in f["obs"] else "nan"
        X = prep.read_csr(f["raw/X"])
        genes = prep.read_index(f["raw/var"])
    obs["tumour_kind"] = [prep.tumour_kind(str(t)) for t in obs.tumor]
    obs["total_counts"] = np.asarray(X.sum(1)).ravel()
    obs["detected_genes"] = np.diff(X.indptr)
    obs.to_csv(a.out / "obs_all.tsv.gz", sep="\t", index=False, compression="gzip")
    norm = obs.is_normal.astype(str).str.lower().isin(["true", "1", "1.0"])
    lab = obs.cluster_name.astype(str)
    keep = (~norm & ~lab.isin(["nan", "", "None"]) & obs.tumour_kind.isin(["primary", "metastasis"])).to_numpy()
    print(f"{len(obs)} cells; analysed {keep.sum()} (normal {norm.sum()}, no label {lab.isin(['nan', '', 'None']).sum()})", flush=True)
    Xa = X[keep].astype(np.float32).tocsr(); ids = obs.cell.to_numpy()[keep]; la = lab.to_numpy()[keep]
    tot = np.asarray(Xa.sum(1)).ravel()
    L = sparse.diags(1e4 / np.maximum(tot, 1)) @ Xa; L.data = np.log1p(L.data); L = L.tocsr()
    P, sel = rb.pca_coordinates(L, 2000, 30, False, False, SEED)
    np.savez_compressed(a.out / "pca_logcpm30.npz", cell=ids.astype(str), label=la.astype(str), X=P.astype(np.float32),
                        hvg=np.asarray(genes)[sel].astype(str))
    import anndata as ad
    import scanpy as sc
    A = ad.AnnData(L, obs=pd.DataFrame({"label": pd.Categorical(la)}, index=ids), var=pd.DataFrame(index=np.asarray(genes).astype(str)))
    A.uns["log1p"] = {"base": None}
    sc.tl.rank_genes_groups(A, "label", method="wilcoxon", n_genes=TOP_KEEP)
    mk = []
    for g in A.obs.label.cat.categories:
        d = sc.get.rank_genes_groups_df(A, g).head(TOP_KEEP); d.insert(0, "label", g); d["rank"] = np.arange(len(d)); mk.append(d)
    mk = pd.concat(mk); mk.to_csv(a.out / "markers.tsv.gz", sep="\t", index=False, compression="gzip")
    union = sorted(set(mk[mk["rank"] < TOP_UNION].names))
    gi = pd.Index(A.var_names).get_indexer(union)
    Lu = L[:, gi].toarray()
    lm = pd.DataFrame(Lu).groupby(la).mean(); lm.columns = union
    lm.to_csv(a.out / "label_means.tsv.gz", sep="\t", compression="gzip")
    o = obs[keep].reset_index(drop=True)
    key = o.mouse.astype(str) + "|" + o.tumor.astype(str) + "|" + o.cluster_name.astype(str)
    vc = key.value_counts(); ok = key.isin(vc[vc >= MIN_GROUP].index).to_numpy()
    gm = pd.DataFrame(Lu[ok]).groupby(key[ok].to_numpy()).mean(); gm.columns = union
    gm.index.name = "group"; gm.to_csv(a.out / "group_means.tsv.gz", sep="\t", compression="gzip")
    print(f"markers for {A.obs.label.nunique()} labels; union {len(union)} genes; {len(gm)} groups", flush=True)


if __name__ == "__main__":
    main()

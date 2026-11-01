"""DOWNSAMPLING_INFORMATION_LOSS.tsv: what equalising LTL331 to 2,817 UMIs removes, per time point and cell type.

    python realdata_benchmark/02_downsampling_information_loss.py --run-root RUN_ROOT --converted CONVERTED --out OUT_DIR

For every time point, the original counts of the analysed cells (inputs/rawqc) are equalised again with
confidenceot.equalise_depth (the function 27_downsample_counts.py calls) to the run's target, 2,817, with seeds 1-5,
beside the frozen equalised file (inputs/ds; its seed belongs to 27_ and is not reused here). Per cell:
  detected genes before / after and the fraction lost;
  top-256 membership of the rank_no_median encoding (stable argsort, the production tie rule) after equalisation
    against the original counts' top-256 (Jaccard), and seed against seed (seed 1 against seeds 2-5: stability);
  the same for the cell's top-256 by value with ties broken at random (tie-aware membership: how much of the change
    is the tie rule rather than the reads);
  Pearson correlation of log1p(1e4 * counts / total) between the original and the equalised profile over the genes
    detected in the original.
Writes OUT/DOWNSAMPLING_INFORMATION_LOSS.tsv (per time point x coarse_celltype: medians and quartiles) and
OUT/source/downsampling_cells.tsv.gz (per cell).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from confidenceot import equalise_depth

TOP = 256
TARGET = 2817
SEEDS = [1, 2, 3, 4, 5]


def top_sets(X: sp.csr_matrix, random_ties: np.random.Generator | None = None) -> sp.csr_matrix:
    """Binary cells x genes: the cell's top TOP genes by value. Ties keep column order (production) or break at random."""
    rows, cols = [], []
    for i in range(X.shape[0]):
        s, e = X.indptr[i], X.indptr[i + 1]
        d, g = X.data[s:e], X.indices[s:e]
        if random_ties is None:
            order = np.argsort(-d, kind="stable")
        else:
            order = np.lexsort((random_ties.random(d.size), -d))
        k = order[:TOP]
        rows.append(np.full(k.size, i)); cols.append(g[k])
    r, c = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(r.size), (r, c)), shape=X.shape)


def jaccard(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    inter = np.asarray(A.multiply(B).sum(1)).ravel()
    union = np.asarray(A.sum(1)).ravel() + np.asarray(B.sum(1)).ravel() - inter
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def lognorm(X: sp.csr_matrix) -> sp.csr_matrix:
    t = np.asarray(X.sum(1)).ravel()
    Y = sp.diags(1e4 / np.maximum(t, 1)) @ X
    Y = sp.csr_matrix(Y); Y.data = np.log1p(Y.data)
    return Y


def rowwise_pearson(A: sp.csr_matrix, B: sp.csr_matrix) -> np.ndarray:
    """Per row, Pearson correlation of A and B over A's nonzero columns (B is zero where it has no entry)."""
    A = sp.csr_matrix(A); B = sp.csr_matrix(B); A.sort_indices(); B.sort_indices()
    out = np.full(A.shape[0], np.nan)
    for i in range(A.shape[0]):
        g = A.indices[A.indptr[i]:A.indptr[i + 1]]; a = A.data[A.indptr[i]:A.indptr[i + 1]]
        bi = B.indices[B.indptr[i]:B.indptr[i + 1]]; bd = B.data[B.indptr[i]:B.indptr[i + 1]]
        pos = np.searchsorted(bi, g); ok = pos < bi.size
        ok[ok] = bi[pos[ok]] == g[ok]
        b = np.zeros(g.size); b[ok] = bd[pos[ok]]
        if a.size > 2 and a.std() > 0 and b.std() > 0:
            out[i] = np.corrcoef(a, b)[0, 1]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    target = a.out / "DOWNSAMPLING_INFORMATION_LOSS.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    (a.out / "source").mkdir(parents=True, exist_ok=True)
    ctype = pd.concat([ad.read_h5ad(f, backed="r").obs["coarse_celltype"].astype(str) for f in sorted((a.converted / "h5ad").glob("*.h5ad"))])
    cells = []
    for f in sorted((a.run_root / "inputs" / "rawqc" / "h5ad").glob("*.h5ad")):
        sample = f.stem
        raw = ad.read_h5ad(f)
        frozen = ad.read_h5ad(a.run_root / "inputs" / "ds" / "h5ad" / f"LTL331__{sample}.h5ad")
        # 27_ drops the repeated-symbol copies (make_index_unique '-1' names): 29,594 of 29,622 genes, the set OT used
        if list(frozen.obs_names) != list(raw.obs_names) or not set(frozen.var_names) <= set(raw.var_names):
            sys.exit(f"{sample}: equalised and original files differ in cells, or the equalised file has genes the original lacks")
        raw = raw[:, frozen.var_names].copy()
        X = sp.csr_matrix(raw.X).astype(np.int64); X.eliminate_zeros()
        F = sp.csr_matrix(frozen.X).astype(np.int64); F.eliminate_zeros()
        eq = {"frozen": F}
        for seed in SEEDS:
            E = sp.csr_matrix(equalise_depth(X, rng=np.random.default_rng(seed), target=TARGET)); E.eliminate_zeros(); eq[seed] = E
        top_raw = top_sets(X); top_raw_random = top_sets(X, np.random.default_rng(0))
        tops = {k: top_sets(v) for k, v in eq.items()}
        tops_random = {k: top_sets(v, np.random.default_rng(100 + (k if isinstance(k, int) else 0))) for k, v in eq.items()}
        detected = np.diff(X.indptr)
        L, LF = lognorm(X), lognorm(F)
        d = pd.DataFrame({"sample_id": sample, "observation_id": raw.obs_names.astype(str),
                          "coarse_celltype": ctype.loc[raw.obs_names.astype(str)].to_numpy(),
                          "original_depth": np.asarray(X.sum(1)).ravel(), "original_detected": detected,
                          "frozen_detected": np.diff(F.indptr),
                          "untouched": np.asarray(X.sum(1)).ravel() <= TARGET})
        d["frozen_fraction_detected_lost"] = 1 - d.frozen_detected / d.original_detected
        d["seed_detected_mean"] = np.mean([np.diff(eq[s].indptr) for s in SEEDS], 0)
        d["top256_jaccard_frozen_vs_original"] = jaccard(tops["frozen"], top_raw)
        d["top256_jaccard_seed_vs_original_mean"] = np.mean([jaccard(tops[s], top_raw) for s in SEEDS], 0)
        d["top256_jaccard_seed1_vs_other_seeds_mean"] = np.mean([jaccard(tops[1], tops[s]) for s in SEEDS[1:]], 0)
        d["top256_jaccard_randomties_seed_vs_original_mean"] = np.mean([jaccard(tops_random[s], top_raw_random) for s in SEEDS], 0)
        d["top256_jaccard_randomties_seed1_vs_other_seeds_mean"] = np.mean([jaccard(tops_random[1], tops_random[s]) for s in SEEDS[1:]], 0)
        d["top256_jaccard_original_stable_vs_randomties"] = jaccard(top_raw, top_raw_random)
        d["top256_jaccard_frozen_stable_vs_randomties"] = jaccard(tops["frozen"], tops_random["frozen"])
        d["lognorm_pearson_frozen_vs_original"] = rowwise_pearson(L, LF)
        cells.append(d)
        print(sample, len(d), "cells", flush=True)
    cells = pd.concat(cells, ignore_index=True)
    cells.to_csv(a.out / "source" / "downsampling_cells.tsv.gz", sep="\t", index=False, compression="gzip")
    metrics = [c for c in cells.columns if c not in ("sample_id", "observation_id", "coarse_celltype", "untouched")]
    rows = []
    for (sample, t), g in list(cells.groupby(["sample_id", "coarse_celltype"])) + [((s, "all"), g) for s, g in cells.groupby("sample_id")]:
        r = {"sample_id": sample, "coarse_celltype": t, "cells": len(g), "untouched_fraction": g.untouched.mean()}
        for m in metrics:
            r[f"{m}_median"] = g[m].median(); r[f"{m}_q25"] = g[m].quantile(0.25); r[f"{m}_q75"] = g[m].quantile(0.75)
        rows.append(r)
    pd.DataFrame(rows).sort_values(["sample_id", "coarse_celltype"]).to_csv(target, sep="\t", index=False)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
